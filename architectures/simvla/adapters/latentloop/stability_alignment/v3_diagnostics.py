"""Bounded immutable-v2 diagnostics required before V3 training."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    _drop_unused_vlm,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    freeze_module,
    load_frozen_simvla,
    move_batch,
)
from architectures.simvla.adapters.latentloop.stability_alignment.checkpoint import (
    load_checkpoint,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment.model import (
    StabilityAlignedModules,
    configure_condition_only_stage,
    load_warm_start,
)
from architectures.simvla.adapters.latentloop.stability_alignment.objectives import (
    LOSS_NAMES,
    condition_paths,
    first_r_per_sequence,
    masked_nrms,
)
from architectures.simvla.adapters.latentloop.stability_alignment.trainer import (
    _assert_parent_contract,
    _forward,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_contracts import (
    evaluate_v3_scientific_gate,
    select_v2_checkpoint,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_data import (
    V3StabilityExactTeacherDataset,
    collate_v3_sequences,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_runtime import (
    DEFAULT_CACHE,
    DEFAULT_GENERATION_30K,
    DEFAULT_NORM,
    cleanup_distributed,
    condition_updater_parameters,
    configure_determinism,
    cosine,
    decode_full_actions,
    distributed_pair,
    flatten_gradients,
    source_locked_indices,
    summary,
    v3_source_lock,
    zero_code_ng3_actions,
)


def _mean(values: Sequence[float]) -> float:
    return float(np.asarray(tuple(values), dtype=np.float64).mean())


def _action_error(prediction: Tensor, target: Tensor) -> float:
    return float(first_r_per_sequence(prediction, target).item())


def _transition_row(
    predictions: Sequence[Tensor],
    targets: Sequence[Tensor],
    anchor_target: Tensor,
    *,
    age_index: int,
) -> dict[str, int]:
    prediction = predictions[int(age_index)][:, :5, 6].float()
    target = targets[int(age_index)][:, :5, 6].float()
    sign_mismatch = (prediction >= 0.0) != (target >= 0.0)
    if int(age_index) == 0:
        previous_prediction = anchor_target[:, 4, 6].float()
        previous_target = previous_prediction
    else:
        previous_prediction = predictions[int(age_index) - 1][:, 4, 6].float()
        previous_target = targets[int(age_index) - 1][:, 4, 6].float()
    predicted_before = torch.cat(
        (previous_prediction[:, None], prediction[:, :-1]), dim=1
    )
    target_before = torch.cat((previous_target[:, None], target[:, :-1]), dim=1)
    predicted_switch = (predicted_before >= 0.0) != (prediction >= 0.0)
    target_switch = (target_before >= 0.0) != (target >= 0.0)
    switch_mismatch = predicted_switch != target_switch
    return {
        "sign_mismatch_sequence": int(sign_mismatch.any(dim=1).sum().item()),
        "sign_mismatch_positions": int(sign_mismatch.sum().item()),
        "switch_mismatch_sequence": int(switch_mismatch.any(dim=1).sum().item()),
        "switch_mismatch_positions": int(switch_mismatch.sum().item()),
    }


def _adapter_row(
    *,
    modules: StabilityAlignedModules,
    frozen_model: Any,
    action_adapter: Any,
    batch: Mapping[str, Any],
    dataset_index: int,
) -> dict[str, Any]:
    paths = condition_paths(modules.condition, batch)
    exact_conditions = tuple(
        batch["teacher_conditions"][:, index] for index in range(3)
    )
    exact_actions = tuple(batch["teacher_actions"][:, index] for index in range(3))
    proprio = tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3))
    noises = tuple(batch["explicit_noises"][:, index] for index in range(3))
    decoded = decode_full_actions(
        action_adapter,
        conditions=(*paths.teacher_forced, *paths.recursive),
        proprio=(*proprio, *proprio),
        noises=(*noises, *noises),
        requires_grad=False,
    )
    teacher_full = decoded[:3]
    recursive_full = decoded[3:]
    ng3 = zero_code_ng3_actions(
        modules=modules,
        frozen_model=frozen_model,
        action_adapter=action_adapter,
        conditions=(*paths.recursive, *exact_conditions),
        proprio=(*proprio, *proprio),
        noises=(*noises, *noises),
        valid_mask=batch["valid_mask"],
        optimizer_step=int(dataset_index),
        requires_grad=False,
    )
    recursive_ng3 = ng3[:3]
    exact_ng3 = ng3[3:]
    row: dict[str, Any] = {
        "dataset_index": int(dataset_index),
        "task_id": int(batch["task_id"].item()),
        "episode_id": str(batch["episode_id"][0]),
        "anchor_query_index": int(batch["anchor_query_index"].item()),
    }
    for index, age in enumerate((1, 2, 3)):
        teacher_error = _action_error(teacher_full[index], exact_actions[index])
        recursive_error = _action_error(recursive_full[index], exact_actions[index])
        row.update(
            {
                f"age{age}_condition_nrms": float(
                    masked_nrms(
                        paths.recursive[index],
                        exact_conditions[index],
                        batch["valid_mask"],
                    ).item()
                ),
                f"age{age}_teacher_first_r": teacher_error,
                f"age{age}_recursive_first_r": recursive_error,
                f"age{age}_recurrence_excess": recursive_error - teacher_error,
                f"age{age}_joint_first_r": _action_error(
                    recursive_ng3[index], exact_actions[index]
                ),
                f"age{age}_exact_ng3_first_r": _action_error(
                    exact_ng3[index], exact_actions[index]
                ),
            }
        )
        transitions = _transition_row(
            recursive_ng3,
            exact_actions,
            batch["anchor_teacher_action"],
            age_index=index,
        )
        row.update(
            {
                f"age{age}_{name}": value for name, value in transitions.items()
            }
        )
    return row


def _evaluate_adapter(
    *,
    modules: StabilityAlignedModules,
    frozen_model: Any,
    action_adapter: Any,
    dataset: V3StabilityExactTeacherDataset,
    rank: int,
    world: int,
    device: torch.device,
    description: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_index in tqdm(
        range(rank, len(dataset), world),
        desc=f"{description} rank{rank}",
        dynamic_ncols=True,
    ):
        batch = move_batch(collate_v3_sequences([dataset[dataset_index]]), device)
        with torch.no_grad():
            rows.append(
                _adapter_row(
                    modules=modules,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    dataset_index=dataset_index,
                )
            )
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _merge_rank_rows(output: Path, stem: str, world: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for rank in range(int(world)):
        merged.extend(load_json(output / "shards" / f"rank_{rank}_{stem}.json"))
    merged.sort(key=lambda row: int(row["dataset_index"]))
    return merged


def _comparison(
    parent: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    branch: str,
    split: str,
    optimizer_step: int,
    checkpoint: str,
) -> dict[str, Any]:
    if len(parent) != len(candidate):
        raise RuntimeError("parent/candidate evaluation rows differ in length")
    parent_by_index = {int(row["dataset_index"]): row for row in parent}
    candidate_by_index = {int(row["dataset_index"]): row for row in candidate}
    if set(parent_by_index) != set(candidate_by_index):
        raise RuntimeError("parent/candidate dataset indices differ")

    def values(source: Mapping[int, Mapping[str, Any]], field: str) -> list[float]:
        return [float(source[index][field]) for index in sorted(source)]

    epsilon = 1e-12
    parent_age1 = values(parent_by_index, "age1_joint_first_r")
    candidate_age1 = values(candidate_by_index, "age1_joint_first_r")
    parent_age2_excess = values(parent_by_index, "age2_recurrence_excess")
    candidate_age2_excess = values(candidate_by_index, "age2_recurrence_excess")
    parent_age3_excess = values(parent_by_index, "age3_recurrence_excess")
    candidate_age3_excess = values(candidate_by_index, "age3_recurrence_excess")
    parent_age3_recursive = values(parent_by_index, "age3_recursive_first_r")
    candidate_age3_recursive = values(candidate_by_index, "age3_recursive_first_r")
    parent_exact = [
        sum(float(parent_by_index[index][f"age{age}_exact_ng3_first_r"]) for age in (1, 2, 3))
        / 3.0
        for index in sorted(parent_by_index)
    ]
    candidate_exact = [
        sum(float(candidate_by_index[index][f"age{age}_exact_ng3_first_r"]) for age in (1, 2, 3))
        / 3.0
        for index in sorted(candidate_by_index)
    ]
    parent_mismatch_sequences = sum(
        int(parent_by_index[index]["age3_sign_mismatch_sequence"])
        for index in parent_by_index
    )
    candidate_mismatch_sequences = sum(
        int(candidate_by_index[index]["age3_sign_mismatch_sequence"])
        for index in candidate_by_index
    )
    parent_switch_sequences = sum(
        int(parent_by_index[index]["age3_switch_mismatch_sequence"])
        for index in parent_by_index
    )
    candidate_switch_sequences = sum(
        int(candidate_by_index[index]["age3_switch_mismatch_sequence"])
        for index in candidate_by_index
    )
    metrics = {
        "age2_recurrence_improvement": (
            _mean(parent_age2_excess) - _mean(candidate_age2_excess)
        )
        / max(abs(_mean(parent_age2_excess)), epsilon),
        "age3_recurrence_improvement": (
            _mean(parent_age3_excess) - _mean(candidate_age3_excess)
        )
        / max(abs(_mean(parent_age3_excess)), epsilon),
        "age3_first_r_p95_ratio": float(np.quantile(candidate_age3_recursive, 0.95))
        / max(float(np.quantile(parent_age3_recursive, 0.95)), epsilon),
        "age3_first_r_p99_ratio": float(np.quantile(candidate_age3_recursive, 0.99))
        / max(float(np.quantile(parent_age3_recursive, 0.99)), epsilon),
        "age1_first_r_ratio_to_parent": _mean(candidate_age1)
        / max(_mean(parent_age1), epsilon),
        "exact_ng3_ratio_to_parent": _mean(candidate_exact)
        / max(_mean(parent_exact), epsilon),
        "parent_age3_mismatch_sequences": int(parent_mismatch_sequences),
        "candidate_age3_mismatch_sequences": int(candidate_mismatch_sequences),
        "parent_age3_mismatch_positions": int(
            sum(int(row["age3_sign_mismatch_positions"]) for row in parent)
        ),
        "candidate_age3_mismatch_positions": int(
            sum(int(row["age3_sign_mismatch_positions"]) for row in candidate)
        ),
        "parent_age3_switch_mismatch_sequences": int(parent_switch_sequences),
        "candidate_age3_switch_mismatch_sequences": int(candidate_switch_sequences),
        "parent_age3_switch_mismatch_positions": int(
            sum(int(row["age3_switch_mismatch_positions"]) for row in parent)
        ),
        "candidate_age3_switch_mismatch_positions": int(
            sum(int(row["age3_switch_mismatch_positions"]) for row in candidate)
        ),
        "original_simvla_frozen": True,
    }
    return {
        "branch": str(branch),
        "split": str(split),
        "optimizer_step": int(optimizer_step),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sequences": len(parent),
        "candidate_age3_mismatch_sequences": int(candidate_mismatch_sequences),
        "candidate_age3_first_r_p95": float(
            np.quantile(candidate_age3_recursive, 0.95)
        ),
        "candidate_age3_recurrence_excess_mean": _mean(candidate_age3_excess),
        "age1_first_r_ratio_to_parent": metrics[
            "age1_first_r_ratio_to_parent"
        ],
        "parent_age3_first_r_p95": float(np.quantile(parent_age3_recursive, 0.95)),
        "parent_age3_recurrence_excess_mean": _mean(parent_age3_excess),
        "metrics": metrics,
        "distributions": {
            "parent_age1_joint_first_r": summary(parent_age1),
            "candidate_age1_joint_first_r": summary(candidate_age1),
            "parent_age2_recurrence_excess": summary(parent_age2_excess),
            "candidate_age2_recurrence_excess": summary(candidate_age2_excess),
            "parent_age3_recurrence_excess": summary(parent_age3_excess),
            "candidate_age3_recurrence_excess": summary(candidate_age3_excess),
            "parent_age3_recursive_first_r": summary(parent_age3_recursive),
            "candidate_age3_recursive_first_r": summary(candidate_age3_recursive),
        },
    }


def _sweep_csv_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "branch": payload["branch"],
        "split": payload["split"],
        "optimizer_step": payload["optimizer_step"],
        "checkpoint": payload["checkpoint"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "sequences": payload["sequences"],
        "candidate_age3_mismatch_sequences": payload[
            "candidate_age3_mismatch_sequences"
        ],
        "candidate_age3_mismatch_positions": metrics[
            "candidate_age3_mismatch_positions"
        ],
        "candidate_age3_switch_mismatch_sequences": metrics[
            "candidate_age3_switch_mismatch_sequences"
        ],
        "candidate_age3_first_r_p95": payload["candidate_age3_first_r_p95"],
        "candidate_age3_recurrence_excess_mean": payload[
            "candidate_age3_recurrence_excess_mean"
        ],
        "age1_first_r_ratio_to_parent": payload[
            "age1_first_r_ratio_to_parent"
        ],
        "age2_recurrence_improvement": metrics["age2_recurrence_improvement"],
        "age3_recurrence_improvement": metrics["age3_recurrence_improvement"],
        "age3_first_r_p95_ratio": metrics["age3_first_r_p95_ratio"],
        "age3_first_r_p99_ratio": metrics["age3_first_r_p99_ratio"],
        "exact_ng3_ratio_to_parent": metrics["exact_ng3_ratio_to_parent"],
        "evaluation_generation_change_code": "zero_128d",
    }


def command_checkpoint_sweep(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, world, device = distributed_pair()
    try:
        configure_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"refusing existing sweep output: {output}")
            (output / "shards").mkdir(parents=True)
        dist.barrier()
        checkpoints = tuple(
            Path(value).expanduser().resolve()
            for value in str(args.checkpoints).split(",")
            if value
        )
        if len(checkpoints) != 5 or any(not path.is_file() for path in checkpoints):
            raise ValueError("checkpoint sweep requires existing 2/4/6/8/10K files")
        source = v3_source_lock(
            repository=args.repository,
            cache=args.cache,
            condition_parent=args.condition_parent,
            generation_parent=args.generation_parent,
            norm_stats=args.norm_stats,
            checkpoint=args.checkpoint,
            smolvlm_model=args.smolvlm_model,
            split_seed=args.split_seed,
            training_seed=args.seed,
        )
        modules, parent_condition, parent_generation, payloads = load_warm_start(
            condition_checkpoint=args.condition_parent,
            generation_checkpoint=args.generation_parent,
            device=device,
        )
        _assert_parent_contract(payloads)
        parent_modules = StabilityAlignedModules(
            parent_condition, parent_generation
        ).to(device)
        freeze_module(parent_modules)
        freeze_module(modules)
        frozen_model, processor, action_adapter = load_frozen_simvla(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            device=device,
        )
        del processor
        _drop_unused_vlm(frozen_model)

        validation = V3StabilityExactTeacherDataset(
            args.cache, split="checkpoint_validation", split_seed=args.split_seed
        )
        parent_rows = _evaluate_adapter(
            modules=parent_modules,
            frozen_model=frozen_model,
            action_adapter=action_adapter,
            dataset=validation,
            rank=rank,
            world=world,
            device=device,
            description=f"{args.branch} parent validation",
        )
        atomic_write_json(
            output / "shards" / f"rank_{rank}_parent_validation.json", parent_rows
        )
        dist.barrier()
        merged_parent = (
            _merge_rank_rows(output, "parent_validation", world)
            if rank == 0
            else []
        )
        validation_summaries: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            payload = load_checkpoint(checkpoint, modules=modules)
            freeze_module(modules)
            step = int(payload["optimizer_step"])
            rows = _evaluate_adapter(
                modules=modules,
                frozen_model=frozen_model,
                action_adapter=action_adapter,
                dataset=validation,
                rank=rank,
                world=world,
                device=device,
                description=f"{args.branch} {step // 1000}K validation",
            )
            stem = f"candidate_validation_{step:06d}"
            atomic_write_json(output / "shards" / f"rank_{rank}_{stem}.json", rows)
            dist.barrier()
            if rank == 0:
                merged_candidate = _merge_rank_rows(output, stem, world)
                validation_summaries.append(
                    _comparison(
                        merged_parent,
                        merged_candidate,
                        branch=args.branch,
                        split="checkpoint_validation",
                        optimizer_step=step,
                        checkpoint=str(checkpoint),
                    )
                )

        selection_holder: list[Any] = [None]
        if rank == 0:
            selection_holder[0] = select_v2_checkpoint(validation_summaries)
        dist.broadcast_object_list(selection_holder, src=0)
        selection = selection_holder[0]
        if not selection.get("selected_checkpoint"):
            raise RuntimeError(json.dumps(selection, indent=2, sort_keys=True))
        selected_checkpoint = Path(selection["selected_checkpoint"])
        selected_payload = load_checkpoint(selected_checkpoint, modules=modules)
        freeze_module(modules)
        final_dataset = V3StabilityExactTeacherDataset(
            args.cache, split="final_offline", split_seed=args.split_seed
        )
        final_parent_rows = _evaluate_adapter(
            modules=parent_modules,
            frozen_model=frozen_model,
            action_adapter=action_adapter,
            dataset=final_dataset,
            rank=rank,
            world=world,
            device=device,
            description=f"{args.branch} selected parent final",
        )
        final_candidate_rows = _evaluate_adapter(
            modules=modules,
            frozen_model=frozen_model,
            action_adapter=action_adapter,
            dataset=final_dataset,
            rank=rank,
            world=world,
            device=device,
            description=f"{args.branch} selected candidate final",
        )
        atomic_write_json(
            output / "shards" / f"rank_{rank}_parent_final.json", final_parent_rows
        )
        atomic_write_json(
            output / "shards" / f"rank_{rank}_candidate_final.json",
            final_candidate_rows,
        )
        dist.barrier()
        result: dict[str, Any] = {}
        if rank == 0:
            final_summary = _comparison(
                _merge_rank_rows(output, "parent_final", world),
                _merge_rank_rows(output, "candidate_final", world),
                branch=args.branch,
                split="final_offline",
                optimizer_step=int(selected_payload["optimizer_step"]),
                checkpoint=str(selected_checkpoint),
            )
            gate = evaluate_v3_scientific_gate(final_summary["metrics"])
            final_summary["corrected_scientific_gate"] = gate.to_dict()
            all_summaries = [*validation_summaries, final_summary]
            rows = [_sweep_csv_row(payload) for payload in all_summaries]
            _write_rows(output / "stability_v2_checkpoint_sweep.csv", rows)
            result = {
                "schema_version": "simvla_stability_v2_checkpoint_sweep_v1",
                "verdict": "V2_CHECKPOINT_SWEEP_COMPLETE",
                "branch": str(args.branch),
                "source_lock": source,
                "validation_dataset": validation.contract(),
                "final_offline_dataset": final_dataset.contract(),
                "validation_selection": selection,
                "final_offline": final_summary,
                "historical_v2_artifacts_modified": False,
                "evaluation_generation_change_code": "zero_128d",
                "legacy_candidate_change_code_bug_reused": False,
            }
            result["combined_sha256"] = canonical_sha256(result)
            atomic_write_json(output / "checkpoint_sweep_summary.json", result)
            atomic_write_json(output / "source_lock.json", source)
        dist.barrier()
        return result
    finally:
        cleanup_distributed()


def command_gradient_audit(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, world, device = distributed_pair()
    try:
        configure_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"refusing existing gradient output: {output}")
            output.mkdir(parents=True)
        dist.barrier()
        source = v3_source_lock(
            repository=args.repository,
            cache=args.cache,
            condition_parent=args.condition_parent,
            generation_parent=args.generation_parent,
            norm_stats=args.norm_stats,
            checkpoint=args.checkpoint,
            smolvlm_model=args.smolvlm_model,
            split_seed=args.split_seed,
            training_seed=args.seed,
        )
        modules, parent_condition, parent_generation, payloads = load_warm_start(
            condition_checkpoint=args.condition_parent,
            generation_checkpoint=args.generation_parent,
            device=device,
        )
        _assert_parent_contract(payloads)
        checkpoint_payload = load_checkpoint(args.candidate, modules=modules)
        configure_condition_only_stage(modules)
        freeze_module(parent_condition)
        freeze_module(parent_generation)
        frozen_model, processor, action_adapter = load_frozen_simvla(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            device=device,
        )
        del processor
        _drop_unused_vlm(frozen_model)
        dataset = V3StabilityExactTeacherDataset(
            args.cache, split="train", split_seed=args.split_seed
        )
        fixed = source_locked_indices(
            dataset.identities, count=args.audit_batches, seed=args.audit_seed
        )
        local_indices = fixed[rank::world]
        parameters = condition_updater_parameters(modules)
        local_gradient_sums = {
            name: torch.zeros(
                sum(parameter.numel() for parameter in parameters),
                device=device,
                dtype=torch.float32,
            )
            for name in LOSS_NAMES
        }
        local_norm_sums = {name: 0.0 for name in LOSS_NAMES}
        local_raw_sums = {name: 0.0 for name in LOSS_NAMES}
        local_rows: list[dict[str, Any]] = []
        local_age_rows: list[dict[str, Any]] = []
        for sequence_number, dataset_index in enumerate(
            tqdm(local_indices, desc=f"v2 gradient rank{rank}", dynamic_ncols=True)
        ):
            batch = move_batch(collate_v3_sequences([dataset[dataset_index]]), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                raw, diagnostics, _ = _forward(
                    modules=modules,
                    parent_condition=parent_condition,
                    parent_generation=parent_generation,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    optimizer_step=int(dataset_index),
                    requires_grad=True,
                )
            vectors: dict[str, Tensor] = {}
            for loss_index, name in enumerate(LOSS_NAMES):
                vector = flatten_gradients(
                    raw[name],
                    parameters,
                    retain_graph=loss_index < len(LOSS_NAMES) - 1,
                )
                vectors[name] = vector
                local_gradient_sums[name].add_(vector)
                local_norm_sums[name] += float(vector.norm().item())
                local_raw_sums[name] += float(raw[name].detach().item())
            for left_index, left in enumerate(LOSS_NAMES):
                for right in LOSS_NAMES[left_index:]:
                    local_rows.append(
                        {
                            "rank": rank,
                            "dataset_index": int(dataset_index),
                            "loss_a": left,
                            "loss_b": right,
                            "cosine": cosine(vectors[left], vectors[right]),
                            "norm_a": float(vectors[left].norm().item()),
                            "norm_b": float(vectors[right].norm().item()),
                            "raw_a": float(raw[left].detach().item()),
                            "raw_b": float(raw[right].detach().item()),
                        }
                    )
            for name, value in diagnostics.items():
                if str(name).startswith("age"):
                    local_age_rows.append(
                        {
                            "rank": rank,
                            "dataset_index": int(dataset_index),
                            "metric": str(name),
                            "value": float(value.detach().item()),
                        }
                    )
        count_tensor = torch.tensor(float(len(local_indices)), device=device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        total_count = int(count_tensor.item())
        if total_count != int(args.audit_batches):
            raise RuntimeError("gradient audit did not consume the fixed batch count")
        for vector in local_gradient_sums.values():
            dist.all_reduce(vector, op=dist.ReduceOp.SUM)
            vector.div_(float(total_count))
        scalar_sums = torch.tensor(
            [
                *[local_norm_sums[name] for name in LOSS_NAMES],
                *[local_raw_sums[name] for name in LOSS_NAMES],
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(scalar_sums, op=dist.ReduceOp.SUM)
        scalar_means = scalar_sums.div(float(total_count)).cpu().tolist()
        gathered_rows: list[Any] = [None for _ in range(world)] if rank == 0 else []
        gathered_age: list[Any] = [None for _ in range(world)] if rank == 0 else []
        dist.gather_object(local_rows, gathered_rows if rank == 0 else None, dst=0)
        dist.gather_object(local_age_rows, gathered_age if rank == 0 else None, dst=0)
        result: dict[str, Any] = {}
        if rank == 0:
            per_batch = [row for shard in gathered_rows for row in shard]
            age_rows = [row for shard in gathered_age for row in shard]
            weights_payload = load_json(args.loss_weights)
            weights = {
                name: float(weights_payload["weights"][name]) for name in LOSS_NAMES
            }
            aggregate_gradient_norms = {
                name: float(local_gradient_sums[name].norm().item())
                for name in LOSS_NAMES
            }
            gradient_norms = {
                name: float(scalar_means[index])
                for index, name in enumerate(LOSS_NAMES)
            }
            raw_means = {
                name: float(scalar_means[len(LOSS_NAMES) + index])
                for index, name in enumerate(LOSS_NAMES)
            }
            weighted_norms = {
                name: weights[name] * gradient_norms[name] for name in LOSS_NAMES
            }
            weighted_total = sum(weighted_norms.values())
            weighted_shares = {
                name: weighted_norms[name] / max(weighted_total, 1e-12)
                for name in LOSS_NAMES
            }
            cosine_rows: list[dict[str, Any]] = []
            for left_index, left in enumerate(LOSS_NAMES):
                for right in LOSS_NAMES[left_index:]:
                    selected = [
                        float(row["cosine"])
                        for row in per_batch
                        if row["loss_a"] == left and row["loss_b"] == right
                        and math.isfinite(float(row["cosine"]))
                    ]
                    if not selected:
                        raise RuntimeError(
                            f"no finite per-batch cosine for {left} vs {right}"
                        )
                    cosine_rows.append(
                        {
                            "branch": str(args.branch),
                            "loss_a": left,
                            "loss_b": right,
                            "fixed_batches": total_count,
                            "mean_batch_cosine": _mean(selected),
                            "p05_batch_cosine": float(np.quantile(selected, 0.05)),
                            "p95_batch_cosine": float(np.quantile(selected, 0.95)),
                            "aggregate_gradient_cosine": cosine(
                                local_gradient_sums[left], local_gradient_sums[right]
                            ),
                            "mean_batch_gradient_norm_a": gradient_norms[left],
                            "mean_batch_gradient_norm_b": gradient_norms[right],
                            "aggregate_gradient_norm_a": aggregate_gradient_norms[left],
                            "aggregate_gradient_norm_b": aggregate_gradient_norms[right],
                            "weight_a": weights[left],
                            "weight_b": weights[right],
                            "weighted_gradient_share_a": weighted_shares[left],
                            "weighted_gradient_share_b": weighted_shares[right],
                        }
                    )
            _write_rows(output / "stability_v2_gradient_cosine.csv", cosine_rows)
            age_summary = {
                metric: summary(
                    [float(row["value"]) for row in age_rows if row["metric"] == metric]
                )
                for metric in sorted({str(row["metric"]) for row in age_rows})
            }
            fixed_contract = {
                "count": total_count,
                "indices": list(fixed),
                "identities": [dataset.identities[index] for index in fixed],
            }
            fixed_contract["combined_sha256"] = canonical_sha256(fixed_contract)
            result = {
                "schema_version": "simvla_stability_v2_gradient_audit_v1",
                "verdict": "STABILITY_V2_GRADIENT_AUDIT_COMPLETE",
                "candidate": str(Path(args.candidate).expanduser().resolve()),
                "branch": str(args.branch),
                "candidate_sha256": sha256_file(args.candidate),
                "candidate_optimizer_step": int(checkpoint_payload["optimizer_step"]),
                "fixed_batch_contract": fixed_contract,
                "condition_updater_parameters": sum(
                    parameter.numel() for parameter in parameters
                ),
                "gradient_norms": gradient_norms,
                "gradient_norm_semantics": "mean of per-batch Condition-updater gradient norms",
                "aggregate_gradient_norms": aggregate_gradient_norms,
                "aggregate_gradient_norm_semantics": "norm of the mean Condition-updater gradient vector",
                "raw_loss_means": raw_means,
                "historical_weights": weights,
                "actual_weighted_gradient_norms": weighted_norms,
                "actual_weighted_gradient_shares": weighted_shares,
                "per_age_raw_metrics": age_summary,
                "source_lock": source,
                "historical_objectives_unchanged": True,
            }
            result["combined_sha256"] = canonical_sha256(result)
            atomic_write_json(output / "gradient_audit_summary.json", result)
            atomic_write_json(output / "source_lock.json", source)
        dist.barrier()
        return result
    finally:
        cleanup_distributed()


def command_merge(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing merge output: {output}")
    output.mkdir(parents=True)
    branches = {
        "S50": Path(args.s50).expanduser().resolve(),
        "S150": Path(args.s150).expanduser().resolve(),
    }
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for branch, root in branches.items():
        with (root / "stability_v2_checkpoint_sweep.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
        summaries[branch] = load_json(root / "checkpoint_sweep_summary.json")
    _write_rows(output / "stability_v2_checkpoint_sweep.csv", rows)
    payload = {
        "schema_version": "simvla_stability_v2_checkpoint_sweep_merged_v1",
        "verdict": "STABILITY_V2_SWEEPS_MERGED",
        "branches": {
            branch: {
                "selected_step": summary["validation_selection"]["selected_step"],
                "selected_checkpoint": summary["validation_selection"][
                    "selected_checkpoint"
                ],
                "final_offline_gate": summary["final_offline"][
                    "corrected_scientific_gate"
                ]["verdict"],
            }
            for branch, summary in summaries.items()
        },
    }
    payload["combined_sha256"] = canonical_sha256(payload)
    atomic_write_json(output / "merged_sweep_summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(value: argparse.ArgumentParser) -> None:
        value.add_argument("--output", required=True)
        value.add_argument("--repository", required=True)
        value.add_argument("--cache", default=DEFAULT_CACHE)
        value.add_argument("--condition-parent", required=True)
        value.add_argument("--generation-parent", default=DEFAULT_GENERATION_30K)
        value.add_argument("--norm-stats", default=DEFAULT_NORM)
        value.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
        value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
        value.add_argument("--split-seed", type=int, default=20260822)
        value.add_argument("--seed", type=int, default=20260825)
        value.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)

    sweep = subparsers.add_parser("checkpoint-sweep")
    common(sweep)
    sweep.add_argument("--branch", choices=("S50", "S150"), required=True)
    sweep.add_argument("--checkpoints", required=True)

    audit = subparsers.add_parser("gradient-audit")
    common(audit)
    audit.add_argument("--branch", choices=("S50", "S150"), required=True)
    audit.add_argument("--candidate", required=True)
    audit.add_argument("--loss-weights", required=True)
    audit.add_argument("--audit-batches", type=int, default=64)
    audit.add_argument("--audit-seed", type=int, default=20260826)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--output", required=True)
    merge.add_argument("--s50", required=True)
    merge.add_argument("--s150", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "checkpoint-sweep":
        result = command_checkpoint_sweep(args)
    elif args.command == "gradient-audit":
        result = command_gradient_audit(args)
    else:
        result = command_merge(args)
    if not dist.is_available() or not dist.is_initialized() or int(
        __import__("os").environ.get("RANK", "0")
    ) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
