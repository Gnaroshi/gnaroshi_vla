"""Frozen-parent hard-pool preparation and V3 gradient calibration."""

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
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    atomic_write_json,
    canonical_sha256,
    load_json,
)
from architectures.simvla.adapters.latentloop.stability_alignment.model import (
    configure_condition_only_stage,
    load_warm_start,
)
from architectures.simvla.adapters.latentloop.stability_alignment.objectives import (
    first_r_per_sequence,
)
from architectures.simvla.adapters.latentloop.stability_alignment.trainer import (
    _assert_parent_contract,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_contracts import (
    V3_GAMMA_QUANTILE,
    V3_HARD_POOL_SCHEMA,
    V3_LOSS_NAMES,
    calibrate_v3_gradient_weights,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_data import (
    V3PoolSampler,
    V3StabilityExactTeacherDataset,
    build_v3_hard_pool_contract,
    collate_v3_sequences,
    gripper_event_contract,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_objectives import (
    parent_recurrence_gains,
    v3_condition_paths,
    v3_raw_losses,
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
    summary,
    v3_source_lock,
    weighted_quantile,
    zero_code_ng3_actions,
)


def _percentile_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted((float(value), index) for index, value in enumerate(values))
    result = [0.0 for _ in indexed]
    offset = 0
    while offset < len(indexed):
        stop = offset + 1
        while stop < len(indexed) and indexed[stop][0] == indexed[offset][0]:
            stop += 1
        average_rank = 0.5 * (offset + stop - 1)
        percentile = average_rank / max(len(indexed) - 1, 1)
        for _, original in indexed[offset:stop]:
            result[original] = percentile
        offset = stop
    return result


def _parent_transition_counts(
    predictions: Sequence[Tensor], targets: Sequence[Tensor]
) -> dict[str, int | bool]:
    prediction = predictions[2][:, :5, 6].float()
    target = targets[2][:, :5, 6].float()
    mismatch = (prediction >= 0.0) != (target >= 0.0)
    previous_prediction = predictions[1][:, 4, 6].float()
    previous_target = targets[1][:, 4, 6].float()
    predicted_before = torch.cat(
        (previous_prediction[:, None], prediction[:, :-1]), dim=1
    )
    target_before = torch.cat((previous_target[:, None], target[:, :-1]), dim=1)
    switch_mismatch = (
        ((predicted_before >= 0.0) != (prediction >= 0.0))
        != ((target_before >= 0.0) != (target >= 0.0))
    )
    return {
        "parent_age3_sign_mismatch_sequence": bool(mismatch.any().item()),
        "parent_age3_sign_mismatch_positions": int(mismatch.sum().item()),
        "parent_age3_switch_mismatch_sequence": bool(
            switch_mismatch.any().item()
        ),
        "parent_age3_switch_mismatch_positions": int(switch_mismatch.sum().item()),
    }


def _score_parent_batch(
    *,
    modules: Any,
    parent: Any,
    frozen_model: Any,
    action_adapter: Any,
    batch: Mapping[str, Any],
    dataset_index: int,
) -> dict[str, Any]:
    with torch.no_grad():
        paths = v3_condition_paths(modules.condition, parent, batch)
        exact = tuple(batch["teacher_conditions"][:, index] for index in range(3))
        actions = tuple(batch["teacher_actions"][:, index] for index in range(3))
        gains = parent_recurrence_gains(paths, exact, batch["valid_mask"])
        proprio = tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3))
        noises = tuple(batch["explicit_noises"][:, index] for index in range(3))
        ng3 = zero_code_ng3_actions(
            modules=modules,
            frozen_model=frozen_model,
            action_adapter=action_adapter,
            conditions=paths.frozen_parent_recursive,
            proprio=proprio,
            noises=noises,
            valid_mask=batch["valid_mask"],
            optimizer_step=int(dataset_index),
            requires_grad=False,
        )
        age3_action = float(first_r_per_sequence(ng3[2], actions[2]).item())
    event = gripper_event_contract(
        batch["anchor_teacher_action"][0], batch["teacher_actions"][0]
    )
    transition = _parent_transition_counts(ng3, actions)
    return {
        "dataset_index": int(dataset_index),
        "task_id": int(batch["task_id"].item()),
        "episode_id": str(batch["episode_id"][0]),
        "anchor_query_index": int(batch["anchor_query_index"].item()),
        "parent_gain_age2": float(gains[2].item()),
        "parent_gain_age3": float(gains[3].item()),
        "parent_age3_recurrence_divergence": float(
            paths.frozen_parent_recursive[2]
            .float()
            .sub(paths.frozen_teacher_targets[2].float())
            .square()
            .mean()
            .sqrt()
            .item()
        ),
        "parent_age3_ng3_first_r": age3_action,
        "has_any_gripper_event": bool(event["has_any_event"]),
        "has_cross_query_gripper_event": bool(event["has_cross_query_event"]),
        "within_query_event_positions": int(event["within_query_event_positions"]),
        "cross_query_event_positions": int(event["cross_query_event_positions"]),
        **transition,
    }


def _merge_shards(output: Path, stem: str, world: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in range(int(world)):
        rows.extend(load_json(output / "shards" / f"rank_{rank}_{stem}.json"))
    rows.sort(key=lambda row: int(row["dataset_index"]))
    return rows


def command_build_pools(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, world, device = distributed_pair()
    try:
        configure_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"refusing existing pool output: {output}")
            (output / "shards").mkdir(parents=True)
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
        modules, parent, _, payloads = load_warm_start(
            condition_checkpoint=args.condition_parent,
            generation_checkpoint=args.generation_parent,
            device=device,
        )
        identity = _assert_parent_contract(payloads)
        freeze_module(modules)
        freeze_module(parent)
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
        local_rows: list[dict[str, Any]] = []
        for dataset_index in tqdm(
            range(rank, len(dataset), world),
            desc=f"V3 parent scoring rank{rank}",
            dynamic_ncols=True,
        ):
            batch = move_batch(collate_v3_sequences([dataset[dataset_index]]), device)
            local_rows.append(
                _score_parent_batch(
                    modules=modules,
                    parent=parent,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    dataset_index=dataset_index,
                )
            )
        atomic_write_json(
            output / "shards" / f"rank_{rank}_parent_scores.json", local_rows
        )
        dist.barrier()
        result: dict[str, Any] = {}
        if rank == 0:
            rows = _merge_shards(output, "parent_scores", world)
            recurrence_percentile = _percentile_ranks(
                [float(row["parent_age3_recurrence_divergence"]) for row in rows]
            )
            action_percentile = _percentile_ranks(
                [float(row["parent_age3_ng3_first_r"]) for row in rows]
            )
            for row, recurrence_rank, action_rank in zip(
                rows, recurrence_percentile, action_percentile
            ):
                row["recurrence_percentile"] = float(recurrence_rank)
                row["action_percentile"] = float(action_rank)
                row["tail_score"] = 0.5 * (
                    float(recurrence_rank) + float(action_rank)
                )
            gains = [
                float(row[f"parent_gain_age{age}"])
                for row in rows
                for age in (2, 3)
            ]
            gain_weights = [
                float(weight) for _ in rows for weight in (1.0, 2.0)
            ]
            gamma = weighted_quantile(
                gains, gain_weights, q=float(V3_GAMMA_QUANTILE)
            )
            pool = build_v3_hard_pool_contract(
                rows,
                dataset_contract=dataset.contract(),
                source_lock=source,
            )
            pool["parent_identity"] = identity
            pool["gamma_contract"] = {
                "selection": "frozen-parent age-weighted median",
                "quantile": float(V3_GAMMA_QUANTILE),
                "age_weights": {"2": 1.0, "3": 2.0},
                "gamma": float(gamma),
                "measured_before_optimizer_step_zero": True,
                "frozen_for_all_v3_training": True,
                "parent_gain_age2": summary(
                    [float(row["parent_gain_age2"]) for row in rows]
                ),
                "parent_gain_age3": summary(
                    [float(row["parent_gain_age3"]) for row in rows]
                ),
            }
            pool["combined_sha256"] = canonical_sha256(
                {key: value for key, value in pool.items() if key != "combined_sha256"}
            )
            atomic_write_json(output / "stability_v3_hard_pool_contract.json", pool)
            atomic_write_json(output / "parent_score_rows.json", rows)
            atomic_write_json(output / "source_lock.json", source)
            result = {
                "schema_version": "simvla_stability_v3_pool_preparation_v1",
                "verdict": "STABILITY_V3_HARD_POOLS_READY",
                "hard_pool_contract": str(
                    output / "stability_v3_hard_pool_contract.json"
                ),
                "gamma": float(gamma),
                "training_sequences": len(rows),
                "source_combined_sha256": source["combined_sha256"],
            }
            atomic_write_json(output / "pool_preparation_summary.json", result)
        dist.barrier()
        return result
    finally:
        cleanup_distributed()


def _v3_forward(
    *,
    modules: Any,
    parent: Any,
    frozen_model: Any,
    action_adapter: Any,
    batch: Mapping[str, Any],
    optimizer_step: int,
    gamma: float,
    hard_sample: bool,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    paths = v3_condition_paths(modules.condition, parent, batch)
    exact_conditions = tuple(
        batch["teacher_conditions"][:, index] for index in range(3)
    )
    exact_actions = tuple(batch["teacher_actions"][:, index] for index in range(3))
    proprio = tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3))
    noises = tuple(batch["explicit_noises"][:, index] for index in range(3))
    ng3_actions = zero_code_ng3_actions(
        modules=modules,
        frozen_model=frozen_model,
        action_adapter=action_adapter,
        conditions=paths.student_recursive,
        proprio=proprio,
        noises=noises,
        valid_mask=batch["valid_mask"],
        optimizer_step=int(optimizer_step),
        requires_grad=True,
    )
    rotating_index = int(optimizer_step) % 3
    rotating_action = decode_full_actions(
        action_adapter,
        conditions=(paths.student_recursive[rotating_index],),
        proprio=(proprio[rotating_index],),
        noises=(noises[rotating_index],),
        requires_grad=True,
    )[0]
    return v3_raw_losses(
        paths=paths,
        exact_conditions=exact_conditions,
        ng3_actions=ng3_actions,
        exact_actions=exact_actions,
        rotating_full_action=rotating_action,
        rotating_age_index=rotating_index,
        anchor_teacher_action=batch["anchor_teacher_action"],
        valid_mask=batch["valid_mask"],
        gamma=float(gamma),
        hard_sample=bool(hard_sample),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_calibrate(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, world, device = distributed_pair()
    try:
        configure_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"refusing existing V3 calibration output: {output}")
            output.mkdir(parents=True)
        dist.barrier()
        pool = load_json(args.hard_pool)
        if pool.get("schema_version") != V3_HARD_POOL_SCHEMA:
            raise ValueError("V3 hard-pool schema changed")
        gamma = float(pool["gamma_contract"]["gamma"])
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
        if pool["source_lock"]["combined_sha256"] != source["combined_sha256"]:
            raise RuntimeError("hard pool and V3 calibration source locks differ")
        modules, parent, _, payloads = load_warm_start(
            condition_checkpoint=args.condition_parent,
            generation_checkpoint=args.generation_parent,
            device=device,
        )
        _assert_parent_contract(payloads)
        configure_condition_only_stage(modules)
        freeze_module(parent)
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
        sampler = V3PoolSampler(
            pool,
            seed=args.audit_seed,
            start_step=0,
            stop_step=args.audit_batches,
        )
        schedule = [
            (step, *sampler.index(step)) for step in range(args.audit_batches)
        ]
        local_schedule = schedule[rank::world]
        parameters = condition_updater_parameters(modules)
        dimension = sum(parameter.numel() for parameter in parameters)
        vector_sums = {
            name: torch.zeros(dimension, device=device, dtype=torch.float32)
            for name in V3_LOSS_NAMES
        }
        norm_sums = {name: 0.0 for name in V3_LOSS_NAMES}
        raw_sums = {name: 0.0 for name in V3_LOSS_NAMES}
        local_cosines: list[dict[str, Any]] = []
        for optimizer_step, dataset_index, stream in tqdm(
            local_schedule, desc=f"V3 gradient rank{rank}", dynamic_ncols=True
        ):
            batch = move_batch(collate_v3_sequences([dataset[dataset_index]]), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                raw, _ = _v3_forward(
                    modules=modules,
                    parent=parent,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    optimizer_step=optimizer_step,
                    gamma=gamma,
                    hard_sample=stream == "recurrence_action_tail",
                )
            vectors: dict[str, Tensor] = {}
            for loss_index, name in enumerate(V3_LOSS_NAMES):
                vector = flatten_gradients(
                    raw[name],
                    parameters,
                    retain_graph=loss_index < len(V3_LOSS_NAMES) - 1,
                )
                vectors[name] = vector
                vector_sums[name].add_(vector)
                norm_sums[name] += float(vector.norm().item())
                raw_sums[name] += float(raw[name].detach().item())
            for left_index, left in enumerate(V3_LOSS_NAMES):
                for right in V3_LOSS_NAMES[left_index:]:
                    local_cosines.append(
                        {
                            "optimizer_step": int(optimizer_step),
                            "dataset_index": int(dataset_index),
                            "stream": str(stream),
                            "loss_a": left,
                            "loss_b": right,
                            "cosine": cosine(vectors[left], vectors[right]),
                        }
                    )
        count = torch.tensor(float(len(local_schedule)), device=device)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        total_count = int(count.item())
        if total_count != int(args.audit_batches):
            raise RuntimeError("V3 calibration did not consume fixed batch count")
        for name in V3_LOSS_NAMES:
            dist.all_reduce(vector_sums[name], op=dist.ReduceOp.SUM)
            vector_sums[name].div_(float(total_count))
        scalar = torch.tensor(
            [
                *[norm_sums[name] for name in V3_LOSS_NAMES],
                *[raw_sums[name] for name in V3_LOSS_NAMES],
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        means = scalar.div(float(total_count)).cpu().tolist()
        gradient_norms = {
            name: float(means[index]) for index, name in enumerate(V3_LOSS_NAMES)
        }
        raw_means = {
            name: float(means[len(V3_LOSS_NAMES) + index])
            for index, name in enumerate(V3_LOSS_NAMES)
        }
        gathered: list[Any] = [None for _ in range(world)] if rank == 0 else []
        dist.gather_object(local_cosines, gathered if rank == 0 else None, dst=0)
        result: dict[str, Any] = {}
        if rank == 0:
            reference = 0.5 * (
                vector_sums["exact_condition_reference"]
                + vector_sums["frozen_teacher_preservation"]
            )
            pairwise_conflicts = {
                "recurrence_vs_frozen_ng3": cosine(
                    vector_sums["recurrence_gain"],
                    vector_sums["frozen_ng3_execution"],
                ),
                "recurrence_vs_exact_teacher_group": cosine(
                    vector_sums["recurrence_gain"], reference
                ),
                "recurrence_vs_gripper_transition": cosine(
                    vector_sums["recurrence_gain"],
                    vector_sums["gripper_transition"],
                ),
            }
            preliminary = calibrate_v3_gradient_weights(
                gradient_norms,
                recurrence_cosines_with_dominant_losses=pairwise_conflicts,
            )
            weighted_total = sum(
                preliminary["weights"][name] * vector_sums[name]
                for name in V3_LOSS_NAMES
            )
            weighted_total_cosines_all = {
                name: cosine(vector_sums[name], weighted_total)
                for name in V3_LOSS_NAMES
            }
            protected_total_cosines = {
                "recurrence_gain": weighted_total_cosines_all["recurrence_gain"],
                "frozen_ng3_execution": weighted_total_cosines_all[
                    "frozen_ng3_execution"
                ],
                "exact_teacher_group": cosine(reference, weighted_total),
                "gripper_transition": weighted_total_cosines_all[
                    "gripper_transition"
                ],
            }
            calibrated = calibrate_v3_gradient_weights(
                gradient_norms,
                recurrence_cosines_with_dominant_losses=pairwise_conflicts,
                weighted_total_cosines_with_protected_losses=protected_total_cosines,
            )
            all_cosines = [row for shard in gathered for row in shard]
            all_cosines.sort(
                key=lambda row: (
                    int(row["optimizer_step"]),
                    str(row["loss_a"]),
                    str(row["loss_b"]),
                )
            )
            _write_csv(
                output / "stability_v3_gradient_cosine_batches.csv", all_cosines
            )
            cosine_rows: list[dict[str, Any]] = []
            for left_index, left in enumerate(V3_LOSS_NAMES):
                for right in V3_LOSS_NAMES[left_index:]:
                    selected = [
                        float(row["cosine"])
                        for row in all_cosines
                        if row["loss_a"] == left
                        and row["loss_b"] == right
                        and math.isfinite(float(row["cosine"]))
                    ]
                    if not selected:
                        raise RuntimeError(
                            f"no finite per-batch cosine for {left} vs {right}"
                        )
                    cosine_rows.append(
                        {
                            "loss_a": left,
                            "loss_b": right,
                            "fixed_batches": total_count,
                            "mean_batch_cosine": float(np.mean(selected)),
                            "batch_mean_standard_error": float(
                                np.std(selected, ddof=1) / math.sqrt(len(selected))
                            ),
                            "p05_batch_cosine": float(np.quantile(selected, 0.05)),
                            "p95_batch_cosine": float(np.quantile(selected, 0.95)),
                            "aggregate_gradient_cosine": cosine(
                                vector_sums[left], vector_sums[right]
                            ),
                        }
                    )
            _write_csv(output / "stability_v3_gradient_cosine.csv", cosine_rows)
            stream_counts = {
                name: sum(stream == name for _, _, stream in schedule)
                for name in (
                    "base",
                    "gripper_transition",
                    "recurrence_action_tail",
                )
            }
            result = {
                **calibrated,
                "weighted_total_cosines_all_losses": weighted_total_cosines_all,
                "weighted_total_gradient_norm": float(weighted_total.norm().item()),
                "raw_means": raw_means,
                "gamma": gamma,
                "gamma_contract": pool["gamma_contract"],
                "hard_pool_combined_sha256": pool["combined_sha256"],
                "source_lock": source,
                "source_contract_sha256": source["combined_sha256"],
                "audit_batches": total_count,
                "audit_stream_counts": stream_counts,
                "audit_schedule_sha256": canonical_sha256(schedule),
                "condition_updater_parameters": dimension,
                "parent_preservation_contribution": 0.0,
                "weights_frozen_before_optimizer_step_zero": True,
            }
            result["combined_sha256"] = canonical_sha256(result)
            atomic_write_json(output / "stability_v3_loss_weights.json", result)
            atomic_write_json(output / "source_lock.json", source)
        dist.barrier()
        return result
    finally:
        cleanup_distributed()


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

    pools = subparsers.add_parser("build-pools")
    common(pools)

    calibrate = subparsers.add_parser("calibrate")
    common(calibrate)
    calibrate.add_argument("--hard-pool", required=True)
    calibrate.add_argument("--audit-batches", type=int, default=64)
    calibrate.add_argument("--audit-seed", type=int, default=20260826)
    calibrate.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = (
        command_build_pools(args)
        if args.command == "build-pools"
        else command_calibrate(args)
    )
    if int(__import__("os").environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
