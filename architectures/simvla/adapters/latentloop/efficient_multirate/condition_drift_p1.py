"""Measure recursive versus teacher-forced SimVLA condition drift.

This diagnostic reuses the frozen native V0 checkpoint and its episode-disjoint
held-out training split.  It never changes model parameters and decodes all
action comparisons with the exact same cached flow noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_condition_hook import GROUP_NAMES
from architectures.simvla.adapters.latentloop.native_v0_dataset import (
    NativeV0SequenceDataset,
    collate_native_v0_sequences,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    cached_batch_token_layout,
    configure_strict_torch_determinism,
    load_frozen_simvla,
    move_batch,
    native_v0_source_manifest,
    require_gate,
    write_json,
)
from architectures.simvla.adapters.latentloop.source_lock import sha256_file
from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0ObservationPair,
    NativeV0UpdateOutput,
)
from methods.latentloop.training.native_simvla_v0 import decode_age_conditions


ROOT = Path(__file__).resolve().parents[5]
DIAGNOSTIC_FILES = (
    "architectures/simvla/adapters/latentloop/efficient_multirate/condition_drift_p1.py",
    "architectures/simvla/wrappers/run_condition_drift_p1.sh",
)
PATHS = ("recursive", "teacher_forced")
AGES = (1, 2, 3)
CONDITION_FIELDS = (
    "condition_token_cosine_mean",
    "condition_flat_cosine",
    "condition_normalized_mse",
    "condition_raw_mse",
)
ACTION_FIELDS = (
    "action_first5_l1",
    "action_full_chunk_l1",
    "action_translation_l1",
    "action_rotation_l1",
    "action_gripper_l1",
    "action_gripper_sign_mismatch_rate",
    "action_gripper_switch_mismatch_rate",
    "action_gripper_max_abs_error",
)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", "-C", str(ROOT), *args), text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def diagnostic_source_lock() -> dict[str, Any]:
    files = {name: sha256_file(ROOT / name) for name in DIAGNOSTIC_FILES}
    return {
        "git_branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "files": files,
        "command": list(sys.argv),
    }


def validate_scientific_source_compatibility(
    current: dict[str, Any], checkpoint_source: dict[str, Any]
) -> dict[str, Any]:
    """Allow proven path relocation and GPU rebinding, but no scientific change."""

    derived = {"combined_sha256", "complete_source_lock"}
    hardware = {"selected_physical_gpu_ids"}
    relocation_evidence = {
        "norm_stats_path": "norm_stats_sha256",
        "cache_manifest_path": "cache_manifest_sha256",
        "libero_root": "libero_commit",
    }
    relocations: dict[str, dict[str, Any]] = {}
    for path_field, identity_field in relocation_evidence.items():
        if current.get(path_field) == checkpoint_source.get(path_field):
            continue
        current_identity = current.get(identity_field)
        checkpoint_identity = checkpoint_source.get(identity_field)
        if current_identity is None or current_identity != checkpoint_identity:
            raise RuntimeError(
                f"native V0 relocation identity mismatch for {path_field}: "
                f"{identity_field}={current_identity!r} != {checkpoint_identity!r}"
            )
        relocations[path_field] = {
            "checkpoint": checkpoint_source.get(path_field),
            "current": current.get(path_field),
            "identity_field": identity_field,
            "identity": current_identity,
        }
    ignored = derived | hardware | set(relocation_evidence)
    differences = {
        key: {"checkpoint": checkpoint_source.get(key), "current": current.get(key)}
        for key in sorted((set(current) | set(checkpoint_source)) - ignored)
        if current.get(key) != checkpoint_source.get(key)
    }
    if differences:
        raise RuntimeError(f"native V0 scientific source mismatch: {differences}")
    return {
        "verdict": "SCIENTIFIC_SOURCE_MATCH_RELOCATED_HARDWARE_REBOUND",
        "checkpoint_source_sha256": checkpoint_source["combined_sha256"],
        "current_source_sha256": current["combined_sha256"],
        "checkpoint_gpu_ids": checkpoint_source.get("selected_physical_gpu_ids"),
        "current_gpu_ids": current.get("selected_physical_gpu_ids"),
        "derived_fields_excluded_from_comparison": sorted(derived),
        "hardware_rebinding_fields": sorted(hardware),
        "verified_path_relocations": relocations,
        "scientific_differences": differences,
    }


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        raise ValueError("cannot summarize an empty metric")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def condition_fidelity_metrics(
    prediction: Tensor, target: Tensor, valid_mask: Tensor
) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("condition tensors must share [B,T,D] shape")
    if prediction.shape[0] != 1 or valid_mask.shape != prediction.shape[:2]:
        raise ValueError("P1 condition metrics require one sample and a [1,T] mask")
    prediction_f = prediction.float()
    target_f = target.float()
    mask = valid_mask.to(device=prediction.device, dtype=torch.bool)
    selected_prediction = prediction_f[mask]
    selected_target = target_f[mask]
    if not selected_prediction.numel():
        raise ValueError("condition mask selected no tokens")
    token_cosine = F.cosine_similarity(selected_prediction, selected_target, dim=-1)
    normalized_prediction = F.layer_norm(prediction_f, (prediction_f.shape[-1],))
    normalized_target = F.layer_norm(target_f, (target_f.shape[-1],))
    normalized_difference = normalized_prediction[mask] - normalized_target[mask]
    difference = selected_prediction - selected_target
    return {
        "condition_token_cosine_mean": float(token_cosine.mean().item()),
        "condition_token_cosine_p05": float(torch.quantile(token_cosine, 0.05).item()),
        "condition_flat_cosine": float(
            F.cosine_similarity(
                selected_prediction.reshape(1, -1),
                selected_target.reshape(1, -1),
                dim=-1,
            ).item()
        ),
        "condition_normalized_mse": float(normalized_difference.square().mean().item()),
        "condition_raw_mse": float(difference.square().mean().item()),
        "condition_max_abs": float(difference.abs().max().item()),
        "tokens": int(mask.sum().item()),
    }


def action_fidelity_metrics(prediction: Tensor, target: Tensor) -> dict[str, float | int]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("action tensors must share [B,H,7] shape")
    if prediction.shape[0] != 1 or prediction.shape[1] < 5 or prediction.shape[2] != 7:
        raise ValueError("P1 action metrics require [1,H>=5,7]")
    difference = (prediction.float() - target.float()).abs()
    first5 = difference[:, :5]
    prediction_gripper = prediction[:, :5, 6].float()
    target_gripper = target[:, :5, 6].float()
    prediction_open = prediction_gripper >= 0.0
    target_open = target_gripper >= 0.0
    prediction_switch = prediction_open[:, 1:] != prediction_open[:, :-1]
    target_switch = target_open[:, 1:] != target_open[:, :-1]
    switch_false_positive = prediction_switch & ~target_switch
    switch_false_negative = ~prediction_switch & target_switch
    return {
        "action_first5_l1": float(first5.mean().item()),
        "action_full_chunk_l1": float(difference.mean().item()),
        "action_translation_l1": float(first5[..., :3].mean().item()),
        "action_rotation_l1": float(first5[..., 3:6].mean().item()),
        "action_gripper_l1": float(first5[..., 6].mean().item()),
        "action_gripper_max_abs_error": float(first5[..., 6].max().item()),
        "action_gripper_sign_mismatch_count": int((prediction_open != target_open).sum().item()),
        "action_gripper_sign_mismatch_rate": float(
            (prediction_open != target_open).float().mean().item()
        ),
        "action_gripper_switch_mismatch_count": int(
            (prediction_switch != target_switch).sum().item()
        ),
        "action_gripper_switch_mismatch_rate": float(
            (prediction_switch != target_switch).float().mean().item()
        ),
        "action_gripper_switch_false_positive_count": int(
            switch_false_positive.sum().item()
        ),
        "action_gripper_switch_false_negative_count": int(
            switch_false_negative.sum().item()
        ),
        "teacher_gripper_switch_count": int(target_switch.sum().item()),
        "predicted_gripper_switch_count": int(prediction_switch.sum().item()),
    }


def teacher_forced_updates(
    model: NativeSimVLAV0,
    batch: dict[str, Any],
    *,
    valid_mask: Tensor,
    group_ids: Tensor,
) -> tuple[NativeV0UpdateOutput, NativeV0UpdateOutput, NativeV0UpdateOutput]:
    """Apply each one-step update from the exact previous condition."""

    updates: list[NativeV0UpdateOutput] = []
    for age in AGES:
        previous_condition = (
            batch["anchor_condition"]
            if age == 1
            else batch["teacher_conditions"][:, age - 2]
        )
        pair = NativeV0ObservationPair(
            previous_images=batch["image_sequence"][:, age - 1],
            current_images=batch["image_sequence"][:, age],
            previous_proprio=batch["proprio_sequence"][:, age - 1],
            current_proprio=batch["proprio_sequence"][:, age],
        )
        updates.append(
            model.update_once(
                previous_condition,
                pair,
                valid_mask=valid_mask,
                group_ids=group_ids,
                age=age,
            )
        )
    return updates[0], updates[1], updates[2]


def _decode_conditions(
    action_adapter: Any,
    conditions: Sequence[Tensor],
    batch: dict[str, Any],
    *,
    mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    return decode_age_conditions(
        lambda condition, proprio, noise: action_adapter.decode_action_from_condition(
            condition,
            proprio,
            steps=10,
            initial_noise=noise,
        ),
        conditions,
        tuple(batch["proprio_sequence"][:, age] for age in AGES),
        tuple(batch["explicit_noises"][:, age - 1] for age in AGES),
        mode=mode,
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"inconsistent CSV fields for {path}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _path_metrics(row: dict[str, Any], path: str) -> dict[str, float]:
    return {
        field: float(row[f"{path}/{field}"])
        for field in (*CONDITION_FIELDS, *ACTION_FIELDS)
    }


def _summarize_rows(rows: Sequence[dict[str, Any]], path: str) -> dict[str, Any]:
    return {
        field: _summary(float(row[f"{path}/{field}"]) for row in rows)
        for field in (*CONDITION_FIELDS, *ACTION_FIELDS)
    }


def _aggregate_output(output: Path) -> dict[str, Any]:
    contract = json.loads((output / "run_contract.json").read_text(encoding="utf-8"))
    world_size = int(contract["world_size"])
    expected_sequences = int(contract["evaluated_sequences"])
    sequence_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    shard_summaries = []
    for rank in range(world_size):
        shard = output / "shards" / f"rank_{rank}"
        shard_summary = json.loads((shard / "shard_summary.json").read_text(encoding="utf-8"))
        if shard_summary.get("verdict") != "CONDITION_DRIFT_P1_SHARD_COMPLETE":
            raise RuntimeError(f"incomplete P1 shard: {shard}")
        shard_summaries.append(shard_summary)
        sequence_rows.extend(_read_csv(shard / "sequence_age_metrics.csv"))
        group_rows.extend(_read_csv(shard / "token_group_metrics.csv"))

    identities = {
        (int(row["dataset_index"]), int(row["age"])) for row in sequence_rows
    }
    expected_identities = {
        (dataset_index, age)
        for dataset_index in range(expected_sequences)
        for age in AGES
    }
    if identities != expected_identities or len(sequence_rows) != len(expected_identities):
        raise RuntimeError("P1 shard union does not cover each evaluated sequence and age once")
    sequence_rows.sort(key=lambda row: (int(row["dataset_index"]), int(row["age"])))
    group_rows.sort(
        key=lambda row: (
            int(row["dataset_index"]),
            int(row["age"]),
            row["path"],
            int(row["group_id"]),
        )
    )
    _write_csv(output / "sequence_age_metrics.csv", sequence_rows)
    _write_csv(output / "token_group_metrics.csv", group_rows)

    age_path_summary: dict[str, Any] = {}
    for path in PATHS:
        age_path_summary[path] = {}
        for age in AGES:
            selected = [row for row in sequence_rows if int(row["age"]) == age]
            age_path_summary[path][str(age)] = _summarize_rows(selected, path)

    task_age_path_summary: dict[str, Any] = {}
    for task_id in sorted({int(row["task_id"]) for row in sequence_rows}):
        task_age_path_summary[str(task_id)] = {}
        for path in PATHS:
            task_age_path_summary[str(task_id)][path] = {}
            for age in AGES:
                selected = [
                    row
                    for row in sequence_rows
                    if int(row["task_id"]) == task_id and int(row["age"]) == age
                ]
                task_age_path_summary[str(task_id)][path][str(age)] = _summarize_rows(
                    selected, path
                )

    token_group_summary: dict[str, Any] = {}
    group_keys = sorted({(int(row["group_id"]), row["group_name"]) for row in group_rows})
    for group_id, group_name in group_keys:
        token_group_summary[group_name] = {}
        for path in PATHS:
            token_group_summary[group_name][path] = {}
            for age in AGES:
                selected = [
                    row
                    for row in group_rows
                    if int(row["group_id"]) == group_id
                    and row["path"] == path
                    and int(row["age"]) == age
                ]
                token_group_summary[group_name][path][str(age)] = {
                    field: _summary(float(row[field]) for row in selected)
                    for field in CONDITION_FIELDS
                }

    recurrence_excess: dict[str, Any] = {}
    recursive_teacher_summary: dict[str, Any] = {}
    for age in AGES:
        recursive = age_path_summary["recursive"][str(age)]
        teacher_forced = age_path_summary["teacher_forced"][str(age)]
        recurrence_excess[str(age)] = {}
        for field in (*CONDITION_FIELDS, *ACTION_FIELDS):
            recursive_mean = float(recursive[field]["mean"])
            teacher_mean = float(teacher_forced[field]["mean"])
            recurrence_excess[str(age)][field] = {
                "recursive_mean": recursive_mean,
                "teacher_forced_mean": teacher_mean,
                "recursive_minus_teacher_forced": recursive_mean - teacher_mean,
                "recursive_over_teacher_forced": (
                    recursive_mean / teacher_mean if abs(teacher_mean) > 1e-12 else None
                ),
            }
        selected = [row for row in sequence_rows if int(row["age"]) == age]
        recursive_teacher_summary[str(age)] = {
            field: _summary(
                float(row[f"recursive_vs_teacher/{field}"]) for row in selected
            )
            for field in (
                "condition_token_cosine_mean",
                "condition_flat_cosine",
                "condition_normalized_mse",
                "condition_raw_mse",
                "condition_max_abs",
                "action_first5_l1",
                "action_max_abs",
            )
        }

    tail_rows = sorted(
        sequence_rows,
        key=lambda row: float(row["recursive/action_gripper_max_abs_error"]),
        reverse=True,
    )[:100]
    _write_csv(output / "top100_recursive_gripper_tail.csv", tail_rows)
    finite = all(
        math.isfinite(float(value))
        for row in sequence_rows
        for key, value in row.items()
        if "/" in key and key not in {"episode_id"}
    )
    age1_condition_parity = max(
        float(row["recursive_vs_teacher/condition_max_abs"])
        for row in sequence_rows
        if int(row["age"]) == 1
    )
    age1_action_parity = max(
        float(row["recursive_vs_teacher/action_max_abs"])
        for row in sequence_rows
        if int(row["age"]) == 1
    )
    result = {
        "verdict": "P1_CONDITION_DRIFT_DIAGNOSTIC_COMPLETE",
        "classification": "OFFLINE_HELDOUT_CAUSAL_DIAGNOSTIC",
        "paper_table_eligible": False,
        "evaluated_sequences": expected_sequences,
        "rows": len(sequence_rows),
        "world_size": world_size,
        "dataset_split": contract["dataset_split"],
        "source_compatibility": contract["source_compatibility"],
        "diagnostic_source_lock": contract["diagnostic_source_lock"],
        "age_path_summary": age_path_summary,
        "task_age_path_summary": task_age_path_summary,
        "token_group_summary": token_group_summary,
        "recurrence_excess": recurrence_excess,
        "recursive_teacher_direct_summary": recursive_teacher_summary,
        "checks": {
            "all_metrics_finite": finite,
            "age1_recursive_teacher_condition_max_abs": age1_condition_parity,
            "age1_recursive_teacher_action_max_abs": age1_action_parity,
            "age1_paths_exact": age1_condition_parity == 0.0 and age1_action_parity == 0.0,
            "all_shards_complete": len(shard_summaries) == world_size,
        },
        "interpretation_contract": {
            "teacher_forced": "exact previous condition -> one updater step",
            "recursive": "previous predicted condition -> next updater step",
            "same_noise_actions": True,
            "causal_scope": (
                "Differences between recursive and teacher-forced paths isolate propagation "
                "from the previous predicted condition on the fixed offline observation sequence."
            ),
            "not_measured": "online closed-loop observation-distribution feedback",
        },
    }
    write_json(output / "condition_drift_p1_summary.json", result)
    return result


def _runtime() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _close_runtime(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, world_size, device = _runtime()
    try:
        selected_gpu_ids = [
            int(value) for value in os.environ.get("SIMVLA_GPU_IDS", "").split(",") if value
        ]
        visible_gpu_ids = [
            int(value)
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value
        ]
        if selected_gpu_ids != visible_gpu_ids or len(selected_gpu_ids) != world_size:
            raise RuntimeError(
                "SIMVLA_GPU_IDS, CUDA_VISIBLE_DEVICES, and DDP world size must agree"
            )
        configure_strict_torch_determinism(args.seed)
        model, checkpoint = load_native_v0_checkpoint(
            args.v0_checkpoint, device=device, require_final_150k=True
        )
        current_source = native_v0_source_manifest(
            checkpoint=args.checkpoint, norm_stats=args.norm_stats, cache=args.cache
        )
        source_compatibility = validate_scientific_source_compatibility(
            current_source, checkpoint["source_lock"]
        )
        checkpoint_source_hash = checkpoint["source_lock"]["combined_sha256"]
        require_gate(
            args.parity_gate,
            verdicts=("K1_HOOK_PARITY_PASS",),
            source_combined_sha256=checkpoint_source_hash,
        )
        require_gate(
            args.parameter_gate,
            verdicts=("PARAMETER_AUDIT_PASS",),
            source_combined_sha256=checkpoint_source_hash,
        )
        training_gate = require_gate(
            args.training_gate,
            verdicts=("FINAL_150K_TRAINING_COMPLETE",),
            source_combined_sha256=checkpoint_source_hash,
        )
        if int(training_gate.get("global_optimizer_step", -1)) != 150_000:
            raise RuntimeError("P1 diagnostic requires the final 150K condition checkpoint")

        dataset = NativeV0SequenceDataset(
            args.cache,
            split="heldout",
            heldout_fraction=args.heldout_fraction,
            split_seed=args.split_seed,
        )
        expected_splits = checkpoint["training_config"].get("dataset_splits")
        if training_gate.get("dataset_splits") != expected_splits:
            raise RuntimeError("training summary and checkpoint split contracts differ")
        if checkpoint["training_config"].get("dataset_contract_heldout") != dataset.contract():
            raise RuntimeError("P1 heldout split differs from checkpoint training contract")
        evaluated_sequences = (
            len(dataset)
            if int(args.max_sequences) <= 0
            else min(len(dataset), int(args.max_sequences))
        )
        output = Path(args.output).expanduser().resolve()
        exists = torch.tensor([int(output.exists())], device=device)
        if world_size > 1:
            dist.all_reduce(exists, op=dist.ReduceOp.MAX)
        if int(exists.item()):
            raise FileExistsError(f"refusing existing output: {output}")
        if rank == 0:
            output.mkdir(parents=True)
            write_json(
                output / "run_contract.json",
                {
                    "verdict": "P1_CONDITION_DRIFT_RUN_LOCKED",
                    "classification": "OFFLINE_HELDOUT_CAUSAL_DIAGNOSTIC",
                    "world_size": world_size,
                    "selected_physical_gpu_ids": selected_gpu_ids,
                    "evaluated_sequences": evaluated_sequences,
                    "full_heldout_sequences": len(dataset),
                    "max_sequences": int(args.max_sequences),
                    "dataset_split": dataset.contract(),
                    "checkpoint": str(Path(args.v0_checkpoint).expanduser().resolve()),
                    "checkpoint_step": int(checkpoint["global_optimizer_step"]),
                    "action_decode_mode": str(checkpoint["training_config"]["mode"]),
                    "source_compatibility": source_compatibility,
                    "diagnostic_source_lock": diagnostic_source_lock(),
                },
            )
        _barrier(world_size)

        _, processor, action_adapter = load_frozen_simvla(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            device=device,
        )
        decode_mode = str(checkpoint["training_config"]["mode"])
        sequence_rows: list[dict[str, Any]] = []
        group_rows: list[dict[str, Any]] = []
        shard = output / "shards" / f"rank_{rank}"
        shard.mkdir(parents=True)
        local_indices = list(range(rank, evaluated_sequences, world_size))
        with torch.no_grad():
            for local_count, dataset_index in enumerate(local_indices, start=1):
                batch = move_batch(
                    collate_native_v0_sequences([dataset[dataset_index]]), device
                )
                layout = cached_batch_token_layout(
                    condition=batch["anchor_condition"],
                    language_instructions=batch["language_instruction"],
                    processor=processor,
                )
                recursive = model(
                    batch["anchor_condition"],
                    batch["image_sequence"],
                    batch["proprio_sequence"],
                    valid_mask=layout.valid_mask,
                    group_ids=layout.group_ids,
                )
                teacher_updates = teacher_forced_updates(
                    model,
                    batch,
                    valid_mask=layout.valid_mask,
                    group_ids=layout.group_ids,
                )
                recursive_conditions = recursive.conditions
                teacher_conditions = tuple(update.condition for update in teacher_updates)
                exact_conditions = tuple(
                    batch["teacher_conditions"][:, age - 1] for age in AGES
                )
                recursive_actions = _decode_conditions(
                    action_adapter, recursive_conditions, batch, mode=decode_mode
                )
                teacher_actions = _decode_conditions(
                    action_adapter, teacher_conditions, batch, mode=decode_mode
                )
                exact_actions = _decode_conditions(
                    action_adapter, exact_conditions, batch, mode=decode_mode
                )

                identity = {
                    "dataset_index": dataset_index,
                    "task_id": int(batch["task_id"][0].item()),
                    "episode_id": str(batch["episode_id"][0]),
                    "anchor_query_index": int(batch["anchor_query_index"][0].item()),
                }
                for offset, age in enumerate(AGES):
                    row: dict[str, Any] = {**identity, "age": age}
                    path_values = {
                        "recursive": (
                            recursive_conditions[offset],
                            recursive_actions[offset],
                            recursive.updates[offset],
                        ),
                        "teacher_forced": (
                            teacher_conditions[offset],
                            teacher_actions[offset],
                            teacher_updates[offset],
                        ),
                    }
                    for path, (condition, action, update) in path_values.items():
                        condition_metrics = condition_fidelity_metrics(
                            condition, exact_conditions[offset], layout.valid_mask
                        )
                        action_metrics = action_fidelity_metrics(
                            action, exact_actions[offset]
                        )
                        for key, value in (*condition_metrics.items(), *action_metrics.items()):
                            row[f"{path}/{key}"] = value
                        valid = layout.valid_mask.unsqueeze(-1)
                        row[f"{path}/gate_mean"] = float(
                            update.gate.masked_select(valid.expand_as(update.gate))
                            .float()
                            .mean()
                            .item()
                        )
                        row[f"{path}/residual_rms"] = float(
                            update.residual.masked_select(valid.expand_as(update.residual))
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        )
                        for group_id, group_name in sorted(GROUP_NAMES.items()):
                            group_mask = layout.valid_mask & (layout.group_ids == group_id)
                            if not bool(group_mask.any()):
                                continue
                            group_metric = condition_fidelity_metrics(
                                condition, exact_conditions[offset], group_mask
                            )
                            group_rows.append(
                                {
                                    **identity,
                                    "age": age,
                                    "path": path,
                                    "group_id": group_id,
                                    "group_name": group_name,
                                    **{
                                        key: group_metric[key]
                                        for key in CONDITION_FIELDS
                                    },
                                    "tokens": group_metric["tokens"],
                                }
                            )
                    recursive_teacher_condition = condition_fidelity_metrics(
                        recursive_conditions[offset],
                        teacher_conditions[offset],
                        layout.valid_mask,
                    )
                    recursive_teacher_action = (
                        recursive_actions[offset].float() - teacher_actions[offset].float()
                    ).abs()
                    row["recursive_vs_teacher/condition_max_abs"] = (
                        recursive_teacher_condition["condition_max_abs"]
                    )
                    row["recursive_vs_teacher/condition_token_cosine_mean"] = (
                        recursive_teacher_condition["condition_token_cosine_mean"]
                    )
                    row["recursive_vs_teacher/condition_flat_cosine"] = (
                        recursive_teacher_condition["condition_flat_cosine"]
                    )
                    row["recursive_vs_teacher/condition_normalized_mse"] = (
                        recursive_teacher_condition["condition_normalized_mse"]
                    )
                    row["recursive_vs_teacher/condition_raw_mse"] = (
                        recursive_teacher_condition["condition_raw_mse"]
                    )
                    row["recursive_vs_teacher/action_max_abs"] = float(
                        recursive_teacher_action.max().item()
                    )
                    row["recursive_vs_teacher/action_first5_l1"] = float(
                        recursive_teacher_action[:, :5].mean().item()
                    )
                    sequence_rows.append(row)

                if local_count % int(args.flush_interval) == 0:
                    _write_csv(shard / "sequence_age_metrics.csv", sequence_rows)
                    _write_csv(shard / "token_group_metrics.csv", group_rows)
                if local_count % int(args.log_interval) == 0 or local_count == len(local_indices):
                    print(
                        f"rank={rank} sequences={local_count}/{len(local_indices)} "
                        f"global_target={evaluated_sequences}",
                        flush=True,
                    )

        _write_csv(shard / "sequence_age_metrics.csv", sequence_rows)
        _write_csv(shard / "token_group_metrics.csv", group_rows)
        write_json(
            shard / "shard_summary.json",
            {
                "verdict": "CONDITION_DRIFT_P1_SHARD_COMPLETE",
                "rank": rank,
                "world_size": world_size,
                "dataset_indices": local_indices,
                "sequences": len(local_indices),
                "sequence_age_rows": len(sequence_rows),
                "token_group_rows": len(group_rows),
            },
        )
        _barrier(world_size)
        result: dict[str, Any] = {}
        if rank == 0:
            result = _aggregate_output(output)
            print(json.dumps(result["checks"], indent=2, sort_keys=True), flush=True)
        _barrier(world_size)
        return result
    finally:
        _close_runtime(world_size)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--cache", required=True)
    evaluate_parser.add_argument("--v0-checkpoint", required=True)
    evaluate_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    evaluate_parser.add_argument("--norm-stats", required=True)
    evaluate_parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    evaluate_parser.add_argument("--parity-gate", required=True)
    evaluate_parser.add_argument("--parameter-gate", required=True)
    evaluate_parser.add_argument("--training-gate", required=True)
    evaluate_parser.add_argument("--heldout-fraction", type=float, default=0.2)
    evaluate_parser.add_argument("--split-seed", type=int, default=20260822)
    evaluate_parser.add_argument("--seed", type=int, default=20260815)
    evaluate_parser.add_argument("--max-sequences", type=int, default=0)
    evaluate_parser.add_argument("--flush-interval", type=int, default=25)
    evaluate_parser.add_argument("--log-interval", type=int, default=25)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args)
    else:
        result = _aggregate_output(Path(args.output).expanduser().resolve())
        print(json.dumps(result["checks"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
