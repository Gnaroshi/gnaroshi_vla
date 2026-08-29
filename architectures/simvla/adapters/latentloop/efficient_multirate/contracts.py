"""Pure contracts for the efficient SimVLA multirate campaign.

This module intentionally has no torch dependency. It is shared by command
guards, report generation, and lightweight tests. Scientific stages are
fail-closed: a stage is runnable only when every declared prerequisite gate
exists, has the expected verdict, and carries the same source lock.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CACHE_SCHEMA = "simvla_exact_teacher_query_cache_v1"
CACHE_SHARD_SCHEMA = "simvla_exact_teacher_query_shard_v1"
CACHE_MARKER_SCHEMA = "simvla_exact_teacher_shard_complete_v1"
SOURCE_LOCK_SCHEMA = "simvla_efficient_multirate_source_lock_v1"
STAGE_GRAPH_SCHEMA = "simvla_efficient_multirate_stage_graph_v1"

CONDITION_TOKENS = 122
CONDITION_DIM = 960
ACTION_HORIZON = 10
ACTION_DIM = 7
PROPRIO_DIM = 8
EXECUTION_HORIZON = 5
FIXED_K_C = 4
REFERENCE_EFFECTIVE_GLOBAL_BATCH = 1

PERMANENT_CACHE_LIMIT_BYTES = 80 * 2**30
TEMPORARY_CACHE_LIMIT_BYTES = 150 * 2**30
MINIMUM_FREE_AFTER_CACHE_BYTES = 1 * 2**40

GENERATION_SCHEDULES: dict[int, tuple[int, ...]] = {
    10: tuple(range(10)),
    5: (0, 2, 4, 6, 8),
    3: (0, 4, 8),
    2: (0, 5),
}
DISABLED_GENERATION_SCHEDULES: dict[int, tuple[int, ...]] = {
    1: (0,),
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@dataclass(frozen=True)
class ExactCacheProjection:
    query_count: int
    window_count: int
    condition_tokens: int
    condition_dim: int
    condition_dtype: str
    shard_queries: int
    shard_count: int
    condition_bytes: int
    valid_mask_bytes: int
    group_id_bytes: int
    teacher_action_bytes: int
    proprio_bytes: int
    query_seed_bytes: int
    query_id_bytes: int
    window_index_bytes: int
    tensor_bytes: int
    metadata_reserve_bytes: int
    projected_permanent_bytes: int
    projected_peak_temporary_bytes: int
    free_bytes_before: int | None
    projected_free_bytes_after: int | None
    permanent_under_80_gib: bool
    temporary_under_150_gib: bool
    at_least_1_tib_free_after: bool | None
    exact_fp32_storage_gate_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_exact_teacher_cache(
    *,
    query_count: int,
    window_count: int,
    condition_tokens: int = CONDITION_TOKENS,
    condition_dim: int = CONDITION_DIM,
    condition_element_bytes: int = 4,
    shard_queries: int = 1024,
    free_bytes_before: int | None = None,
    metadata_reserve_bytes_per_query: int = 1024,
) -> ExactCacheProjection:
    """Project a deduplicated query cache using source-native FP32 tensors."""

    integers = {
        "query_count": query_count,
        "window_count": window_count,
        "condition_tokens": condition_tokens,
        "condition_dim": condition_dim,
        "condition_element_bytes": condition_element_bytes,
        "shard_queries": shard_queries,
        "metadata_reserve_bytes_per_query": metadata_reserve_bytes_per_query,
    }
    if any(int(value) <= 0 for value in integers.values()):
        raise ValueError(f"cache projection values must be positive: {integers}")
    if condition_element_bytes != 4:
        raise ValueError("primary exact cache projection is fixed to source-native FP32")

    condition_bytes = query_count * condition_tokens * condition_dim * condition_element_bytes
    valid_mask_bytes = query_count * condition_tokens
    group_id_bytes = query_count * condition_tokens
    teacher_action_bytes = query_count * ACTION_HORIZON * ACTION_DIM * 4
    proprio_bytes = query_count * PROPRIO_DIM * 4
    query_seed_bytes = query_count * 8
    query_id_bytes = query_count * 8
    window_index_bytes = window_count * FIXED_K_C * 8
    tensor_bytes = sum(
        (
            condition_bytes,
            valid_mask_bytes,
            group_id_bytes,
            teacher_action_bytes,
            proprio_bytes,
            query_seed_bytes,
            query_id_bytes,
            window_index_bytes,
        )
    )
    metadata_reserve_bytes = query_count * metadata_reserve_bytes_per_query
    projected_permanent = tensor_bytes + metadata_reserve_bytes
    largest_shard_queries = min(query_count, shard_queries)
    largest_shard_bytes = (
        largest_shard_queries
        * (
            condition_tokens * condition_dim * condition_element_bytes
            + condition_tokens
            + condition_tokens
            + ACTION_HORIZON * ACTION_DIM * 4
            + PROPRIO_DIM * 4
            + 16
            + metadata_reserve_bytes_per_query
        )
    )
    projected_temporary = projected_permanent + 2 * largest_shard_bytes
    free_after = (
        None
        if free_bytes_before is None
        else int(free_bytes_before) - projected_permanent
    )
    permanent_pass = projected_permanent <= PERMANENT_CACHE_LIMIT_BYTES
    temporary_pass = projected_temporary <= TEMPORARY_CACHE_LIMIT_BYTES
    free_pass = None if free_after is None else free_after >= MINIMUM_FREE_AFTER_CACHE_BYTES
    overall = permanent_pass and temporary_pass and free_pass is True
    return ExactCacheProjection(
        query_count=int(query_count),
        window_count=int(window_count),
        condition_tokens=int(condition_tokens),
        condition_dim=int(condition_dim),
        condition_dtype="torch.float32",
        shard_queries=int(shard_queries),
        shard_count=math.ceil(query_count / shard_queries),
        condition_bytes=condition_bytes,
        valid_mask_bytes=valid_mask_bytes,
        group_id_bytes=group_id_bytes,
        teacher_action_bytes=teacher_action_bytes,
        proprio_bytes=proprio_bytes,
        query_seed_bytes=query_seed_bytes,
        query_id_bytes=query_id_bytes,
        window_index_bytes=window_index_bytes,
        tensor_bytes=tensor_bytes,
        metadata_reserve_bytes=metadata_reserve_bytes,
        projected_permanent_bytes=projected_permanent,
        projected_peak_temporary_bytes=projected_temporary,
        free_bytes_before=free_bytes_before,
        projected_free_bytes_after=free_after,
        permanent_under_80_gib=permanent_pass,
        temporary_under_150_gib=temporary_pass,
        at_least_1_tib_free_after=free_pass,
        exact_fp32_storage_gate_pass=overall,
    )


def query_identity(task_id: int, episode_id: str, query_index: int) -> str:
    if int(task_id) < 0 or int(query_index) < 0 or not str(episode_id):
        raise ValueError("invalid query identity")
    return f"{int(task_id):04d}|{episode_id}|{int(query_index):08d}"


def validate_query_windows(
    windows: Sequence[Sequence[str]],
    *,
    require_global_deduplication: bool = True,
) -> dict[str, Any]:
    """Validate q0-q1-q2-q3 windows and query-level deduplication."""

    flattened: list[str] = []
    errors: list[str] = []
    for index, window in enumerate(windows):
        if len(window) != FIXED_K_C:
            errors.append(f"window {index} has {len(window)} IDs, expected {FIXED_K_C}")
        if len(set(window)) != len(window):
            errors.append(f"window {index} repeats a query ID")
        flattened.extend(str(item) for item in window)
    unique = set(flattened)
    if require_global_deduplication and len(unique) != len(flattened):
        errors.append("query tensor would be duplicated across windows")
    return {
        "windows": len(windows),
        "query_references": len(flattened),
        "unique_queries": len(unique),
        "global_query_deduplication": len(unique) == len(flattened),
        "errors": errors,
        "passed": not errors,
    }


def balanced_mode_d_age(zero_based_optimizer_step: int) -> int:
    """Deterministic rank-independent age cycle 1 -> 2 -> 3."""

    if int(zero_based_optimizer_step) < 0:
        raise ValueError("optimizer step must be non-negative")
    return int(zero_based_optimizer_step) % 3 + 1


def mode_d_age_counts(start_step: int, measured_steps: int) -> dict[int, int]:
    if int(start_step) < 0 or int(measured_steps) < 1:
        raise ValueError("invalid Mode D measurement range")
    counts = {1: 0, 2: 0, 3: 0}
    for step in range(int(start_step), int(start_step) + int(measured_steps)):
        counts[balanced_mode_d_age(step)] += 1
    return counts


def native_nfe_time_grid(nfe: int) -> tuple[float, ...]:
    """Return the exact Euler evaluation times used by upstream SimVLA."""

    if int(nfe) < 1:
        raise ValueError("NFE must be positive")
    value = int(nfe)
    return tuple(1.0 - index / value for index in range(value))


def validate_generation_schedule(n_g: int, full_steps: Sequence[int]) -> dict[str, Any]:
    expected = GENERATION_SCHEDULES.get(int(n_g))
    disabled = DISABLED_GENERATION_SCHEDULES.get(int(n_g))
    observed = tuple(int(value) for value in full_steps)
    return {
        "n_g": int(n_g),
        "full_steps": list(observed),
        "expected_full_steps": list(expected or disabled or ()),
        "enabled": expected is not None,
        "matches_contract": expected == observed,
        "disabled_until_n_g3_passes": disabled is not None,
    }


def effective_batch_contract(
    *,
    local_unique_batch: int,
    gradient_accumulation_steps: int,
    world_size: int = 2,
    replicated_logical_sample: bool = True,
    reference_effective_global_batch: int = REFERENCE_EFFECTIVE_GLOBAL_BATCH,
) -> dict[str, Any]:
    """Compute unique samples/update without mistaking replicas for samples."""

    values = (local_unique_batch, gradient_accumulation_steps, world_size)
    if any(int(value) < 1 for value in values):
        raise ValueError("batch, accumulation, and world size must be positive")
    rank_multiplier = 1 if replicated_logical_sample else int(world_size)
    effective = int(local_unique_batch) * int(gradient_accumulation_steps) * rank_multiplier
    return {
        "local_unique_batch": int(local_unique_batch),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "world_size": int(world_size),
        "replicated_logical_sample": bool(replicated_logical_sample),
        "physical_examples_per_update": (
            int(local_unique_batch) * int(gradient_accumulation_steps) * int(world_size)
        ),
        "effective_unique_global_batch": effective,
        "reference_effective_global_batch": int(reference_effective_global_batch),
        "preserves_reference": effective == int(reference_effective_global_batch),
    }


def mode_ab_pass(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "total_loss_relative_difference_le_0_005": (
            float(metrics["max_total_loss_relative_difference"]) <= 0.005
        ),
        "first_r_loss_relative_difference_le_0_005": (
            float(metrics["max_first5_loss_relative_difference"]) <= 0.005
        ),
        "gradient_cosine_ge_0_999": float(metrics["min_gradient_cosine"]) >= 0.999,
        "gradient_relative_error_le_0_01": (
            float(metrics["max_gradient_relative_error"]) <= 0.01
        ),
        "all_ages_represented": bool(metrics["all_ages_represented"]),
        "all_finite": bool(metrics["all_finite"]),
        "speedup_ge_1_5": float(metrics["median_speedup"]) >= 1.5,
        "peak_vram_fits": bool(metrics["mode_b_peak_vram_fits"]),
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "verdict": "MODE_B_APPROVED" if passed else "MODE_A_REQUIRED",
    }


def mode_d_pass(metrics: Mapping[str, Any]) -> dict[str, Any]:
    counts = {int(key): int(value) for key, value in metrics["age_counts"].items()}
    balanced = max(counts.values()) - min(counts.values()) <= 1 and set(counts) == {1, 2, 3}
    checks = {
        "heldout_first_r_mean_le_1_05x": float(metrics["heldout_first_r_ratio"]) <= 1.05,
        "age3_first_r_p95_le_1_10x": float(metrics["age3_first_r_p95_ratio"]) <= 1.10,
        "no_gripper_collapse": bool(metrics["no_gripper_collapse"]),
        "stable_training_curve": bool(metrics["stable_training_curve"]),
        "balanced_age_cycle": balanced,
        "speedup_ge_1_5": float(metrics["step_time_speedup"]) >= 1.5,
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "verdict": "MODE_D_APPROVED" if passed else "MODE_D_REJECTED",
    }


def wallclock_projection(
    *,
    mean_step_seconds: float,
    measured_steps: int,
    total_steps: int = 150_000,
    amortized_validation_checkpoint_seconds: float = 0.0,
    scientific_parity_gates_pass: bool,
    objective_mode_approved: bool,
    target_hours: float = 12.0,
) -> dict[str, Any]:
    if not math.isfinite(mean_step_seconds) or mean_step_seconds <= 0:
        raise ValueError("mean step time must be finite and positive")
    if int(measured_steps) < 1_000:
        raise ValueError("wall-clock gate requires at least 1,000 measured steps")
    training_seconds = float(mean_step_seconds) * int(total_steps)
    projected_seconds = training_seconds + float(amortized_validation_checkpoint_seconds)
    projected_hours = projected_seconds / 3600.0
    checks = {
        "measurement_window_ge_1000": int(measured_steps) >= 1_000,
        "scientific_parity_gates_pass": bool(scientific_parity_gates_pass),
        "objective_mode_approved": bool(objective_mode_approved),
        "projected_hours_le_target": projected_hours <= float(target_hours),
    }
    approved = all(checks.values())
    return {
        "mean_step_seconds": float(mean_step_seconds),
        "measured_steps": int(measured_steps),
        "total_steps": int(total_steps),
        "training_seconds": training_seconds,
        "amortized_validation_checkpoint_seconds": float(
            amortized_validation_checkpoint_seconds
        ),
        "projected_seconds": projected_seconds,
        "projected_hours": projected_hours,
        "target_hours": float(target_hours),
        "checks": checks,
        "verdict": (
            "TRAIN_150K_APPROVED"
            if approved
            else "TRAINING_OPTIMIZATION_INSUFFICIENT"
        ),
    }


def libero_long_500_episode_keys(seed: int = 20260815) -> list[dict[str, int | str]]:
    return [
        {
            "suite": "libero_10",
            "task_id": task_id,
            "trial_id": trial_id,
            "init_state_index": trial_id,
            "environment_seed": int(seed),
        }
        for task_id in range(10)
        for trial_id in range(50)
    ]


@dataclass(frozen=True)
class StageSpec:
    stage: str
    lane: str
    description: str
    prerequisites: tuple[str, ...]
    pass_verdicts: tuple[str, ...]
    automatic_launch: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prerequisites"] = list(self.prerequisites)
        payload["pass_verdicts"] = list(self.pass_verdicts)
        return payload


STAGE_GRAPH: tuple[StageSpec, ...] = (
    StageSpec("0", "condition", "artifact/source/current-run audit", (), ("STAGE0_AUDIT_PASS",)),
    StageSpec("1", "condition", "current streaming V0 profile", ("0",), ("STREAMING_PROFILE_COMPLETE",)),
    StageSpec("2", "condition", "exact teacher-cache projection and pilot", ("0",), ("EXACT_TEACHER_CACHE_PASS",)),
    StageSpec("3", "condition", "exact teacher-cache generation", ("2",), ("EXACT_TEACHER_CACHE_COMPLETE",)),
    StageSpec("4", "condition", "cache-backed Mode A/B benchmark", ("3",), ("MODE_B_APPROVED", "MODE_A_REQUIRED")),
    StageSpec("5", "condition", "batch-size throughput benchmark", ("4",), ("BATCH_CONFIGURATION_SELECTED",)),
    StageSpec("6", "condition", "optional Mode D benchmark", ("5",), ("MODE_D_APPROVED", "MODE_D_REJECTED", "MODE_D_NOT_REQUIRED")),
    StageSpec("7", "condition", "V0 wall-clock gate", ("5", "6"), ("TRAIN_150K_APPROVED", "TRAINING_OPTIMIZATION_INSUFFICIENT")),
    StageSpec("8", "condition", "optimized V0 150K", ("7",), ("FINAL_150K_TRAINING_COMPLETE",)),
    StageSpec("9", "condition", "V0 offline gate", ("8",), ("OFFLINE_K4_GATE_PASS",)),
    StageSpec("10", "condition", "V0 LIBERO-Long 500", ("9",), ("LIBERO_LONG_500_COMPLETE",)),
    StageSpec("G0", "generation", "generation hidden-hook parity", ("0",), ("GENERATOR_HIDDEN_HOOK_PASS",)),
    StageSpec("G1", "generation", "naive NFE audit", ("G0",), ("NAIVE_NFE_AUDIT_COMPLETE",)),
    StageSpec("G2", "generation", "Generation Loop implementation", ("G0",), ("GENERATION_LOOP_IMPLEMENTATION_PASS",)),
    StageSpec("G3", "generation", "Generation Loop training", ("G1", "G2", "3"), ("GENERATION_LOOP_TRAINING_COMPLETE",)),
    StageSpec("G4", "generation", "Generation Loop offline gate", ("G3",), ("GENERATION_LOOP_OFFLINE_PASS",)),
    StageSpec("G5", "generation", "Generation Loop-only LIBERO-Long 500", ("G4",), ("GENERATION_LOOP_LONG_500_COMPLETE",)),
    StageSpec("C0", "coupled", "condition-change-code parity", ("9", "G4"), ("CONDITION_CHANGE_CODE_PARITY_PASS",)),
    StageSpec("C1", "coupled", "coupling adapter", ("C0",), ("COUPLING_ADAPTER_PASS",)),
    StageSpec("C2", "coupled", "fixed-budget 2x2 LIBERO-Long", ("10", "G5", "C1"), ("FIXED_BUDGET_2X2_COMPLETE",)),
    StageSpec("C3", "coupled", "other-suite expansion", ("C2",), ("OTHER_SUITE_EXPANSION_COMPLETE",)),
    StageSpec("C4", "coupled", "dynamic N_G preparation", ("C2",), ("DYNAMIC_NG_PREPARATION_COMPLETE",)),
)


def stage_graph_payload() -> dict[str, Any]:
    return {
        "schema_version": STAGE_GRAPH_SCHEMA,
        "automatic_stage_launch": False,
        "dynamic_k_c_enabled": False,
        "dynamic_n_g_enabled": False,
        "stages": [stage.to_dict() for stage in STAGE_GRAPH],
    }


def stage_by_name(stage: str) -> StageSpec:
    matches = [item for item in STAGE_GRAPH if item.stage == str(stage)]
    if len(matches) != 1:
        raise KeyError(f"unknown stage {stage!r}")
    return matches[0]


def stage_readiness(stage: str, completed_verdicts: Mapping[str, str]) -> dict[str, Any]:
    """Evaluate prerequisites without launching any stage."""

    requested = stage_by_name(stage)
    missing: list[str] = []
    invalid: list[dict[str, Any]] = []
    for prerequisite_name in requested.prerequisites:
        prerequisite = stage_by_name(prerequisite_name)
        observed = completed_verdicts.get(prerequisite_name)
        if observed is None:
            missing.append(prerequisite_name)
        elif observed not in prerequisite.pass_verdicts:
            invalid.append(
                {
                    "stage": prerequisite_name,
                    "observed": observed,
                    "allowed": list(prerequisite.pass_verdicts),
                }
            )
    ready = not missing and not invalid
    return {
        "stage": requested.stage,
        "ready": ready,
        "verdict": "STAGE_READY" if ready else "STAGE_BLOCKED",
        "missing_prerequisites": missing,
        "invalid_prerequisites": invalid,
        "automatic_launch": False,
    }


def reference_noninterference(
    *,
    active_result_root: str | Path,
    optimized_result_root: str | Path,
    active_source_files: Sequence[str | Path],
    modified_files: Sequence[str | Path],
) -> dict[str, Any]:
    """Prove the optimized path neither writes nor edits the active reference."""

    active_result = Path(active_result_root).expanduser().resolve()
    optimized_result = Path(optimized_result_root).expanduser().resolve()
    result_disjoint = (
        active_result != optimized_result
        and active_result not in optimized_result.parents
        and optimized_result not in active_result.parents
    )
    active_sources = {Path(value).expanduser().resolve() for value in active_source_files}
    modified = {Path(value).expanduser().resolve() for value in modified_files}
    source_overlap = sorted(str(value) for value in active_sources & modified)
    passed = result_disjoint and not source_overlap
    return {
        "verdict": "STREAMING_REFERENCE_UNTOUCHED" if passed else "REFERENCE_INTERFERENCE",
        "active_result_root": str(active_result),
        "optimized_result_root": str(optimized_result),
        "result_roots_disjoint": result_disjoint,
        "active_source_files_modified": source_overlap,
        "passed": passed,
    }


def require_gate_payload(
    path: str | Path,
    *,
    verdicts: Iterable[str],
    source_combined_sha256: str,
) -> dict[str, Any]:
    gate_path = Path(path).expanduser().resolve()
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    allowed = set(str(value) for value in verdicts)
    if str(payload.get("verdict")) not in allowed:
        raise RuntimeError(
            f"gate {gate_path} verdict={payload.get('verdict')!r}, expected {sorted(allowed)}"
        )
    if payload.get("source_combined_sha256") != source_combined_sha256:
        raise RuntimeError(f"gate {gate_path} uses a different source lock")
    return payload


def build_source_lock(
    *,
    repository: str | Path,
    parent_source_lock: str | Path,
    parent_training_config: str | Path,
    checkpoint: str,
    checkpoint_revision: str,
    norm_stats: str | Path,
    compact_cache_manifest: str | Path,
    source_files: Sequence[str | Path],
) -> dict[str, Any]:
    root = Path(repository).expanduser().resolve()
    parent_path = Path(parent_source_lock).expanduser().resolve()
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_combined = str(parent_payload.get("combined_sha256", ""))
    if not parent_combined:
        raise ValueError("parent source lock has no combined_sha256")
    norm_path = Path(norm_stats).expanduser().resolve()
    compact_path = Path(compact_cache_manifest).expanduser().resolve()
    compact_payload = json.loads(compact_path.read_text(encoding="utf-8"))
    parent_training_path = Path(parent_training_config).expanduser().resolve()
    parent_training_payload = json.loads(
        parent_training_path.read_text(encoding="utf-8")
    )
    files = {
        str(Path(path).resolve().relative_to(root)): sha256_file(path)
        for path in sorted((Path(value).expanduser().resolve() for value in source_files))
    }
    scientific = {
        "schema_version": SOURCE_LOCK_SCHEMA,
        "experiment_identifier": "simvla_efficient_coupled_multirate_latentloop",
        "architecture": "SimVLA",
        "method": "LatentLoop",
        "parent_source_lock_path": str(parent_path),
        "parent_source_lock_sha256": sha256_file(parent_path),
        "parent_source_combined_sha256": parent_combined,
        "parent_training_config_path": str(parent_training_path),
        "parent_training_config_sha256": sha256_file(parent_training_path),
        "dataset_splits": parent_training_payload["dataset_splits"],
        "checkpoint": str(checkpoint),
        "checkpoint_revision": str(checkpoint_revision),
        "norm_stats_path": str(norm_path),
        "norm_stats_sha256": sha256_file(norm_path),
        "compact_cache_manifest_path": str(compact_path),
        "compact_cache_manifest_sha256": sha256_file(compact_path),
        "action_noise_seed_base": int(
            compact_payload["metadata"]["action_noise_seed_base"]
        ),
        "action_horizon": ACTION_HORIZON,
        "execution_horizon": EXECUTION_HORIZON,
        "flow_steps": 10,
        "fixed_k_c": FIXED_K_C,
        "condition_tokens": CONDITION_TOKENS,
        "condition_dim": CONDITION_DIM,
        "source_file_sha256": files,
        "frozen_modules": ["vlm", "action_transformer", "action_decoder"],
        "forbidden_method_changes": [
            "action_correction",
            "new_action_head",
            "executed_action_history",
            "direct_long_gap_prediction",
            "dynamic_k_c",
            "joint_base_finetuning",
        ],
    }
    scientific["combined_sha256"] = canonical_sha256(scientific)
    return scientific
