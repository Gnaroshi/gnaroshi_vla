"""Resumable EGL LIBERO-Long evaluation for action-equivalent refresh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.action_equivalent_refresh.checkpoint import (
    load_action_fidelity_checkpoint,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.policy import (
    ActionEquivalentRefreshSimVLAPolicy,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_CHECKPOINT_REVISION,
    FROZEN_GENERATION_CHECKPOINT_SHA256,
    FROZEN_GENERATION_SOURCE_SHA256,
    FROZEN_NORM_STATS_SHA256,
    FROZEN_UPSTREAM_COMMIT,
    atomic_write_json,
    load_json,
    require_egl_preflight,
    runtime_versions,
    sha256_file,
    validate_manifest_identity,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import (
    FROZEN_CONDITION_CHECKPOINT_SHA256,
    FROZEN_CONDITION_SOURCE_SHA256,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_long_eval import (
    _trajectory_metrics,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
)
from architectures.simvla.wrappers.dcld_eval import rollout_runner as rollout_runtime
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)


ROOT = Path(__file__).resolve().parents[5]
UPSTREAM = Path(
    os.environ.get(
        "SIMVLA_UPSTREAM_ROOT",
        ROOT / "architectures" / "simvla" / "upstream",
    )
).expanduser().resolve()

ROW = "action_equivalent_refresh_ng3"
EPISODE_SCHEMA = "simvla_action_equivalent_refresh_online_episode_v1"
SHARD_VERDICT = "ACTION_EQUIVALENT_REFRESH_ONLINE_SHARD_COMPLETE"
SMOKE_VERDICT = "ACTION_EQUIVALENT_REFRESH_ONLINE_SMOKE_COMPLETE"
EXPECTED_RISK_CHECKPOINT_SHA256 = (
    "7203bc7214dad3cd793feaa156c3cb89547ee8ba20faa3ba894d98008bdf4654"
)

SOURCE_FILES = (
    "methods/latentloop/modules/action_equivalent_refresh.py",
    "methods/latentloop/modules/native_simvla_v0.py",
    "methods/latentloop/modules/simvla_generation_loop.py",
    "architectures/simvla/adapters/latentloop/action_equivalent_refresh/checkpoint.py",
    "architectures/simvla/adapters/latentloop/action_equivalent_refresh/features.py",
    "architectures/simvla/adapters/latentloop/action_equivalent_refresh/policy.py",
    "architectures/simvla/adapters/latentloop/action_equivalent_refresh/online_evaluator.py",
    "architectures/simvla/adapters/latentloop/action_equivalent_refresh/online_aggregate.py",
    "architectures/simvla/adapters/latentloop/action_equivalent_refresh/three_seed_aggregate.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_checkpoint.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_hidden.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_policy.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/fixed_2x2_eval.py",
    "architectures/simvla/adapters/latentloop/native_v0_checkpoint.py",
    "architectures/simvla/adapters/latentloop/native_v0_policy.py",
    "architectures/simvla/wrappers/dcld_eval/rollout_runner.py",
    "architectures/simvla/wrappers/run_action_equivalent_refresh_online_sd1.sh",
    "architectures/simvla/wrappers/run_action_equivalent_refresh_three_seed_rb2.sh",
)

PRODUCTION_GPU_IDS = {
    "SD1_HOST_LOCAL_EGL_LONG500": {4, 5, 6, 7},
    "RB2_HOST_LOCAL_EGL_LONG500": {0},
}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_task_ids(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("task IDs must be a non-empty unique CSV")
    if any(value < 0 or value > 9 for value in values):
        raise ValueError("LIBERO-Long task IDs must be in [0,9]")
    return values


def _episode_identity(task_id: int, trial_id: int) -> str:
    return f"task_{int(task_id):02d}_trial_{int(trial_id):03d}"


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _sum(values: Sequence[float]) -> float:
    return float(np.sum(values)) if values else 0.0


def _gripper_metrics(actions: Sequence[np.ndarray]) -> dict[str, float | int]:
    if len(actions) < 2:
        return {"gripper_switches": 0, "gripper_switch_rate": 0.0}
    array = np.asarray(actions, dtype=np.float32)
    switches = int(
        np.count_nonzero((array[1:, 6] >= 0) != (array[:-1, 6] >= 0))
    )
    return {
        "gripper_switches": switches,
        "gripper_switch_rate": float(switches / max(1, len(actions) - 1)),
    }


def _validate_selective_counters(
    counters: Mapping[str, int], *, decision_count: int
) -> dict[str, Any]:
    observed = {
        "policy_queries": int(counters.get("num_policy_queries", 0)),
        "full_vlm_calls": int(counters.get("num_full_vlm_calls", 0)),
        "candidate_condition_updates": int(
            counters.get("num_candidate_condition_updates", 0)
        ),
        "condition_updater_calls": int(
            counters.get("num_condition_updater_calls", 0)
        ),
        "risk_head_calls": int(counters.get("num_risk_head_calls", 0)),
        "accepted_condition_updates": int(
            counters.get("num_accepted_condition_updates", 0)
        ),
        "rejected_condition_updates": int(
            counters.get("num_rejected_condition_updates", 0)
        ),
        "forced_age_refreshes": int(
            counters.get("num_forced_age_refreshes", 0)
        ),
        "risk_triggered_refreshes": int(
            counters.get("num_risk_triggered_refreshes", 0)
        ),
        "action_transformer_decodes": int(
            counters.get("num_action_transformer_decodes", 0)
        ),
        "full_action_transformer_evaluations": int(
            counters.get("num_action_transformer_calls", 0)
        ),
        "generation_loop_updates": int(
            counters.get("num_generation_decoder_only_steps", 0)
        ),
    }
    q = observed["policy_queries"]
    f = observed["full_vlm_calls"]
    c = observed["candidate_condition_updates"]
    a = observed["accepted_condition_updates"]
    r = observed["rejected_condition_updates"]
    forced = observed["forced_age_refreshes"]
    risk_exact = observed["risk_triggered_refreshes"]
    checks = {
        "positive_query_count": q > 0,
        "one_condition_source_per_query": q == f + a,
        "candidate_equals_condition_updater": c
        == observed["condition_updater_calls"],
        "candidate_equals_risk_calls": c == observed["risk_head_calls"],
        "candidate_partition": c == a + r,
        "rejected_equals_risk_exact": r == risk_exact,
        "exact_reason_partition": f == 1 + forced + risk_exact,
        "one_action_decode_per_query": observed["action_transformer_decodes"] == q,
        "ng3_full_evaluations": observed[
            "full_action_transformer_evaluations"
        ]
        == 3 * q,
        "ng3_learned_updates": observed["generation_loop_updates"] == 7 * q,
        "ten_integration_updates": (
            observed["full_action_transformer_evaluations"]
            + observed["generation_loop_updates"]
        )
        == 10 * q,
        "one_decision_per_query": int(decision_count) == q,
    }
    return {
        "verdict": (
            "ACTION_EQUIVALENT_REFRESH_COUNTER_PASS"
            if all(checks.values())
            else "ACTION_EQUIVALENT_REFRESH_COUNTER_FAIL"
        ),
        "observed": observed,
        "checks": checks,
    }


def _verify_offline_gate(root: Path, risk_checkpoint: Path) -> dict[str, Any]:
    status = (root / "pipeline.status").read_text(encoding="utf-8").strip()
    training = load_json(root / "risk_head_2k" / "training_summary.json")
    comparison = load_json(root / "final_offline_comparison.json")
    checks = {
        "pipeline_complete": status == "ACTION_EQUIVALENT_REFRESH_OFFLINE_COMPLETE",
        "training_complete": training.get("verdict")
        == "ACTION_FIDELITY_HEAD_TRAINING_COMPLETE",
        "sequential_calibration": training.get("calibration_mode")
        == "reachable_all_anchor_q0_q3",
        "offline_complete": comparison.get("verdict")
        == "OFFLINE_ACTION_FIDELITY_COMPARISON_COMPLETE",
        "manual_candidate_supported": bool(
            comparison.get("comparison", {}).get("online_candidate")
        ),
        "risk_checkpoint_filename": Path(
            str(comparison.get("checkpoint", ""))
        ).name
        == risk_checkpoint.name,
    }
    if not all(checks.values()):
        raise RuntimeError(f"offline selector gate failed: {checks}")
    return {
        "verdict": "ACTION_EQUIVALENT_REFRESH_OFFLINE_GATE_PASS",
        "checks": checks,
        "training_summary": str(root / "risk_head_2k" / "training_summary.json"),
        "offline_comparison": str(root / "final_offline_comparison.json"),
        "target_exact_fraction": float(comparison["target_exact_fraction"]),
    }


def _verify_provenance(args: argparse.Namespace) -> dict[str, Any]:
    root_commit = _git(ROOT, "rev-parse", "HEAD")
    tracked_status = _git(ROOT, "status", "--porcelain", "--untracked-files=no")
    upstream_commit = _git(UPSTREAM, "rev-parse", "HEAD")
    expected_root_commit = str(args.expected_root_commit)
    failures: list[str] = []
    if root_commit != expected_root_commit:
        failures.append(f"root commit mismatch: {root_commit} != {expected_root_commit}")
    if tracked_status:
        failures.append("tracked worktree changes are present")
    if upstream_commit != FROZEN_UPSTREAM_COMMIT:
        failures.append(
            f"upstream commit mismatch: {upstream_commit} != {FROZEN_UPSTREAM_COMMIT}"
        )

    artifacts = {
        "risk_checkpoint": Path(args.risk_checkpoint).expanduser().resolve(),
        "condition_checkpoint": Path(args.condition_checkpoint).expanduser().resolve(),
        "generation_checkpoint": Path(args.generation_checkpoint).expanduser().resolve(),
        "norm_stats": Path(args.norm_stats).expanduser().resolve(),
    }
    expected_hashes = {
        "risk_checkpoint": EXPECTED_RISK_CHECKPOINT_SHA256,
        "condition_checkpoint": FROZEN_CONDITION_CHECKPOINT_SHA256,
        "generation_checkpoint": FROZEN_GENERATION_CHECKPOINT_SHA256,
        "norm_stats": FROZEN_NORM_STATS_SHA256,
    }
    observed_hashes = {
        name: sha256_file(path) if path.is_file() else None
        for name, path in artifacts.items()
    }
    for name, expected in expected_hashes.items():
        if observed_hashes[name] != expected:
            failures.append(
                f"{name} SHA-256 mismatch: {observed_hashes[name]} != {expected}"
            )

    hf_ref = (
        Path(os.environ.get("HF_HOME", "")).expanduser().resolve()
        / "hub"
        / "models--YuankaiLuo--SimVLA-LIBERO"
        / "refs"
        / "main"
    )
    hf_revision = hf_ref.read_text(encoding="utf-8").strip() if hf_ref.is_file() else None
    if hf_revision != FROZEN_CHECKPOINT_REVISION:
        failures.append(
            f"local SimVLA HF revision mismatch: {hf_revision} != {FROZEN_CHECKPOINT_REVISION}"
        )

    file_hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing locked source file: {relative}")
        else:
            file_hashes[relative] = sha256_file(path)

    offline_gate = _verify_offline_gate(
        Path(args.offline_root).expanduser().resolve(), artifacts["risk_checkpoint"]
    )
    scientific = {
        "schema_version": "simvla_action_equivalent_refresh_online_source_v1",
        "root_commit": root_commit,
        "upstream_commit": upstream_commit,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": hf_revision,
        "artifact_sha256": observed_hashes,
        "source_file_sha256": file_hashes,
        "manifest_sha256": args.expected_manifest_sha256,
        "offline_gate": offline_gate,
        "policy_contract": {
            "action_horizon": 10,
            "execution_horizon": 5,
            "generation_n_g": 3,
            "maximum_approximate_age": 3,
            "target_exact_fraction": 1.0 / 3.0,
            "dynamic_action_execution": False,
            "dynamic_generation_n_g": False,
        },
    }
    scientific["combined_sha256"] = _canonical_sha256(scientific)
    report = {
        **scientific,
        "verdict": (
            "ACTION_EQUIVALENT_REFRESH_PROVENANCE_PASS"
            if not failures
            else "ACTION_EQUIVALENT_REFRESH_PROVENANCE_FAIL"
        ),
        "tracked_status": tracked_status,
        "runtime": runtime_versions(),
        "paths": {name: str(path) for name, path in artifacts.items()},
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


def _load_completed_episode(
    path: Path,
    *,
    manifest_sha256: str,
    source_sha256: str,
    task_id: int,
    trial_id: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = load_json(path)
    checks = {
        "schema": payload.get("schema_version") == EPISODE_SCHEMA,
        "verdict": payload.get("verdict")
        == "ACTION_EQUIVALENT_REFRESH_EPISODE_COMPLETE",
        "manifest": payload.get("manifest_sha256") == manifest_sha256,
        "source": payload.get("source_combined_sha256") == source_sha256,
        "task": int(payload.get("metrics", {}).get("task_id", -1)) == int(task_id),
        "trial": int(payload.get("metrics", {}).get("trial_id", -1))
        == int(trial_id),
        "counter_gate": payload.get("metrics", {}).get("counter_gate")
        == "ACTION_EQUIVALENT_REFRESH_COUNTER_PASS",
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid completed episode {path}: {checks}")
    return payload


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty episode table")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finalize_shard(
    *,
    output: Path,
    episode_specs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    physical_gpu_id: int,
    task_ids: Sequence[int],
    classification: str,
) -> dict[str, Any]:
    episode_payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in episode_specs:
        identity = _episode_identity(int(spec["task_id"]), int(spec["trial_id"]))
        path = output / "episodes" / f"{identity}.json"
        payload = _load_completed_episode(
            path,
            manifest_sha256=str(manifest["manifest_sha256"]),
            source_sha256=str(provenance["combined_sha256"]),
            task_id=int(spec["task_id"]),
            trial_id=int(spec["trial_id"]),
        )
        if payload is None:
            missing.append(identity)
        else:
            episode_payloads.append(payload)
    if missing:
        raise RuntimeError(
            f"shard is missing {len(missing)} episode files; first={missing[:5]}"
        )

    episode_payloads.sort(
        key=lambda item: (
            int(item["metrics"]["task_id"]),
            int(item["metrics"]["trial_id"]),
        )
    )
    metrics = [dict(item["metrics"]) for item in episode_payloads]
    decisions: list[dict[str, Any]] = []
    for payload in episode_payloads:
        base = {
            "task_id": int(payload["metrics"]["task_id"]),
            "trial_id": int(payload["metrics"]["trial_id"]),
        }
        decisions.extend({**base, **row} for row in payload["route_decisions"])
    _write_csv_atomic(output / "episode_metrics.csv", metrics)
    _write_jsonl_atomic(output / "route_decisions.jsonl", decisions)

    total_queries = sum(int(row["num_policy_queries"]) for row in metrics)
    total_full = sum(int(row["num_full_vlm_calls"]) for row in metrics)
    total_candidates = sum(
        int(row["num_candidate_condition_updates"]) for row in metrics
    )
    total_accepted = sum(
        int(row["num_accepted_condition_updates"]) for row in metrics
    )
    total_rejected = sum(
        int(row["num_rejected_condition_updates"]) for row in metrics
    )
    verdict = SMOKE_VERDICT if classification == "SMOKE_EGL" else SHARD_VERDICT
    summary = {
        "verdict": verdict,
        "row": ROW,
        "classification": classification,
        "physical_gpu_id": int(physical_gpu_id),
        "task_ids": list(task_ids),
        "episodes": len(metrics),
        "successes": sum(int(row["success"]) for row in metrics),
        "success_rate": float(np.mean([int(row["success"]) for row in metrics])),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_combined_sha256": provenance["combined_sha256"],
        "total_policy_queries": total_queries,
        "total_full_vlm_calls": total_full,
        "total_candidate_condition_updates": total_candidates,
        "total_accepted_condition_updates": total_accepted,
        "total_rejected_condition_updates": total_rejected,
        "observed_exact_fraction": float(total_full / max(1, total_queries)),
        "effective_k_c": float(total_queries / max(1, total_full)),
        "total_full_action_transformer_evaluations": sum(
            int(row["num_full_action_transformer_evaluations"]) for row in metrics
        ),
        "total_generation_loop_updates": sum(
            int(row["num_generation_loop_updates"]) for row in metrics
        ),
        "all_counter_gates_pass": all(
            row["counter_gate"] == "ACTION_EQUIVALENT_REFRESH_COUNTER_PASS"
            for row in metrics
        ),
        "latency_per_policy_query_ms": float(
            sum(float(row["policy_wall_time_seconds"]) for row in metrics)
            * 1000.0
            / max(1, total_queries)
        ),
        "latency_per_executed_action_ms": float(
            sum(float(row["policy_wall_time_seconds"]) for row in metrics)
            * 1000.0
            / max(1, sum(int(row["episode_length"]) for row in metrics))
        ),
        "route_decisions": len(decisions),
        "scientific_contract": episode_payloads[0]["scientific_contract"],
    }
    atomic_write_json(output / "shard_summary.json", summary)
    return summary


def evaluate_shard(args: argparse.Namespace) -> dict[str, Any]:
    physical_gpu_id = int(args.physical_gpu_id)
    allowed_gpu_ids = PRODUCTION_GPU_IDS.get(
        args.classification,
        set(range(8)) if args.classification == "SMOKE_EGL" else set(),
    )
    if physical_gpu_id not in allowed_gpu_ids:
        raise RuntimeError(
            f"physical GPU {physical_gpu_id} is invalid for {args.classification}; "
            f"allowed={sorted(allowed_gpu_ids)}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu_id):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly the physical GPU")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(physical_gpu_id):
        raise RuntimeError("MUJOCO_EGL_DEVICE_ID must equal the physical GPU ID")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get(
        "PYOPENGL_PLATFORM"
    ) != "egl":
        raise RuntimeError("action-equivalent online evaluation is EGL-only")
    task_ids = _parse_task_ids(args.task_ids)
    if args.classification != "SMOKE_EGL" and (
        int(args.max_episodes) != 0 or int(args.max_policy_actions) != 0
    ):
        raise RuntimeError("production evaluation forbids bounded smoke overrides")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    (output / "videos").mkdir(exist_ok=True)

    provenance = _verify_provenance(args)
    source_path = output / "source_lock.json"
    if source_path.is_file():
        previous = load_json(source_path)
        if previous.get("combined_sha256") != provenance["combined_sha256"]:
            raise RuntimeError("resume source lock differs from the current clean commit")
    atomic_write_json(source_path, provenance)

    preflight = require_egl_preflight(args.egl_preflight, physical_gpu_id)
    manifest = load_json(args.manifest)
    manifest_report = validate_manifest_identity(
        manifest, expected_manifest_sha256=args.expected_manifest_sha256
    )
    if manifest_report["verdict"] != "EPISODE_MANIFEST_PASS":
        raise RuntimeError(json.dumps(manifest_report, indent=2, sort_keys=True))
    renderer_mismatches = {
        name: {"expected": value, "observed": os.environ.get(name)}
        for name, value in manifest["renderer"].items()
        if os.environ.get(name) != value
    }
    if renderer_mismatches:
        raise RuntimeError(f"renderer contract mismatch: {renderer_mismatches}")

    specs_by_task: dict[int, list[dict[str, Any]]] = {}
    for task_id in task_ids:
        specs = sorted(
            (
                item
                for item in manifest["episodes"]
                if int(item["task_id"]) == task_id
            ),
            key=lambda item: int(item["trial_id"]),
        )
        if len(specs) != int(manifest["trials_per_task"]):
            raise RuntimeError(f"task {task_id} does not have exactly 50 episodes")
        specs_by_task[task_id] = specs

    ordered_specs = [
        spec for task_id in reversed(task_ids) for spec in specs_by_task[task_id]
    ]
    if int(args.max_episodes) > 0:
        ordered_specs = ordered_specs[: int(args.max_episodes)]
    atomic_write_json(output / "egl_preflight.json", preflight)
    atomic_write_json(output / "manifest_validation.json", manifest_report)
    atomic_write_json(
        output / "shard_contract.json",
        {
            "verdict": "ACTION_EQUIVALENT_REFRESH_SHARD_CONTRACT_PASS",
            "physical_gpu_id": physical_gpu_id,
            "task_ids": list(task_ids),
            "episode_count": len(ordered_specs),
            "manifest_gpu_assignment_ignored": True,
            "reason": "The immutable episode identities are retained while sd1 task sharding relocates compute only.",
        },
    )

    if args.finalize_only:
        return _finalize_shard(
            output=output,
            episode_specs=ordered_specs,
            manifest=manifest,
            provenance=provenance,
            physical_gpu_id=physical_gpu_id,
            task_ids=task_ids,
            classification=args.classification,
        )

    completed = 0
    for spec in ordered_specs:
        path = output / "episodes" / (
            _episode_identity(int(spec["task_id"]), int(spec["trial_id"])) + ".json"
        )
        if _load_completed_episode(
            path,
            manifest_sha256=str(manifest["manifest_sha256"]),
            source_sha256=str(provenance["combined_sha256"]),
            task_id=int(spec["task_id"]),
            trial_id=int(spec["trial_id"]),
        ):
            completed += 1
    if completed == len(ordered_specs):
        return _finalize_shard(
            output=output,
            episode_specs=ordered_specs,
            manifest=manifest,
            provenance=provenance,
            physical_gpu_id=physical_gpu_id,
            task_ids=task_ids,
            classification=args.classification,
        )

    configure_strict_torch_determinism(int(manifest["determinism_seed"]))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each shard requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    for field in ("condition_updater_ms", "risk_head_ms", "generation_loop_ms"):
        if field not in rollout_runtime.LATENCY_FIELDS:
            rollout_runtime.LATENCY_FIELDS.append(field)

    model, processor, _ = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    freeze_module(model)
    condition_updater, condition_payload = load_native_v0_checkpoint(
        args.condition_checkpoint, device=device, require_final_150k=True
    )
    if int(condition_payload.get("global_optimizer_step", -1)) != 150_000:
        raise RuntimeError("Condition checkpoint is not optimizer step 150,000")
    if (
        condition_payload.get("source_lock", {}).get("combined_sha256")
        != FROZEN_CONDITION_SOURCE_SHA256
    ):
        raise RuntimeError("Condition checkpoint source lock changed")
    freeze_module(condition_updater)
    generation_updater, generation_payload = load_generation_checkpoint(
        args.generation_checkpoint, device=device
    )
    if int(generation_payload.get("optimizer_step", -1)) != 30_000:
        raise RuntimeError("Generation checkpoint is not optimizer step 30,000")
    if (
        generation_payload.get("source_lock", {}).get("combined_sha256")
        != FROZEN_GENERATION_SOURCE_SHA256
    ):
        raise RuntimeError("Generation checkpoint source lock changed")
    freeze_module(generation_updater)
    risk_head, feature_config, calibration, _ = load_action_fidelity_checkpoint(
        args.risk_checkpoint, device=device
    )
    freeze_module(risk_head)

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[str(manifest["suite"])]()
    progress = tqdm(
        total=len(ordered_specs),
        initial=completed,
        desc=f"{ROW} gpu{physical_gpu_id}",
        dynamic_ncols=True,
        mininterval=float(args.tqdm_mininterval),
    )
    action_limit = int(args.max_policy_actions) or int(manifest["max_policy_actions"])
    for task_id in reversed(task_ids):
        task_specs = [
            spec for spec in ordered_specs if int(spec["task_id"]) == task_id
        ]
        if not task_specs:
            continue
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, prompt = get_libero_env(
            task,
            int(manifest["environment_resolution"]),
            int(manifest["environment_seed"]),
        )
        try:
            for spec in task_specs:
                trial_id = int(spec["trial_id"])
                identity = _episode_identity(task_id, trial_id)
                episode_path = output / "episodes" / f"{identity}.json"
                existing = _load_completed_episode(
                    episode_path,
                    manifest_sha256=str(manifest["manifest_sha256"]),
                    source_sha256=str(provenance["combined_sha256"]),
                    task_id=task_id,
                    trial_id=trial_id,
                )

                # Replaying reset/wait for completed trials preserves the original
                # per-task environment-reset sequence before any resumed episode.
                env.reset()
                observation = env.set_init_state(
                    initial_states[int(spec["init_state_index"]) % len(initial_states)]
                )
                environment_ms = 0.0
                for _ in range(int(manifest["num_wait_steps"])):
                    started = time.perf_counter()
                    observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    environment_ms += (time.perf_counter() - started) * 1000.0
                if existing is not None:
                    continue

                episode_started = time.perf_counter()
                policy = ActionEquivalentRefreshSimVLAPolicy(
                    risk_head=risk_head,
                    calibration=calibration,
                    feature_config=feature_config,
                    generation_updater=generation_updater,
                    model=model,
                    processor=processor,
                    adapter=condition_updater,
                    checkpoint_id=args.checkpoint,
                    device=device,
                    suite=str(manifest["suite"]),
                    task_id=task_id,
                    trial_id=trial_id,
                    action_noise_seed_base=int(manifest["action_noise_seed_base"]),
                    client_resize_size=int(manifest["client_resize_size"]),
                    image_size=int(manifest["model_image_size"]),
                    flow_steps=int(manifest["flow_steps"]),
                    log_action_chunks=False,
                )
                actions: list[np.ndarray] = []
                policy_ms: list[float] = []
                query_policy_ms: list[float] = []
                video_root = output / "videos" / f"task_{task_id:02d}"
                capture_video = bool(args.save_video) and len(
                    list(video_root.glob("*.mp4"))
                ) < int(args.video_max_per_task)
                frames: list[np.ndarray] = []
                success = False
                for action_index in range(action_limit):
                    if capture_video and action_index % int(args.video_stride) == 0:
                        frames.append(video_frame_from_obs(observation))
                    image0, image1, proprio = build_env_obs(observation)
                    is_policy_query = not policy.action_queue
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    step = policy.act(image0, image1, proprio, prompt)
                    torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter() - started) * 1000.0
                    policy_ms.append(elapsed)
                    if is_policy_query:
                        query_policy_ms.append(elapsed)
                    started = time.perf_counter()
                    observation, _, done, _ = env.step(step.action.tolist())
                    environment_ms += (time.perf_counter() - started) * 1000.0
                    actions.append(step.action.copy())
                    if done:
                        success = True
                        break

                gate = _validate_selective_counters(
                    policy.metrics.counters,
                    decision_count=len(policy.refresh_decisions),
                )
                if gate["verdict"] != "ACTION_EQUIVALENT_REFRESH_COUNTER_PASS":
                    raise RuntimeError(json.dumps(gate, indent=2, sort_keys=True))
                observed = gate["observed"]
                trajectory = _trajectory_metrics(actions)
                gripper = _gripper_metrics(actions)
                latency = policy.metrics.latencies
                metrics = {
                    "row": ROW,
                    "classification": args.classification,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "init_state_index": int(spec["init_state_index"]),
                    "success": int(success),
                    "episode_length": len(actions),
                    "num_policy_queries": observed["policy_queries"],
                    "num_full_vlm_calls": observed["full_vlm_calls"],
                    "num_candidate_condition_updates": observed[
                        "candidate_condition_updates"
                    ],
                    "num_condition_updater_calls": observed[
                        "condition_updater_calls"
                    ],
                    "num_risk_head_calls": observed["risk_head_calls"],
                    "num_accepted_condition_updates": observed[
                        "accepted_condition_updates"
                    ],
                    "num_rejected_condition_updates": observed[
                        "rejected_condition_updates"
                    ],
                    "num_forced_age_refreshes": observed["forced_age_refreshes"],
                    "num_risk_triggered_refreshes": observed[
                        "risk_triggered_refreshes"
                    ],
                    "num_full_action_transformer_evaluations": observed[
                        "full_action_transformer_evaluations"
                    ],
                    "num_generation_loop_updates": observed[
                        "generation_loop_updates"
                    ],
                    "num_integration_updates": observed[
                        "full_action_transformer_evaluations"
                    ]
                    + observed["generation_loop_updates"],
                    "num_action_queue_steps": int(
                        policy.metrics.counters.get("num_action_queue_steps", 0)
                    ),
                    "observed_exact_fraction": float(
                        observed["full_vlm_calls"] / max(1, observed["policy_queries"])
                    ),
                    "effective_k_c": float(
                        observed["policy_queries"] / max(1, observed["full_vlm_calls"])
                    ),
                    "latency_per_policy_query_ms": _mean(query_policy_ms),
                    "latency_per_executed_action_ms": _mean(policy_ms),
                    "model_vlm_encoder_per_exact_ms": _mean(
                        latency.get("VLM_encoder_ms", [])
                    ),
                    "model_condition_updater_per_candidate_ms": _mean(
                        latency.get("condition_updater_ms", [])
                    ),
                    "model_risk_head_per_candidate_ms": _mean(
                        latency.get("risk_head_ms", [])
                    ),
                    "model_generation_loop_per_query_ms": _mean(
                        latency.get("generation_loop_ms", [])
                    ),
                    "vlm_encoder_total_ms": _sum(latency.get("VLM_encoder_ms", [])),
                    "condition_updater_total_ms": _sum(
                        latency.get("condition_updater_ms", [])
                    ),
                    "risk_head_total_ms": _sum(latency.get("risk_head_ms", [])),
                    "generation_loop_total_ms": _sum(
                        latency.get("generation_loop_ms", [])
                    ),
                    "policy_wall_time_seconds": float(sum(policy_ms) / 1000.0),
                    "environment_wall_time_seconds": float(environment_ms / 1000.0),
                    "episode_wall_time_seconds": float(
                        time.perf_counter() - episode_started
                    ),
                    "normalized_second_difference": trajectory[
                        "normalized_second_difference"
                    ],
                    "short_reversal": trajectory["short_reversal"],
                    **gripper,
                    "counter_gate": gate["verdict"],
                }
                if capture_video and not success:
                    save_episode_video(
                        frames,
                        video_root / f"trial_{trial_id:03d}_failure.mp4",
                        int(args.video_fps),
                    )
                payload = {
                    "schema_version": EPISODE_SCHEMA,
                    "verdict": "ACTION_EQUIVALENT_REFRESH_EPISODE_COMPLETE",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "source_combined_sha256": provenance["combined_sha256"],
                    "physical_gpu_id": physical_gpu_id,
                    "metrics": metrics,
                    "counter_audit": gate,
                    "route_decisions": policy.refresh_decisions,
                    "scientific_contract": policy.scientific_contract(),
                }
                atomic_write_json(episode_path, payload)
                completed += 1
                progress.update(1)
                success_count = 0
                for path in (output / "episodes").glob("*.json"):
                    success_count += int(load_json(path).get("metrics", {}).get("success", 0))
                progress.set_postfix(
                    successes=success_count,
                    sr=f"{100.0 * success_count / max(1, completed):.1f}%",
                )
        finally:
            env.close()
    progress.close()
    return _finalize_shard(
        output=output,
        episode_specs=ordered_specs,
        manifest=manifest,
        provenance=provenance,
        physical_gpu_id=physical_gpu_id,
        task_ids=task_ids,
        classification=args.classification,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--offline-root", required=True)
    parser.add_argument("--risk-checkpoint", required=True)
    parser.add_argument("--condition-checkpoint", required=True)
    parser.add_argument("--generation-checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--expected-root-commit", required=True)
    parser.add_argument("--egl-preflight", required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument(
        "--classification",
        choices=(
            "SD1_HOST_LOCAL_EGL_LONG500",
            "RB2_HOST_LOCAL_EGL_LONG500",
            "SMOKE_EGL",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-max-per-task", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-policy-actions", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def main() -> int:
    result = evaluate_shard(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
