"""Run the independent R150 V3 control through 10K without soft-gate stopping."""

from __future__ import annotations

import fcntl
import json
import os
import traceback
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop.stability_alignment import v3_pipeline as base
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    atomic_write_json,
    load_json,
)


BRANCH = "R150"
STEPS = (500, 2_000, 5_000, 10_000)
PORTS = {
    500: (30000, 30010),
    2_000: (30020, 30030),
    5_000: (30040, 30050),
    10_000: (30060, 30070),
}


def _label(step: int) -> str:
    return "500" if int(step) == 500 else f"{int(step) // 1000}k"


def _checkpoint(step: int) -> Path:
    return (
        base.RESULT_ROOT
        / "training/r150/checkpoints"
        / f"stability_v3_step_{int(step):06d}.pt"
    )


def _summary(step: int) -> Path:
    return (
        base.RESULT_ROOT
        / "training/r150"
        / f"run_summary_step_{int(step):06d}.json"
    )


def _gate(step: int) -> Path:
    return (
        base.RESULT_ROOT
        / "offline/r150"
        / f"{_label(step)}_validation/offline_gate.json"
    )


def _decision(step: int, gate_path: Path) -> dict[str, Any]:
    training = load_json(_summary(step))
    gate = load_json(gate_path)
    audit = dict(training.get("moving_window_audit", {}))
    measurements = dict(gate.get("gate", {}).get("measurements", {}))
    checks = {
        "training_segment_complete": training.get("verdict")
        in {
            "STABILITY_V3_TRAINING_SEGMENT_COMPLETE",
            "STABILITY_V3_TRAINING_SEGMENT_COMPLETE_WITH_AUDIT_WARNING",
        },
        "optimizer_step_exact": int(training.get("optimizer_step", -1)) == int(step),
        "moving_window_numerical_safety": bool(audit.get("hard_safety_passed")),
        "original_simvla_frozen": bool(measurements.get("original_simvla_frozen")),
        "checkpoint_hash_matches_gate": (
            training.get("checkpoint_sha256") == gate.get("candidate_sha256")
        ),
        "source_lock_matches_gate": (
            training.get("source_combined_sha256")
            == gate.get("source_combined_sha256")
        ),
    }
    payload = {
        "schema_version": "simvla_stability_v3_r150_continuation_decision_v1",
        "branch": BRANCH,
        "optimizer_step": int(step),
        "passed_hard_safety": all(checks.values()),
        "checks": checks,
        "soft_stage_gate_passed": bool(gate.get("passed")),
        "soft_stage_gate_verdict": gate.get("verdict"),
        "moving_window_target_balance_passed": bool(
            audit.get("target_balance_passed")
        ),
    }
    atomic_write_json(
        base.RESULT_ROOT / "continuation_decisions" / f"step_{int(step):06d}.json",
        payload,
    )
    return payload


def _run_segment(step: int, previous: int | None, pool: tuple[int, ...]) -> Path:
    train_port, gate_port = PORTS[int(step)]
    resume = _checkpoint(previous) if previous is not None else None
    train = base._train_job(BRANCH, step, resume, train_port)
    base._run_jobs([train], stage=f"R150_TRAIN_{int(step):06d}", pool=pool)
    previous_gate = _gate(previous) if previous in {500, 2_000} else None
    offline = base._offline_job(
        BRANCH,
        step,
        "checkpoint_validation",
        previous_gate,
        gate_port,
    )
    base._run_jobs(
        [offline], stage=f"R150_GATE_{int(step):06d}", pool=pool
    )
    return offline.success_file


def run() -> dict[str, Any]:
    if os.environ.get("SIMVLA_STABILITY_V3_R150_RUN") != "1":
        raise RuntimeError("set SIMVLA_STABILITY_V3_R150_RUN=1 to approve R150 control")
    pool = base._gpu_pool()
    if len(pool) != 2:
        raise ValueError("R150 control requires one dedicated GPU pair")
    for path in (base.WORKTREE, base.PYTHON, base.CACHE / "manifest.json", base.GENERATION, base.NORM, base.CONDITION["S150"]):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    base.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        base.RESULT_ROOT / "run_contract.json",
        {
            "schema_version": "simvla_stability_v3_r150_continuous_contract_v1",
            "branch": BRANCH,
            "gpu_pool": list(pool),
            "condition_parent": str(base.CONDITION["S150"]),
            "scheduler_horizon": 30_000,
            "stop_step": 10_000,
            "soft_gates_are_diagnostic": True,
            "hard_safety_only_stopping": True,
            "online_checkpoint_selection": False,
        },
    )

    base._run_jobs(
        base._prepare_jobs("build-pools", 30080, (BRANCH,)),
        stage="R150_BUILD_POOLS",
        pool=pool,
    )
    base._run_jobs(
        base._prepare_jobs("calibrate", 30090, (BRANCH,)),
        stage="R150_CALIBRATE",
        pool=pool,
    )
    calibration_path = (
        base.RESULT_ROOT
        / "v3_preparation/r150/calibration/stability_v3_loss_weights.json"
    )
    calibration = load_json(calibration_path)
    if not bool(calibration.get("approved_for_bounded_pilot")):
        return {
            "verdict": "STABILITY_V3_R150_CALIBRATION_BLOCKED",
            "optimizer_step": 0,
            "calibration": str(calibration_path),
            "reason": calibration.get("severe_conflicts"),
        }

    decisions: dict[str, Any] = {}
    previous: int | None = None
    for step in STEPS:
        gate_path = _run_segment(step, previous, pool)
        decision = _decision(step, gate_path)
        decisions[str(step)] = decision
        if not decision["passed_hard_safety"]:
            raise RuntimeError(f"R150 hard-safety failure at step {step}")
        previous = step

    final = base._offline_job(BRANCH, 10_000, "final_offline", None, 30100)
    base._run_jobs([final], stage="R150_FINAL_OFFLINE_010000", pool=pool)
    final_gate = load_json(final.success_file)
    return {
        "schema_version": "simvla_stability_v3_r150_continuous_summary_v1",
        "verdict": (
            "STABILITY_V3_R150_10K_STRICT_PASS"
            if final_gate.get("passed")
            else "STABILITY_V3_R150_10K_DIAGNOSTIC_COMPLETE"
        ),
        "branch": BRANCH,
        "optimizer_step": 10_000,
        "decisions": decisions,
        "strict_final_gate": str(final.success_file),
        "strict_final_gate_passed": bool(final_gate.get("passed")),
        "classification": "CONTROL_BRANCH_NOT_ONLINE_SELECTED",
    }


def main() -> int:
    base.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with (base.RESULT_ROOT / "pipeline.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run()
            atomic_write_json(base.RESULT_ROOT / "pipeline_summary.json", result)
            failure = base.RESULT_ROOT / "pipeline_failure.json"
            if failure.is_file():
                failure.unlink()
    except Exception as error:
        atomic_write_json(
            base.RESULT_ROOT / "pipeline_failure.json",
            {
                "verdict": "STABILITY_V3_R150_RUNTIME_OR_HARD_FAILURE",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
