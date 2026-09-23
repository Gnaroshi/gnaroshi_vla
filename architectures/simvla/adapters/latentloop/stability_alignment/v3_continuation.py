"""Resume V3 through 10K while treating early scientific gates as diagnostics."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    atomic_write_json,
    load_json,
)
from architectures.simvla.adapters.latentloop.stability_alignment import v3_pipeline as base
from architectures.simvla.adapters.latentloop.stability_alignment.v3_diagnostic_bundle import (
    build as build_diagnostic_bundle,
)


BRANCH = "R50"
STEPS = (500, 2_000, 5_000, 10_000)
PORTS = {
    2_000: (29900, 29910),
    5_000: (29920, 29930),
    10_000: (29940, 29950),
}
TRAJECTORY_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_TRAJECTORY_ROOT",
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/"
        "transfers/simvla_stability_v3_trajectory",
    )
).resolve()
REMOTE_TRAJECTORY_ROOT = os.environ.get(
    "RB2_V3_TRAJECTORY_DESTINATION",
    "rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/"
    "simvla_stability_v3_trajectory",
)


def _step_label(step: int) -> str:
    return "500" if int(step) == 500 else f"{int(step) // 1000}k"


def _checkpoint(step: int) -> Path:
    return (
        base.RESULT_ROOT
        / "training/r50/checkpoints"
        / f"stability_v3_step_{int(step):06d}.pt"
    )


def _training_summary(step: int) -> Path:
    return (
        base.RESULT_ROOT
        / "training/r50"
        / f"run_summary_step_{int(step):06d}.json"
    )


def _validation_gate(step: int) -> Path:
    return (
        base.RESULT_ROOT
        / "offline/r50"
        / f"{_step_label(step)}_validation/offline_gate.json"
    )


def _hard_safety(step: int, gate_path: Path) -> dict[str, Any]:
    training = load_json(_training_summary(step))
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
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "soft_stage_gate_passed": bool(gate.get("passed")),
        "soft_stage_gate_verdict": gate.get("verdict"),
        "moving_window_target_balance_passed": bool(
            audit.get("target_balance_passed")
        ),
        "policy": (
            "continue through 10K on soft metric misses; stop only on runtime, "
            "nonfinite/numerical-safety, freeze, checkpoint-hash, or source-lock failure"
        ),
    }


def _record_decision(step: int, gate_path: Path) -> dict[str, Any]:
    decision = _hard_safety(step, gate_path)
    decision.update(
        {
            "schema_version": "simvla_stability_v3_continuation_decision_v1",
            "branch": BRANCH,
            "optimizer_step": int(step),
            "training_summary": str(_training_summary(step)),
            "offline_gate": str(gate_path),
        }
    )
    atomic_write_json(
        base.RESULT_ROOT
        / "continuation_decisions"
        / f"step_{int(step):06d}.json",
        decision,
    )
    return decision


def _transfer_bundle(bundle: Path, step: int) -> dict[str, Any]:
    if ":" not in REMOTE_TRAJECTORY_ROOT:
        raise ValueError("RB2 trajectory destination must use host:/absolute/path")
    remote_host, remote_root = REMOTE_TRAJECTORY_ROOT.split(":", 1)
    destination = (
        f"{REMOTE_TRAJECTORY_ROOT.rstrip('/')}/r50_step_{int(step):06d}/"
    )
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, 4):
        mkdir = subprocess.run(
            ("ssh", remote_host, "mkdir", "-p", remote_root),
            text=True,
            capture_output=True,
            check=False,
        )
        payload = (
            subprocess.run(
                (
                    "rsync",
                    "-a",
                    "--exclude=READY_SHORT_DIAGNOSTIC_FOR_RB2.json",
                    f"{bundle}/",
                    destination,
                ),
                text=True,
                capture_output=True,
                check=False,
            )
            if mkdir.returncode == 0
            else None
        )
        readiness = (
            subprocess.run(
                (
                    "rsync",
                    "-a",
                    str(bundle / "READY_SHORT_DIAGNOSTIC_FOR_RB2.json"),
                    destination,
                ),
                text=True,
                capture_output=True,
                check=False,
            )
            if payload is not None and payload.returncode == 0
            else None
        )
        returncode = int(
            mkdir.returncode
            or (payload.returncode if payload is not None else 1)
            or (readiness.returncode if readiness is not None else 1)
        )
        attempts.append(
            {
                "attempt": attempt,
                "returncode": returncode,
                "mkdir_stdout": mkdir.stdout,
                "mkdir_stderr": mkdir.stderr,
                "payload_stdout": payload.stdout if payload else "",
                "payload_stderr": (
                    payload.stderr if payload else "remote mkdir failed"
                ),
                "readiness_stdout": readiness.stdout if readiness else "",
                "readiness_stderr": (
                    readiness.stderr if readiness else "payload transfer failed"
                ),
                "readiness_sent_last": readiness is not None,
            }
        )
        if returncode == 0:
            break
        time.sleep(30)
    result = {
        "schema_version": "simvla_stability_v3_trajectory_transfer_v1",
        "optimizer_step": int(step),
        "source": str(bundle),
        "destination": destination,
        "passed": bool(attempts and attempts[-1]["returncode"] == 0),
        "attempts": attempts,
    }
    atomic_write_json(bundle / "rb2_transfer_status.json", result)
    return result


def _publish(step: int, gate_path: Path) -> dict[str, Any]:
    bundle = TRAJECTORY_ROOT / f"r50_step_{int(step):06d}"
    ready = build_diagnostic_bundle(
        checkpoint=_checkpoint(step),
        offline_gate=gate_path,
        norm_stats=base.NORM,
        training_root=base.RESULT_ROOT / "training/r50",
        output=bundle,
        branch=BRANCH,
        optimizer_step=step,
    )
    transfer = _transfer_bundle(bundle, step)
    return {"ready": ready, "transfer": transfer}


def _retry_transfer_backlog(current_step: int) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    for step in STEPS:
        if step > int(current_step):
            continue
        bundle = TRAJECTORY_ROOT / f"r50_step_{int(step):06d}"
        ready = bundle / "READY_SHORT_DIAGNOSTIC_FOR_RB2.json"
        status = bundle / "rb2_transfer_status.json"
        if not ready.is_file():
            continue
        if status.is_file() and bool(load_json(status).get("passed")):
            continue
        outcomes[str(step)] = _transfer_bundle(bundle, step)
    return outcomes


def _ensure_existing_500() -> None:
    required = (_checkpoint(500), _training_summary(500), _validation_gate(500))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V3 500-step state is incomplete: {missing}")


def _run_segment(step: int, previous_step: int, pool: tuple[int, ...]) -> Path:
    train_port, offline_port = PORTS[int(step)]
    train_job = base._train_job(BRANCH, step, _checkpoint(previous_step), train_port)
    base._run_jobs(
        [train_job], stage=f"CONTINUE_TRAIN_{int(step):06d}", pool=pool
    )
    previous_gate = _validation_gate(previous_step) if step in {2_000, 5_000} else None
    offline_job = base._offline_job(
        BRANCH,
        step,
        "checkpoint_validation",
        previous_gate,
        offline_port,
    )
    base._run_jobs(
        [offline_job], stage=f"CONTINUE_GATE_{int(step):06d}", pool=pool
    )
    return offline_job.success_file


def run() -> dict[str, Any]:
    if os.environ.get("SIMVLA_STABILITY_V3_CONTINUATION_RUN") != "1":
        raise RuntimeError(
            "set SIMVLA_STABILITY_V3_CONTINUATION_RUN=1 to approve continuation"
        )
    _ensure_existing_500()
    pool = base._gpu_pool()
    decisions: dict[str, Mapping[str, Any]] = {}
    transfers: dict[str, Mapping[str, Any]] = {}

    gate = _validation_gate(500)
    decision = _record_decision(500, gate)
    decisions["500"] = decision
    transfers["500"] = _publish(500, gate)
    _retry_transfer_backlog(500)
    if not decision["passed"]:
        raise RuntimeError("500-step hard-safety contract failed")

    previous = 500
    for step in (2_000, 5_000, 10_000):
        print(
            f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] "
            f"R50 exact resume {previous}->{step}",
            flush=True,
        )
        gate = _run_segment(step, previous, pool)
        decision = _record_decision(step, gate)
        decisions[str(step)] = decision
        if not decision["passed"]:
            raise RuntimeError(f"{step}-step hard-safety contract failed")
        if step < 10_000:
            transfers[str(step)] = _publish(step, gate)
            _retry_transfer_backlog(step)
        previous = step

    final_job = base._offline_job(BRANCH, 10_000, "final_offline", None, 29960)
    base._run_jobs([final_job], stage="CONTINUE_FINAL_OFFLINE_010000", pool=pool)
    final_gate = final_job.success_file
    transfers["10000"] = _publish(10_000, final_gate)
    _retry_transfer_backlog(10_000)
    final_payload = load_json(final_gate)
    exported: Mapping[str, Any] | None = None
    export_error: str | None = None
    if bool(final_payload.get("passed")):
        try:
            exported = base._export(BRANCH, final_gate)
        except Exception as error:
            export_error = f"{type(error).__name__}: {error}"
            atomic_write_json(
                base.RESULT_ROOT / "continuation_export_recovery_required.json",
                {
                    "verdict": "STRICT_GATE_PASS_EXPORT_RECOVERY_REQUIRED",
                    "error": export_error,
                    "checkpoint": str(_checkpoint(10_000)),
                    "final_gate": str(final_gate),
                    "training_and_final_gate_remain_valid": True,
                },
            )

    return {
        "schema_version": "simvla_stability_v3_continuation_summary_v1",
        "verdict": (
            "STABILITY_V3_CONTINUOUS_10K_STRICT_PASS"
            if final_payload.get("passed")
            else "STABILITY_V3_CONTINUOUS_10K_DIAGNOSTIC_COMPLETE"
        ),
        "branch": BRANCH,
        "gpu_pool": list(pool),
        "optimizer_step": 10_000,
        "soft_gates_are_diagnostic": True,
        "hard_safety_only_stopping": True,
        "decisions": decisions,
        "trajectory_transfers": transfers,
        "strict_final_gate": str(final_gate),
        "strict_final_gate_passed": bool(final_payload.get("passed")),
        "strict_bundle_exported": bool(exported),
        "strict_export_error": export_error,
    }


def main() -> int:
    base.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = base.RESULT_ROOT / "pipeline.lock"
    try:
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run()
            atomic_write_json(base.RESULT_ROOT / "continuation_summary.json", result)
            failure = base.RESULT_ROOT / "continuation_failure.json"
            if failure.is_file():
                failure.unlink()
    except Exception as error:
        atomic_write_json(
            base.RESULT_ROOT / "continuation_failure.json",
            {
                "verdict": "STABILITY_V3_CONTINUATION_RUNTIME_OR_HARD_FAILURE",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
