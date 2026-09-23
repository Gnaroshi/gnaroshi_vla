"""Continuously evaluate V3 trajectory checkpoints on rb2's fixed 500 episodes."""

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
    BUNDLE_SCHEMA,
    atomic_write_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment import (
    v3_rb2_pipeline as runtime,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_runtime import (
    V3_SOURCE_FILES,
)


STEPS = (500, 2_000, 5_000, 10_000)
ROWS = ((3, "learned_ng3"), (4, "learned_ng3"))
INCOMING_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_TRAJECTORY_INCOMING",
        "/home/mingyujung/private/gnaroshi_vla_storage/incoming/"
        "simvla_stability_v3_trajectory",
    )
).resolve()
RESULT_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_TRAJECTORY_RESULT_ROOT",
        "/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/"
        "stability_alignment/recurrence_v3_trajectory_rb2",
    )
).resolve()
STRICT_BUNDLE = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_BUNDLE",
        "/home/mingyujung/private/gnaroshi_vla_storage/incoming/"
        "simvla_stability_v3_selected",
    )
).resolve()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _bundle(step: int) -> Path:
    return INCOMING_ROOT / f"r50_step_{int(step):06d}"


def _verify_bundle(bundle: Path, step: int) -> dict[str, Any]:
    ready_path = bundle / "READY_SHORT_DIAGNOSTIC_FOR_RB2.json"
    ready = load_json(ready_path)
    if (
        ready.get("schema_version") != BUNDLE_SCHEMA
        or ready.get("verdict") != "READY_SHORT_DIAGNOSTIC_FOR_RB2"
        or ready.get("classification") != "DIAGNOSTIC_ONLY"
        or ready.get("diagnostic_only") is not True
        or ready.get("offline_gate_passed") is not False
        or ready.get("online_must_not_select_checkpoint") is not True
        or ready.get("method_version") != "stability_v3"
        or ready.get("selected_branch") != "R50"
        or int(ready.get("optimizer_step", -1)) != int(step)
    ):
        raise RuntimeError(f"invalid V3 trajectory bundle contract: {bundle}")
    expected_combined = ready.get("combined_sha256")
    observed_combined = canonical_sha256(
        {key: value for key, value in ready.items() if key != "combined_sha256"}
    )
    if expected_combined != observed_combined:
        raise RuntimeError("V3 trajectory readiness hash changed")
    files = ready.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("V3 trajectory file manifest is missing")
    mismatches = {
        name: {
            "expected": digest,
            "observed": sha256_file(bundle / name) if (bundle / name).is_file() else None,
        }
        for name, digest in files.items()
        if not (bundle / name).is_file() or sha256_file(bundle / name) != digest
    }
    if mismatches:
        raise RuntimeError(f"V3 trajectory bundle hash mismatch: {mismatches}")
    if sha256_file(bundle / str(ready["checkpoint"])) != ready["checkpoint_sha256"]:
        raise RuntimeError("V3 trajectory checkpoint hash changed")
    return ready


def _wait_bundle(step: int) -> tuple[Path, dict[str, Any]]:
    bundle = _bundle(step)
    ready_path = bundle / "READY_SHORT_DIAGNOSTIC_FOR_RB2.json"
    while True:
        if ready_path.is_file():
            try:
                return bundle, _verify_bundle(bundle, step)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        print(
            f"[{_timestamp()}] waiting for sd1 V3 step {step} bundle without GPU use",
            flush=True,
        )
        time.sleep(60)


def _verify_runtime_source(bundle: Path) -> dict[str, Any]:
    source_lock = load_json(bundle / "source_lock.json")
    expected = source_lock.get("source_files")
    if not isinstance(expected, dict) or set(expected) != set(V3_SOURCE_FILES):
        raise RuntimeError("V3 trajectory source-lock file set changed")
    observed = {
        relative: (
            sha256_file(runtime.WORKTREE / relative)
            if (runtime.WORKTREE / relative).is_file()
            else None
        )
        for relative in V3_SOURCE_FILES
    }
    mismatches = {
        relative: {"expected": expected[relative], "observed": observed[relative]}
        for relative in V3_SOURCE_FILES
        if observed[relative] != expected[relative]
    }
    if mismatches:
        raise RuntimeError(f"rb2 V3 trajectory runtime source mismatch: {mismatches}")
    combined = canonical_sha256(observed)
    if combined != source_lock.get("source_files_combined_sha256"):
        raise RuntimeError("rb2 V3 trajectory source combined hash changed")
    return {
        "verdict": "STABILITY_V3_TRAJECTORY_SOURCE_LOCK_PASS",
        "source_files": len(observed),
        "source_files_combined_sha256": combined,
    }


def _run_row(
    *, bundle: Path, ready: Mapping[str, Any], step: int, k_c: int, mode: str
) -> dict[str, Any]:
    output = RESULT_ROOT / f"step_{int(step):06d}" / f"kc{k_c}_{mode}"
    summary_path = output / "row_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        if (
            summary.get("verdict") == "RB2_STABILITY_DIAGNOSTIC_ROW_COMPLETE"
            and int(summary.get("episodes", -1)) == 500
            and summary.get("checkpoint_sha256") == ready["checkpoint_sha256"]
            and int(summary.get("k_c", -1)) == int(k_c)
        ):
            return summary
        raise RuntimeError(f"incompatible V3 trajectory row exists: {output}")
    runtime._wait_gpu()
    print(
        f"[{_timestamp()}] start V3 trajectory step={step} K_C={k_c} mode={mode}",
        flush=True,
    )
    subprocess.run(
        (
            str(runtime.PYTHON),
            "-m",
            runtime.EVALUATOR,
            "--output",
            str(output),
            "--bundle",
            str(bundle),
            "--manifest",
            str(runtime.MANIFEST),
            "--k-c",
            str(k_c),
            "--generation-mode",
            mode,
            "--physical-gpu-id",
            "0",
            "--diagnostic-only",
        ),
        cwd=runtime.WORKTREE,
        env=runtime._environment(),
        check=True,
    )
    summary = load_json(summary_path)
    print(
        f"[{_timestamp()}] complete step={step} K_C={k_c} "
        f"success={summary['successes']}/{summary['episodes']}",
        flush=True,
    )
    return summary


def _write_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = runtime._baseline_with_tasks()
    table = []
    for row in rows:
        table.append(
            {
                "optimizer_step": int(row["trajectory_optimizer_step"]),
                "k_c": int(row["k_c"]),
                "generation_mode": row["generation_mode"],
                "episodes": int(row["episodes"]),
                "successes": int(row["successes"]),
                "success_rate": float(row["success_rate"]),
                "delta_success_rate_vs_fixed_baseline": (
                    float(row["success_rate"]) - float(baseline["success_rate"])
                ),
                "mean_policy_query_latency_ms": row["mean_policy_query_latency_ms"],
                "mean_vlm_latency_ms": row["mean_vlm_latency_ms"],
                "mean_condition_latency_ms": row["mean_condition_latency_ms"],
                "mean_generation_latency_ms": row["mean_generation_latency_ms"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "classification": "DIAGNOSTIC_ONLY",
            }
        )
    result = {
        "schema_version": "simvla_stability_v3_trajectory_rb2_summary_v1",
        "verdict": (
            "STABILITY_V3_TRAJECTORY_RB2_COMPLETE"
            if len(table) == len(STEPS) * len(ROWS)
            else "STABILITY_V3_TRAJECTORY_RB2_IN_PROGRESS"
        ),
        "classification": "DIAGNOSTIC_ONLY",
        "online_results_must_not_select_checkpoint": True,
        "manifest_sha256": runtime.MANIFEST_SHA256,
        "fixed_baseline": baseline,
        "expected_rows": len(STEPS) * len(ROWS),
        "completed_rows": len(table),
        "rows": table,
    }
    atomic_write_json(RESULT_ROOT / "trajectory_summary.json", result)
    return result


def _run_strict_if_ready() -> dict[str, Any] | None:
    ready = STRICT_BUNDLE / "READY_SHORT_FOR_RB2.json"
    if not ready.is_file():
        return None
    strict_result = RESULT_ROOT.parent / "recurrence_v3_rb2"
    environment = dict(os.environ)
    environment.update(
        {
            "SIMVLA_STABILITY_V3_RB2_RUN": "1",
            "SIMVLA_STABILITY_V3_BUNDLE": str(STRICT_BUNDLE),
            "SIMVLA_STABILITY_V3_RB2_RESULT_ROOT": str(strict_result),
        }
    )
    completed = subprocess.run(
        (
            str(runtime.PYTHON),
            "-m",
            "architectures.simvla.adapters.latentloop.stability_alignment.v3_rb2_pipeline",
        ),
        cwd=runtime.WORKTREE,
        env=environment,
        check=False,
    )
    return {
        "attempted": True,
        "returncode": int(completed.returncode),
        "result_root": str(strict_result),
    }


def run() -> dict[str, Any]:
    if os.environ.get("SIMVLA_STABILITY_V3_TRAJECTORY_RUN") != "1":
        raise RuntimeError(
            "set SIMVLA_STABILITY_V3_TRAJECTORY_RUN=1 to approve rb2 trajectory evaluation"
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_audits: dict[str, Any] = {}
    for step in STEPS:
        bundle, ready = _wait_bundle(step)
        source_audits[str(step)] = _verify_runtime_source(bundle)
        for k_c, mode in ROWS:
            row = _run_row(
                bundle=bundle,
                ready=ready,
                step=step,
                k_c=k_c,
                mode=mode,
            )
            rows.append({**row, "trajectory_optimizer_step": step})
            _write_summary(rows)
    summary = _write_summary(rows)
    summary["source_audits"] = source_audits
    summary["strict_pipeline"] = _run_strict_if_ready()
    atomic_write_json(RESULT_ROOT / "trajectory_summary.json", summary)
    return summary


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = RESULT_ROOT / "pipeline.lock"
    try:
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run()
            failure = RESULT_ROOT / "pipeline_failure.json"
            if failure.is_file():
                failure.unlink()
    except Exception as error:
        atomic_write_json(
            RESULT_ROOT / "pipeline_failure.json",
            {
                "verdict": "STABILITY_V3_TRAJECTORY_RB2_FAILURE",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
