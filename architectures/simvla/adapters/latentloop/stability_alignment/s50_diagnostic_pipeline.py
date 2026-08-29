"""Run the failed-gate S50 10K checkpoint on the locked rb2 LIBERO axis."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    atomic_write_json,
    load_json,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment.rb2_pipeline import (
    BASELINE,
    MANIFEST,
    MANIFEST_SHA256,
    PYTHON,
    WORKTREE,
    _environment,
    _gpu_snapshot,
)


STORAGE = Path("/home/mingyujung/private/gnaroshi_vla_storage").resolve()
BUNDLE = STORAGE / "artifacts/simvla/stability_alignment/s50_10k_diagnostic"
RESULT_ROOT = STORAGE / "results/simvla/stability_alignment/s50_10k_diagnostic_rb2"
EVALUATOR = "architectures.simvla.adapters.latentloop.stability_alignment.online_eval"
SOURCE_FILES = {
    "online_eval.py": WORKTREE
    / "architectures/simvla/adapters/latentloop/stability_alignment/online_eval.py",
    "s50_diagnostic_pipeline.py": WORKTREE
    / "architectures/simvla/adapters/latentloop/stability_alignment/s50_diagnostic_pipeline.py",
}
ROWS = tuple(
    (k_c, mode)
    for k_c in (3, 4)
    for mode in ("nfe10", "naive_nfe3", "learned_ng3")
)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _wait_for_gpu() -> None:
    while True:
        snapshot = _gpu_snapshot()
        atomic_write_json(RESULT_ROOT / "gpu_wait_state.json", snapshot)
        if snapshot["free"]:
            return
        print(f"[{_timestamp()}] rb2 GPU is occupied; waiting 120s", flush=True)
        time.sleep(120)


def _verify_bundle() -> dict[str, Any]:
    ready = load_json(BUNDLE / "READY_SHORT_DIAGNOSTIC_FOR_RB2.json")
    if ready.get("verdict") != "READY_SHORT_DIAGNOSTIC_FOR_RB2":
        raise RuntimeError("S50 diagnostic readiness verdict is missing")
    if not ready.get("diagnostic_only") or ready.get("offline_gate_passed") is not False:
        raise RuntimeError("S50 bundle does not preserve its failed offline gate")
    checkpoint = BUNDLE / str(ready["checkpoint"])
    if sha256_file(checkpoint) != ready.get("checkpoint_sha256"):
        raise RuntimeError("S50 diagnostic checkpoint hash changed")
    gate = load_json(BUNDLE / "offline_gate.json")
    if gate.get("verdict") != "STABILITY_10K_GATE_FAIL":
        raise RuntimeError("S50 diagnostic bundle lost the failed gate evidence")
    manifest = load_json(MANIFEST)
    if manifest.get("manifest_sha256") != MANIFEST_SHA256:
        raise RuntimeError("rb2 episode manifest changed")
    baseline = load_json(BASELINE)
    if int(baseline.get("episodes", -1)) != 500:
        raise RuntimeError("rb2 baseline is not the locked 500-episode row")
    return ready


def _row_output(k_c: int, mode: str) -> Path:
    return RESULT_ROOT / "rows" / f"kc{k_c}_{mode}"


def _run_row(k_c: int, mode: str) -> dict[str, Any]:
    output = _row_output(k_c, mode)
    summary_path = output / "row_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        if (
            summary.get("verdict") == "RB2_STABILITY_DIAGNOSTIC_ROW_COMPLETE"
            and int(summary.get("episodes", -1)) == 500
            and summary.get("manifest_sha256") == MANIFEST_SHA256
        ):
            return summary
        raise RuntimeError(f"incompatible diagnostic row exists: {output}")
    _wait_for_gpu()
    print(f"[{_timestamp()}] start K_C={k_c} mode={mode}", flush=True)
    subprocess.run(
        (
            str(PYTHON),
            "-m",
            EVALUATOR,
            "--output",
            str(output),
            "--bundle",
            str(BUNDLE),
            "--manifest",
            str(MANIFEST),
            "--k-c",
            str(k_c),
            "--generation-mode",
            mode,
            "--physical-gpu-id",
            "0",
            "--diagnostic-only",
        ),
        cwd=WORKTREE,
        env=_environment(),
        check=True,
    )
    summary = load_json(summary_path)
    print(
        f"[{_timestamp()}] complete K_C={k_c} mode={mode} "
        f"success={summary['successes']}/{summary['episodes']}",
        flush=True,
    )
    return summary


def _aggregate(ready: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = load_json(BASELINE)
    baseline_rate = float(baseline["success_rate"])
    table = []
    for row in rows:
        rate = float(row["success_rate"])
        table.append(
            {
                "row": row["row"],
                "k_c": int(row["k_c"]),
                "generation_mode": row["generation_mode"],
                "episodes": int(row["episodes"]),
                "successes": int(row["successes"]),
                "success_rate": rate,
                "delta_vs_rb2_baseline_pp": 100.0 * (rate - baseline_rate),
                "mean_policy_query_latency_ms": row["mean_policy_query_latency_ms"],
                "classification": "DIAGNOSTIC_ONLY",
            }
        )
    return {
        "schema_version": "simvla_s50_10k_rb2_diagnostic_summary_v1",
        "verdict": "S50_10K_RB2_DIAGNOSTIC_COMPLETE",
        "classification": "DIAGNOSTIC_ONLY",
        "offline_gate_passed": False,
        "offline_gate_verdict": "STABILITY_10K_GATE_FAIL",
        "checkpoint_sha256": ready["checkpoint_sha256"],
        "evaluation_source_sha256": {
            name: sha256_file(path) for name, path in SOURCE_FILES.items()
        },
        "manifest_sha256": MANIFEST_SHA256,
        "baseline": {
            "successes": int(baseline["successes"]),
            "episodes": int(baseline["episodes"]),
            "success_rate": baseline_rate,
            "path": str(BASELINE),
        },
        "rows": table,
        "claim_boundary": (
            "These rows diagnose offline-online correlation only. They do not "
            "retroactively pass the failed S50 offline gate."
        ),
    }


def run() -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = RESULT_ROOT / "pipeline.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready = _verify_bundle()
        atomic_write_json(
            RESULT_ROOT / "run_contract.json",
            {
                "classification": "DIAGNOSTIC_ONLY",
                "rows": [f"kc{k_c}_{mode}" for k_c, mode in ROWS],
                "checkpoint_sha256": ready["checkpoint_sha256"],
                "evaluation_source_sha256": {
                    name: sha256_file(path) for name, path in SOURCE_FILES.items()
                },
                "manifest_sha256": MANIFEST_SHA256,
                "renderer": "egl",
                "inference_seed": "seed02",
                "episodes_per_row": 500,
                "h": 10,
                "r": 5,
            },
        )
        rows = [_run_row(k_c, mode) for k_c, mode in ROWS]
        summary = _aggregate(ready, rows)
        atomic_write_json(RESULT_ROOT / "diagnostic_summary.json", summary)
        return summary


def main() -> int:
    try:
        result = run()
    except Exception as error:
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            RESULT_ROOT / "failure.json",
            {
                "verdict": "S50_10K_RB2_DIAGNOSTIC_FAILED",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
