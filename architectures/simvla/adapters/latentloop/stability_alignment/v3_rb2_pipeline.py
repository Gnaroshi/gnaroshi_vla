"""rb2 evaluator that preserves the immutable S50 diagnostic and gates V3 rows."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import shutil
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
from architectures.simvla.adapters.latentloop.stability_alignment.rb2_pipeline import (
    BASELINE,
    MANIFEST,
    MANIFEST_SHA256,
    _acceptance,
    _baseline_with_tasks,
    _gpu_snapshot,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_runtime import (
    V3_SOURCE_FILES,
)


STORAGE = Path("/home/mingyujung/private/gnaroshi_vla_storage").resolve()
WORKTREE = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_WORKTREE",
        "/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_v3",
    )
).resolve()
PYTHON = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_PYTHON",
        str(STORAGE / "envs/simvla/libero_mujoco237/bin/python"),
    )
).resolve()
UPSTREAM = Path(
    os.environ.get(
        "SIMVLA_UPSTREAM_ROOT",
        "/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream",
    )
).resolve()
BUNDLE = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_BUNDLE",
        str(STORAGE / "incoming/simvla_stability_v3_selected"),
    )
).resolve()
RESULT_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_RB2_RESULT_ROOT",
        str(STORAGE / "results/simvla/stability_alignment/recurrence_v3_rb2"),
    )
).resolve()
DIAGNOSTIC_ROOT = Path(
    os.environ.get(
        "SIMVLA_S50_DIAGNOSTIC_ROOT",
        str(STORAGE / "results/simvla/stability_alignment/s50_10k_diagnostic_rb2"),
    )
).resolve()
EVALUATOR = "architectures.simvla.adapters.latentloop.stability_alignment.online_eval"
PRIMARY_ROWS = (
    (2, "learned_ng3"),
    (3, "nfe10"),
    (3, "naive_nfe3"),
    (3, "learned_ng3"),
)
CONDITIONAL_KC4_ROWS = (
    (4, "nfe10"),
    (4, "naive_nfe3"),
    (4, "learned_ng3"),
)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _environment() -> dict[str, str]:
    manifest = load_json(MANIFEST)
    renderer = manifest["renderer"]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "EGL_DEVICE_ID": "0",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "SIMVLA_LIBERO_ROOT": str(STORAGE / "datasets/LIBERO"),
            "LIBERO_CONFIG_PATH": str(
                STORAGE
                / "results/simvla/reproduction/"
                "official_ckpt_mujoco237_official_norm_seed7_n50_r2/"
                "runtime/libero_config"
            ),
            "SIMVLA_UPSTREAM_ROOT": str(UPSTREAM),
            "PYTHONPATH": ":".join(
                (
                    str(WORKTREE),
                    str(UPSTREAM),
                    str(STORAGE / "datasets/LIBERO"),
                    environment.get("PYTHONPATH", ""),
                )
            ),
            "HF_HOME": environment.get(
                "HF_HOME", str(STORAGE / "cache/simvla/huggingface")
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "CUBLAS_WORKSPACE_CONFIG": str(renderer["CUBLAS_WORKSPACE_CONFIG"]),
            "CUDA_DEVICE_MAX_CONNECTIONS": str(
                renderer["CUDA_DEVICE_MAX_CONNECTIONS"]
            ),
            "PYTHONHASHSEED": str(renderer["PYTHONHASHSEED"]),
            "SIMVLA_RENDER_AXIS": str(renderer["SIMVLA_RENDER_AXIS"]),
        }
    )
    environment.pop("GALLIUM_DRIVER", None)
    environment.pop("LIBGL_ALWAYS_SOFTWARE", None)
    return environment


def _wait_for(path: Path, *, label: str) -> None:
    while not path.is_file():
        failure = path.parent / "failure.json"
        if failure.is_file():
            raise RuntimeError(f"{label} failed: {load_json(failure)}")
        print(f"[{_timestamp()}] waiting for {label} without GPU use", flush=True)
        time.sleep(120)


def _verify_diagnostic() -> dict[str, Any]:
    path = DIAGNOSTIC_ROOT / "diagnostic_summary.json"
    _wait_for(path, label="immutable S50 diagnostic aggregation")
    payload = load_json(path)
    if (
        payload.get("verdict") != "S50_10K_RB2_DIAGNOSTIC_COMPLETE"
        or payload.get("classification") != "DIAGNOSTIC_ONLY"
        or payload.get("offline_gate_passed") is not False
        or len(payload.get("rows", [])) != 6
    ):
        raise RuntimeError("S50 diagnostic aggregation lost its immutable contract")
    return payload


def _verify_bundle() -> dict[str, Any]:
    ready_path = BUNDLE / "READY_SHORT_FOR_RB2.json"
    _wait_for(ready_path, label="gate-passing V3 bundle")
    ready = load_json(ready_path)
    if (
        ready.get("schema_version") != BUNDLE_SCHEMA
        or ready.get("verdict") != "READY_SHORT_FOR_RB2"
        or ready.get("method_version") != "stability_v3"
        or ready.get("offline_gate_passed") is not True
    ):
        raise RuntimeError("V3 bundle is not strict-offline approved")
    expected_ready_sha = ready.get("combined_sha256")
    observed_ready_sha = canonical_sha256(
        {key: value for key, value in ready.items() if key != "combined_sha256"}
    )
    if expected_ready_sha != observed_ready_sha:
        raise RuntimeError("V3 readiness contract hash changed")
    manifest = load_json(BUNDLE / "SHA256_MANIFEST.json")
    if manifest != ready.get("files"):
        raise RuntimeError("V3 bundle manifest and readiness contract differ")
    failures = {
        relative: {
            "expected": digest,
            "observed": (
                sha256_file(BUNDLE / relative)
                if (BUNDLE / relative).is_file()
                else None
            ),
        }
        for relative, digest in manifest.items()
        if not (BUNDLE / relative).is_file()
        or sha256_file(BUNDLE / relative) != digest
    }
    if failures:
        raise RuntimeError(f"V3 bundle file hash mismatch: {failures}")
    checkpoint = BUNDLE / str(ready["checkpoint"])
    if sha256_file(checkpoint) != ready.get("checkpoint_sha256"):
        raise RuntimeError("V3 bundle checkpoint hash changed")
    if ready.get("kc8_offline_ready") is not False:
        raise RuntimeError("V3 bundle incorrectly approved K_C=8")
    return ready


def _verify_runtime_source() -> dict[str, Any]:
    source_lock = load_json(BUNDLE / "source_lock.json")
    expected = source_lock.get("source_files")
    if not isinstance(expected, dict) or set(expected) != set(V3_SOURCE_FILES):
        raise RuntimeError("V3 source-lock file set changed")
    observed = {
        relative: (
            sha256_file(WORKTREE / relative)
            if (WORKTREE / relative).is_file()
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
        raise RuntimeError(f"rb2 V3 runtime source mismatch: {mismatches}")
    observed_combined = canonical_sha256(observed)
    if observed_combined != source_lock.get("source_files_combined_sha256"):
        raise RuntimeError("rb2 V3 runtime source combined hash changed")
    return {
        "verdict": "STABILITY_V3_RB2_SOURCE_LOCK_PASS",
        "source_files": len(observed),
        "source_files_combined_sha256": observed_combined,
    }


def _wait_gpu() -> None:
    while True:
        snapshot = _gpu_snapshot()
        atomic_write_json(RESULT_ROOT / "gpu_wait_state.json", snapshot)
        if snapshot["free"]:
            return
        print(f"[{_timestamp()}] rb2 GPU occupied; waiting 120s", flush=True)
        time.sleep(120)


def _run_row(ready: Mapping[str, Any], k_c: int, mode: str) -> dict[str, Any]:
    output = RESULT_ROOT / "rows" / f"kc{k_c}_{mode}"
    summary_path = output / "row_summary.json"
    if summary_path.is_file():
        payload = load_json(summary_path)
        if (
            payload.get("verdict") == "RB2_STABILITY_ROW_COMPLETE"
            and int(payload.get("episodes", -1)) == 500
            and payload.get("checkpoint_sha256") == ready["checkpoint_sha256"]
        ):
            return payload
        raise RuntimeError(f"incompatible V3 online row exists: {output}")
    if output.exists():
        failed_root = RESULT_ROOT / "failed_attempts"
        failed_root.mkdir(parents=True, exist_ok=True)
        destination = failed_root / f"kc{k_c}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(output), destination)
    _wait_gpu()
    print(f"[{_timestamp()}] V3 start K_C={k_c} mode={mode}", flush=True)
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
        ),
        cwd=WORKTREE,
        env=_environment(),
        check=True,
    )
    result = load_json(summary_path)
    print(
        f"[{_timestamp()}] V3 complete K_C={k_c} mode={mode} "
        f"success={result['successes']}/{result['episodes']}",
        flush=True,
    )
    return result


def _aggregate(
    diagnostic: Mapping[str, Any],
    ready: Mapping[str, Any],
    runtime_source: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = _baseline_with_tasks()
    table: list[dict[str, Any]] = []
    learned_acceptance: dict[str, Any] = {}
    for row in rows:
        acceptance = None
        if row["generation_mode"] == "learned_ng3":
            old = {3: 467 / 500, 4: 438 / 500}.get(int(row["k_c"]))
            acceptance = _acceptance(
                baseline=baseline, row=row, old_success_rate=old
            )
            learned_acceptance[str(row["k_c"])] = acceptance
        table.append({**row, "online_acceptance": acceptance})
    fields = (
        "row",
        "k_c",
        "generation_mode",
        "episodes",
        "successes",
        "success_rate",
        "mean_policy_query_latency_ms",
        "mean_vlm_latency_ms",
        "mean_condition_latency_ms",
        "mean_generation_latency_ms",
        "mean_gripper_switches_per_episode",
        "switch_disagreement_p95",
        "manifest_sha256",
        "checkpoint_sha256",
    )
    with (RESULT_ROOT / "v3_online_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    result = {
        "schema_version": "simvla_stability_v3_rb2_summary_v1",
        "verdict": "SIMVLA_STABILITY_V3_RB2_EVALUATION_COMPLETE",
        "checkpoint_sha256": ready["checkpoint_sha256"],
        "selected_branch": ready["selected_branch"],
        "manifest_sha256": MANIFEST_SHA256,
        "runtime_source_audit": dict(runtime_source),
        "baseline": baseline,
        "rows": table,
        "learned_ng3_acceptance": learned_acceptance,
        "immutable_s50_diagnostic": {
            "classification": diagnostic["classification"],
            "offline_gate_passed": diagnostic["offline_gate_passed"],
            "rows": diagnostic["rows"],
        },
        "kc8_status": "BLOCKED_UNTIL_KC4_PASSES",
        "additional_inference_noise_seeds": 0,
    }
    atomic_write_json(RESULT_ROOT / "v3_online_summary.json", result)
    return result


def run() -> dict[str, Any]:
    if os.environ.get("SIMVLA_STABILITY_V3_RB2_RUN") != "1":
        raise RuntimeError("set SIMVLA_STABILITY_V3_RB2_RUN=1 to approve rb2 evaluation")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in (WORKTREE, PYTHON, UPSTREAM, MANIFEST, BASELINE):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    diagnostic = _verify_diagnostic()
    atomic_write_json(
        RESULT_ROOT / "immutable_s50_diagnostic_reference.json", diagnostic
    )
    ready = _verify_bundle()
    runtime_source = _verify_runtime_source()
    atomic_write_json(RESULT_ROOT / "runtime_source_audit.json", runtime_source)
    rows = [_run_row(ready, k_c, mode) for k_c, mode in PRIMARY_ROWS]
    preliminary = _aggregate(diagnostic, ready, runtime_source, rows)
    kc3_passed = bool(
        preliminary.get("learned_ng3_acceptance", {}).get("3", {}).get("passed")
    )
    if kc3_passed:
        rows.extend(
            _run_row(ready, k_c, mode) for k_c, mode in CONDITIONAL_KC4_ROWS
        )
    result = _aggregate(diagnostic, ready, runtime_source, rows)
    result["kc4_executed"] = kc3_passed
    result["kc8_status"] = (
        "BLOCKED_PENDING_KC4_ACCEPTANCE"
        if kc3_passed
        else "BLOCKED_BECAUSE_KC3_FAILED"
    )
    atomic_write_json(RESULT_ROOT / "v3_online_summary.json", result)
    return result


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = RESULT_ROOT / "pipeline.lock"
    try:
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run()
    except Exception as error:
        atomic_write_json(
            RESULT_ROOT / "pipeline_failure.json",
            {
                "verdict": "STABILITY_V3_RB2_RUNTIME_FAILURE",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "existing_s50_diagnostic_altered": False,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
