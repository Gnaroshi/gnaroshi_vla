#!/usr/bin/env python3
"""Prepare, validate, and aggregate the frozen SimVLA four-suite matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SEEDS = ("seed01", "seed02", "seed03")
ROWS = (
    "full_nfe10",
    "generation_ng3",
    "condition_kc2_ng3",
    "condition_kc2_ng10",
    "naive_nfe3",
)
MAX_POLICY_ACTIONS = {
    "libero_spatial": 800,
    "libero_object": 800,
    "libero_goal": 800,
    "libero_10": 900,
}
SEED_CONTRACTS = {
    "seed01": (20260815, 6828326409295398833),
    "seed02": (20260816, 6828326409295398834),
    "seed03": (20260817, 6828326409295398835),
}
EXPECTED_GENERATION_COMMIT = "f73ec564cfc9784366903d62047ca80cd115ac56"
EXPECTED_FIXED_COMMIT = "a30631477b277e2067c7fbe33f23cda10a82b5eb"
EXPECTED_UPSTREAM_COMMIT = "32700d0ad8991996e123e4b685abe370ce6e9aab"
EXPECTED_GENERATION_SOURCE = (
    "9d245d6cec54ea470df431eb3f77614d1f77d22b6d00e019caa51be97a37caa7"
)
EXPECTED_GENERATION_CHECKPOINT = (
    "126458846c11cfae3b47c6027f5538e9f58670d0d237a9eea8cd3c30ad3175bc"
)
EXPECTED_CONDITION_CHECKPOINT = (
    "d19057768a99f8130bbce279d694a1a9aa9896a7953ad613aadc88d1e8b194db"
)
EXPECTED_NORM_STATS = (
    "5e4dcf9026271137e102f6f784d345f0f03c1fd9963b679631b110a16788149e"
)
EXPECTED_CACHE_MANIFEST = (
    "ac2856e6033f025c28ed97773a198802837508a9a72edea3418693cf359ddcfc"
)
EXPECTED_REVISION = "93dc4d90b0596c652ad2840ad743c62b9c4473fb"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def git_commit(path: str | Path) -> str:
    return subprocess.check_output(
        ("git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"), text=True
    ).strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, str | None]:
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mujoco": _package_version("mujoco"),
        "transformers": _package_version("transformers"),
        "numpy": _package_version("numpy"),
    }


def validate_manifest(
    manifest: Mapping[str, Any], *, suite: str | None = None, seed: str | None = None
) -> dict[str, Any]:
    copied = dict(manifest)
    claimed = str(copied.pop("manifest_sha256", ""))
    observed = canonical_sha256(copied)
    failures: list[str] = []
    if claimed != observed:
        failures.append(f"manifest SHA mismatch: {claimed} != {observed}")
    if suite is not None and manifest.get("suite") != suite:
        failures.append(f"suite mismatch: {manifest.get('suite')} != {suite}")
    if seed is not None:
        determinism_seed, action_seed = SEED_CONTRACTS[seed]
        if int(manifest.get("determinism_seed", -1)) != determinism_seed:
            failures.append("determinism seed mismatch")
        if int(manifest.get("action_noise_seed_base", -1)) != action_seed:
            failures.append("action-noise seed mismatch")
        if manifest.get("inference_seed_replica") not in (None, seed):
            failures.append("inference seed label mismatch")
    expected_suite = suite or str(manifest.get("suite"))
    if expected_suite not in SUITES:
        failures.append(f"unsupported suite: {expected_suite}")
    elif int(manifest.get("max_policy_actions", -1)) != MAX_POLICY_ACTIONS[expected_suite]:
        failures.append("max-policy-actions differs from official SimVLA client")
    episodes = manifest.get("episodes", [])
    if len(episodes) != 500 or int(manifest.get("episodes_per_row", -1)) != 500:
        failures.append("manifest must contain exactly 500 episodes")
    if int(manifest.get("trials_per_task", -1)) != 50:
        failures.append("manifest must contain 50 trials per task")
    keys = {
        (int(item["task_id"]), int(item["trial_id"])) for item in episodes
    }
    if keys != {(task, trial) for task in range(10) for trial in range(50)}:
        failures.append("episode task/trial coverage is not the exact 10x50 grid")
    if any(item.get("suite") != expected_suite for item in episodes):
        failures.append("one or more episode suite labels differ")
    fixed_fields = {
        "source_combined_sha256": EXPECTED_GENERATION_SOURCE,
        "checkpoint_revision": EXPECTED_REVISION,
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "num_wait_steps": 10,
        "client_resize_size": 224,
        "model_image_size": 384,
        "environment_resolution": 256,
        "environment_seed": 7,
    }
    for name, expected in fixed_fields.items():
        if manifest.get(name) != expected:
            failures.append(f"{name} mismatch: {manifest.get(name)!r} != {expected!r}")
    renderer = manifest.get("renderer", {})
    if renderer.get("MUJOCO_GL") != "egl" or renderer.get("PYOPENGL_PLATFORM") != "egl":
        failures.append("renderer is not EGL")
    return {
        "verdict": "PAPER_SUITE_MANIFEST_PASS" if not failures else "PAPER_SUITE_MANIFEST_FAIL",
        "manifest_sha256": claimed,
        "suite": expected_suite,
        "failures": failures,
    }


def prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.suite not in SUITES or args.seed not in SEEDS:
        raise ValueError("unsupported suite or seed")
    destination = Path(args.output).expanduser().resolve()
    if args.suite == "libero_10":
        payload = load_json(args.base_manifest)
    else:
        base = load_json(args.base_manifest)
        determinism_seed, action_seed = SEED_CONTRACTS[args.seed]
        payload = {key: value for key, value in base.items() if key != "manifest_sha256"}
        payload.update(
            {
                "schema_version": "simvla_generation_libero_suite_v1",
                "suite": args.suite,
                "max_policy_actions": MAX_POLICY_ACTIONS[args.suite],
                "determinism_seed": determinism_seed,
                "action_noise_seed_base": action_seed,
                "environment_seed": 7,
                "inference_seed_replica": args.seed,
                "training_seed_replica": "fixed_generation_step_030000",
                "same_trained_checkpoint_across_replicas": True,
                "evaluation_axis": {
                    "classification": "rb2_egl_four_suite_three_inference_seed",
                    "suite": args.suite,
                    "inference_seed": args.seed,
                    "source_locked_policy_code_modified": False,
                },
                "renderer": {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "PYTHONHASHSEED": str(determinism_seed),
                    "SIMVLA_RENDER_AXIS": (
                        f"rb2_egl_paper4suite3seed_{args.suite}_{args.seed}_v1"
                    ),
                },
            }
        )
        payload["episodes"] = [
            {
                "suite": args.suite,
                "task_id": task_id,
                "trial_id": trial_id,
                "init_state_index": trial_id,
                "environment_seed": 7,
                "physical_gpu_id": 0,
            }
            for task_id in range(10)
            for trial_id in range(50)
        ]
        payload["episodes_per_row"] = 500
        payload["trials_per_task"] = 50
        payload["selected_physical_gpu_ids"] = [0]
        payload["task_partition"] = {"rank0": list(range(10))}
        payload["task_iteration_order"] = {"rank0": list(reversed(range(10)))}
        payload["manifest_sha256"] = canonical_sha256(payload)
    report = validate_manifest(payload, suite=args.suite, seed=args.seed)
    if report["verdict"] != "PAPER_SUITE_MANIFEST_PASS":
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    if destination.exists():
        existing = load_json(destination)
        if existing != payload:
            raise FileExistsError(f"existing manifest differs: {destination}")
    else:
        atomic_json(destination, payload)
    return {**report, "path": str(destination), "reused_long_manifest": args.suite == "libero_10"}


def manifest_env(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    report = validate_manifest(manifest)
    if report["verdict"] != "PAPER_SUITE_MANIFEST_PASS":
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    renderer = manifest["renderer"]
    result = {
        "PYTHONHASHSEED": str(manifest["determinism_seed"]),
        "SIMVLA_RENDER_AXIS": str(renderer["SIMVLA_RENDER_AXIS"]),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "suite": manifest["suite"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    if args.shell:
        for name in (
            "PYTHONHASHSEED",
            "SIMVLA_RENDER_AXIS",
            "MUJOCO_GL",
            "PYOPENGL_PLATFORM",
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_DEVICE_MAX_CONNECTIONS",
        ):
            print(f"{name}={result[name]}")
    return result


def _verify_hash_map(root: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    for relative, digest in sorted(expected.items()):
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        if observed != digest:
            mismatches[relative] = {"expected": digest, "observed": observed}
    return {
        "verdict": "FILE_HASHES_PASS" if not mismatches else "FILE_HASHES_FAIL",
        "root": str(root),
        "checked_files": len(expected),
        "mismatches": mismatches,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    generation_root = Path(args.generation_root).resolve()
    control_root = Path(args.control_root).resolve()
    fixed_root = Path(args.fixed_root).resolve()
    upstream = Path(args.upstream).resolve()
    storage = Path(args.storage).resolve()
    bundle = Path(args.bundle).resolve()
    condition_checkpoint = Path(args.condition_checkpoint).resolve()
    fixed_lock_path = Path(args.fixed_lock).resolve()
    source_lock = load_json(bundle / "metadata/source_lock.json")
    transfer = load_json(bundle / "transfer_manifest.json")
    fixed_lock = load_json(fixed_lock_path)
    failures: list[str] = []

    commits = {
        "generation": git_commit(generation_root),
        "generation_control": git_commit(control_root),
        "fixed": git_commit(fixed_root),
        "upstream": git_commit(upstream),
    }
    expected_commits = {
        "generation": EXPECTED_GENERATION_COMMIT,
        "generation_control": EXPECTED_GENERATION_COMMIT,
        "fixed": EXPECTED_FIXED_COMMIT,
        "upstream": EXPECTED_UPSTREAM_COMMIT,
    }
    for name, expected in expected_commits.items():
        if commits[name] != expected:
            failures.append(f"{name} commit mismatch: {commits[name]} != {expected}")
    if source_lock.get("combined_sha256") != EXPECTED_GENERATION_SOURCE:
        failures.append("Generation source combined SHA changed")
    if fixed_lock.get("root_commit") != EXPECTED_FIXED_COMMIT:
        failures.append("fixed evaluator source-lock commit changed")
    generation_files = _verify_hash_map(
        generation_root, source_lock.get("relevant_file_sha256", {})
    )
    control_files = _verify_hash_map(
        control_root, transfer.get("control_file_sha256", {})
    )
    fixed_files = _verify_hash_map(fixed_root, fixed_lock.get("file_sha256", {}))
    for name, report in (
        ("generation", generation_files),
        ("control", control_files),
        ("fixed", fixed_files),
    ):
        if report["verdict"] != "FILE_HASHES_PASS":
            failures.append(f"{name} locked file hashes changed")

    artifacts = {
        "generation_checkpoint": (
            bundle / "checkpoint/generation_step_030000.pt",
            EXPECTED_GENERATION_CHECKPOINT,
        ),
        "condition_checkpoint": (
            condition_checkpoint,
            EXPECTED_CONDITION_CHECKPOINT,
        ),
        "norm_stats": (
            bundle / "norm/libero_norm_official_32700d0.json",
            EXPECTED_NORM_STATS,
        ),
        "cache_manifest": (
            bundle / "exact_cache_contract/manifest.json",
            EXPECTED_CACHE_MANIFEST,
        ),
    }
    artifact_report: dict[str, Any] = {}
    for name, (path, expected) in artifacts.items():
        observed = sha256_file(path) if path.is_file() else None
        artifact_report[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "matches": observed == expected,
        }
        if observed != expected:
            failures.append(f"{name} SHA mismatch")

    expected_runtime = source_lock.get("environment", {})
    current_runtime = runtime_versions()
    runtime_mismatches = {
        key: {"expected": value, "observed": current_runtime.get(key)}
        for key, value in expected_runtime.items()
        if current_runtime.get(key) != value
    }
    if runtime_mismatches:
        failures.append("runtime versions differ from the frozen Generation source")

    required_dirs = {
        "libero": storage / "datasets/LIBERO",
        "hf_simvla": (
            storage
            / "cache/simvla/huggingface/hub/models--YuankaiLuo--SimVLA-LIBERO"
            / f"snapshots/{EXPECTED_REVISION}"
        ),
        "hf_smolvlm": (
            storage
            / "cache/simvla/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots"
        ),
    }
    missing_dirs = {name: str(path) for name, path in required_dirs.items() if not path.is_dir()}
    if missing_dirs:
        failures.append("required dataset/model-cache directory is missing")
    usage = shutil.disk_usage(storage)
    minimum_free = int(args.minimum_free_gib) * 2**30
    if usage.free < minimum_free:
        failures.append(
            f"storage free space below {args.minimum_free_gib} GiB: {usage.free / 2**30:.1f} GiB"
        )

    report = {
        "verdict": "PAPER_MATRIX_PREFLIGHT_PASS" if not failures else "PAPER_MATRIX_PREFLIGHT_FAIL",
        "commits": commits,
        "expected_commits": expected_commits,
        "generation_source_combined_sha256": source_lock.get("combined_sha256"),
        "locked_files": {
            "generation": generation_files,
            "generation_control": control_files,
            "fixed_2x2": fixed_files,
        },
        "artifacts": artifact_report,
        "runtime": {"expected": expected_runtime, "observed": current_runtime, "mismatches": runtime_mismatches},
        "required_directories": {name: str(path) for name, path in required_dirs.items()},
        "missing_directories": missing_dirs,
        "storage": {
            "path": str(storage),
            "free_gib": usage.free / 2**30,
            "minimum_free_gib": int(args.minimum_free_gib),
        },
        "driver_files": {
            "helper": {"path": str(Path(args.helper).resolve()), "sha256": sha256_file(args.helper)},
            "launcher": {"path": str(Path(args.launcher).resolve()), "sha256": sha256_file(args.launcher)},
        },
        "failures": failures,
    }
    atomic_json(args.output, report)
    if failures:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _find_metrics(root: Path) -> Path | None:
    candidates = (
        root / "merged/episode_metrics.csv",
        root / "shard_rank0_tasks_0_9/episode_metrics.csv",
        root / "episode_metrics.csv",
    )
    return next((path for path in candidates if path.is_file()), None)


def _find_summary(root: Path) -> Path | None:
    candidates = (
        root / "merged/row_summary.json",
        root / "row_summary.json",
        root / "shard_rank0_tasks_0_9/shard_summary.json",
        root / "shard_summary.json",
    )
    return next((path for path in candidates if path.is_file()), None)


ROW_ALIASES = {"full_nfe10": {"full_nfe10", "baseline_k1"}}


def validate_row_data(
    root: str | Path, manifest_path: str | Path, expected_row: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    row_root = Path(root).resolve()
    manifest = load_json(manifest_path)
    manifest_report = validate_manifest(manifest)
    failures = list(manifest_report["failures"])
    metrics_path = _find_metrics(row_root)
    summary_path = _find_summary(row_root)
    rows: list[dict[str, str]] = []
    if metrics_path is None:
        failures.append("episode_metrics.csv is missing")
    else:
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if summary_path is None:
        failures.append("row summary is missing")
        summary: dict[str, Any] = {}
    else:
        summary = load_json(summary_path)
    accepted_names = ROW_ALIASES.get(expected_row, {expected_row})
    observed_names = {str(item.get("row")) for item in rows}
    if rows and not observed_names.issubset(accepted_names):
        failures.append(f"row name mismatch: {sorted(observed_names)} not in {sorted(accepted_names)}")
    if len(rows) != 500:
        failures.append(f"row has {len(rows)} episodes, expected 500")
    keys = [(int(item["task_id"]), int(item["trial_id"])) for item in rows]
    if len(keys) != len(set(keys)):
        failures.append("duplicate task/trial episode key")
    if set(keys) != {(task, trial) for task in range(10) for trial in range(50)}:
        failures.append("row does not cover the exact 10x50 episode grid")
    if summary and int(summary.get("episodes", -1)) != 500:
        failures.append("summary episode count is not 500")
    if summary and summary.get("manifest_sha256") != manifest.get("manifest_sha256"):
        failures.append("summary and manifest hashes differ")
    counter_values = [item.get("counter_gate", "") for item in rows]
    if any(value and not value.endswith("_PASS") for value in counter_values):
        failures.append("one or more per-episode counter gates failed")
    successes = sum(_bool(item.get("success")) for item in rows)
    if summary and int(summary.get("successes", successes)) != successes:
        failures.append("summary success count differs from CSV")
    actions = sum(int(item.get("episode_length", 0)) for item in rows)
    latency = (
        sum(
            float(item.get("latency_per_executed_action_ms", 0.0))
            * int(item.get("episode_length", 0))
            for item in rows
        )
        / actions
        if actions
        else None
    )
    report = {
        "verdict": "PAPER_ROW_PASS" if not failures else "PAPER_ROW_FAIL",
        "path": str(row_root),
        "row": expected_row,
        "suite": manifest.get("suite"),
        "inference_seed": manifest.get("inference_seed_replica"),
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "executed_actions": actions,
        "latency_per_executed_action_ms": latency,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "metrics_sha256": sha256_file(metrics_path) if metrics_path else None,
        "summary_path": str(summary_path) if summary_path else None,
        "summary_sha256": sha256_file(summary_path) if summary_path else None,
        "failures": failures,
    }
    return report, rows


def validate_row(args: argparse.Namespace) -> dict[str, Any]:
    report, _ = validate_row_data(args.root, args.manifest, args.row)
    if args.output:
        atomic_json(args.output, report)
    if report["verdict"] != "PAPER_ROW_PASS":
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_registry(args: argparse.Namespace) -> dict[str, Any]:
    result_root = Path(args.result_root).resolve()
    storage = Path(args.storage).resolve()
    manifests = {
        (suite, seed): result_root / "manifests" / suite / seed / "episode_manifest.json"
        for suite in SUITES
        for seed in SEEDS
    }
    long_generation = storage / "results/simvla/latentloop/generation_loop_ng2_rb2_v1/online"
    fixed = storage / "results/simvla/fixed_2x2/kc2_ng3_seed02_v1"
    naive = storage / "results/simvla/generation_control/naive_confirmatory_v1"
    reuse: dict[tuple[str, str], Path] = {
        ("seed01", "full_nfe10"): long_generation / "step_010000_long500_egl_paired_v1/baseline_k1",
        ("seed01", "generation_ng3"): long_generation / "step_030000_long500_egl_paired_v1/generation_ng3",
        ("seed02", "full_nfe10"): fixed / "compatibility_from_generation_v1/full_nfe10",
        ("seed02", "generation_ng3"): fixed / "compatibility_from_generation_v1/generation_ng3",
        ("seed02", "condition_kc2_ng3"): fixed / "condition_kc2_ng3",
        ("seed02", "condition_kc2_ng10"): fixed / "condition_kc2_ng10",
        ("seed02", "naive_nfe3"): naive / "seed02/naive_nfe3",
        ("seed03", "full_nfe10"): long_generation / "step_030000_long500_egl_seed03_v1/baseline_k1",
        ("seed03", "generation_ng3"): long_generation / "step_030000_long500_egl_seed03_v1/generation_ng3",
        ("seed03", "naive_nfe3"): naive / "seed03/naive_nfe3",
    }
    entries: list[dict[str, Any]] = []
    for suite in SUITES:
        for seed in SEEDS:
            manifest = manifests[(suite, seed)]
            for row in ROWS:
                planned = result_root / "rows" / suite / seed / row
                reused = False
                path = planned
                reuse_report: dict[str, Any] | None = None
                if suite == "libero_10" and (seed, row) in reuse:
                    candidate = reuse[(seed, row)]
                    reuse_report, _ = validate_row_data(candidate, manifest, row)
                    if reuse_report["verdict"] == "PAPER_ROW_PASS":
                        path = candidate
                        reused = True
                entries.append(
                    {
                        "suite": suite,
                        "seed": seed,
                        "row": row,
                        "manifest": str(manifest),
                        "path": str(path),
                        "planned_path": str(planned),
                        "reused": reused,
                        "reuse_validation": reuse_report,
                    }
                )
    payload = {
        "schema_version": "simvla_paper_four_suite_three_seed_registry_v1",
        "result_root": str(result_root),
        "suites": list(SUITES),
        "seeds": list(SEEDS),
        "rows": list(ROWS),
        "cells": len(entries),
        "episodes_total_including_reuse": len(entries) * 500,
        "episodes_reused": sum(500 for item in entries if item["reused"]),
        "episodes_to_run": sum(500 for item in entries if not item["reused"]),
        "entries": entries,
    }
    atomic_json(args.output, payload)
    return payload


def lookup(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_json(args.registry)
    matches = [
        item
        for item in registry["entries"]
        if item["suite"] == args.suite and item["seed"] == args.seed and item["row"] == args.row
    ]
    if len(matches) != 1:
        raise RuntimeError(f"registry lookup returned {len(matches)} rows")
    result = matches[0]
    if args.field:
        value = result[args.field]
        print(str(value).lower() if isinstance(value, bool) else value)
    return result


def _mcnemar(left_only: int, right_only: int) -> float:
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    smaller = min(int(left_only), int(right_only))
    return min(
        1.0,
        2.0
        * sum(math.comb(discordant, index) for index in range(smaller + 1))
        / (2**discordant),
    )


def _paired(
    baseline: Mapping[tuple[int, int], bool], candidate: Mapping[tuple[int, int], bool]
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise ValueError("paired outcome keys differ")
    counts = {"both_success": 0, "both_fail": 0, "baseline_only": 0, "candidate_only": 0}
    for key, left in baseline.items():
        right = candidate[key]
        if left and right:
            counts["both_success"] += 1
        elif left:
            counts["baseline_only"] += 1
        elif right:
            counts["candidate_only"] += 1
        else:
            counts["both_fail"] += 1
    counts["mcnemar_exact_p"] = _mcnemar(
        counts["baseline_only"], counts["candidate_only"]
    )
    return counts


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_json(args.registry)
    output = Path(args.output).resolve()
    cell_reports: dict[tuple[str, str, str], dict[str, Any]] = {}
    cell_rows: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    missing: list[dict[str, Any]] = []
    for entry in registry["entries"]:
        key = (entry["suite"], entry["seed"], entry["row"])
        report, rows = validate_row_data(entry["path"], entry["manifest"], entry["row"])
        report["reused"] = bool(entry["reused"])
        cell_reports[key] = report
        cell_rows[key] = rows
        if report["verdict"] != "PAPER_ROW_PASS":
            missing.append({"suite": key[0], "seed": key[1], "row": key[2], "failures": report["failures"]})

    suite_summary: dict[str, Any] = {}
    paired_summary: dict[str, Any] = {}
    if not missing:
        for suite in SUITES:
            suite_summary[suite] = {}
            paired_summary[suite] = {}
            for row in ROWS:
                reports = [cell_reports[(suite, seed, row)] for seed in SEEDS]
                rates = [float(item["success_rate"]) for item in reports]
                total_successes = sum(int(item["successes"]) for item in reports)
                total_episodes = sum(int(item["episodes"]) for item in reports)
                total_actions = sum(int(item["executed_actions"]) for item in reports)
                latency = sum(
                    float(item["latency_per_executed_action_ms"])
                    * int(item["executed_actions"])
                    for item in reports
                ) / total_actions
                suite_summary[suite][row] = {
                    "episodes": total_episodes,
                    "successes": total_successes,
                    "success_rate": total_successes / total_episodes,
                    "seed_mean_success_rate": statistics.mean(rates),
                    "seed_sample_std_success_rate": statistics.stdev(rates),
                    "latency_per_executed_action_ms": latency,
                    "per_seed": {
                        seed: {
                            "successes": cell_reports[(suite, seed, row)]["successes"],
                            "episodes": 500,
                            "success_rate": cell_reports[(suite, seed, row)]["success_rate"],
                            "latency_per_executed_action_ms": cell_reports[(suite, seed, row)]["latency_per_executed_action_ms"],
                            "reused": cell_reports[(suite, seed, row)]["reused"],
                        }
                        for seed in SEEDS
                    },
                }
            baseline_outcomes: dict[tuple[int, int, int], bool] = {}
            for seed_index, seed in enumerate(SEEDS):
                for item in cell_rows[(suite, seed, "full_nfe10")]:
                    baseline_outcomes[(seed_index, int(item["task_id"]), int(item["trial_id"]))] = _bool(item["success"])
            for row in ROWS[1:]:
                candidate: dict[tuple[int, int, int], bool] = {}
                for seed_index, seed in enumerate(SEEDS):
                    for item in cell_rows[(suite, seed, row)]:
                        candidate[(seed_index, int(item["task_id"]), int(item["trial_id"]))] = _bool(item["success"])
                paired = _paired(baseline_outcomes, candidate)
                paired.update(
                    {
                        "success_rate_delta_percentage_points": 100.0
                        * (
                            suite_summary[suite][row]["success_rate"]
                            - suite_summary[suite]["full_nfe10"]["success_rate"]
                        ),
                        "latency_reduction_fraction": 1.0
                        - suite_summary[suite][row]["latency_per_executed_action_ms"]
                        / suite_summary[suite]["full_nfe10"]["latency_per_executed_action_ms"],
                    }
                )
                paired_summary[suite][row] = paired

    four_suite: dict[str, Any] = {}
    if not missing:
        for row in ROWS:
            per_seed_macro = []
            for seed in SEEDS:
                per_seed_macro.append(
                    statistics.mean(
                        float(cell_reports[(suite, seed, row)]["success_rate"])
                        for suite in SUITES
                    )
                )
            four_suite[row] = {
                "macro_success_rate": statistics.mean(
                    suite_summary[suite][row]["success_rate"] for suite in SUITES
                ),
                "seed_macro_mean_success_rate": statistics.mean(per_seed_macro),
                "seed_macro_sample_std_success_rate": statistics.stdev(per_seed_macro),
                "per_seed_macro_success_rate": dict(zip(SEEDS, per_seed_macro)),
                "mean_suite_latency_per_executed_action_ms": statistics.mean(
                    suite_summary[suite][row]["latency_per_executed_action_ms"]
                    for suite in SUITES
                ),
            }

    result = {
        "verdict": "PAPER_FOUR_SUITE_THREE_SEED_COMPLETE" if not missing else "PAPER_FOUR_SUITE_THREE_SEED_INCOMPLETE",
        "replication_unit": "inference_noise_seed_on_fixed_trained_checkpoints",
        "training_seed_replication": False,
        "episodes_per_cell": 500,
        "cells": len(registry["entries"]),
        "total_episodes_including_reuse": len(registry["entries"]) * 500,
        "episodes_reused": registry["episodes_reused"],
        "episodes_run_by_this_pipeline": registry["episodes_to_run"],
        "suite_summary": suite_summary,
        "paired_vs_full_nfe10": paired_summary,
        "four_suite_summary": four_suite,
        "cell_reports": {"|".join(key): value for key, value in cell_reports.items()},
        "missing_or_invalid": missing,
    }
    atomic_json(output, result)
    if missing and not args.allow_partial:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("prepare-manifest")
    manifest.add_argument("--base-manifest", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--suite", choices=SUITES, required=True)
    manifest.add_argument("--seed", choices=SEEDS, required=True)
    manifest.set_defaults(handler=prepare_manifest)

    env = commands.add_parser("manifest-env")
    env.add_argument("--manifest", required=True)
    env.add_argument("--shell", action="store_true")
    env.set_defaults(handler=manifest_env)

    check = commands.add_parser("audit")
    check.add_argument("--generation-root", required=True)
    check.add_argument("--control-root", required=True)
    check.add_argument("--fixed-root", required=True)
    check.add_argument("--upstream", required=True)
    check.add_argument("--storage", required=True)
    check.add_argument("--bundle", required=True)
    check.add_argument("--condition-checkpoint", required=True)
    check.add_argument("--fixed-lock", required=True)
    check.add_argument("--helper", required=True)
    check.add_argument("--launcher", required=True)
    check.add_argument("--minimum-free-gib", type=int, default=100)
    check.add_argument("--output", required=True)
    check.set_defaults(handler=audit)

    row = commands.add_parser("validate-row")
    row.add_argument("--root", required=True)
    row.add_argument("--manifest", required=True)
    row.add_argument("--row", choices=ROWS, required=True)
    row.add_argument("--output", default="")
    row.set_defaults(handler=validate_row)

    registry = commands.add_parser("build-registry")
    registry.add_argument("--result-root", required=True)
    registry.add_argument("--storage", required=True)
    registry.add_argument("--output", required=True)
    registry.set_defaults(handler=build_registry)

    find = commands.add_parser("lookup")
    find.add_argument("--registry", required=True)
    find.add_argument("--suite", choices=SUITES, required=True)
    find.add_argument("--seed", choices=SEEDS, required=True)
    find.add_argument("--row", choices=ROWS, required=True)
    find.add_argument("--field", default="")
    find.set_defaults(handler=lookup)

    summary = commands.add_parser("aggregate")
    summary.add_argument("--registry", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--allow-partial", action="store_true")
    summary.set_defaults(handler=aggregate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = args.handler(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not (args.command == "manifest-env" and args.shell) and not (
        args.command == "lookup" and args.field
    ):
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
