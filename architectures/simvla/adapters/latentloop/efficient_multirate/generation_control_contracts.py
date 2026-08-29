"""Contracts for the frozen SimVLA Generation Loop versus naive-NFE control."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


FROZEN_ROOT_COMMIT = "f73ec564cfc9784366903d62047ca80cd115ac56"
FROZEN_UPSTREAM_COMMIT = "32700d0ad8991996e123e4b685abe370ce6e9aab"
FROZEN_CHECKPOINT_REVISION = "93dc4d90b0596c652ad2840ad743c62b9c4473fb"
FROZEN_GENERATION_SOURCE_SHA256 = (
    "9d245d6cec54ea470df431eb3f77614d1f77d22b6d00e019caa51be97a37caa7"
)
FROZEN_GENERATION_CHECKPOINT_SHA256 = (
    "126458846c11cfae3b47c6027f5538e9f58670d0d237a9eea8cd3c30ad3175bc"
)
FROZEN_NORM_STATS_SHA256 = (
    "5e4dcf9026271137e102f6f784d345f0f03c1fd9963b679631b110a16788149e"
)
FROZEN_SEED01_MANIFEST_SHA256 = (
    "d1d9bf5a0ff6b20c235eb92dae80189ed3ebdc9eb1591a51fd0d8d572521e74a"
)
FROZEN_EXACT_CACHE_MANIFEST_SHA256 = (
    "ac2856e6033f025c28ed97773a198802837508a9a72edea3418693cf359ddcfc"
)
FROZEN_TEACHER_CACHE_SOURCE_SHA256 = (
    "40ee185a0af984b028743cecfa96669efa84ca88fb7083f8175dce228c6f0ed9"
)

FULL_ROW = "full_nfe10"
NAIVE_ROW = "naive_nfe3"
GENERATION_ROW = "generation_ng3"
ROWS = (FULL_ROW, NAIVE_ROW, GENERATION_ROW)

ROW_CONTRACTS: dict[str, dict[str, Any]] = {
    FULL_ROW: {
        "full_action_transformer_calls_per_query": 10,
        "generation_loop_updates_per_query": 0,
        "integration_updates_per_query": 10,
        "flow_time_grid": [1.0 - index / 10.0 for index in range(10)],
        "full_step_indices": list(range(10)),
        "k_c": 1,
        "condition_change_code": None,
    },
    NAIVE_ROW: {
        "full_action_transformer_calls_per_query": 3,
        "generation_loop_updates_per_query": 0,
        "integration_updates_per_query": 3,
        "flow_time_grid": [1.0, 2.0 / 3.0, 1.0 / 3.0],
        "full_step_indices": [0, 1, 2],
        "k_c": 1,
        "condition_change_code": None,
    },
    GENERATION_ROW: {
        "full_action_transformer_calls_per_query": 3,
        "generation_loop_updates_per_query": 7,
        "integration_updates_per_query": 10,
        "flow_time_grid": [1.0 - index / 10.0 for index in range(10)],
        "full_step_indices": [0, 4, 8],
        "k_c": 1,
        "condition_change_code": "zero",
    },
}

SEED_CONTRACTS: dict[str, dict[str, Any]] = {
    "seed01": {
        "determinism_seed": 20260815,
        "action_noise_seed_base": 6828326409295398833,
        "environment_seed": 7,
        "role": "exploratory_selection",
    },
    "seed02": {
        "determinism_seed": 20260816,
        "action_noise_seed_base": 6828326409295398834,
        "environment_seed": 7,
        "role": "confirmatory",
    },
    "seed03": {
        "determinism_seed": 20260817,
        "action_noise_seed_base": 6828326409295398835,
        "environment_seed": 7,
        "role": "confirmatory",
    },
}

FORBIDDEN_SOFTWARE_RENDERER_TOKENS = (
    "llvmpipe",
    "softpipe",
    "software rasterizer",
    "osmesa",
    "swrast",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
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


def native_nfe_time_grid(nfe: int) -> tuple[float, ...]:
    value = int(nfe)
    if value < 1:
        raise ValueError("NFE must be positive")
    return tuple(1.0 - index / value for index in range(value))


def runtime_versions() -> dict[str, str | None]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mujoco": version("mujoco"),
        "transformers": version("transformers"),
        "numpy": version("numpy"),
    }


def validate_egl_identity(
    *,
    environment: Mapping[str, str | None],
    gl_vendor: str,
    gl_renderer: str,
) -> dict[str, Any]:
    observed = {
        "MUJOCO_GL": environment.get("MUJOCO_GL"),
        "PYOPENGL_PLATFORM": environment.get("PYOPENGL_PLATFORM"),
        "MUJOCO_EGL_DEVICE_ID": environment.get("MUJOCO_EGL_DEVICE_ID"),
        "GALLIUM_DRIVER": environment.get("GALLIUM_DRIVER"),
        "LIBGL_ALWAYS_SOFTWARE": environment.get("LIBGL_ALWAYS_SOFTWARE"),
        "gl_vendor": str(gl_vendor),
        "gl_renderer": str(gl_renderer),
    }
    failures: list[str] = []
    if observed["MUJOCO_GL"] != "egl":
        failures.append("MUJOCO_GL must equal egl")
    if observed["PYOPENGL_PLATFORM"] != "egl":
        failures.append("PYOPENGL_PLATFORM must equal egl")
    egl_device = str(observed["MUJOCO_EGL_DEVICE_ID"] or "")
    if not egl_device.isdigit():
        failures.append("MUJOCO_EGL_DEVICE_ID must be an explicit non-negative integer")
    if str(observed["LIBGL_ALWAYS_SOFTWARE"] or "").lower() in {
        "1",
        "true",
        "yes",
    }:
        failures.append("LIBGL_ALWAYS_SOFTWARE requests software rendering")
    gallium = str(observed["GALLIUM_DRIVER"] or "").lower()
    if any(token in gallium for token in FORBIDDEN_SOFTWARE_RENDERER_TOKENS):
        failures.append(f"software GALLIUM_DRIVER detected: {gallium}")
    combined = f"{gl_vendor} {gl_renderer}".strip().lower()
    if not str(gl_vendor).strip() or not str(gl_renderer).strip():
        failures.append("GL vendor/renderer must be non-empty")
    if any(token in combined for token in FORBIDDEN_SOFTWARE_RENDERER_TOKENS):
        failures.append(f"software GL renderer detected: {combined}")
    return {
        "verdict": "EGL_PREFLIGHT_PASS" if not failures else "EGL_PREFLIGHT_FAIL",
        "observed": observed,
        "failures": failures,
    }


def require_egl_preflight(path: str | Path, physical_gpu_id: int) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("verdict") != "EGL_PREFLIGHT_PASS":
        raise RuntimeError(f"EGL preflight did not pass: {path}")
    if int(payload.get("physical_gpu_id", -1)) != int(physical_gpu_id):
        raise RuntimeError("EGL preflight physical GPU does not match this shard")
    if payload.get("environment", {}).get("MUJOCO_EGL_DEVICE_ID") != str(
        int(physical_gpu_id)
    ):
        raise RuntimeError("MuJoCo EGL device does not match this shard's physical GPU")
    identity = validate_egl_identity(
        environment=payload.get("environment", {}),
        gl_vendor=str(payload.get("gl_vendor", "")),
        gl_renderer=str(payload.get("gl_renderer", "")),
    )
    if identity["verdict"] != "EGL_PREFLIGHT_PASS":
        raise RuntimeError(f"invalid EGL preflight payload: {identity['failures']}")
    if not payload.get("libero_reset_pass") or not payload.get("libero_step_pass"):
        raise RuntimeError("EGL preflight lacks LIBERO reset/step PASS")
    return payload


def require_stage_gate(
    path: str | Path,
    *,
    verdicts: Iterable[str],
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    payload = load_json(path)
    accepted = set(verdicts)
    if str(payload.get("verdict")) not in accepted:
        raise RuntimeError(
            f"stage gate {path} has verdict={payload.get('verdict')!r}; "
            f"expected one of {sorted(accepted)}"
        )
    if expected_source_sha256 is not None:
        observed = payload.get("source_combined_sha256")
        if observed != expected_source_sha256:
            raise RuntimeError(
                f"stage gate source mismatch: {observed} != {expected_source_sha256}"
            )
    return payload


def validate_row_counters(
    row: str,
    *,
    policy_queries: int,
    full_action_transformer_calls: int,
    generation_loop_updates: int,
    integration_updates: int,
    full_vlm_calls: int,
) -> dict[str, Any]:
    if row not in ROW_CONTRACTS:
        raise ValueError(f"unknown control row: {row}")
    queries = int(policy_queries)
    contract = ROW_CONTRACTS[row]
    expected = {
        "full_action_transformer_calls": (
            queries * int(contract["full_action_transformer_calls_per_query"])
        ),
        "generation_loop_updates": (
            queries * int(contract["generation_loop_updates_per_query"])
        ),
        "integration_updates": queries * int(contract["integration_updates_per_query"]),
        "full_vlm_calls": queries,
    }
    observed = {
        "full_action_transformer_calls": int(full_action_transformer_calls),
        "generation_loop_updates": int(generation_loop_updates),
        "integration_updates": int(integration_updates),
        "full_vlm_calls": int(full_vlm_calls),
    }
    checks = {key: observed[key] == value for key, value in expected.items()}
    checks["k_c_is_one"] = int(contract["k_c"]) == 1
    checks["condition_change_code_zero"] = (
        row != GENERATION_ROW or contract["condition_change_code"] == "zero"
    )
    return {
        "verdict": "ROW_COUNTER_PASS" if all(checks.values()) else "ROW_COUNTER_FAIL",
        "row": row,
        "policy_queries": queries,
        "expected": expected,
        "observed": observed,
        "checks": checks,
    }


def validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    copied = dict(manifest)
    claimed = str(copied.pop("manifest_sha256", ""))
    observed = canonical_sha256(copied)
    failures: list[str] = []
    if not claimed or claimed != observed:
        failures.append(f"canonical manifest hash mismatch: {claimed} != {observed}")
    if expected_manifest_sha256 and claimed != expected_manifest_sha256:
        failures.append(
            f"manifest identity mismatch: {claimed} != {expected_manifest_sha256}"
        )
    if manifest.get("source_combined_sha256") != FROZEN_GENERATION_SOURCE_SHA256:
        failures.append("Generation source identity changed")
    if manifest.get("checkpoint_revision") != FROZEN_CHECKPOINT_REVISION:
        failures.append("SimVLA checkpoint revision changed")
    if int(manifest.get("action_horizon", -1)) != 10:
        failures.append("action horizon must be H=10")
    if int(manifest.get("execution_horizon", -1)) != 5:
        failures.append("execution horizon must be R=5")
    if int(manifest.get("trials_per_task", -1)) != 50:
        failures.append("paper Long manifest must have 50 trials/task")
    if int(manifest.get("episodes_per_row", -1)) != 500:
        failures.append("paper Long manifest must have 500 episodes/row")
    renderer = manifest.get("renderer", {})
    if renderer.get("MUJOCO_GL") != "egl" or renderer.get("PYOPENGL_PLATFORM") != "egl":
        failures.append("manifest renderer is not EGL")
    return {
        "verdict": "EPISODE_MANIFEST_PASS" if not failures else "EPISODE_MANIFEST_FAIL",
        "claimed_manifest_sha256": claimed,
        "observed_manifest_sha256": observed,
        "failures": failures,
    }


def validate_sd1_shard(physical_gpu_id: int, task_ids: Sequence[int]) -> dict[str, Any]:
    gpu = int(physical_gpu_id)
    tasks = tuple(int(value) for value in task_ids)
    expected = (0, 1, 2, 3, 4) if gpu == 2 else (5, 6, 7, 8, 9)
    checks = {
        "allowed_gpu": gpu in {2, 3},
        "forbidden_gpus_excluded": gpu not in {0, 1, 4, 5, 6, 7},
        "task_partition_exact": tasks == expected,
    }
    return {
        "verdict": "SD1_SHARD_PASS" if all(checks.values()) else "SD1_SHARD_FAIL",
        "physical_gpu_id": gpu,
        "task_ids": list(tasks),
        "expected_task_ids": list(expected),
        "checks": checks,
    }


def verify_file_hashes(root: str | Path, expected: Mapping[str, str]) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    rows: dict[str, Any] = {}
    failures: list[str] = []
    for relative, digest in sorted(expected.items()):
        path = base / relative
        observed = sha256_file(path) if path.is_file() else None
        matches = observed == digest
        rows[relative] = {
            "expected_sha256": digest,
            "observed_sha256": observed,
            "matches": matches,
        }
        if not matches:
            failures.append(relative)
    return {
        "verdict": "FILE_HASHES_PASS" if not failures else "FILE_HASHES_FAIL",
        "root": str(base),
        "files": rows,
        "failures": failures,
    }


def exact_mcnemar(baseline_only: int, candidate_only: int) -> float:
    left = int(baseline_only)
    right = int(candidate_only)
    total = left + right
    if total == 0:
        return 1.0
    lower = min(left, right)
    probability = sum(math.comb(total, k) for k in range(lower + 1)) / (2**total)
    return min(1.0, 2.0 * probability)


def hierarchical_bootstrap_difference(
    records: Sequence[Mapping[str, Any]],
    *,
    value_a: str,
    value_b: str,
    seed: int = 20260824,
    samples: int = 10_000,
) -> dict[str, Any]:
    """Bootstrap B-A while resampling inference seed, task, then episode."""

    if int(samples) < 1:
        raise ValueError("bootstrap samples must be positive")
    by_seed: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    for record in records:
        seed_name = str(record["inference_seed"])
        task_id = int(record["task_id"])
        by_seed.setdefault(seed_name, {}).setdefault(task_id, []).append(record)
    if not by_seed:
        raise ValueError("bootstrap requires records")
    rng = np.random.default_rng(int(seed))
    seed_names = sorted(by_seed)
    draws = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        selected_seeds = rng.choice(seed_names, size=len(seed_names), replace=True)
        values: list[float] = []
        for seed_name in selected_seeds:
            tasks = by_seed[str(seed_name)]
            task_ids = sorted(tasks)
            selected_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
            for task_id in selected_tasks:
                episodes = tasks[int(task_id)]
                selected_indices = rng.integers(0, len(episodes), size=len(episodes))
                values.extend(
                    float(episodes[int(item)][value_b]) - float(episodes[int(item)][value_a])
                    for item in selected_indices
                )
        draws[index] = float(np.mean(values))
    return {
        "difference": f"{value_b}-{value_a}",
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "mean": float(draws.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "resampling_hierarchy": ["inference_seed", "task", "episode"],
    }


def git_output(root: str | Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(Path(root).expanduser().resolve()), *args),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
