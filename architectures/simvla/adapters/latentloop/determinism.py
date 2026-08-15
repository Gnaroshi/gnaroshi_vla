"""Strict deterministic runtime and trace contracts for SimVLA evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


DETERMINISM_PROTOCOL = "simvla_online_determinism_v1"
REQUIRED_PROCESS_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def stable_seed(namespace: str, seed: int, *parts: Any, bits: int = 63) -> int:
    """Map a structured key to a stable positive integer seed."""

    if bits < 1 or bits > 63:
        raise ValueError("bits must be in [1, 63]")
    payload = json.dumps(
        [DETERMINISM_PROTOCOL, namespace, int(seed), *parts],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & ((1 << bits) - 1)


@dataclass(frozen=True)
class EvaluationSeedPlan:
    """All stochastic namespaces controlled by one optional experiment seed."""

    experiment_seed: int | None
    process_seed: int
    environment_seed_base: int
    action_noise_seed_base: int
    bootstrap_seed: int
    derivation: str


def resolve_seed_plan(
    *,
    experiment_seed: int | None,
    environment_seed_base: int,
    action_noise_seed_base: int,
    bootstrap_seed: int,
) -> EvaluationSeedPlan:
    """Resolve either a new single-seed contract or a fixed legacy seed tuple."""

    if experiment_seed is None:
        return EvaluationSeedPlan(
            experiment_seed=None,
            process_seed=int(environment_seed_base),
            environment_seed_base=int(environment_seed_base),
            action_noise_seed_base=int(action_noise_seed_base),
            bootstrap_seed=int(bootstrap_seed),
            derivation="legacy_explicit_seed_tuple",
        )
    master = int(experiment_seed)
    return EvaluationSeedPlan(
        experiment_seed=master,
        process_seed=master,
        environment_seed_base=stable_seed("environment_base", master, bits=31),
        action_noise_seed_base=stable_seed("action_noise_base", master),
        bootstrap_seed=stable_seed("bootstrap", master, bits=32),
        derivation="namespace_derived_from_experiment_seed",
    )


def episode_env_seed(base_seed: int, task_id: int, trial_id: int) -> int:
    """Preserve the existing cache seed mapping for backward compatibility."""

    payload = f"{base_seed}|{task_id}|{trial_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def evaluation_episode_seed(
    base_seed: int,
    suite: str,
    task_id: int,
    trial_id: int,
) -> int:
    """Return an order- and row-independent seed for one evaluation episode."""

    return stable_seed(
        "evaluation_episode",
        base_seed,
        str(suite),
        int(task_id),
        int(trial_id),
        bits=31,
    )


def seed_all(seed: int) -> None:
    """Reset every process-local RNG used by the evaluation stack."""

    normalized = int(seed)
    random.seed(normalized)
    np.random.seed(normalized % (2**32))
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)


def _backend_snapshot() -> dict[str, Any]:
    cuda_backend = getattr(torch.backends, "cuda", None)
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "flash_sdp_enabled": bool(cuda_backend.flash_sdp_enabled()) if cuda_backend else None,
        "mem_efficient_sdp_enabled": bool(cuda_backend.mem_efficient_sdp_enabled()) if cuda_backend else None,
        "math_sdp_enabled": bool(cuda_backend.math_sdp_enabled()) if cuda_backend else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def configure_strict_determinism(seed: int) -> dict[str, Any]:
    """Enable fail-closed deterministic execution after validating process env."""

    required = {**REQUIRED_PROCESS_ENV, "PYTHONHASHSEED": str(int(seed))}
    mismatches = {
        name: {"expected": expected, "actual": os.environ.get(name)}
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "strict determinism requires environment variables before Python starts: "
            + json.dumps(mismatches, sort_keys=True)
        )

    seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    snapshot = _backend_snapshot()
    expected_snapshot = {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    bad_backend = {
        name: {"expected": expected, "actual": snapshot.get(name)}
        for name, expected in expected_snapshot.items()
        if snapshot.get(name) != expected
    }
    if bad_backend:
        raise RuntimeError(
            "failed to establish strict PyTorch deterministic backend: "
            + json.dumps(bad_backend, sort_keys=True)
        )
    return {
        "protocol": DETERMINISM_PROTOCOL,
        "seed": int(seed),
        "required_process_environment": required,
        "backend": snapshot,
        "fail_on_nondeterministic_operator": True,
    }


def _hash_update(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        byte_source = tensor.reshape(1) if tensor.ndim == 0 else tensor
        digest.update(byte_source.view(torch.uint8).reshape(-1).numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=str):
            _hash_update(digest, str(key))
            _hash_update(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_update(digest, item)
        return
    if isinstance(value, Path):
        value = str(value)
    digest.update(type(value).__name__.encode("ascii", errors="replace"))
    digest.update(b"\0")
    digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def exact_hash(value: Any) -> str:
    """Hash tensors, arrays, and nested metadata without lossy conversion."""

    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def collect_runtime_identity(device: torch.device) -> dict[str, Any]:
    """Collect numerical-runtime fields used to reject incompatible repeats."""

    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": int(properties.total_memory),
        }
    pip_freeze = _command([sys.executable, "-m", "pip", "freeze", "--all"])
    conda_explicit = _command(["conda", "list", "--explicit"])
    identity = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "cpu_model": _command(
            ["sh", "-c", "sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1"]
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "nvidia_driver": _command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "environment_resolution": {
            "pip_freeze_sha256": canonical_json_hash(pip_freeze),
            "conda_explicit_sha256": canonical_json_hash(conda_explicit),
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "mujoco",
                "robosuite",
                "libero",
                "numpy",
                "transformers",
                "torchvision",
                "Pillow",
            )
        },
        "rendering": {
            name: os.environ.get(name)
            for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM")
        },
    }
    return identity


def build_determinism_manifest(
    *,
    seed_plan: EvaluationSeedPlan,
    strict_runtime: dict[str, Any],
    runtime_identity: dict[str, Any],
    source_identity: dict[str, Any],
    semantic_config: dict[str, Any],
    task_assets: dict[str, Any],
    adapter_checkpoints: dict[str, Any],
) -> dict[str, Any]:
    """Build separate runtime and run-contract hashes for exact repeats."""

    normalized_runtime = dict(strict_runtime)
    normalized_runtime.pop("seed", None)
    normalized_environment = dict(normalized_runtime["required_process_environment"])
    normalized_environment["PYTHONHASHSEED"] = "<experiment-seed>"
    normalized_runtime["required_process_environment"] = normalized_environment
    runtime_contract = {
        "protocol": DETERMINISM_PROTOCOL,
        "strict_runtime": normalized_runtime,
        "runtime_identity": runtime_identity,
        "source_identity": source_identity,
        "adapter_checkpoints": adapter_checkpoints,
    }
    runtime_sha256 = canonical_json_hash(runtime_contract)
    run_contract = {
        "runtime_sha256": runtime_sha256,
        "seed_plan": asdict(seed_plan),
        "semantic_config": semantic_config,
        "task_assets": task_assets,
    }
    return {
        "protocol": DETERMINISM_PROTOCOL,
        "scope": {
            "exact": [
                "episode reset state",
                "policy-query inputs",
                "flow initial noise",
                "condition/action outputs",
                "environment action sequence",
                "terminal state and success",
            ],
            "excluded": [
                "wall-clock duration",
                "latency measurements",
                "progress timestamps",
                "encoded video container bytes",
            ],
        },
        "runtime_contract": runtime_contract,
        "runtime_sha256": runtime_sha256,
        "run_contract": run_contract,
        "run_contract_sha256": canonical_json_hash(run_contract),
    }


def compare_manifest_contracts(
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    """Fail before rollout when a repeat does not share the exact contract."""

    fields = ("protocol", "runtime_sha256", "run_contract_sha256")
    mismatches = {
        field: {"reference": reference.get(field), "current": current.get(field)}
        for field in fields
        if current.get(field) != reference.get(field)
    }
    if mismatches:
        raise RuntimeError(
            "determinism reference contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
