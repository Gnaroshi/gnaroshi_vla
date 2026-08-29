"""Source/checkpoint/environment provenance for every LatentLoop run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = Path(
    os.environ.get(
        "SIMVLA_UPSTREAM_ROOT",
        ROOT / "architectures" / "simvla" / "upstream",
    )
).expanduser().resolve()


def _command(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_huggingface_checkpoint(checkpoint: str) -> dict[str, Any]:
    """Resolve a locally cached HF snapshot without downloading anything."""

    hf_home = Path(os.environ.get("HF_HOME", ROOT / ".cache" / "huggingface"))
    model_key = "models--" + checkpoint.replace("/", "--")
    model_root = hf_home / "hub" / model_key
    ref = model_root / "refs" / "main"
    revision = ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    snapshot = model_root / "snapshots" / revision if revision else None
    weights = snapshot / "model.safetensors" if snapshot else None
    resolved = weights.resolve() if weights and weights.exists() else None
    blob_key = resolved.name if resolved is not None else None
    return {
        "identifier": checkpoint,
        "hf_home": str(hf_home),
        "revision": revision,
        "snapshot_path": str(snapshot) if snapshot else None,
        "weights_path": str(weights) if weights else None,
        "weights_size_bytes": weights.stat().st_size if weights and weights.exists() else None,
        "hf_blob_key_sha256": blob_key if blob_key and len(blob_key) == 64 else None,
        "hash_note": "HF blob key is content-addressed; large weights were not rehashed by source-lock collection.",
    }


def collect_source_lock(
    *,
    checkpoint: str,
    norm_stats_path: str | Path,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Collect immutable axes without importing LIBERO or loading a checkpoint."""

    norm_path = Path(norm_stats_path).resolve()
    return {
        "root": str(ROOT),
        "root_commit": _command(["git", "rev-parse", "HEAD"], ROOT),
        "root_branch": _command(["git", "branch", "--show-current"], ROOT),
        "root_status_short": _command(["git", "status", "--short"], ROOT),
        "simvla_upstream_root": str(UPSTREAM),
        "simvla_upstream_commit": _command(["git", "rev-parse", "HEAD"], UPSTREAM),
        "simvla_upstream_status_short": _command(["git", "status", "--short"], UPSTREAM),
        "checkpoint": resolve_huggingface_checkpoint(checkpoint),
        "norm_stats_path": str(norm_path),
        "norm_stats_sha256": sha256_file(norm_path),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {
            name: _package_version(name)
            for name in (
                "mujoco",
                "robosuite",
                "libero",
                "numpy",
                "transformers",
                "accelerate",
                "torchvision",
            )
        },
        "command": command or sys.argv,
    }


def write_source_lock(
    output_dir: str | Path,
    *,
    checkpoint: str,
    norm_stats_path: str | Path,
    command: list[str] | None = None,
) -> Path:
    """Write source_lock.json under an already-approved unique output root."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "source_lock.json"
    path.write_text(
        json.dumps(
            collect_source_lock(
                checkpoint=checkpoint,
                norm_stats_path=norm_stats_path,
                command=command,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def require_empty_output(path: str | Path) -> Path:
    """Create a unique output directory and refuse any nonempty existing root."""

    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output
