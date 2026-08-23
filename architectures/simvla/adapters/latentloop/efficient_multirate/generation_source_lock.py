"""Git-centered source identity for Generation Loop experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from architectures.simvla.adapters.latentloop.source_lock import sha256_file


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RELEVANT_PATHS = (
    "methods/latentloop/modules/simvla_generation_loop.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/contracts.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_checkpoint.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_eval.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_hidden.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_objective.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_policy.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_source_lock.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_train.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_offline.py",
    "architectures/simvla/wrappers/run_generation_loop_screening.sh",
)


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generation_source_lock(
    *,
    checkpoint: str,
    checkpoint_revision: str,
    norm_stats: str | Path,
    exact_cache: str | Path,
    relevant_paths: Sequence[str] = DEFAULT_RELEVANT_PATHS,
) -> dict[str, Any]:
    """Return a scientific source hash without binding unrelated dirty files.

    The declared experiment paths must be tracked and clean. Branch names,
    unrelated repository changes, output paths, and physical GPU ordinals are
    recorded as runtime metadata but are not part of the scientific hash.
    """

    paths = tuple(str(value) for value in relevant_paths)
    status = _git("status", "--porcelain", "--", *paths)
    if status:
        raise RuntimeError(
            "Generation Loop source paths are not committed and clean:\n" + status
        )
    tracked = set(_git("ls-files", "--", *paths).splitlines())
    missing = sorted(set(paths) - tracked)
    if missing:
        raise RuntimeError(f"Generation Loop source paths are untracked: {missing}")

    norm_path = Path(norm_stats).expanduser().resolve()
    cache_root = Path(exact_cache).expanduser().resolve()
    cache_manifest = cache_root / "manifest.json"
    file_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in sorted(paths)
        if (ROOT / relative).is_file()
    }
    upstream_root = Path(
        os.environ.get(
            "SIMVLA_UPSTREAM_ROOT", ROOT / "architectures" / "simvla" / "upstream"
        )
    ).expanduser().resolve()
    scientific = {
        "schema_version": "simvla_generation_git_source_v1",
        "root_commit": _git("rev-parse", "HEAD"),
        "relevant_file_sha256": file_hashes,
        "simvla_upstream_commit": _git("rev-parse", "HEAD", cwd=upstream_root),
        "checkpoint": str(checkpoint),
        "checkpoint_revision": str(checkpoint_revision),
        "norm_stats_sha256": sha256_file(norm_path),
        "exact_cache_manifest_sha256": sha256_file(cache_manifest),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mujoco": _package_version("mujoco"),
            "transformers": _package_version("transformers"),
            "numpy": _package_version("numpy"),
        },
    }
    combined = _canonical_sha256(scientific)
    return {
        **scientific,
        "combined_sha256": combined,
        "runtime_metadata": {
            "root_branch": _git("branch", "--show-current"),
            "whole_repo_status_short": _git("status", "--short"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "simvla_gpu_ids": os.environ.get("SIMVLA_GPU_IDS"),
            "root": str(ROOT),
            "upstream_root": str(upstream_root),
            "norm_stats_path": str(norm_path),
            "exact_cache_root": str(cache_root),
        },
    }
