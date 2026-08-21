#!/usr/bin/env python3
"""Fail-closed current-source lock for repaired pi0.5 LatentLoop stages."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "architectures" / "openpi" / "upstream"
EXPECTED_UPSTREAM_COMMIT = "813c9b19f8a5058f71de8b462ae69d08bf60d8e1"
EXPECTED_LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
EXPECTED_CHECKPOINT_SHA256 = "2c2183dc53a8493e3fcde3ee2e41aa593b027977a456719312ab96d7127e7ff1"
EXPECTED_CONFIG_SHA256 = "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a"
EXPECTED_NORM_SHA256 = "33d91d210abc2ab612cfd5381321f5d2ad9965ab7b3ec39cae9186cad97691d1"

OURS_SOURCE_ROOTS = (
    "methods/variable_time_latentloop",
    "architectures/openpi/adapters/latentloop",
    "architectures/openpi/wrappers",
    "tools/openpi",
)
UPSTREAM_SOURCE_FILES = (
    "src/openpi/models_pytorch/pi0_pytorch.py",
    "src/openpi/models_pytorch/gemma_pytorch.py",
    "src/openpi/models_pytorch/preprocessing_pytorch.py",
    "src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py",
    "src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py",
    "src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py",
    "src/openpi/policies/policy.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/policies/libero_policy.py",
    "src/openpi/transforms.py",
    "src/openpi/training/config.py",
    "examples/libero/main.py",
    "uv.lock",
)
PREPROCESSING_FILES = (
    "architectures/openpi/adapters/latentloop/policy_io.py",
    "tools/openpi/evaluate_pi05_latentloop_client.py",
    "architectures/openpi/upstream/src/openpi/policies/libero_policy.py",
    "architectures/openpi/upstream/src/openpi/policies/policy_config.py",
    "architectures/openpi/upstream/src/openpi/models_pytorch/preprocessing_pytorch.py",
)
POSTPROCESSING_FILES = (
    "architectures/openpi/adapters/latentloop/policy_io.py",
    "architectures/openpi/upstream/src/openpi/policies/libero_policy.py",
    "architectures/openpi/upstream/src/openpi/transforms.py",
)


class SourceLockError(RuntimeError):
    """Raised when required lock evidence is absent or differs."""


def normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise SourceLockError(f"invalid SHA-256 value: {value!r}")
    return normalized


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _run_bytes(arguments: list[str], cwd: Path) -> bytes:
    result = subprocess.run(arguments, cwd=cwd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise SourceLockError(
            f"missing evidence: {' '.join(arguments)} failed in {cwd}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def git_identity(repository: Path) -> dict[str, Any]:
    top = Path(_run_bytes(["git", "rev-parse", "--show-toplevel"], repository).decode().strip()).resolve()
    if top != repository.resolve():
        raise SourceLockError(f"missing evidence: {repository} is not a dedicated git root")
    head = _run_bytes(["git", "rev-parse", "HEAD"], repository).decode().strip()
    branch = _run_bytes(["git", "branch", "--show-current"], repository).decode().strip()
    tracked = _run_bytes(["git", "diff", "--binary", "HEAD", "--"], repository)
    staged = _run_bytes(["git", "diff", "--cached", "--binary", "HEAD", "--"], repository)
    status = _run_bytes(["git", "status", "--short", "--untracked-files=all"], repository)
    untracked_names = _run_bytes(["git", "ls-files", "--others", "--exclude-standard", "-z"], repository)
    untracked_manifest: dict[str, str] = {}
    for raw_name in untracked_names.split(b"\0"):
        if not raw_name:
            continue
        name = raw_name.decode("utf-8", errors="surrogateescape")
        path = repository / name
        if path.is_file():
            untracked_manifest[name] = sha256_file(path)
    untracked_hash = _canonical_hash(untracked_manifest)
    dirty_payload = (
        b"TRACKED\0"
        + tracked
        + b"\0STAGED\0"
        + staged
        + b"\0STATUS\0"
        + status
        + b"\0UNTRACKED\0"
        + untracked_hash.encode("ascii")
    )
    return {
        "root": str(repository.resolve()),
        "head": head,
        "branch": branch,
        "dirty": bool(status.strip()),
        "dirty_diff_sha256": _sha256_bytes(dirty_payload),
        "status_sha256": _sha256_bytes(status),
        "status_short": status.decode("utf-8", errors="replace").splitlines(),
        "untracked_content_manifest_sha256": untracked_hash,
        "untracked_file_count": len(untracked_manifest),
    }


def _source_files() -> list[Path]:
    paths: set[Path] = set()
    for relative_root in OURS_SOURCE_ROOTS:
        root = ROOT / relative_root
        if not root.is_dir():
            raise SourceLockError(f"missing evidence: Ours source root does not exist: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts:
                paths.add(path)
    for relative in UPSTREAM_SOURCE_FILES:
        paths.add(UPSTREAM / relative)
    return sorted(paths, key=lambda value: str(value.relative_to(ROOT)))


def source_manifest(paths: Iterable[Path]) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise SourceLockError(f"missing evidence: source file does not exist: {path}")
        relative = str(path.resolve().relative_to(ROOT.resolve()))
        if relative in files:
            raise SourceLockError(f"duplicate source evidence key: {relative}")
        files[relative] = sha256_file(path)
    return {"combined_sha256": _canonical_hash(files), "files": files}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_identity() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise SourceLockError(f"missing evidence: torch import failed: {exc}") from exc
    trusted_libero_pickle = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    if trusted_libero_pickle != "1":
        raise SourceLockError(
            "environment mismatch: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 is required for "
            "the pinned vendored LIBERO legacy init-state files"
        )
    return {
        "python": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "pytorch": torch.__version__,
        "transformers": _version("transformers"),
        "cuda_build": torch.version.cuda,
        "torch_force_no_weights_only_load": trusted_libero_pickle,
    }


def _implementation_hash(relative_files: Iterable[str]) -> dict[str, Any]:
    return source_manifest(ROOT / relative for relative in relative_files)


def collect_current(
    *,
    checkpoint_dir: str | Path,
    norm_stats_path: str | Path,
    config_path: str | Path | None = None,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    action_horizon: int = 10,
    execution_horizon: int = 5,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    expected_checkpoint_sha256 = normalize_sha256(expected_checkpoint_sha256)
    checkpoint = checkpoint_dir / "model.safetensors"
    norm_stats = Path(norm_stats_path).resolve()
    config = (
        Path(config_path).resolve()
        if config_path is not None
        else (UPSTREAM / "checkpoints" / "pi05_base_pytorch" / "config.json").resolve()
    )
    for label, path in (("checkpoint", checkpoint), ("config", config), ("norm", norm_stats)):
        if not path.is_file():
            raise SourceLockError(f"missing evidence: {label} file does not exist: {path}")
    root_git = git_identity(ROOT)
    upstream_git = git_identity(UPSTREAM)
    nested_libero_root = UPSTREAM / "third_party" / "libero"
    nested_libero_git = git_identity(nested_libero_root)
    if upstream_git["head"] != EXPECTED_UPSTREAM_COMMIT:
        raise SourceLockError(
            f"source mismatch: upstream commit {upstream_git['head']} != {EXPECTED_UPSTREAM_COMMIT}"
        )
    if nested_libero_git["head"] != EXPECTED_LIBERO_COMMIT:
        raise SourceLockError(
            f"source mismatch: nested LIBERO commit {nested_libero_git['head']} "
            f"!= {EXPECTED_LIBERO_COMMIT}"
        )
    source = source_manifest(_source_files())
    payload = {
        "schema_version": 2,
        # The workspace HEAD is frozen by the generated lock itself. Keeping a
        # source-code constant for the same commit would create a self-reference
        # that cannot survive committing this tool.
        "repository": root_git,
        "upstream": {**upstream_git, "expected_head": EXPECTED_UPSTREAM_COMMIT},
        "nested_libero": {
            **nested_libero_git,
            "expected_head": EXPECTED_LIBERO_COMMIT,
        },
        "ours_and_upstream_source": source,
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "model_path": str(checkpoint),
            "model_sha256": sha256_file(checkpoint),
            "expected_model_sha256": expected_checkpoint_sha256,
            "config_path": str(config),
            "config_sha256": sha256_file(config),
            "expected_config_sha256": EXPECTED_CONFIG_SHA256,
        },
        "normalization": {
            "path": str(norm_stats),
            "sha256": sha256_file(norm_stats),
            "expected_sha256": EXPECTED_NORM_SHA256,
        },
        "environment": environment_identity(),
        "native_intervals": {
            "action_horizon_h": int(action_horizon),
            "execution_horizon_r": int(execution_horizon),
        },
        "preprocessing": _implementation_hash(PREPROCESSING_FILES),
        "postprocessing": _implementation_hash(POSTPROCESSING_FILES),
    }
    if payload["checkpoint"]["model_sha256"] != expected_checkpoint_sha256:
        raise SourceLockError("checkpoint mismatch: baseline model hash differs from the audited checkpoint")
    if payload["checkpoint"]["config_sha256"] != EXPECTED_CONFIG_SHA256:
        raise SourceLockError("checkpoint mismatch: config hash differs from the audited config")
    if payload["normalization"]["sha256"] != EXPECTED_NORM_SHA256:
        raise SourceLockError("norm mismatch: normalization hash differs from the audited norm statistics")
    if (action_horizon, execution_horizon) != (10, 5):
        raise SourceLockError("source mismatch: pinned pi0.5/LIBERO intervals must be H=10 and R=5")
    payload["source_lock_id"] = _canonical_hash(payload)
    payload["source_lock_v2_pass"] = True
    return payload


def _compare_section(name: str, expected: Any, observed: Any, category: str, mismatches: list[dict[str, str]]) -> None:
    if expected != observed:
        mismatches.append({"category": category, "field": name, "expected": repr(expected), "observed": repr(observed)})


def verify_lock(lock_path: str | Path) -> dict[str, Any]:
    path = Path(lock_path).resolve()
    if not path.is_file():
        raise SourceLockError(f"missing evidence: source lock does not exist: {path}")
    locked = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "source_lock_id", "repository", "upstream", "nested_libero", "ours_and_upstream_source", "checkpoint", "normalization", "environment", "native_intervals", "preprocessing", "postprocessing"}
    missing = sorted(required - locked.keys())
    if missing:
        raise SourceLockError(f"missing evidence: source lock lacks keys {missing}")
    expected_checkpoint_sha256 = locked["checkpoint"].get("expected_model_sha256")
    if expected_checkpoint_sha256 is None:
        raise SourceLockError(
            "missing evidence: source lock checkpoint lacks expected_model_sha256"
        )
    observed = collect_current(
        checkpoint_dir=locked["checkpoint"]["directory"],
        norm_stats_path=locked["normalization"]["path"],
        config_path=locked["checkpoint"]["config_path"],
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        action_horizon=locked["native_intervals"]["action_horizon_h"],
        execution_horizon=locked["native_intervals"]["execution_horizon_r"],
    )
    mismatches: list[dict[str, str]] = []
    for section, category in (
        ("repository", "source_mismatch"),
        ("upstream", "source_mismatch"),
        ("nested_libero", "source_mismatch"),
        ("ours_and_upstream_source", "source_mismatch"),
        ("preprocessing", "source_mismatch"),
        ("postprocessing", "source_mismatch"),
        ("checkpoint", "checkpoint_mismatch"),
        ("normalization", "norm_mismatch"),
        ("environment", "environment_mismatch"),
        ("native_intervals", "source_mismatch"),
    ):
        _compare_section(section, locked[section], observed[section], category, mismatches)
    expected_id_payload = {key: value for key, value in locked.items() if key not in {"source_lock_id", "source_lock_v2_pass"}}
    _compare_section(
        "source_lock_id",
        locked["source_lock_id"],
        _canonical_hash(expected_id_payload),
        "source_mismatch",
        mismatches,
    )
    result = {
        "source_lock_v2_pass": not mismatches,
        "source_lock": str(path),
        "source_lock_id": locked["source_lock_id"],
        "mismatches": mismatches,
    }
    if mismatches:
        categories = sorted({item["category"] for item in mismatches})
        raise SourceLockError(f"source lock v2 failed ({', '.join(categories)}): {json.dumps(mismatches, sort_keys=True)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--checkpoint", required=True)
    create.add_argument("--norm-stats", required=True)
    create.add_argument("--config")
    create.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--lock", required=True)
    args = parser.parse_args()
    if args.command == "create":
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite source lock: {output}")
        payload = collect_current(
            checkpoint_dir=args.checkpoint,
            norm_stats_path=args.norm_stats,
            config_path=args.config,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
    else:
        print(json.dumps(verify_lock(args.lock), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
