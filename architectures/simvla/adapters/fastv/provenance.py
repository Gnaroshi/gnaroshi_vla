"""Pinned-source validation for FastV and the frozen SimVLA interfaces."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_REPOSITORY = "https://github.com/pkunlp-icler/FastV.git"
OFFICIAL_COMMIT = "d1659729b5bf1be225e99ee15783deeea80f63b1"
OFFICIAL_FILE_SHA256 = {
    "README.md": "9f6f27bf66f31dcc4b6c65078acd98d363bd379c138fa5f8a9b02c5d1457aaa1",
    "demo-hf.py": "a54900a0e75920df91752c6bf97e504ac4bf356d073d7e12799f78d9d419b1d3",
    (
        "src/FastV/llava-hf/transformers/src/transformers/models/llama/"
        "modeling_llama.py"
    ): "48dfd9351aafede82ccc15bbb772571293a0669e0b7051572dc344c8a40e0726",
}
INTEGRATION_FILES = (
    "architectures/fastv/README.md",
    "architectures/simvla/adapters/fastv/README.md",
    "architectures/simvla/adapters/fastv/__init__.py",
    "architectures/simvla/adapters/fastv/encoder.py",
    "architectures/simvla/adapters/fastv/eval.py",
    "architectures/simvla/adapters/fastv/policy.py",
    "architectures/simvla/adapters/fastv/provenance.py",
    "architectures/simvla/adapters/fastv/recipe.py",
    "architectures/simvla/wrappers/simvla_fastv_eval.sh",
    "architectures/simvla/adapters/fastv/smoke.py",
    "architectures/simvla/wrappers/simvla_fastv_smoke.sh",
)
SIMVLA_CONTRACT_FILES = (
    "models/modeling_smolvlm_vla.py",
    "models/transformer_smolvlm.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def resolve_fastv_upstream(path: str | Path | None = None) -> Path:
    configured = path or os.environ.get("FASTV_UPSTREAM_ROOT")
    root = (
        Path(configured).expanduser()
        if configured
        else ROOT / "architectures/fastv/upstream"
    ).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"official FastV clone not found: {root}; see architectures/fastv/README.md"
        )
    return root


def fastv_source_manifest(
    path: str | Path | None = None,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    root = resolve_fastv_upstream(path)
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--short")
    file_hashes = {
        relative: sha256_file(root / relative) for relative in OFFICIAL_FILE_SHA256
    }
    checks = {
        "commit_matches": commit == OFFICIAL_COMMIT,
        "working_tree_clean": not status,
        "file_hashes_match": file_hashes == OFFICIAL_FILE_SHA256,
    }
    if not checks["commit_matches"]:
        raise RuntimeError(f"FastV commit mismatch: {commit} != {OFFICIAL_COMMIT}")
    if not checks["file_hashes_match"]:
        mismatches = {
            name: {"expected": OFFICIAL_FILE_SHA256[name], "observed": file_hashes[name]}
            for name in file_hashes
            if file_hashes[name] != OFFICIAL_FILE_SHA256[name]
        }
        raise RuntimeError(f"official FastV source hash mismatch: {mismatches}")
    if require_clean and not checks["working_tree_clean"]:
        raise RuntimeError(f"official FastV clone is dirty:\n{status}")
    portable = {
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "file_sha256": file_hashes,
        "integration_label": "official-algorithm SimVLA adaptation",
        "official_simvla_implementation": False,
    }
    canonical = json.dumps(portable, sort_keys=True, separators=(",", ":")).encode()
    return {
        **portable,
        "root": str(root),
        "status_short": status,
        "checks": checks,
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def resolve_simvla_upstream() -> Path:
    return Path(
        os.environ.get("SIMVLA_UPSTREAM_ROOT", ROOT / "architectures/simvla/upstream")
    ).expanduser().resolve()


def simvla_fastv_integration_manifest() -> dict[str, Any]:
    missing = [relative for relative in INTEGRATION_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"FastV integration files are missing: {missing}")
    upstream = resolve_simvla_upstream()
    missing_upstream = [
        relative for relative in SIMVLA_CONTRACT_FILES if not (upstream / relative).is_file()
    ]
    if missing_upstream:
        raise FileNotFoundError(
            f"SimVLA contract files are missing under {upstream}: {missing_upstream}"
        )
    identity = {
        "integration_file_sha256": {
            relative: sha256_file(ROOT / relative) for relative in INTEGRATION_FILES
        },
        "simvla_upstream_commit": _git(upstream, "rev-parse", "HEAD"),
        "simvla_contract_file_sha256": {
            relative: sha256_file(upstream / relative)
            for relative in SIMVLA_CONTRACT_FILES
        },
        "official_fastv_sha256": fastv_source_manifest()["combined_sha256"],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {
        **identity,
        "root": str(ROOT),
        "simvla_upstream_root": str(upstream),
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }
