"""Hash and source locks for the official Latent Bridge dependency."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path


OFFICIAL_REPOSITORY = "https://github.com/1999Lyd/Latent-Bridge"
OFFICIAL_COMMIT = "ed556014aa96bae8ed85768194f02360389b9365"
OFFICIAL_MODEL_SHA256 = "df19ff70722347c071fd08244875867d9b8110b76735e72ee50053fbd3351da7"
PUBLIC_SEER_33_SHA256 = "a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646"
SEER_VIT_MAE_SHA256 = "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"


def expected_base_checkpoint_sha256() -> str:
    """Return the launcher's hash-locked Seer checkpoint identity."""

    value = os.environ.get(
        "SEER_LATENT_BRIDGE_BASE_CHECKPOINT_SHA256", PUBLIC_SEER_33_SHA256
    ).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            "SEER_LATENT_BRIDGE_BASE_CHECKPOINT_SHA256 must be a lowercase SHA256"
        )
    return value


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_hash(path: str | Path, expected: str, label: str) -> str:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected={expected}, actual={actual}, path={path}")
    return actual


def verify_official_source(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root).resolve()
    model_file = repo_root / "qcvla/model/rectified_flow_bridge.py"
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official source commit mismatch: {commit} != {OFFICIAL_COMMIT}")
    if dirty:
        raise RuntimeError(f"official source clone is dirty:\n{dirty}")
    model_hash = require_file_hash(model_file, OFFICIAL_MODEL_SHA256, "official bridge model")
    return {
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "dirty": False,
        "model_file": str(model_file),
        "model_sha256": model_hash,
    }
