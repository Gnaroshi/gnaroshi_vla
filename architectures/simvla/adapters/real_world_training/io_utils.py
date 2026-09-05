"""Small deterministic I/O helpers shared by the real-world pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


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


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash directory contents and relative names in a stable order."""

    root = Path(path).expanduser().resolve()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory contains no files: {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_int_seed(*parts: object) -> int:
    value = "\x1f".join(str(part) for part in parts)
    return int(sha256_text(value)[:16], 16) % (2**63 - 1)
