"""Atomic serialization of one Seer hierarchical execution trace."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence


def save_hierarchical_trace(
    output_dir: str | Path,
    episode_key: str,
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{episode_key}.json"
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = {
        "schema_version": 1,
        "metadata": dict(metadata),
        "steps": [dict(row) for row in rows],
    }
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return path
