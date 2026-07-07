"""Teacher condition-cache utilities for DCLD."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class TeacherCacheMetadata:
    architecture: str
    checkpoint: str
    dataset: str
    condition_key: str = "vlm_features"
    norm_stats_path: str | None = None
    action_mode: str | None = None
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class TeacherCacheShardWriter:
    """Small shard writer for teacher condition targets."""

    def __init__(
        self,
        output_dir: str | Path,
        metadata: TeacherCacheMetadata,
        *,
        samples_per_shard: int = 1024,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.samples_per_shard = int(samples_per_shard)
        self._buffer: list[dict[str, Any]] = []
        self._shard_paths: list[str] = []
        self._shard_index = 0

    def add(self, sample: dict[str, Any]) -> None:
        self._buffer.append(sample)
        if len(self._buffer) >= self.samples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        path = self.output_dir / f"shard_{self._shard_index:06d}.pt"
        torch.save(self._buffer, path)
        self._shard_paths.append(path.name)
        self._buffer = []
        self._shard_index += 1
        self.write_manifest()

    def write_manifest(self) -> None:
        manifest = {
            "metadata": asdict(self.metadata),
            "samples_per_shard": self.samples_per_shard,
            "num_shards": len(self._shard_paths),
            "shards": self._shard_paths,
        }
        with (self.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    def close(self) -> None:
        self.flush()
