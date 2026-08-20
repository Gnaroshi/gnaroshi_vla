"""Episode-disjoint transition sequences loaded from teacher-cache shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset


class LatentLoopSequenceDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: Literal[
            "train",
            "checkpoint_validation",
            "defect_fit",
            "defect_validity",
            "scheduler_calibration",
        ],
        max_delta_q: int,
        one_step_only: bool = False,
        exact_delta_q: int | None = None,
    ) -> None:
        self.root = Path(cache_root).resolve()
        manifest = json.loads((self.root / "pi05_latentloop_cache_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError("teacher cache is not complete")
        self.metadata = manifest["metadata"]
        self.examples: list[tuple[Path, int, int]] = []
        self._episode_cache: dict[Path, list[dict[str, Any]]] = {}
        for entry in manifest["episodes"]:
            if entry["split"] != split:
                continue
            path = self.root / entry["path"]
            count = int(entry["records"])
            for start in range(count - 1):
                largest = min(max_delta_q, count - start - 1)
                if exact_delta_q is not None:
                    deltas = (exact_delta_q,) if largest >= exact_delta_q else ()
                else:
                    deltas = (1,) if one_step_only else range(1, largest + 1)
                self.examples.extend((path, start, delta) for delta in deltas)
        if not self.examples:
            raise ValueError(f"no {split} transition sequences found")

    def __len__(self) -> int:
        return len(self.examples)

    def _records(self, path: Path) -> list[dict[str, Any]]:
        records = self._episode_cache.get(path)
        if records is None:
            records = torch.load(path, map_location="cpu", weights_only=False)["records"]
            # Bound the in-process cache to one episode; shards contain large KV tensors.
            self._episode_cache = {path: records}
        return records

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, start, delta = self.examples[index]
        records = self._records(path)
        return {
            "records": records[start : start + delta + 1],
            "delta_q": delta,
            "cache_path": str(path),
        }
