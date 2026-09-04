"""Datasets linking cached SimVLA conditions to real query transitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .condition_cache import RealConditionCacheDataset
from .dataset import RealSimVLADataset, resize_with_pad


class RealConditionPairDataset(Dataset[dict[str, Any]]):
    """Adjacent policy queries separated by the unchanged R=5 execution horizon."""

    def __init__(self, cache_root: str | Path, *, split: str, execution_horizon: int = 5) -> None:
        self.cache = RealConditionCacheDataset(cache_root, split=split)
        dataset_manifest = self.cache.manifest["dataset_manifest"]
        self.images = RealSimVLADataset(dataset_manifest, split=split, training=False)
        self.image_index = {sample: index for index, sample in enumerate(self.images.samples)}
        by_sample = {
            (str(self.cache.records[index]["episode_id"]), int(self.cache.records[index]["frame_index"])): index
            for index in self.cache.indices
        }
        self.pairs = []
        for (episode_id, frame_index), previous in sorted(by_sample.items()):
            current = by_sample.get((episode_id, frame_index + execution_horizon))
            if current is not None:
                self.pairs.append((previous, current))
        if not self.pairs:
            raise ValueError("no same-episode R=5 condition pairs were found")

    def __len__(self) -> int:
        return len(self.pairs)

    def _raw_images(self, cache_index: int) -> torch.Tensor:
        record = self.cache.records[cache_index]
        sample = (str(record["episode_id"]), int(record["frame_index"]))
        raw = self.images.raw_sample(self.image_index[sample])
        views = [
            np.asarray(resize_with_pad(raw[name], 224), dtype=np.uint8)
            for name in ("base_rgb", "wrist_rgb")
        ]
        return torch.from_numpy(np.stack(views, axis=0))

    def __getitem__(self, index: int) -> dict[str, Any]:
        previous, current = self.pairs[index]
        return {
            "previous_cache_index": previous,
            "current_cache_index": current,
            "previous_condition": torch.from_numpy(
                np.array(self.cache.condition[previous], copy=True)
            ),
            "current_condition": torch.from_numpy(
                np.array(self.cache.condition[current], copy=True)
            ),
            "previous_proprio": torch.from_numpy(
                np.array(self.cache.proprio[previous], copy=True)
            ),
            "current_proprio": torch.from_numpy(
                np.array(self.cache.proprio[current], copy=True)
            ),
            "previous_images": self._raw_images(previous),
            "current_images": self._raw_images(current),
        }

