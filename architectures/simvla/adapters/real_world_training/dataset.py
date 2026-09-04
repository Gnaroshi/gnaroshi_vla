"""Compact HDF5 dataset and exact SimVLA preprocessing for real demonstrations."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .geometry import ACTION_HORIZON


DATASET_SCHEMA = "simvla_real_hdf5_v1"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def resize_with_pad(image: np.ndarray, size: int = 224) -> Image.Image:
    source = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    width, height = source.size
    scale = min(float(size) / max(width, 1), float(size) / max(height, 1))
    resized = source.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def decode_jpeg(value: np.ndarray) -> np.ndarray:
    payload = np.asarray(value, dtype=np.uint8).tobytes()
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


class RealEpisodeStore:
    """Worker-local HDF5 handles and random access to compact episodes."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != DATASET_SCHEMA:
            raise ValueError("unsupported real-world SimVLA dataset manifest")
        self.episodes = {
            str(item["episode_id"]): (self.manifest_path.parent / item["path"]).resolve()
            for item in self.manifest["episodes"]
        }
        self._handles: dict[str, h5py.File] = {}

    def handle(self, episode_id: str) -> h5py.File:
        if episode_id not in self._handles:
            self._handles[episode_id] = h5py.File(self.episodes[episode_id], "r")
        return self._handles[episode_id]

    def images(self, episode_id: str, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        handle = self.handle(episode_id)
        return (
            decode_jpeg(handle["base_rgb_jpeg"][frame_index]),
            decode_jpeg(handle["wrist_rgb_jpeg"][frame_index]),
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


class RealSimVLADataset(Dataset[dict[str, Any]]):
    """Finite episode-aware dataset with fresh H=10 action targets."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        training: bool,
        sample_stride: int = 1,
        action_horizon: int = ACTION_HORIZON,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        self.store = RealEpisodeStore(manifest_path)
        self.manifest = self.store.manifest
        self.training = bool(training)
        self.action_horizon = int(action_horizon)
        selected = set(self.manifest["splits"][split])
        self.samples: list[tuple[str, int]] = []
        for episode in self.manifest["episodes"]:
            episode_id = str(episode["episode_id"])
            if episode_id not in selected:
                continue
            usable = int(episode["frames"]) - self.action_horizon
            self.samples.extend(
                (episode_id, index) for index in range(0, max(0, usable), sample_stride)
            )
        if not self.samples:
            raise ValueError(f"split {split!r} contains no full H={action_horizon} samples")
        augmentation = []
        if self.training:
            augmentation.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
                )
            )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (384, 384),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                *augmentation,
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def raw_sample(self, index: int) -> dict[str, Any]:
        episode_id, frame_index = self.samples[index]
        handle = self.store.handle(episode_id)
        base, wrist = self.store.images(episode_id, frame_index)
        action_end = frame_index + self.action_horizon
        return {
            "episode_id": episode_id,
            "frame_index": frame_index,
            "instruction": str(handle.attrs["instruction"]),
            "base_rgb": base,
            "wrist_rgb": wrist,
            "proprio": np.asarray(handle["state"][frame_index], dtype=np.float32),
            "action": np.asarray(
                handle["step_action"][frame_index:action_end], dtype=np.float32
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.raw_sample(index)
        base = self.image_transform(resize_with_pad(raw["base_rgb"], 224))
        wrist = self.image_transform(resize_with_pad(raw["wrist_rgb"], 224))
        image_input = torch.stack((base, wrist, torch.zeros_like(base)), dim=0)
        return {
            "episode_id": raw["episode_id"],
            "frame_index": int(raw["frame_index"]),
            "language_instruction": raw["instruction"],
            "image_input": image_input,
            "image_mask": torch.tensor([True, True, False], dtype=torch.bool),
            "proprio": torch.from_numpy(raw["proprio"]),
            "action": torch.from_numpy(raw["action"]),
        }


def query_lookup(dataset: RealSimVLADataset) -> dict[tuple[str, int], int]:
    return {sample: index for index, sample in enumerate(dataset.samples)}
