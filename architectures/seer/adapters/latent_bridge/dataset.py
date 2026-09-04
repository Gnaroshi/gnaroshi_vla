"""Hash-locked, episode-disjoint bridge transition datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .provenance import sha256_file


DATASET_FORMAT_VERSION = 1


@dataclass
class BridgeTransition:
    previous_condition: np.ndarray
    target_condition: np.ndarray
    stable_context: np.ndarray
    current_state: np.ndarray
    previous_executed_action: np.ndarray
    episode_id: str
    task_id: int
    step: int
    success: int
    source: str


class BridgeTransitionWriter:
    """Write fixed-shape transition records and an immutable manifest."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.records: list[BridgeTransition] = []

    def append(self, record: BridgeTransition) -> None:
        self.records.append(record)

    def close(self, metadata: dict) -> dict:
        if not self.records:
            raise RuntimeError("refusing to write an empty bridge transition dataset")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "previous_condition": np.stack([r.previous_condition for r in self.records]).astype(np.float32),
            "target_condition": np.stack([r.target_condition for r in self.records]).astype(np.float32),
            "stable_context": np.stack([r.stable_context for r in self.records]).astype(np.float32),
            "current_state": np.stack([r.current_state for r in self.records]).astype(np.float32),
            "previous_executed_action": np.stack(
                [r.previous_executed_action for r in self.records]
            ).astype(np.float32),
            "task_id": np.asarray([r.task_id for r in self.records], dtype=np.int32),
            "step": np.asarray([r.step for r in self.records], dtype=np.int32),
            "success": np.asarray([r.success for r in self.records], dtype=np.int8),
        }
        strings = h5py.string_dtype(encoding="utf-8")
        with h5py.File(self.output_path, "w") as handle:
            handle.attrs["format_version"] = DATASET_FORMAT_VERSION
            for key, value in arrays.items():
                handle.create_dataset(key, data=value, compression="gzip", compression_opts=1)
            handle.create_dataset(
                "episode_id", data=np.asarray([r.episode_id for r in self.records], dtype=object), dtype=strings
            )
            handle.create_dataset(
                "source", data=np.asarray([r.source for r in self.records], dtype=object), dtype=strings
            )
        digest = sha256_file(self.output_path)
        manifest = {
            "format_version": DATASET_FORMAT_VERSION,
            "dataset_path": str(self.output_path),
            "dataset_sha256": digest,
            "num_transitions": len(self.records),
            "tensor_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "metadata": metadata,
        }
        manifest_path = self.output_path.with_suffix(self.output_path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


class StreamingBridgeTransitionWriter:
    """Append transitions to an interruption-visible HDF5 partial file."""

    _ARRAY_FIELDS = (
        "previous_condition",
        "target_condition",
        "stable_context",
        "current_state",
        "previous_executed_action",
    )

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.partial_path = self.output_path.with_suffix(self.output_path.suffix + ".partial")
        if self.output_path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite bridge dataset: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial_path, "w")
        self.handle.attrs["format_version"] = DATASET_FORMAT_VERSION
        self.count = 0
        self._initialized = False

    def _initialize(self, record: BridgeTransition) -> None:
        for field in self._ARRAY_FIELDS:
            value = np.asarray(getattr(record, field), dtype=np.float32)
            self.handle.create_dataset(
                field,
                shape=(0, *value.shape),
                maxshape=(None, *value.shape),
                chunks=(1, *value.shape),
                dtype=np.float32,
                compression="gzip",
                compression_opts=1,
            )
        for field, dtype in (("task_id", np.int32), ("step", np.int32), ("success", np.int8)):
            self.handle.create_dataset(field, shape=(0,), maxshape=(None,), chunks=True, dtype=dtype)
        strings = h5py.string_dtype(encoding="utf-8")
        self.handle.create_dataset("episode_id", shape=(0,), maxshape=(None,), chunks=True, dtype=strings)
        self.handle.create_dataset("source", shape=(0,), maxshape=(None,), chunks=True, dtype=strings)
        self._initialized = True

    def append(self, record: BridgeTransition) -> None:
        if not self._initialized:
            self._initialize(record)
        row = self.count
        for field in self._ARRAY_FIELDS:
            dataset = self.handle[field]
            dataset.resize(row + 1, axis=0)
            dataset[row] = np.asarray(getattr(record, field), dtype=np.float32)
        scalar_values = {
            "task_id": record.task_id,
            "step": record.step,
            "success": record.success,
            "episode_id": record.episode_id,
            "source": record.source,
        }
        for field, value in scalar_values.items():
            dataset = self.handle[field]
            dataset.resize(row + 1, axis=0)
            dataset[row] = value
        self.count += 1
        if self.count % 128 == 0:
            self.handle.flush()

    def close(self, metadata: dict) -> dict:
        if self.count == 0:
            self.handle.close()
            raise RuntimeError("refusing to finalize an empty bridge transition dataset")
        shapes = {key: list(value.shape) for key, value in self.handle.items()}
        self.handle.attrs["num_transitions"] = self.count
        self.handle.flush()
        self.handle.close()
        self.partial_path.replace(self.output_path)
        digest = sha256_file(self.output_path)
        manifest = {
            "format_version": DATASET_FORMAT_VERSION,
            "dataset_path": str(self.output_path),
            "dataset_sha256": digest,
            "num_transitions": self.count,
            "tensor_shapes": shapes,
            "metadata": metadata,
        }
        manifest_path = self.output_path.with_suffix(self.output_path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def episode_split(episode_id: str, *, validation_fraction: float, seed: int) -> str:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    key = f"{seed}:{episode_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    return "validation" if bucket < validation_fraction else "train"


class BridgeTransitionDataset(Dataset):
    _TENSOR_FIELDS = (
        "previous_condition",
        "target_condition",
        "stable_context",
        "current_state",
        "previous_executed_action",
    )

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        split: str,
        validation_fraction: float = 0.1,
        split_seed: int = 42,
        expected_sha256: str | None = None,
        preload: bool = False,
    ):
        self.path = Path(dataset_path)
        if expected_sha256 is not None:
            actual = sha256_file(self.path)
            if actual != expected_sha256:
                raise RuntimeError(f"dataset SHA256 mismatch: {actual} != {expected_sha256}")
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown split {split!r}")
        with h5py.File(self.path, "r") as handle:
            if int(handle.attrs["format_version"]) != DATASET_FORMAT_VERSION:
                raise RuntimeError("unsupported bridge dataset format")
            episode_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["episode_id"]]
        self.indices = [
            index
            for index, episode_id in enumerate(episode_ids)
            if episode_split(
                episode_id, validation_fraction=validation_fraction, seed=split_seed
            ) == split
        ]
        self._handle = None
        self._preloaded: dict[str, torch.Tensor] | None = None
        self.preloaded_bytes = 0
        if preload and self.indices:
            selected = np.asarray(self.indices, dtype=np.int64)
            with h5py.File(self.path, "r") as handle:
                arrays = {
                    key: np.asarray(handle[key][selected], dtype=np.float32)
                    for key in self._TENSOR_FIELDS
                }
            self._preloaded = {
                key: torch.from_numpy(value) for key, value in arrays.items()
            }
            self.preloaded_bytes = sum(value.nbytes for value in arrays.values())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._preloaded is not None:
            return {key: value[index] for key, value in self._preloaded.items()}
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        row = self.indices[index]
        return {
            key: torch.from_numpy(self._handle[key][row])
            for key in self._TENSOR_FIELDS
        }
