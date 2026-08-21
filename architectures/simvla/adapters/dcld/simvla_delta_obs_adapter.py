"""Build DCLD delta observations from SimVLA/LIBERO samples.

Production DCLD visual deltas should use raw LIBERO RGB, not SimVLA's
ImageNet-normalized ``image_input`` tensor. The normalized tensor remains a
fallback for legacy smoke checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from copy import copy
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from methods.dcld.modules import DeltaObservation


LIBERO_RAW_RGB_CAMERAS: tuple[str, ...] = ("agentview_rgb", "eye_in_hand_rgb")
LIBERO_DCLD_CAMERA_ALIASES: tuple[str, ...] = ("front", "wrist")


@dataclass(frozen=True)
class LiberoRawSampleRef:
    """Serializable reference to one LIBERO HDF5 timestep."""

    hdf5_path: str
    demo_key: str
    timestep: int
    task_name: str
    language_instruction: str
    episode_id: str
    sample_id: str
    camera_names: tuple[str, ...] = LIBERO_RAW_RGB_CAMERAS
    camera_aliases: tuple[str, ...] = LIBERO_DCLD_CAMERA_ALIASES
    rotate_180: bool = True
    action_horizon: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "hdf5_path": self.hdf5_path,
            "demo_key": self.demo_key,
            "timestep": self.timestep,
            "task_name": self.task_name,
            "language_instruction": self.language_instruction,
            "episode_id": self.episode_id,
            "sample_id": self.sample_id,
            "camera_names": list(self.camera_names),
            "camera_aliases": list(self.camera_aliases),
            "rotate_180": self.rotate_180,
            "action_horizon": self.action_horizon,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiberoRawSampleRef":
        return cls(
            hdf5_path=str(payload["hdf5_path"]),
            demo_key=str(payload["demo_key"]),
            timestep=int(payload["timestep"]),
            task_name=str(payload.get("task_name", "")),
            language_instruction=str(payload.get("language_instruction", payload.get("task_name", ""))),
            episode_id=str(payload.get("episode_id", "")),
            sample_id=str(payload.get("sample_id", "")),
            camera_names=tuple(payload.get("camera_names", LIBERO_RAW_RGB_CAMERAS)),
            camera_aliases=tuple(payload.get("camera_aliases", LIBERO_DCLD_CAMERA_ALIASES)),
            rotate_180=bool(payload.get("rotate_180", True)),
            action_horizon=int(payload.get("action_horizon", 10)),
        )


def _quat2axisangle_single(quat: np.ndarray) -> np.ndarray:
    import math

    quat = quat.astype(np.float32, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def euler_to_axisangle(euler: np.ndarray) -> np.ndarray:
    """Convert LIBERO XYZ Euler state to SimVLA's axis-angle proprio field."""

    rot = R.from_euler("xyz", euler)
    quats = rot.as_quat()
    if quats.ndim == 1:
        return _quat2axisangle_single(quats)
    out = np.zeros((len(quats), 3), dtype=np.float32)
    for i, quat in enumerate(quats):
        out[i] = _quat2axisangle_single(quat)
    return out


def _resolve_hdf5_path(path: str | Path, *, upstream_root: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    root = Path(upstream_root) if upstream_root is not None else Path.cwd()
    return (root / path).resolve()


def _load_meta(metas_path: str | Path) -> dict[str, Any]:
    with Path(metas_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_libero_raw_sample_refs(
    metas_path: str | Path,
    *,
    upstream_root: str | Path | None = None,
    num_actions: int = 10,
    max_episodes: int | None = None,
    max_samples: int | None = None,
) -> Iterable[LiberoRawSampleRef]:
    """Yield raw RGB/proprio references in the same deterministic order as SimVLA eval loading."""

    meta = _load_meta(metas_path)
    yielded_episodes = 0
    yielded_samples = 0
    for file_index, item in enumerate(meta["datalist"]):
        if isinstance(item, dict):
            hdf5_path = _resolve_hdf5_path(item["path"], upstream_root=upstream_root)
            task = str(item.get("task", ""))
        else:
            hdf5_path = _resolve_hdf5_path(item, upstream_root=upstream_root)
            task = hdf5_path.stem.replace("_demo", "").replace("_", " ")

        with h5py.File(hdf5_path, "r") as h5:
            for demo_key in list(h5["data"].keys()):
                if max_episodes is not None and yielded_episodes >= max_episodes:
                    return
                demo = h5["data"][demo_key]
                length = min(
                    len(demo["actions"]),
                    len(demo["obs/agentview_rgb"]),
                    len(demo["obs/eye_in_hand_rgb"]),
                )
                episode_id = f"{file_index:04d}:{demo_key}"
                yielded_episodes += 1
                for timestep in range(max(0, length - num_actions)):
                    if max_samples is not None and yielded_samples >= max_samples:
                        return
                    yielded_samples += 1
                    yield LiberoRawSampleRef(
                        hdf5_path=str(hdf5_path),
                        demo_key=demo_key,
                        timestep=timestep,
                        task_name=task,
                        language_instruction=task,
                        episode_id=episode_id,
                        sample_id=f"{episode_id}:{timestep:06d}",
                        action_horizon=num_actions,
                    )


def load_libero_raw_sample(ref: LiberoRawSampleRef | dict[str, Any]) -> dict[str, Any]:
    """Load raw RGB and proprio for a referenced LIBERO timestep."""

    if isinstance(ref, dict):
        ref = LiberoRawSampleRef.from_dict(ref)
    with h5py.File(ref.hdf5_path, "r") as h5:
        demo = h5["data"][ref.demo_key]
        t = int(ref.timestep)
        images: list[np.ndarray] = []
        for camera in ref.camera_names:
            arr = np.asarray(demo[f"obs/{camera}"][t])
            if ref.rotate_180:
                arr = arr[::-1, ::-1].copy()
            images.append(arr)

        ee_pos = np.asarray(demo["obs/ee_pos"][t], dtype=np.float32)
        ee_ori = np.asarray(demo["obs/ee_ori"][t], dtype=np.float32)
        gripper = np.asarray(demo["obs/gripper_states"][t], dtype=np.float32)
        axis_angle = euler_to_axisangle(ee_ori)
        proprio = np.concatenate([ee_pos, axis_angle, gripper], axis=-1).astype(np.float32)
        action = np.asarray(demo["actions"][t], dtype=np.float32)

    return {
        "ref": ref.to_dict(),
        "rgb": np.stack(images, axis=0),
        "proprio": proprio,
        "action": action,
    }


def raw_rgb_to_tensor(rgb: np.ndarray | torch.Tensor, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Convert raw RGB `[V,H,W,C]` or `[B,V,H,W,C]` to float `[B,V,H,W,C]` in `[0,1]`."""

    tensor = torch.as_tensor(rgb)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"Expected raw RGB [V,H,W,C] or [B,V,H,W,C], got {tuple(tensor.shape)}")
    if not torch.is_floating_point(tensor):
        tensor = tensor.float().div(255.0)
    else:
        tensor = tensor.float()
        if tensor.detach().numel() and float(tensor.detach().max().item()) > 2.0:
            tensor = tensor.div(255.0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.clamp(0.0, 1.0)


def proprio_to_tensor(proprio: np.ndarray | torch.Tensor, *, device: torch.device | str | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(proprio, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class SimVLADeltaObsAdapter:
    """Convert key/current SimVLA batches into ``DeltaObservation``."""

    def make_delta_observation(
        self,
        key_batch: dict[str, torch.Tensor],
        cur_batch: dict[str, torch.Tensor],
        *,
        age: int | float | torch.Tensor | None = None,
        metadata: dict | None = None,
    ) -> DeltaObservation:
        key_images = key_batch.get("raw_rgb", key_batch.get("image_input"))
        cur_images = cur_batch.get("raw_rgb", cur_batch.get("image_input"))
        return DeltaObservation(
            key_images=key_images,
            cur_images=cur_images,
            key_proprio=key_batch.get("proprio"),
            cur_proprio=cur_batch.get("proprio"),
            age=age,
            metadata=copy(metadata) if metadata else None,
        )

    def make_delta_observation_from_raw_samples(
        self,
        key_sample: dict[str, Any],
        cur_sample: dict[str, Any],
        *,
        device: torch.device | str | None = None,
        age: int | float | torch.Tensor | None = None,
        metadata: dict | None = None,
    ) -> DeltaObservation:
        meta = copy(metadata) if metadata else {}
        meta.update(
            {
                "visual_input": "raw_libero_rgb",
                "rgb_value_range": "[0,1]",
                "camera_names": key_sample["ref"].get("camera_names", list(LIBERO_RAW_RGB_CAMERAS)),
                "camera_aliases": key_sample["ref"].get("camera_aliases", list(LIBERO_DCLD_CAMERA_ALIASES)),
                "key_ref": key_sample["ref"],
                "cur_ref": cur_sample["ref"],
            }
        )
        return DeltaObservation(
            key_images=raw_rgb_to_tensor(key_sample["rgb"], device=device),
            cur_images=raw_rgb_to_tensor(cur_sample["rgb"], device=device),
            key_proprio=proprio_to_tensor(key_sample["proprio"], device=device),
            cur_proprio=proprio_to_tensor(cur_sample["proprio"], device=device),
            age=age,
            metadata=meta,
        )

    def clone_cache_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cached = {}
        for key, value in batch.items():
            cached[key] = value.detach().clone() if torch.is_tensor(value) else value
        return cached
