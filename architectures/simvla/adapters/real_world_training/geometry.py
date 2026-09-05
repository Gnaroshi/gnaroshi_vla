"""UR5e TCP and SimVLA action/state conversion contracts."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


PROPRIO_DIM = 8
ACTION_DIM = 7
ACTION_HORIZON = 10
EXECUTION_HORIZON = 5


@dataclass(frozen=True)
class RealActionScales:
    """Physical magnitude represented by one normalized pose-action unit."""

    translation_m: float = 0.02
    rotation_rad: float = 0.05
    clip_abs: float = 1.0
    gripper_max_opening_m: float = 0.04

    def validate(self) -> None:
        values = (
            self.translation_m,
            self.rotation_rad,
            self.clip_abs,
            self.gripper_max_opening_m,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("all real-world action/state scales must be finite and positive")


def _pose6d(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("TCP pose must be a finite [xyz, rotation-vector] six-vector")
    return pose


def pose6d_to_matrix(value: np.ndarray) -> np.ndarray:
    pose = _pose6d(value)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def transition_to_normalized_action(
    current_pose6d: np.ndarray,
    next_pose6d: np.ndarray,
    command_t_gripper_control: float,
    scales: RealActionScales = RealActionScales(),
    *,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert consecutive absolute TCP poses to the deployed local delta action.

    The relative transform is ``inv(T_current) @ T_next``. Its XYZ translation
    and XYZ Euler rotation are exactly the quantities consumed by the copied
    deployment runtime before it composes ``T_current @ T_delta``.
    """

    scales.validate()
    current = pose6d_to_matrix(current_pose6d)
    following = pose6d_to_matrix(next_pose6d)
    relative = np.linalg.inv(current) @ following
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        relative_euler = Rotation.from_matrix(relative[:3, :3]).as_euler("xyz")
    gripper_control = float(command_t_gripper_control)
    if not np.isfinite(gripper_control) or not 0.0 <= gripper_control <= 1.0:
        raise ValueError("command_t gripper target must be normalized to [0,1]")
    raw = np.concatenate(
        (
            relative[:3, 3] / scales.translation_m,
            relative_euler / scales.rotation_rad,
            np.asarray([1.0 - 2.0 * gripper_control]),
        )
    ).astype(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("derived normalized action is not finite")
    clipped = np.clip(raw, -scales.clip_abs, scales.clip_abs).astype(np.float32)
    return (clipped if clip else raw.copy()), raw


def encode_opposed_finger_state(
    pose6d: np.ndarray,
    gripper_position: float,
    scales: RealActionScales = RealActionScales(),
) -> np.ndarray:
    """Encode UR TCP pose and Robotiq position using LIBERO's opposed fingers."""

    scales.validate()
    pose = _pose6d(pose6d)
    position = float(gripper_position)
    if not np.isfinite(position):
        raise ValueError("gripper_position must be finite")
    open_fraction = 1.0 - float(np.clip(position, 0.0, 1.0))
    opening = open_fraction * scales.gripper_max_opening_m
    state = np.concatenate((pose, [opening, -opening])).astype(np.float32)
    if state.shape != (PROPRIO_DIM,):
        raise AssertionError("real SimVLA proprioception must be eight-dimensional")
    return state


def apply_normalized_action(
    current_pose6d: np.ndarray,
    normalized_action: np.ndarray,
    scales: RealActionScales = RealActionScales(),
) -> np.ndarray:
    """Apply the pose part of a normalized action for conversion round-trip tests."""

    scales.validate()
    action = np.asarray(normalized_action, dtype=np.float64).reshape(-1)
    if action.shape != (ACTION_DIM,):
        raise ValueError("normalized_action must have seven elements")
    delta = np.eye(4, dtype=np.float64)
    delta[:3, 3] = action[:3] * scales.translation_m
    delta[:3, :3] = Rotation.from_euler(
        "xyz", action[3:6] * scales.rotation_rad
    ).as_matrix()
    target = pose6d_to_matrix(current_pose6d) @ delta
    return np.concatenate(
        (target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_rotvec())
    )
