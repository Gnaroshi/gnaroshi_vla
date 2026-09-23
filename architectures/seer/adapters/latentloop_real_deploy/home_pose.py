"""Validate shell-owned task home settings before creating a robot interface."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


def configure_home_pose(cfg, artifact_manifest, environ=None):
    env = os.environ if environ is None else environ
    required = ("SEER_HOME_POSE", "SEER_HOME_TASK", "SEER_HOME_POSE_SOURCE")
    if any(not env.get(key, "").strip() for key in required):
        raise ValueError(
            "Task home settings are required. Use deploy_ll_gui_unified.sh; "
            "legacy launchers do not configure task-specific home poses."
        )
    manifest = json.loads(Path(artifact_manifest).read_text(encoding="utf-8"))
    task = env["SEER_HOME_TASK"]
    if manifest.get("task") != task:
        raise ValueError(
            f"Home task {task!r} differs from checkpoint task {manifest.get('task')!r}"
        )
    pose = json.loads(env["SEER_HOME_POSE"])
    if not isinstance(pose, list) or len(pose) != 7:
        raise ValueError("Home pose must contain six joint radians and one gripper target")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in pose):
        raise ValueError("Home pose values must be finite numbers")
    if not 0 <= pose[6] <= 1:
        raise ValueError("Home gripper target must be normalized to [0, 1]")
    cfg.home_pose = [float(value) for value in pose]
    cfg.home_configuration = {
        "task": task,
        "joint_positions_rad": cfg.home_pose[:6],
        "gripper_target_normalized": cfg.home_pose[6],
        "gripper_convention": "0=open, 1=closed",
        "source": env["SEER_HOME_POSE_SOURCE"],
        "home_move_duration_s": float(cfg.home_move_duration),
        "home_move_fps": float(cfg.home_move_fps),
    }
    return cfg.home_configuration
