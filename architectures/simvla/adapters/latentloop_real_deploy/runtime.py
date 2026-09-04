"""Instrumented read-only hardware path that cannot issue robot commands."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hardware import legacy_deploy


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray
    frame_number: int
    sensor_timestamp_ms: float
    host_capture_monotonic_s: float


class InstrumentedRealSenseCamera:
    def __init__(self, serial: str, *, width: int, height: int, fps: int):
        import pyrealsense2 as rs

        self.serial_number = str(serial)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = self.pipeline.start(config)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.active_profile = {
            "serial": self.serial_number,
            "width": int(stream.width()),
            "height": int(stream.height()),
            "fps": int(stream.fps()),
            "format": str(stream.format()),
        }
        self.last_read_metadata: dict[str, Any] | None = None

    def read(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        if color is None:
            raise RuntimeError(f"Failed to read RGB frame from {self.serial_number}")
        captured = time.perf_counter()
        image = np.asanyarray(color.get_data()).copy()
        self.last_read_metadata = {
            **self.active_profile,
            "frame_number": int(color.get_frame_number()),
            "sensor_timestamp_ms": float(color.get_timestamp()),
            "host_capture_monotonic_s": captured,
        }
        return image

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


class ReadOnlyDeployEnvironment:
    """Receive-only RTDE and camera client; no RTDEControlInterface is constructed."""

    def __init__(self, cfg):
        import rtde_receive

        self.cfg = cfg
        self.rtde_rec = rtde_receive.RTDEReceiveInterface(cfg.robot_ip)
        self.gripper = legacy_deploy.RobotiqGripper()
        self.gripper.connect(cfg.robot_ip, 63352)
        serials = dict(cfg.camera_serial_cache)
        width = int(os.environ["SEER_CAMERA_WIDTH"])
        height = int(os.environ["SEER_CAMERA_HEIGHT"])
        fps = int(os.environ["SEER_CAMERA_FPS"])
        self.exterior_camera = InstrumentedRealSenseCamera(
            serials["exterior"], width=width, height=height, fps=fps
        )
        self.wrist_camera = InstrumentedRealSenseCamera(
            serials["wrist"], width=width, height=height, fps=fps
        )
        self.observer_camera = None
        self.deploy_camera_serials = serials
        self.camera_serials = {**serials, "observer": None}

    def get_robot_state(self) -> dict[str, np.ndarray]:
        tcp = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=np.float64)
        pose6d = legacy_deploy._ur_tcp_to_pose6d(tcp)
        position = float(self.gripper.get_current_position()) / 255.0
        open_state = 1.0 if position <= self.cfg.gripper_open_threshold else -1.0
        return {
            "pose": legacy_deploy._6d_to_pose(pose6d),
            "pose6d": pose6d.astype(np.float32),
            "tcp_rotvec": tcp[3:].astype(np.float32),
            "gripper_open_state": np.asarray([open_state], dtype=np.float32),
            "gripper_position": np.asarray([position], dtype=np.float32),
            "joint_positions": np.asarray(
                self.rtde_rec.getActualQ(), dtype=np.float32
            ),
        }

    def get_color_images(self) -> list[np.ndarray]:
        return [self.exterior_camera.read(), self.wrist_camera.read()]

    def close(self) -> None:
        self.exterior_camera.close()
        self.wrist_camera.close()
        try:
            self.gripper.disconnect()
        except Exception:
            pass
        try:
            self.rtde_rec.disconnect()
        except Exception:
            pass


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def run_read_only_profile(
    *, controller, env: ReadOnlyDeployEnvironment, output: str | Path, steps: int
) -> dict[str, Any]:
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    controller.attach_session_dir(output_dir)
    instruction = str(controller.contract.runtime["instructions"][0])
    rows = []
    for step in range(int(steps)):
        started = time.perf_counter()
        state_started = time.perf_counter()
        robot_state = env.get_robot_state()
        state_ms = (time.perf_counter() - state_started) * 1000.0
        camera_started = time.perf_counter()
        images = env.get_color_images()
        camera_ms = (time.perf_counter() - camera_started) * 1000.0
        policy_started = time.perf_counter()
        _, _, _, info = controller.forward(
            {
                "robot_state": robot_state,
                "color_image": images,
                "language_instruction": instruction,
            },
            include_info=True,
            timestep=step,
        )
        rows.append(
            {
                **info["record"],
                "state_read_ms": state_ms,
                "camera_pair_read_ms": camera_ms,
                "policy_call_ms": (time.perf_counter() - policy_started) * 1000.0,
                "tick_compute_ms": (time.perf_counter() - started) * 1000.0,
                "robot_command_issued": False,
                "exterior_camera": env.exterior_camera.last_read_metadata,
                "wrist_camera": env.wrist_camera.last_read_metadata,
            }
        )
    controller.write_runtime_summary()
    with (output_dir / "read_only_steps.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fields = ("state_read_ms", "camera_pair_read_ms", "policy_call_ms", "tick_compute_ms")
    summary = {
        "verdict": "READ_ONLY_PROFILE_COMPLETE",
        "robot_command_issued": False,
        "steps": len(rows),
        "latency_ms": {
            field: _distribution([float(row[field]) for row in rows]) for field in fields
        },
        "controller": controller.runtime_summary(),
    }
    (output_dir / "read_only_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
