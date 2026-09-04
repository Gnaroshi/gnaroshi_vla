"""Read-only real-hardware profiling and instrumented camera runtime for deploy v2."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from architectures.seer.adapters.latentloop_real_deploy.hardware import legacy_deploy


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray
    frame_number: int
    sensor_timestamp_ms: float
    timestamp_domain: str
    host_capture_monotonic_s: float


class InstrumentedRealSenseCameraV2:
    """RealSense reader with synchronous and background latest-frame modes."""

    def __init__(
        self,
        serial_number: str,
        *,
        width: int,
        height: int,
        fps: int,
        mode: str,
    ):
        import pyrealsense2 as rs

        if mode not in {"sync", "async_latest"}:
            raise ValueError(f"Unsupported camera mode: {mode}")
        self._rs = rs
        self.serial_number = str(serial_number)
        self.mode = mode
        self._read_lock = threading.Lock()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._capture_error: BaseException | None = None
        self._latest_packet: FramePacket | None = None
        self._last_consumed_frame_number: int | None = None
        self.last_read_metadata: dict[str, Any] | None = None

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = self.pipeline.start(config)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.active_profile = {
            "serial_number": self.serial_number,
            "width": int(stream.width()),
            "height": int(stream.height()),
            "fps": int(stream.fps()),
            "format": str(stream.format()),
            "camera_mode": self.mode,
        }

        self._capture_thread = None
        if self.mode == "async_latest":
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name=f"realsense-{self.serial_number}",
                daemon=True,
            )
            self._capture_thread.start()

    def _capture_once(self) -> FramePacket:
        frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        if color is None:
            raise RuntimeError(f"Failed to read RGB frame from RealSense {self.serial_number}")
        return FramePacket(
            image=np.asanyarray(color.get_data()).copy(),
            frame_number=int(color.get_frame_number()),
            sensor_timestamp_ms=float(color.get_timestamp()),
            timestamp_domain=str(color.get_frame_timestamp_domain()),
            host_capture_monotonic_s=time.perf_counter(),
        )

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                packet = self._capture_once()
                with self._condition:
                    self._latest_packet = packet
                    self._condition.notify_all()
        except BaseException as exc:
            if not self._stop_event.is_set():
                with self._condition:
                    self._capture_error = exc
                    self._condition.notify_all()

    def _latest(self) -> FramePacket:
        deadline = time.perf_counter() + 3.0
        with self._condition:
            while self._latest_packet is None and self._capture_error is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for RealSense {self.serial_number} first frame"
                    )
                self._condition.wait(timeout=remaining)
            if self._capture_error is not None:
                raise RuntimeError(
                    f"RealSense {self.serial_number} capture thread failed"
                ) from self._capture_error
            return self._latest_packet

    def read(self) -> np.ndarray:
        with self._read_lock:
            packet = self._capture_once() if self.mode == "sync" else self._latest()
            consumed_at = time.perf_counter()
            duplicate = packet.frame_number == self._last_consumed_frame_number
            self._last_consumed_frame_number = packet.frame_number
            self.last_read_metadata = {
                **self.active_profile,
                "frame_number": packet.frame_number,
                "sensor_timestamp_ms": packet.sensor_timestamp_ms,
                "timestamp_domain": packet.timestamp_domain,
                "host_capture_monotonic_s": packet.host_capture_monotonic_s,
                "host_consume_monotonic_s": consumed_at,
                "frame_age_ms": (consumed_at - packet.host_capture_monotonic_s) * 1000.0,
                "duplicate": bool(duplicate),
            }
            return packet.image

    def close(self) -> None:
        self._stop_event.set()
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3.5)


class ReadOnlyDeployEnvV2:
    """Real camera and robot-state reader that creates no motion command interface."""

    def __init__(self, cfg, *, camera_mode: str):
        import rtde_receive

        self.cfg = cfg
        self.rtde_rec = rtde_receive.RTDEReceiveInterface(cfg.robot_ip)
        self.gripper = legacy_deploy.RobotiqGripper()
        self.gripper.connect(cfg.robot_ip, 63352)

        camera_names = [cfg.exterior_camera_name, cfg.wrist_camera_name]
        serial_map = legacy_deploy._resolve_realsense_serials(
            camera_names,
            require_explicit_when_extra=True,
            serial_cache=cfg.camera_serial_cache,
        )
        width = int(os.getenv("SEER_CAMERA_WIDTH", "640"))
        height = int(os.getenv("SEER_CAMERA_HEIGHT", "480"))
        fps = int(os.getenv("SEER_CAMERA_FPS", "30"))
        self.exterior_camera = InstrumentedRealSenseCameraV2(
            serial_map[cfg.exterior_camera_name],
            width=width,
            height=height,
            fps=fps,
            mode=camera_mode,
        )
        self.wrist_camera = InstrumentedRealSenseCameraV2(
            serial_map[cfg.wrist_camera_name],
            width=width,
            height=height,
            fps=fps,
            mode=camera_mode,
        )
        self.observer_camera = None
        self.deploy_camera_serials = {
            name: serial_map[name] for name in camera_names
        }
        self.camera_serials = {
            "exterior": serial_map[cfg.exterior_camera_name],
            "wrist": serial_map[cfg.wrist_camera_name],
            "observer": None,
        }

    def get_robot_state(self):
        tcp_pose = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=np.float64)
        pose6d = legacy_deploy._ur_tcp_to_pose6d(tcp_pose)
        pose = legacy_deploy._6d_to_pose(pose6d)
        joints = np.asarray(self.rtde_rec.getActualQ(), dtype=np.float64)
        gripper_position = float(self.gripper.get_current_position()) / 255.0
        gripper_open_state = (
            1.0 if gripper_position <= self.cfg.gripper_open_threshold else -1.0
        )
        return {
            "pose": pose,
            "pose6d": pose6d.astype(np.float32),
            "gripper_open_state": np.array([gripper_open_state], dtype=np.float32),
            "gripper_position": np.array([gripper_position], dtype=np.float32),
            "joint_positions": joints.astype(np.float32),
        }

    def get_color_images(self):
        images = [self.exterior_camera.read(), self.wrist_camera.read()]
        return images

    def camera_metadata(self) -> dict[str, Any]:
        pair_returned_at = time.perf_counter()
        primary = dict(self.exterior_camera.last_read_metadata or {})
        wrist = dict(self.wrist_camera.last_read_metadata or {})
        for metadata in (primary, wrist):
            metadata["frame_age_at_pair_return_ms"] = (
                pair_returned_at - float(metadata["host_capture_monotonic_s"])
            ) * 1000.0
        return {"primary": primary, "wrist": wrist}

    def close(self) -> None:
        for camera in (self.exterior_camera, self.wrist_camera):
            camera.close()
        try:
            self.gripper.disconnect()
        except Exception:
            pass
        try:
            self.rtde_rec.disconnect()
        except Exception:
            pass


def _distribution(values) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def run_read_only_profile(
    *,
    controller,
    env: ReadOnlyDeployEnvV2,
    instruction: str,
    steps: int,
    warmup_steps: int,
    control_freq: float,
    output_dir: str,
) -> dict[str, Any]:
    """Profile the complete read path without constructing any robot command client."""

    if steps < 2:
        raise ValueError(f"Read-only profile requires at least two steps, got {steps}")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=False)

    for _ in range(warmup_steps):
        observation = {
            "robot_state": env.get_robot_state(),
            "color_image": env.get_color_images(),
            "language_instruction": instruction,
        }
        controller.forward(
            observation, include_info=True, timestep=0, record_step=False
        )
    controller.reset(write_previous=False)
    controller.attach_session_dir(str(output_path))

    target_period_s = 1.0 / float(control_freq)
    records = []
    profile_started = time.perf_counter()
    for timestep in range(steps):
        tick_started = time.perf_counter()

        state_started = time.perf_counter()
        robot_state = env.get_robot_state()
        state_ms = (time.perf_counter() - state_started) * 1000.0

        camera_started = time.perf_counter()
        color_images = env.get_color_images()
        camera_ms = (time.perf_counter() - camera_started) * 1000.0
        camera_metadata = env.camera_metadata()

        policy_started = time.perf_counter()
        _, _, _, policy_record = controller.forward(
            {
                "robot_state": robot_state,
                "color_image": color_images,
                "language_instruction": instruction,
            },
            include_info=True,
            timestep=timestep,
            record_step=False,
        )
        policy_call_ms = (time.perf_counter() - policy_started) * 1000.0

        no_op_boundary = time.perf_counter()
        controller.record_control_command(no_op_boundary)
        compute_ms = (no_op_boundary - tick_started) * 1000.0
        sleep_s = max(0.0, target_period_s - (no_op_boundary - tick_started))
        if sleep_s:
            time.sleep(sleep_s)
        tick_finished = time.perf_counter()

        primary = camera_metadata["primary"]
        wrist = camera_metadata["wrist"]
        records.append(
            {
                **policy_record,
                "state_read_ms": state_ms,
                "camera_pair_read_ms": camera_ms,
                "policy_call_ms": policy_call_ms,
                "tick_compute_ms": compute_ms,
                "sleep_ms": sleep_s * 1000.0,
                "tick_total_ms": (tick_finished - tick_started) * 1000.0,
                "robot_command_issued": False,
                "primary_camera": primary,
                "wrist_camera": wrist,
                "camera_host_capture_skew_ms": abs(
                    float(primary["host_capture_monotonic_s"])
                    - float(wrist["host_capture_monotonic_s"])
                )
                * 1000.0,
            }
        )

    profile_elapsed_s = time.perf_counter() - profile_started
    controller.write_runtime_summary()
    jsonl_path = output_path / "read_only_profile_steps.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    primary_frames = [record["primary_camera"]["frame_number"] for record in records]
    wrist_frames = [record["wrist_camera"]["frame_number"] for record in records]
    target_period_ms = 1000.0 / float(control_freq)

    def deadline_summary(selected_records):
        selected_records = list(selected_records)
        if not selected_records:
            return {
                "count": 0,
                "policy_budget_miss_count": 0,
                "policy_budget_miss_rate": 0.0,
                "compute_budget_miss_count": 0,
                "compute_budget_miss_rate": 0.0,
            }
        policy_misses = np.asarray(
            [record["policy_ms"] > target_period_ms for record in selected_records],
            dtype=np.bool_,
        )
        compute_misses = np.asarray(
            [record["tick_compute_ms"] > target_period_ms for record in selected_records],
            dtype=np.bool_,
        )
        return {
            "count": len(selected_records),
            "policy_budget_miss_count": int(policy_misses.sum()),
            "policy_budget_miss_rate": float(policy_misses.mean()),
            "compute_budget_miss_count": int(compute_misses.sum()),
            "compute_budget_miss_rate": float(compute_misses.mean()),
        }

    controller_summary = controller.runtime_summary()
    summary = {
        "schema_version": 1,
        "execution_mode": "read_only_profile",
        "robot_command_issued": False,
        "profile_steps": int(steps),
        "warmup_steps": int(warmup_steps),
        "control_freq": float(control_freq),
        "control_period_ms": target_period_ms,
        "controller": controller_summary,
        "latency_ms": {
            key: _distribution(record[key] for record in records)
            for key in (
                "state_read_ms",
                "camera_pair_read_ms",
                "policy_call_ms",
                "tick_compute_ms",
                "sleep_ms",
                "tick_total_ms",
            )
        },
        "camera": {
            "primary_active_profile": env.exterior_camera.active_profile,
            "wrist_active_profile": env.wrist_camera.active_profile,
            "primary_duplicate_rate": float(
                np.mean([record["primary_camera"]["duplicate"] for record in records])
            ),
            "wrist_duplicate_rate": float(
                np.mean([record["wrist_camera"]["duplicate"] for record in records])
            ),
            "primary_unique_frames": len(set(primary_frames)),
            "wrist_unique_frames": len(set(wrist_frames)),
            "primary_consumed_unique_hz": float(
                len(set(primary_frames)) / profile_elapsed_s
            ),
            "wrist_consumed_unique_hz": float(
                len(set(wrist_frames)) / profile_elapsed_s
            ),
            "host_capture_skew_ms": _distribution(
                record["camera_host_capture_skew_ms"] for record in records
            ),
            "primary_frame_age_at_pair_return_ms": _distribution(
                record["primary_camera"]["frame_age_at_pair_return_ms"]
                for record in records
            ),
            "wrist_frame_age_at_pair_return_ms": _distribution(
                record["wrist_camera"]["frame_age_at_pair_return_ms"]
                for record in records
            ),
        },
        "deadline": {
            "budget_ms": target_period_ms,
            "all": deadline_summary(records),
            "full": deadline_summary(
                record for record in records if record["mode"] == "full"
            ),
            "latentloop": deadline_summary(
                record for record in records if record["mode"] == "latentloop"
            ),
            "hold_action": deadline_summary(
                record for record in records if record["mode"] == "hold_action"
            ),
            "hold_latent": deadline_summary(
                record for record in records if record["mode"] == "hold_latent"
            ),
            "achieved_control_hz": controller_summary["achieved_control_hz"],
            "target_achievement_ratio": (
                controller_summary["achieved_control_hz"] / float(control_freq)
            ),
            "legacy_boundary_miss_rate_without_tolerance": controller_summary[
                "strict_deadline_miss_rate"
            ],
        },
        "profile_elapsed_s": float(profile_elapsed_s),
    }
    with (output_path / "read_only_profile_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
