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
        try:
            profile = self.pipeline.start(config)
            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.active_profile = {
                "serial": self.serial_number,
                "width": int(stream.width()),
                "height": int(stream.height()),
                "fps": int(stream.fps()),
                "format": str(stream.format()),
            }
        except BaseException:
            self.close()
            raise
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
        self.rtde_rec = self.gripper = None
        self.exterior_camera = self.wrist_camera = None
        try:
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
        except BaseException:
            self.close()
            raise
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
        for name, operation in (("exterior_camera", "close"), ("wrist_camera", "close"),
                                ("gripper", "disconnect"), ("rtde_rec", "disconnect")):
            resource = getattr(self, name, None)
            if resource is not None:
                try:
                    getattr(resource, operation)()
                except Exception:
                    pass
                setattr(self, name, None)


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


def _validate_robot_state(
    state: dict[str, np.ndarray], *, workspace_min: np.ndarray | None,
    workspace_max: np.ndarray | None,
) -> None:
    expected = {
        "pose6d": (6,),
        "tcp_rotvec": (3,),
        "gripper_open_state": (1,),
        "gripper_position": (1,),
        "joint_positions": (6,),
    }
    for key, shape in expected.items():
        value = np.asarray(state.get(key))
        if value.shape != shape or not np.isfinite(value).all():
            raise RuntimeError(f"read-only robot state {key} must be finite {shape}")
    position = np.asarray(state["pose6d"], dtype=np.float64)[:3]
    if (workspace_min is None) != (workspace_max is None):
        raise ValueError("workspace bounds must be supplied together")
    if workspace_min is not None and (
        np.any(position < workspace_min) or np.any(position > workspace_max)
    ):
        raise RuntimeError(
            "read-only actual TCP is outside the reviewed workspace: "
            f"xyz={position.tolist()}"
        )
    gripper_position = float(np.asarray(state["gripper_position"])[0])
    if not 0.0 <= gripper_position <= 1.0:
        raise RuntimeError("read-only gripper position must be normalized to [0,1]")


def _validate_camera_sample(
    *,
    role: str,
    image: np.ndarray,
    metadata: dict[str, Any] | None,
    expected_serial: str,
    expected_width: int,
    expected_height: int,
    expected_fps: int,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    value = np.asarray(image)
    if value.shape != (expected_height, expected_width, 3) or value.dtype != np.uint8:
        raise RuntimeError(
            f"{role} camera must produce uint8 RGB "
            f"{(expected_height, expected_width, 3)}, got {value.shape}/{value.dtype}"
        )
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{role} camera did not expose frame metadata")
    expected = {
        "serial": expected_serial,
        "width": expected_width,
        "height": expected_height,
        "fps": expected_fps,
    }
    mismatches = {
        key: {"observed": metadata.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if "rgb8" not in str(metadata.get("format", "")).lower():
        mismatches["format"] = {
            "observed": metadata.get("format"),
            "expected": "rgb8",
        }
    if mismatches:
        raise RuntimeError(f"{role} camera profile mismatch: {mismatches}")
    frame_number = int(metadata["frame_number"])
    sensor_timestamp = float(metadata["sensor_timestamp_ms"])
    host_timestamp = float(metadata["host_capture_monotonic_s"])
    if not all(np.isfinite(value) for value in (sensor_timestamp, host_timestamp)):
        raise RuntimeError(f"{role} camera timestamps must be finite")
    if previous is not None:
        if frame_number <= int(previous["frame_number"]):
            raise RuntimeError(f"{role} camera frame number did not increase")
        if sensor_timestamp <= float(previous["sensor_timestamp_ms"]):
            raise RuntimeError(f"{role} camera sensor timestamp did not increase")
        if host_timestamp <= float(previous["host_capture_monotonic_s"]):
            raise RuntimeError(f"{role} camera host timestamp did not increase")
    return dict(metadata)


def _expected_policy_counters(method: str, steps: int) -> dict[str, int]:
    queries = (int(steps) + 4) // 5
    expected = {
        "num_policy_queries": queries,
        "num_action_queue_steps": int(steps),
    }
    if method == "baseline":
        expected.update(
            {
                "num_full_vlm_calls": queries,
                "num_condition_updater_calls": 0,
                "num_action_transformer_calls": queries * 10,
            }
        )
    elif method == "condition_loop":
        expected.update(
            {
                "num_full_vlm_calls": (queries + 1) // 2,
                "num_condition_updater_calls": queries // 2,
                "num_action_transformer_calls": queries * 10,
            }
        )
    elif method == "latentloop":
        expected.update(
            {
                "num_full_vlm_calls": (queries + 1) // 2,
                "num_condition_updater_calls": queries // 2,
                "num_condition_change_code_queries": queries // 2,
                "num_action_transformer_calls": queries * 3,
            }
        )
    return expected


def run_read_only_profile(
    *, controller, env: ReadOnlyDeployEnvironment, output: str | Path, steps: int,
    target_hz: float | None = None, camera_fps: int | None = None,
) -> dict[str, Any]:
    target_hz = float(controller.contract.runtime["control_frequency_hz"]
                      if target_hz is None else target_hz)
    if not np.isfinite(target_hz) or target_hz <= 0:
        raise ValueError("profile target Hz must be positive and finite")
    if steps < 11:
        raise ValueError("read-only profile needs at least 11 steps")
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    controller.attach_session_dir(output_dir)
    instruction = str(controller.contract.runtime["instructions"][0])
    cameras = controller.contract.hardware["cameras"]
    workspace = controller.contract.hardware["robot"]["workspace_m"]
    review = controller.contract.payload["safety_review"]
    workspace_reviewed = bool(review["workspace_bounds_verified"]) and not str(
        controller.contract.hardware["robot"]["workspace_source"]
    ).startswith("replace-with-")
    workspace_min = np.asarray(workspace["min"], dtype=np.float64) if workspace_reviewed else None
    workspace_max = np.asarray(workspace["max"], dtype=np.float64) if workspace_reviewed else None
    expected_width = int(cameras["width"])
    expected_height = int(cameras["height"])
    expected_fps = int(cameras["fps"] if camera_fps is None else camera_fps)
    if expected_fps <= 0:
        raise ValueError("profile camera FPS must be positive")
    previous_camera: dict[str, dict[str, Any] | None] = {
        "exterior": None,
        "wrist": None,
    }
    warmup_steps = int(controller.contract.runtime["warmup_steps"])

    for warmup in range(warmup_steps):
        robot_state = env.get_robot_state()
        _validate_robot_state(
            robot_state, workspace_min=workspace_min, workspace_max=workspace_max
        )
        images = env.get_color_images()
        for role, image, camera in (
            ("exterior", images[0], env.exterior_camera),
            ("wrist", images[1], env.wrist_camera),
        ):
            previous_camera[role] = _validate_camera_sample(
                role=role,
                image=image,
                metadata=camera.last_read_metadata,
                expected_serial=str(cameras[role]["serial"]),
                expected_width=expected_width,
                expected_height=expected_height,
                expected_fps=expected_fps,
                previous=previous_camera[role],
            )
        controller.forward(
            {
                "robot_state": robot_state,
                "color_image": images,
                "language_instruction": instruction,
            },
            include_info=True,
            timestep=warmup,
            record_step=False,
        )
    controller.reset()

    rows = []
    tick_starts: list[float] = []
    query_starts: list[float] = []
    preview_images = None
    control_period_s = 1.0 / target_hz
    for step in range(int(steps)):
        started = time.perf_counter()
        tick_starts.append(started)
        state_started = time.perf_counter()
        robot_state = env.get_robot_state()
        _validate_robot_state(
            robot_state, workspace_min=workspace_min, workspace_max=workspace_max
        )
        state_ms = (time.perf_counter() - state_started) * 1000.0
        camera_started = time.perf_counter()
        images = env.get_color_images()
        camera_ms = (time.perf_counter() - camera_started) * 1000.0
        camera_metadata: dict[str, dict[str, Any]] = {}
        for role, image, camera in (
            ("exterior", images[0], env.exterior_camera),
            ("wrist", images[1], env.wrist_camera),
        ):
            camera_metadata[role] = _validate_camera_sample(
                role=role,
                image=image,
                metadata=camera.last_read_metadata,
                expected_serial=str(cameras[role]["serial"]),
                expected_width=expected_width,
                expected_height=expected_height,
                expected_fps=expected_fps,
                previous=previous_camera[role],
            )
            previous_camera[role] = camera_metadata[role]
        policy_started = time.perf_counter()
        is_query = controller.needs_policy_query
        if is_query:
            query_starts.append(policy_started)
        _, _, _, info = controller.forward(
            {
                "robot_state": robot_state,
                "color_image": images,
                "language_instruction": instruction,
            },
            include_info=True,
            timestep=step,
        )
        policy_ms = (time.perf_counter() - policy_started) * 1000.0
        compute_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                **info["record"],
                "state_read_ms": state_ms,
                "camera_pair_read_ms": camera_ms,
                "policy_call_ms": policy_ms,
                "tick_compute_ms": compute_ms,
                "is_policy_query": is_query,
                "nominal_deadline_missed": compute_ms > control_period_s * 1000.0,
                "robot_command_issued": False,
                "actual_tcp_rotvec": np.concatenate(
                    (
                        np.asarray(robot_state["pose6d"], dtype=np.float64)[:3],
                        np.asarray(robot_state["tcp_rotvec"], dtype=np.float64),
                    )
                ).tolist(),
                "camera_host_capture_skew_ms": abs(
                    camera_metadata["exterior"]["host_capture_monotonic_s"]
                    - camera_metadata["wrist"]["host_capture_monotonic_s"]
                )
                * 1000.0,
                "exterior_camera": camera_metadata["exterior"],
                "wrist_camera": camera_metadata["wrist"],
            }
        )
        if preview_images is None:
            preview_images = images
        sleep_left = control_period_s - (time.perf_counter() - started)
        if sleep_left > 0:
            time.sleep(sleep_left)
    controller.write_runtime_summary()
    from PIL import Image

    for role, image in zip(("exterior", "wrist"), preview_images):
        Image.fromarray(image).save(output_dir / f"first_{role}.png")
    with (output_dir / "read_only_steps.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fields = (
        "state_read_ms",
        "camera_pair_read_ms",
        "policy_call_ms",
        "tick_compute_ms",
        "camera_host_capture_skew_ms",
    )
    counters = dict(controller.policy.metrics.counters)
    expected_counters = _expected_policy_counters(
        controller.deployment_method, len(rows)
    )
    counter_mismatches = {
        key: {"observed": int(counters.get(key, 0)), "expected": expected}
        for key, expected in expected_counters.items()
        if int(counters.get(key, 0)) != expected
    }
    if counter_mismatches:
        raise RuntimeError(f"read-only policy schedule drift: {counter_mismatches}")
    tick_intervals_ms = np.diff(np.asarray(tick_starts, dtype=np.float64)) * 1000.0
    deadline_misses = sum(bool(row["nominal_deadline_missed"]) for row in rows)
    summary = {
        "verdict": "READ_ONLY_PROFILE_PASS",
        "robot_command_issued": False,
        "sensor_contract_validated": True,
        "workspace_validated": workspace_reviewed,
        "live_authorization_granted": False,
        "deployment_target_hz": controller.contract.runtime["control_frequency_hz"],
        "profile_target_hz": target_hz,
        "profile_camera_fps": expected_fps,
        "read_only_tick_hz": (len(tick_starts) - 1) / (tick_starts[-1] - tick_starts[0]),
        "policy_query_start_hz": (len(query_starts) - 1) / (query_starts[-1] - query_starts[0])
        if len(query_starts) > 1 else None,
        "actual_robot_command_hz": None,
        "observed_tcp_xyz_m": {
            "min": np.min([row["actual_tcp_rotvec"][:3] for row in rows], axis=0).tolist(),
            "max": np.max([row["actual_tcp_rotvec"][:3] for row in rows], axis=0).tolist(),
            "not_a_safe_workspace_estimate": True,
        },
        "policy_schedule_validated": True,
        "nominal_timing_is_measurement_not_authorization": True,
        "warmup_steps": warmup_steps,
        "steps": len(rows),
        "expected_policy_counters": expected_counters,
        "observed_policy_counters": counters,
        "nominal_control_period_ms": control_period_s * 1000.0,
        "nominal_deadline_misses": deadline_misses,
        "nominal_deadline_miss_fraction": deadline_misses / max(len(rows), 1),
        "tick_interval_ms": _distribution(tick_intervals_ms.tolist()),
        "latency_ms": {
            field: _distribution([float(row[field]) for row in rows]) for field in fields
        },
        "controller": controller.runtime_summary(),
    }
    (output_dir / "read_only_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
