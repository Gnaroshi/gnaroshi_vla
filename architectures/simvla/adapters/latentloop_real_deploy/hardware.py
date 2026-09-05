"""Adapter around the copied, known-working 3DFlow UR5e hardware source."""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import DeploymentContract


ROOT = Path(__file__).resolve().parents[4]
LEGACY_ROOT = (
    ROOT / "architectures" / "simvla" / "third_party" / "3dflow_real_deploy"
)


def _install_controller_stub() -> None:
    package = sys.modules.get("real_controller")
    if package is None:
        package = types.ModuleType("real_controller")
        package.__path__ = [str(LEGACY_ROOT / "real_controller")]
        sys.modules["real_controller"] = package
    elif str(LEGACY_ROOT / "real_controller") not in list(
        getattr(package, "__path__", [])
    ):
        raise RuntimeError("An unrelated real_controller package is already imported")
    existing = sys.modules.get("real_controller.controller")
    if existing is not None and not getattr(existing, "_simvla_disabled_stub", False):
        raise RuntimeError("An unrelated real_controller.controller is already imported")
    module = types.ModuleType("real_controller.controller")

    class RemovedSeerController:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "The copied Seer controller is disabled; use SimVLARealController"
            )

    module.SeerController = RemovedSeerController
    module._simvla_disabled_stub = True
    sys.modules["real_controller.controller"] = module


def load_legacy_deploy():
    name = "_simvla_copied_3dflow_deploy"
    if name in sys.modules:
        return sys.modules[name]
    source = LEGACY_ROOT / "deploy.py"
    if not source.is_file():
        raise FileNotFoundError(f"Copied hardware deployment source is missing: {source}")
    _install_controller_stub()
    legacy_path = str(LEGACY_ROOT)
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load copied deployment source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    existing = sys.modules.get("deploy")
    if existing is not None and existing is not module:
        raise RuntimeError("An unrelated top-level deploy module is already imported")
    sys.modules["deploy"] = module
    return module


def load_legacy_gui():
    name = "_simvla_copied_3dflow_deploy_gui"
    if name in sys.modules:
        return sys.modules[name]
    load_legacy_deploy()
    source = LEGACY_ROOT / "deploy_gui.py"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load copied deployment GUI: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy_deploy = load_legacy_deploy()

# Returned only when an operator outcome/abort arrives while a policy inference
# is in flight.  The caller treats this as a command-free control tick.
POLICY_STEP_CANCELLED = object()


def tcp_tracking_error(
    actual_tcp_rotvec: np.ndarray, commanded_tcp_rotvec: np.ndarray
) -> dict[str, float]:
    actual = np.asarray(actual_tcp_rotvec, dtype=np.float64).reshape(-1)
    commanded = np.asarray(commanded_tcp_rotvec, dtype=np.float64).reshape(-1)
    if actual.shape != (6,) or commanded.shape != (6,):
        raise ValueError("TCP tracking comparison requires two six-vectors")
    if not np.isfinite(actual).all() or not np.isfinite(commanded).all():
        raise ValueError("TCP tracking comparison requires finite poses")
    relative = Rotation.from_rotvec(actual[3:]).inv() * Rotation.from_rotvec(
        commanded[3:]
    )
    return {
        "translation_m": float(np.linalg.norm(actual[:3] - commanded[:3])),
        "rotation_rad": float(np.linalg.norm(relative.as_rotvec())),
    }


def rebase_incremental_target_on_actual_tcp(
    previous_requested_pose6d: np.ndarray,
    requested_pose6d: np.ndarray,
    actual_tcp_rotvec: np.ndarray,
) -> np.ndarray:
    """Apply the upstream local delta to current TCP feedback.

    The preserved GUI accumulates local model actions into an absolute requested
    pose. Recover that one-step local delta, then compose it on the actual TCP so
    deployment matches labels defined by ``inv(T_current) @ T_next``.
    """

    previous = np.asarray(previous_requested_pose6d, dtype=np.float64).reshape(-1)
    requested = np.asarray(requested_pose6d, dtype=np.float64).reshape(-1)
    actual_tcp = np.asarray(actual_tcp_rotvec, dtype=np.float64).reshape(-1)
    if previous.shape != (6,) or requested.shape != (6,) or actual_tcp.shape != (6,):
        raise ValueError("requested and actual TCP poses must be six-vectors")
    if not all(np.isfinite(value).all() for value in (previous, requested, actual_tcp)):
        raise ValueError("requested and actual TCP poses must be finite")
    previous_matrix = legacy_deploy._6d_to_pose(previous)
    requested_matrix = legacy_deploy._6d_to_pose(requested)
    local_delta = np.linalg.inv(previous_matrix) @ requested_matrix
    actual_pose6d = legacy_deploy._ur_tcp_to_pose6d(actual_tcp)
    actual_matrix = legacy_deploy._6d_to_pose(actual_pose6d)
    return np.asarray(
        legacy_deploy.pose_to_6d(actual_matrix @ local_delta), dtype=np.float64
    )


def _set_compatibility_environment(contract: DeploymentContract) -> None:
    """Set private aliases consumed only by the preserved 3DFlow snapshot."""

    cameras = contract.hardware["cameras"]
    runtime = contract.runtime
    aliases = {
        "SEER_EXTERIOR_CAMERA_SERIAL": str(cameras["exterior"]["serial"]),
        "SEER_WRIST_CAMERA_SERIAL": str(cameras["wrist"]["serial"]),
        "SEER_CAMERA_WIDTH": str(cameras["width"]),
        "SEER_CAMERA_HEIGHT": str(cameras["height"]),
        "SEER_CAMERA_FPS": str(cameras["fps"]),
        "SEER_LANGUAGE_INSTRUCTIONS": "||".join(runtime["instructions"]),
    }
    observer = cameras.get("observer")
    if isinstance(observer, dict) and observer.get("serial"):
        aliases["SEER_OBSERVER_CAMERA_SERIAL"] = str(observer["serial"])
    for key, value in aliases.items():
        os.environ[key] = value


def build_deploy_config(contract: DeploymentContract):
    _set_compatibility_environment(contract)
    robot = contract.hardware["robot"]
    cameras = contract.hardware["cameras"]
    runtime = contract.runtime
    action = contract.action
    control = robot["control"]
    gripper = robot["gripper"]
    cfg = legacy_deploy.DeployConfig()
    cfg.robot_ip = str(robot["ip"])
    cfg.language_instructions = list(runtime["instructions"])
    cfg.language_instruction = cfg.language_instructions[0]
    cfg.home_pose = [float(value) for value in robot["home_pose"]]
    cfg.home_move_duration = float(control["home_move_duration_s"])
    cfg.home_move_fps = int(control["home_move_hz"])
    cfg.control_freq = float(runtime["control_frequency_hz"])
    cfg.max_rel_pos = float(action["translation_scale_m"])
    cfg.max_rel_orn = float(action["rotation_scale_rad"])
    cfg.exterior_camera_name = "exterior"
    cfg.wrist_camera_name = "wrist"
    cfg.observer_camera_name = "observer"
    cfg.gripper_open_threshold = float(gripper["open_position_threshold"])
    cfg.gripper_speed = int(gripper["speed"])
    cfg.gripper_force = int(gripper["force"])
    cfg.gripper_min_delta = int(gripper["min_command_delta"])
    cfg.gripper_min_period_s = float(gripper["min_command_period_s"])
    cfg.gripper_force_open_steps = int(gripper["startup_force_open_steps"])
    cfg.gripper_status_lock_steps = int(gripper["status_lock_steps"])
    cfg.arm_acceleration = float(control["acceleration"])
    cfg.arm_velocity = float(control["velocity"])
    cfg.servoj_time = float(control["servoj_time_s"])
    cfg.servoj_lookahead = float(control["servoj_lookahead_s"])
    cfg.servoj_gain = float(control["servoj_gain"])
    cfg.warmup_steps = int(runtime["warmup_steps"])
    cfg.num_rollouts = int(runtime["num_rollouts_per_instruction"])
    def runtime_path(key: str) -> Path:
        path = Path(runtime[key]).expanduser()
        if not path.is_absolute():
            path = contract.path.parent / path
        return path.resolve()

    cfg.results_dir = str(runtime_path("results_directory"))
    cfg.enable_rollout_media = bool(runtime["save_rollout_media"])
    cfg.enable_observer_media = bool(
        cfg.enable_rollout_media
        and isinstance(cameras.get("observer"), dict)
        and cameras["observer"].get("serial")
    )
    cfg.camera_serials_file = str(runtime_path("camera_serials_file"))
    cfg.camera_serial_cache = {
        "exterior": str(cameras["exterior"]["serial"]),
        "wrist": str(cameras["wrist"]["serial"]),
    }
    return cfg


class SafeUR5eDeployEnv(legacy_deploy.UR5eDeployEnv):
    """Copied live environment plus an explicit Cartesian workspace guard."""

    def __init__(
        self,
        cfg,
        *,
        workspace_min: list[float],
        workspace_max: list[float],
        tracking_error_guard: dict[str, Any] | None = None,
    ):
        self._workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self._workspace_max = np.asarray(workspace_max, dtype=np.float64)
        tracking = dict(tracking_error_guard or {})
        self._tracking_guard_enabled = bool(tracking.get("enabled", False))
        self._max_tracking_translation_m = float(
            tracking.get("max_translation_error_m") or float("inf")
        )
        self._max_tracking_rotation_rad = float(
            tracking.get("max_rotation_error_rad") or float("inf")
        )
        self._last_commanded_tcp: np.ndarray | None = None
        self._upstream_reference_target: np.ndarray | None = None
        self._command_lock = threading.RLock()
        self._motion_abort = threading.Event()
        self._policy_commands_armed = False
        self._cancel_next_disarmed_policy_step = False
        self.last_tracking_error: dict[str, float] | None = None
        self.last_emergency_stop_report: dict[str, Any] | None = None
        self._closed = False
        self.cfg = cfg
        for name in ("rtde_ctrl", "rtde_rec", "gripper", "exterior_camera",
                     "wrist_camera", "observer_camera"):
            setattr(self, name, None)
        try:
            super().__init__(cfg)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self.emergency_stop()
        try:
            super().close()
        finally:
            gripper = getattr(self, "gripper", None)
            if gripper is not None:
                try:
                    gripper.disconnect()
                except Exception:
                    pass
            self._closed = True

    def arm_policy_commands(self) -> None:
        """Allow policy-driven Cartesian steps for one active GUI rollout."""

        with self._command_lock:
            self._motion_abort.clear()
            self._cancel_next_disarmed_policy_step = False
            self._policy_commands_armed = True

    def disarm_policy_commands(self) -> None:
        """Reject late policy outputs without affecting an explicit home move."""

        with self._command_lock:
            self._policy_commands_armed = False

    def cancel_pending_policy_step(self) -> None:
        """Turn one in-flight inference result into a command-free no-op."""

        with self._command_lock:
            self._policy_commands_armed = False
            self._cancel_next_disarmed_policy_step = True

    def move_to_home(self):
        if len(self.cfg.home_pose) < 6:
            raise ValueError("home_pose must contain at least 6 joint values")

        print("[simvla-deploy] moving robot to reviewed home pose")
        current_joints = np.asarray(self.rtde_rec.getActualQ(), dtype=np.float64)
        target_joints = np.asarray(self.cfg.home_pose[:6], dtype=np.float64)
        gripper_target = (
            float(np.clip(self.cfg.home_pose[6], 0.0, 1.0))
            if len(self.cfg.home_pose) > 6
            else float(self.gripper.get_current_position()) / 255.0
        )
        steps = max(1, int(self.cfg.home_move_duration * self.cfg.home_move_fps))
        completed = True
        for index in range(steps):
            if self._motion_abort.is_set():
                completed = False
                break
            loop_start = time.perf_counter()
            alpha = (index + 1) / steps
            interpolated = current_joints + alpha * (target_joints - current_joints)
            with self._command_lock:
                if self._motion_abort.is_set():
                    completed = False
                    break
                period = self.rtde_ctrl.initPeriod()
                self.rtde_ctrl.servoJ(
                    interpolated.tolist(),
                    self.cfg.arm_velocity,
                    self.cfg.arm_acceleration,
                    self.cfg.servoj_time,
                    self.cfg.servoj_lookahead,
                    self.cfg.servoj_gain,
                )
                self._command_gripper_normalized(gripper_target)
                self.rtde_ctrl.waitPeriod(period)
            sleep_left = (1.0 / self.cfg.home_move_fps) - (
                time.perf_counter() - loop_start
            )
            if sleep_left > 0 and self._motion_abort.wait(sleep_left):
                completed = False
                break

        if completed:
            time.sleep(0.5)
        self._step_count = 0
        self._gripper_lock_counter = 0
        self._last_commanded_tcp = None
        self._upstream_reference_target = None
        self.last_tracking_error = None
        return completed

    def emergency_stop(self) -> dict[str, Any]:
        """Stop Cartesian servoing at the current RTDE command boundary."""

        self._motion_abort.set()
        report: dict[str, Any] = {
            "servo_stop_called": False,
            "stop_j_fallback_called": False,
            "stopped": False,
            "errors": [],
        }
        with self._command_lock:
            self._policy_commands_armed = False
            self._last_commanded_tcp = None
            self._upstream_reference_target = None
            control = getattr(self, "rtde_ctrl", None)
            if control is None:
                report["errors"].append("RTDE control interface is unavailable")
                self.last_emergency_stop_report = report
                return report
            try:
                result = control.servoStop()
                report["servo_stop_called"] = True
                report["servo_stop_return"] = result
                if result is False:
                    raise RuntimeError("servoStop returned false")
                report["stopped"] = True
            except Exception as error:
                report["errors"].append(f"servoStop: {type(error).__name__}: {error}")
                try:
                    result = control.stopJ(float(self.cfg.arm_acceleration))
                    report["stop_j_fallback_called"] = True
                    report["stop_j_return"] = result
                    if result is False:
                        raise RuntimeError("stopJ returned false")
                    report["stopped"] = True
                except Exception as fallback_error:
                    report["errors"].append(
                        f"stopJ: {type(fallback_error).__name__}: {fallback_error}"
                    )
        self.last_emergency_stop_report = report
        return report

    def _command_gripper(self, target_gripper):
        """Make the inherited gripper policy explicit and manifest-controlled."""

        self._step_count += 1
        model_target = float(np.clip(target_gripper, -1.0, 1.0))
        target_normalized = (1.0 - model_target) / 2.0
        if self._step_count <= int(self.cfg.gripper_force_open_steps):
            target_normalized = 0.0
        desired_status = "open" if target_normalized < 0.5 else "close"
        if self._gripper_lock_counter > 0:
            self._gripper_lock_counter -= 1
        if self._gripper_status is not None and desired_status != self._gripper_status:
            if self._gripper_lock_counter > 0:
                desired_status = self._gripper_status
                if self._last_gripper_pos is not None:
                    target_normalized = self._last_gripper_pos / 255.0
                else:
                    target_normalized = 0.0 if desired_status == "open" else 1.0
            else:
                self._gripper_lock_counter = int(self.cfg.gripper_status_lock_steps)
                self._gripper_status = desired_status
        elif self._gripper_status is None:
            self._gripper_status = desired_status

        target_position = int(np.clip(round(target_normalized * 255.0), 0, 255))
        now = time.time()
        if self._last_gripper_pos is not None:
            if abs(target_position - self._last_gripper_pos) < self.cfg.gripper_min_delta:
                return
            if now - self._last_gripper_cmd_time < self.cfg.gripper_min_period_s:
                return
        self.gripper.move(target_position, self.cfg.gripper_speed, self.cfg.gripper_force)
        self._last_gripper_pos = target_position
        self._last_gripper_cmd_time = now

    def step(self, target_pose6d, target_gripper):
        with self._command_lock:
            if self._motion_abort.is_set():
                self._policy_commands_armed = False
                self._cancel_next_disarmed_policy_step = False
                return POLICY_STEP_CANCELLED
            if not self._policy_commands_armed:
                if self._cancel_next_disarmed_policy_step:
                    self._cancel_next_disarmed_policy_step = False
                    return POLICY_STEP_CANCELLED
                raise RuntimeError(
                    "Rejected policy command because no live rollout is armed"
                )
            target = np.asarray(target_pose6d, dtype=np.float64).reshape(-1)
            if target.shape != (6,) or not np.isfinite(target).all():
                raise RuntimeError("Rejected non-finite or malformed target TCP pose")
            actual_tcp = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=np.float64)
            if actual_tcp.shape != (6,) or not np.isfinite(actual_tcp).all():
                raise RuntimeError("Rejected invalid actual TCP feedback")
            if np.any(actual_tcp[:3] < self._workspace_min) or np.any(
                actual_tcp[:3] > self._workspace_max
            ):
                raise RuntimeError(
                    "Actual TCP left the reviewed workspace: "
                    f"xyz={actual_tcp[:3].tolist()}"
                )
            if self._tracking_guard_enabled and self._last_commanded_tcp is not None:
                self.last_tracking_error = tcp_tracking_error(
                    actual_tcp, self._last_commanded_tcp
                )
                translation_error = self.last_tracking_error["translation_m"]
                rotation_error = self.last_tracking_error["rotation_rad"]
                if (
                    translation_error > self._max_tracking_translation_m
                    or rotation_error > self._max_tracking_rotation_rad
                ):
                    raise RuntimeError(
                        "Rejected command after TCP tracking error exceeded reviewed limits: "
                        f"translation_m={translation_error:.6f}, "
                        f"rotation_rad={rotation_error:.6f}"
                    )
            previous_requested = self._upstream_reference_target
            if previous_requested is None:
                previous_requested = legacy_deploy._ur_tcp_to_pose6d(actual_tcp)
            rebased_target = rebase_incremental_target_on_actual_tcp(
                previous_requested,
                target,
                actual_tcp,
            )
            xyz = rebased_target[:3]
            if np.any(xyz < self._workspace_min) or np.any(
                xyz > self._workspace_max
            ):
                raise RuntimeError(
                    "Rejected rebased target outside reviewed workspace: "
                    f"xyz={xyz.tolist()} min={self._workspace_min.tolist()} "
                    f"max={self._workspace_max.tolist()}"
                )
            # Stop may be requested while validation is in progress. Recheck at
            # the last command boundary so a completed, late inference cannot
            # enter the copied hardware implementation after an operator stop.
            if self._motion_abort.is_set():
                self._policy_commands_armed = False
                self._cancel_next_disarmed_policy_step = False
                return POLICY_STEP_CANCELLED
            result = super().step(rebased_target, float(target_gripper))
            self._upstream_reference_target = target.copy()
            self._last_commanded_tcp = legacy_deploy._pose6d_to_ur_tcp(rebased_target)
            return result

    def get_robot_state(self):
        state = super().get_robot_state()
        tcp = legacy_deploy._pose6d_to_ur_tcp(state["pose6d"])
        state["tcp_rotvec"] = tcp[3:].astype(np.float32)
        return state


def build_live_environment(contract: DeploymentContract, cfg):
    workspace = contract.hardware["robot"]["workspace_m"]
    tracking = contract.hardware["robot"]["control"]["tracking_error_guard"]
    return SafeUR5eDeployEnv(
        cfg,
        workspace_min=workspace["min"],
        workspace_max=workspace["max"],
        tracking_error_guard=dict(tracking),
    )
