"""Adapter around the copied, known-working 3DFlow UR5e hardware source."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

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
    cfg.arm_acceleration = float(control["acceleration"])
    cfg.arm_velocity = float(control["velocity"])
    cfg.servoj_time = float(control["servoj_time_s"])
    cfg.servoj_lookahead = float(control["servoj_lookahead_s"])
    cfg.servoj_gain = float(control["servoj_gain"])
    cfg.warmup_steps = int(runtime["warmup_steps"])
    cfg.num_rollouts = int(runtime["num_rollouts_per_instruction"])
    cfg.results_dir = str(Path(runtime["results_directory"]).expanduser().resolve())
    cfg.enable_rollout_media = bool(runtime["save_rollout_media"])
    cfg.enable_observer_media = bool(
        cfg.enable_rollout_media
        and isinstance(cameras.get("observer"), dict)
        and cameras["observer"].get("serial")
    )
    cfg.camera_serials_file = str(
        Path(runtime["camera_serials_file"]).expanduser().resolve()
    )
    cfg.camera_serial_cache = {
        "exterior": str(cameras["exterior"]["serial"]),
        "wrist": str(cameras["wrist"]["serial"]),
    }
    return cfg


class SafeUR5eDeployEnv(legacy_deploy.UR5eDeployEnv):
    """Copied live environment plus an explicit Cartesian workspace guard."""

    def __init__(self, cfg, *, workspace_min: list[float], workspace_max: list[float]):
        self._workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self._workspace_max = np.asarray(workspace_max, dtype=np.float64)
        super().__init__(cfg)

    def step(self, target_pose6d, target_gripper):
        target = np.asarray(target_pose6d, dtype=np.float64).reshape(-1)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise RuntimeError("Rejected non-finite or malformed target TCP pose")
        xyz = target[:3]
        if np.any(xyz < self._workspace_min) or np.any(xyz > self._workspace_max):
            raise RuntimeError(
                "Rejected target outside reviewed workspace: "
                f"xyz={xyz.tolist()} min={self._workspace_min.tolist()} "
                f"max={self._workspace_max.tolist()}"
            )
        return super().step(target, float(target_gripper))

    def get_robot_state(self):
        state = super().get_robot_state()
        tcp = legacy_deploy._pose6d_to_ur_tcp(state["pose6d"])
        state["tcp_rotvec"] = tcp[3:].astype(np.float32)
        return state


def build_live_environment(contract: DeploymentContract, cfg):
    workspace = contract.hardware["robot"]["workspace_m"]
    return SafeUR5eDeployEnv(
        cfg,
        workspace_min=workspace["min"],
        workspace_max=workspace["max"],
    )
