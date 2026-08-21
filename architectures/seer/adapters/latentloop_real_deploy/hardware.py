"""Load the preserved 3DFlow-Seer hardware and GUI implementation.

The source snapshot is intentionally left byte-identical. This compatibility
module loads it under a private module name so LatentLoop can reuse the known
camera, UR5e, gripper, media, and GUI behavior without replacing Seer's upstream
``deploy.py`` or ``real_controller`` files.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[4]
SEER_UPSTREAM_ROOT = REPO_ROOT / "architectures" / "seer" / "upstream"
LEGACY_ROOT = (
    REPO_ROOT / "architectures" / "seer" / "third_party" / "3dflow_real_deploy"
)


def _load_legacy_deploy() -> ModuleType:
    source = LEGACY_ROOT / "deploy.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing preserved 3DFlow deploy source: {source}")

    for path in (str(SEER_UPSTREAM_ROOT), str(LEGACY_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    module_name = "_gnaroshi_3dflow_real_deploy"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load preserved deploy source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


legacy_deploy = _load_legacy_deploy()

DeployConfig = legacy_deploy.DeployConfig
DeployResult = legacy_deploy.DeployResult
RealSenseCamera = legacy_deploy.RealSenseCamera
RolloutMediaCapture = legacy_deploy.RolloutMediaCapture
UR5eDeployEnv = legacy_deploy.UR5eDeployEnv
_6d_to_pose = legacy_deploy._6d_to_pose
_camera_serials_file = legacy_deploy._camera_serials_file
_checkpoint_results_dir = legacy_deploy._checkpoint_results_dir
_discover_realsense_devices = legacy_deploy._discover_realsense_devices
_known_deploy_camera_serials = legacy_deploy._known_deploy_camera_serials
_load_camera_serial_cache = legacy_deploy._load_camera_serial_cache
_maybe_cuda_sync = legacy_deploy._maybe_cuda_sync
_prompt_enter = legacy_deploy._prompt_enter
_prompt_yes_no = legacy_deploy._prompt_yes_no
_resolve_realsense_serials = legacy_deploy._resolve_realsense_serials
_save_camera_serial_cache = legacy_deploy._save_camera_serial_cache
_write_instruction_markers = legacy_deploy._write_instruction_markers
pose_to_6d = legacy_deploy.pose_to_6d
save_deploy_results = legacy_deploy.save_deploy_results


def load_legacy_gui() -> ModuleType:
    """Load the preserved GUI after binding its legacy ``deploy`` import."""

    module_name = "_gnaroshi_3dflow_real_deploy_gui"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    source = LEGACY_ROOT / "deploy_gui.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing preserved 3DFlow GUI source: {source}")

    # deploy_gui.py imports ``deploy`` by its historical top-level name.
    previous_deploy = sys.modules.get("deploy")
    sys.modules["deploy"] = legacy_deploy
    try:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load preserved GUI source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_deploy is None:
            sys.modules.pop("deploy", None)
        else:
            sys.modules["deploy"] = previous_deploy
    return module
