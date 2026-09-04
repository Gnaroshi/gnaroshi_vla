"""Strict artifact, policy, sensor, action, and live-safety contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 2
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOURCE_UPSTREAM_COMMIT = "32700d0ad8991996e123e4b685abe370ce6e9aab"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory contains no files: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedArtifact:
    name: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class DeploymentContract:
    path: Path
    payload: dict[str, Any]
    artifacts: dict[str, VerifiedArtifact]

    @property
    def deployment_id(self) -> str:
        return str(self.payload["deployment_id"])

    @property
    def policy(self) -> Mapping[str, Any]:
        return self.payload["policy"]

    @property
    def state(self) -> Mapping[str, Any]:
        return self.payload["state"]

    @property
    def action(self) -> Mapping[str, Any]:
        return self.payload["action"]

    @property
    def hardware(self) -> Mapping[str, Any]:
        return self.payload["hardware"]

    @property
    def runtime(self) -> Mapping[str, Any]:
        return self.payload["runtime"]

    @property
    def pairing(self) -> Mapping[str, Any]:
        return self.payload["pairing"]

    @property
    def live_authorized(self) -> bool:
        return bool(self.payload["safety_review"]["live_authorized"])


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _require_number(parent: Mapping[str, Any], key: str, *, positive: bool = False) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ValueError(f"{key} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{key} must be positive")
    return number


def _resolve_path(value: Any, manifest_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} path must be a non-empty string")
    path = Path(value).expanduser()
    return (manifest_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _verify_artifacts(
    payload: Mapping[str, Any], manifest_dir: Path, *, verify_files: bool
) -> dict[str, VerifiedArtifact]:
    raw = _require_mapping(payload, "artifacts")
    file_names = (
        "official_base_model_weights",
        "norm_stats",
        "real_action_transformer",
        "condition_updater",
        "generation_updater",
    )
    verified: dict[str, VerifiedArtifact] = {}
    for name in file_names:
        spec = _require_mapping(raw, name)
        path = _resolve_path(spec.get("path"), manifest_dir, name)
        expected = str(spec.get("sha256", "")).lower()
        if not SHA256_PATTERN.fullmatch(expected):
            raise ValueError(f"artifacts.{name}.sha256 must be a 64-character SHA-256")
        if verify_files:
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name}: {path}")
            observed = sha256_file(path)
            if observed != expected:
                raise ValueError(
                    f"{name} SHA-256 mismatch: observed={observed} expected={expected}"
                )
            size = path.stat().st_size
        else:
            size = path.stat().st_size if path.is_file() else 0
        verified[name] = VerifiedArtifact(name, path, expected, size)

    for name in ("official_base_model_directory", "processor_directory"):
        spec = _require_mapping(raw, name)
        path = _resolve_path(spec.get("path"), manifest_dir, name)
        expected = str(spec.get("sha256", "")).lower()
        if not SHA256_PATTERN.fullmatch(expected):
            raise ValueError(f"artifacts.{name}.sha256 must be a 64-character SHA-256")
        if verify_files:
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {name}: {path}")
            observed = sha256_directory(path)
            if observed != expected:
                raise ValueError(
                    f"{name} SHA-256 mismatch: observed={observed} expected={expected}"
                )
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        else:
            size = 0
        verified[name] = VerifiedArtifact(name, path, expected, size)
    return verified


def _validate_policy(payload: Mapping[str, Any]) -> None:
    policy = _require_mapping(payload, "policy")
    expected = {
        "action_mode": "libero_joint",
        "proprio_dim": 8,
        "action_dim": 7,
        "condition_tokens": 122,
        "condition_dim": 960,
        "model_image_size": 384,
        "client_resize_size": 224,
        "image_interpolation": "bicubic",
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "condition_refresh_interval": 2,
        "generation_full_evaluations": 3,
    }
    mismatches = {
        key: {"observed": policy.get(key), "required": value}
        for key, value in expected.items()
        if policy.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported SimVLA deployment policy contract: {mismatches}")
    if policy.get("control_protocol") != "fresh_h10_execute_r5":
        raise ValueError("policy.control_protocol must be fresh_h10_execute_r5")
    if policy.get("generation_condition_coupling") != "uncoupled_zero_code":
        raise ValueError("Only the selected uncoupled Generation Loop is supported")
    if policy.get("deterministic_action_noise") is not True:
        raise ValueError("policy.deterministic_action_noise must be true")


def _validate_state_action(payload: Mapping[str, Any]) -> None:
    state = _require_mapping(payload, "state")
    if state.get("encoding") != "opposed_finger_positions":
        raise ValueError("state.encoding must be opposed_finger_positions")
    _require_number(state, "gripper_max_opening_m", positive=True)
    if state.get("tcp_orientation") not in {
        "axis_angle_radians",
        "euler_xyz_radians",
    }:
        raise ValueError(
            "state.tcp_orientation must explicitly be axis_angle_radians or "
            "euler_xyz_radians"
        )

    action = _require_mapping(payload, "action")
    if action.get("representation") != "normalized_delta_pose":
        raise ValueError("action.representation must be normalized_delta_pose")
    _require_number(action, "translation_scale_m", positive=True)
    _require_number(action, "rotation_scale_rad", positive=True)
    _require_number(action, "clip_abs", positive=True)
    if action.get("model_positive_gripper_means") not in {"open", "close"}:
        raise ValueError("action.model_positive_gripper_means must be open or close")
    preflight = state.get("preflight_vector")
    if not isinstance(preflight, list) or len(preflight) != 8:
        raise ValueError("state.preflight_vector must contain eight values")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) == float(value)
        and abs(float(value)) != float("inf")
        for value in preflight
    ):
        raise ValueError("state.preflight_vector must be finite and numeric")


def _validate_hardware_runtime(payload: Mapping[str, Any]) -> None:
    hardware = _require_mapping(payload, "hardware")
    robot = _require_mapping(hardware, "robot")
    cameras = _require_mapping(hardware, "cameras")
    if robot.get("type") != "ur5e_robotiq_2f85":
        raise ValueError("hardware.robot.type must be ur5e_robotiq_2f85")
    if not str(robot.get("ip", "")).strip():
        raise ValueError("hardware.robot.ip is required")
    home = robot.get("home_pose")
    if not isinstance(home, list) or len(home) not in {6, 7}:
        raise ValueError("hardware.robot.home_pose must contain 6 joints and optional gripper")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in home):
        raise ValueError("hardware.robot.home_pose values must be numeric")
    workspace = _require_mapping(robot, "workspace_m")
    for key in ("min", "max"):
        values = workspace.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"hardware.robot.workspace_m.{key} must contain xyz")
    if any(float(lo) >= float(hi) for lo, hi in zip(workspace["min"], workspace["max"])):
        raise ValueError("hardware.robot.workspace_m min must be below max")
    control = _require_mapping(robot, "control")
    for key in (
        "home_move_duration_s",
        "home_move_hz",
        "acceleration",
        "velocity",
        "servoj_time_s",
        "servoj_lookahead_s",
        "servoj_gain",
    ):
        _require_number(control, key, positive=True)
    gripper = _require_mapping(robot, "gripper")
    _require_number(gripper, "open_position_threshold")
    for key in ("speed", "force", "min_command_delta"):
        value = gripper.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"hardware.robot.gripper.{key} must be a non-negative integer")
    _require_number(gripper, "min_command_period_s", positive=True)
    for name in ("exterior", "wrist"):
        camera = _require_mapping(cameras, name)
        if not str(camera.get("serial", "")).strip():
            raise ValueError(f"hardware.cameras.{name}.serial is required")
    if cameras.get("model_input_order") != ["exterior", "wrist"]:
        raise ValueError("hardware.cameras.model_input_order must be [exterior, wrist]")
    for key in ("width", "height", "fps"):
        value = cameras.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"hardware.cameras.{key} must be a positive integer")

    runtime = _require_mapping(payload, "runtime")
    _require_number(runtime, "control_frequency_hz", positive=True)
    if not isinstance(runtime.get("max_steps"), int) or runtime["max_steps"] < 1:
        raise ValueError("runtime.max_steps must be a positive integer")
    if not isinstance(runtime.get("seed"), int):
        raise ValueError("runtime.seed must be an integer")
    if not isinstance(runtime.get("warmup_steps"), int) or runtime["warmup_steps"] < 0:
        raise ValueError("runtime.warmup_steps must be a non-negative integer")
    if (
        not isinstance(runtime.get("num_rollouts_per_instruction"), int)
        or runtime["num_rollouts_per_instruction"] < 1
    ):
        raise ValueError("runtime.num_rollouts_per_instruction must be positive")
    for key in ("results_directory", "camera_serials_file"):
        if not str(runtime.get(key, "")).strip():
            raise ValueError(f"runtime.{key} is required")
    if not isinstance(runtime.get("save_rollout_media"), bool):
        raise ValueError("runtime.save_rollout_media must be boolean")
    instructions = runtime.get("instructions")
    if not isinstance(instructions, list) or not instructions or not all(
        isinstance(value, str) and value.strip() for value in instructions
    ):
        raise ValueError("runtime.instructions must contain at least one instruction")


def _validate_pairing(payload: Mapping[str, Any]) -> None:
    pairing = _require_mapping(payload, "pairing")
    for key in (
        "official_base_model_identity",
        "real_baseline_identity",
        "condition_source_real_baseline_identity",
        "generation_source_real_baseline_identity",
    ):
        if not str(pairing.get(key, "")).strip():
            raise ValueError(f"pairing.{key} is required")
    baseline = pairing["real_baseline_identity"]
    if pairing["condition_source_real_baseline_identity"] != baseline:
        raise ValueError("Condition updater was not paired with this real baseline")
    if pairing["generation_source_real_baseline_identity"] != baseline:
        raise ValueError("Generation updater was not paired with this real baseline")


def load_deployment_contract(
    path: str | Path, *, verify_artifacts: bool = True
) -> DeploymentContract:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Deployment manifest must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported deployment schema {payload.get('schema_version')}; expected {SCHEMA_VERSION}"
        )
    if not str(payload.get("deployment_id", "")).strip():
        raise ValueError("deployment_id is required")
    if payload.get("simvla_upstream_commit") != SOURCE_UPSTREAM_COMMIT:
        raise ValueError("Deployment manifest does not select the vendored SimVLA revision")

    _validate_policy(payload)
    _validate_state_action(payload)
    _validate_hardware_runtime(payload)
    _validate_pairing(payload)
    safety = _require_mapping(payload, "safety_review")
    if not isinstance(safety.get("live_authorized"), bool):
        raise ValueError("safety_review.live_authorized must be boolean")
    artifacts = _verify_artifacts(payload, manifest_path.parent, verify_files=verify_artifacts)
    return DeploymentContract(manifest_path, payload, artifacts)


def require_live_authorization(contract: DeploymentContract) -> None:
    review = contract.payload["safety_review"]
    failures = []
    if not contract.live_authorized:
        failures.append("manifest safety_review.live_authorized is false")
    if not bool(review.get("model_preflight_passed")):
        failures.append("model_preflight_passed is not true")
    if not bool(review.get("read_only_profile_passed")):
        failures.append("read_only_profile_passed is not true")
    if not str(review.get("approved_by", "")).strip():
        failures.append("approved_by is empty")
    if not str(review.get("approved_at", "")).strip():
        failures.append("approved_at is empty")
    if os.environ.get("SIMVLA_REAL_LIVE_RUN") != "1":
        failures.append("SIMVLA_REAL_LIVE_RUN=1 is absent")
    if os.environ.get("SIMVLA_REAL_DEPLOYMENT_ID") != contract.deployment_id:
        failures.append("SIMVLA_REAL_DEPLOYMENT_ID does not match the manifest")
    if failures:
        raise PermissionError("Live deployment rejected: " + "; ".join(failures))
