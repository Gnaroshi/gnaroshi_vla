"""Strict artifact, policy, sensor, action, and live-safety contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 4
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOURCE_UPSTREAM_COMMIT = "32700d0ad8991996e123e4b685abe370ce6e9aab"
REAL_DATASET_SCHEMA = "simvla_real_hdf5_v3"
REAL_CONDITION_CACHE_SCHEMA = "simvla_real_condition_cache_v2"
REAL_IMAGE_PREPROCESSING = (
    "resize_with_pad_224_then_bicubic_384_imagenet_norm"
)
REAL_IMAGE_STORAGE = "JPEG quality=95 subsampling=0"
REAL_STATE_REPRESENTATION = "tcp_xyz_rotvec_plus_opposed_finger_positions"
REAL_CONDITION_ROTATION_DELTA = (
    "current rotvec mapped to equivalent 2pi branch nearest previous rotvec"
)
REAL_ACTION_REPRESENTATION = (
    "inv(T_current)@T_next local xyz plus xyz_euler plus continuous gripper target"
)
REAL_DATA_SOURCE_FORMAT = "3dflow_teleoperation_pickle"
REAL_DATA_TCP_POSE_SOURCE = "ee_pos_quat interpreted as xyz+rotation_vector"
REAL_DATA_OBSERVATION_ACTION_ALIGNMENT = (
    "frame_t stores observation_t and command_t before env.step(command_t)"
)
REAL_DATA_GRIPPER_COMMAND_SOURCE = (
    "current_frame.control[6] recorded with observation_t before "
    "env.step(command_t); 0=open and 1=close continuous target"
)
REAL_DATA_POSE_LABEL_SOURCE = (
    "measured transition observation_t to observation_t+1"
)
REAL_DATA_GRIPPER_LABEL_SOURCE = "1 - 2 * command_t stored in current frame"
REAL_DATA_SAMPLING_MODE = "native_capture_order_with_timing_valid_windows"
REAL_DATA_SPLIT_SEED = 20260904

# Conservatively pin every repository source tree that can affect a deployed
# SimVLA action. Vendored upstream and copied hardware sources are locked by
# source_manifest.json; this list covers the adapter/runtime implementation.
RUNTIME_SOURCE_PATHS = (
    "architectures/simvla/adapters/latentloop_real_deploy",
    "architectures/simvla/adapters/dcld",
    "methods/dcld",
    "architectures/simvla/adapters/latentloop/native_v0_condition_hook.py",
    "architectures/simvla/adapters/latentloop/native_v0_policy.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/contracts.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/coupled_condition_generation.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_hidden.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_policy.py",
    "methods/latentloop/modules/__init__.py",
    "methods/latentloop/modules/native_simvla_v0.py",
    "methods/latentloop/modules/simvla_generation_loop.py",
    "architectures/simvla/wrappers/dcld_eval/invariants.py",
    "architectures/simvla/wrappers/dcld_eval/rollout_runner.py",
    "architectures/simvla/adapters/real_world_training",
)


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


def sha256_json(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def runtime_source_identity() -> dict[str, Any]:
    """Return a deterministic identity for code that can affect robot actions."""

    root = Path(__file__).resolve().parents[4]
    files: set[Path] = set()
    for relative in RUNTIME_SOURCE_PATHS:
        path = root / relative
        if path.is_dir():
            files.update(item for item in path.rglob("*.py") if item.is_file())
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"Runtime source is missing: {path}")
    digests = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(files)
    }
    if not digests:
        raise RuntimeError("Runtime source identity contains no files")
    return {
        "combined_sha256": sha256_json(digests),
        "file_count": len(digests),
        "files": digests,
    }


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
        "dataset_manifest",
        "condition_cache_manifest",
        "condition_cache_attestation",
        "real_action_transformer",
        "condition_updater",
        "generation_updater",
        "coupled_generation_updater",
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
    if policy.get("generation_condition_coupling") != "condition_delta_code":
        raise ValueError(
            "policy.generation_condition_coupling must be condition_delta_code"
        )
    if policy.get("deterministic_action_noise") is not True:
        raise ValueError("policy.deterministic_action_noise must be true")


def _validate_state_action(payload: Mapping[str, Any]) -> None:
    state = _require_mapping(payload, "state")
    if state.get("encoding") != "opposed_finger_positions":
        raise ValueError("state.encoding must be opposed_finger_positions")
    _require_number(state, "gripper_max_opening_m", positive=True)
    if state.get("tcp_orientation") != "axis_angle_radians":
        raise ValueError("state.tcp_orientation must be axis_angle_radians")

    action = _require_mapping(payload, "action")
    if action.get("representation") != "normalized_delta_pose":
        raise ValueError("action.representation must be normalized_delta_pose")
    _require_number(action, "translation_scale_m", positive=True)
    _require_number(action, "rotation_scale_rad", positive=True)
    _require_number(action, "clip_abs", positive=True)
    if action.get("model_positive_gripper_means") != "open":
        raise ValueError("action.model_positive_gripper_means must be open")
    if action.get("gripper_representation") != "continuous_normalized_position":
        raise ValueError(
            "action.gripper_representation must be continuous_normalized_position"
        )
    preflight = _require_mapping(state, "preflight_robot_state")
    for key, length in (("pose6d_euler_xyz", 6), ("tcp_rotvec", 3)):
        values = preflight.get(key)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"state.preflight_robot_state.{key} must contain {length} values")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) == float(value)
            and abs(float(value)) != float("inf")
            for value in values
        ):
            raise ValueError(f"state.preflight_robot_state.{key} must be finite")
    _require_number(preflight, "gripper_open_state")
    gripper_position = _require_number(preflight, "gripper_position_normalized")
    if not 0.0 <= gripper_position <= 1.0:
        raise ValueError("preflight gripper_position_normalized must be in [0,1]")


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
    for index, value in enumerate(home):
        _require_number({str(index): value}, str(index))
    if len(home) == 7 and not 0.0 <= home[6] <= 1.0:
        raise ValueError("hardware.robot.home_pose gripper target must be in [0,1]")
    for key in ("home_pose_source", "workspace_source"):
        source = str(robot.get(key, "")).strip()
        if not source:
            raise ValueError(f"hardware.robot.{key} is required")
    workspace = _require_mapping(robot, "workspace_m")
    for key in ("min", "max"):
        values = workspace.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"hardware.robot.workspace_m.{key} must contain xyz")
        for index, value in enumerate(values):
            _require_number({f"workspace.{key}.{index}": value}, f"workspace.{key}.{index}")
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
    tracking = _require_mapping(control, "tracking_error_guard")
    if not isinstance(tracking.get("enabled"), bool):
        raise ValueError("hardware.robot.control.tracking_error_guard.enabled must be boolean")
    if tracking["enabled"]:
        _require_number(tracking, "max_translation_error_m", positive=True)
        _require_number(tracking, "max_rotation_error_rad", positive=True)
    gripper = _require_mapping(robot, "gripper")
    if gripper.get("command_mode") != "continuous_position_with_crossing_lock":
        raise ValueError(
            "hardware.robot.gripper.command_mode must preserve continuous position "
            "with the reviewed open/close crossing lock"
        )
    if not 0.0 <= _require_number(gripper, "open_position_threshold") <= 1.0:
        raise ValueError("hardware.robot.gripper.open_position_threshold must be in [0,1]")
    for key in ("speed", "force", "min_command_delta"):
        value = gripper.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"hardware.robot.gripper.{key} must be a non-negative integer")
        if value > 255:
            raise ValueError(f"hardware.robot.gripper.{key} must be at most 255")
    _require_number(gripper, "min_command_period_s", positive=True)
    for key in ("startup_force_open_steps", "status_lock_steps"):
        value = gripper.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"hardware.robot.gripper.{key} must be a non-negative integer")
    for name in ("exterior", "wrist"):
        camera = _require_mapping(cameras, name)
        if not str(camera.get("serial", "")).strip():
            raise ValueError(f"hardware.cameras.{name}.serial is required")
    if cameras["exterior"]["serial"] == cameras["wrist"]["serial"]:
        raise ValueError("exterior and wrist must use different camera serials")
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
        "norm_stats_identity",
        "dataset_identity",
        "condition_cache_identity",
        "condition_cache_attestation_identity",
        "condition_source_real_baseline_identity",
        "generation_source_real_baseline_identity",
        "coupled_source_real_baseline_identity",
        "coupled_parent_generation_identity",
        "coupled_condition_updater_identity",
        "coupled_generation_identity",
    ):
        if not str(pairing.get(key, "")).strip():
            raise ValueError(f"pairing.{key} is required")
    baseline = pairing["real_baseline_identity"]
    if pairing["condition_source_real_baseline_identity"] != baseline:
        raise ValueError("Condition updater was not paired with this real baseline")
    if pairing["generation_source_real_baseline_identity"] != baseline:
        raise ValueError("Generation updater was not paired with this real baseline")
    if pairing["coupled_source_real_baseline_identity"] != baseline:
        raise ValueError("Coupled Generation updater was not paired with this real baseline")


def _validate_artifact_pairing(
    pairing: Mapping[str, Any], artifacts: Mapping[str, VerifiedArtifact]
) -> None:
    direct = {
        "official_base_model_identity": "official_base_model_weights",
        "real_baseline_identity": "real_action_transformer",
        "norm_stats_identity": "norm_stats",
        "coupled_parent_generation_identity": "generation_updater",
        "coupled_condition_updater_identity": "condition_updater",
        "coupled_generation_identity": "coupled_generation_updater",
    }
    mismatches = {
        key: {
            "declared": pairing.get(key),
            "artifact": artifacts[artifact_name].sha256,
        }
        for key, artifact_name in direct.items()
        if pairing.get(key) != artifacts[artifact_name].sha256
    }
    dataset = json.loads(
        artifacts["dataset_manifest"].path.read_text(encoding="utf-8")
    )
    cache = json.loads(
        artifacts["condition_cache_manifest"].path.read_text(encoding="utf-8")
    )
    if pairing.get("dataset_identity") != dataset.get("dataset_identity_sha256"):
        mismatches["dataset_identity"] = {
            "declared": pairing.get("dataset_identity"),
            "manifest": dataset.get("dataset_identity_sha256"),
        }
    if pairing.get("condition_cache_identity") != cache.get(
        "condition_cache_identity_sha256"
    ):
        mismatches["condition_cache_identity"] = {
            "declared": pairing.get("condition_cache_identity"),
            "manifest": cache.get("condition_cache_identity_sha256"),
        }
    if cache.get("dataset_identity_sha256") != pairing.get("dataset_identity"):
        mismatches["cache_dataset_identity"] = {
            "cache": cache.get("dataset_identity_sha256"),
            "declared": pairing.get("dataset_identity"),
        }
    if mismatches:
        raise ValueError(f"deployment artifact pairing mismatch: {mismatches}")


def _validate_dataset_identity(dataset: Mapping[str, Any]) -> None:
    episodes = dataset.get("episodes")
    splits = dataset.get("splits")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("dataset manifest must contain episodes")
    if not isinstance(splits, Mapping):
        raise ValueError("dataset manifest must contain train/validation splits")
    train = splits.get("train")
    validation = splits.get("validation")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("dataset train/validation splits must be lists")
    train_ids = [str(value) for value in train]
    validation_ids = [str(value) for value in validation]
    if len(train_ids) != 32 or len(validation_ids) != 8:
        raise ValueError("real deployment requires the predeclared 32/8 episode split")
    if set(train_ids).intersection(validation_ids):
        raise ValueError("dataset train and validation episodes overlap")
    episode_ids = [str(item.get("episode_id", "")) for item in episodes]
    if len(episode_ids) != 40 or len(set(episode_ids)) != 40:
        raise ValueError("real deployment requires 40 unique source episodes")
    if set(episode_ids) != set(train_ids).union(validation_ids):
        raise ValueError("dataset splits do not cover exactly the manifest episodes")
    episode_sha256: dict[str, str] = {}
    for item in episodes:
        episode_id = str(item.get("episode_id", ""))
        digest = str(item.get("sha256", "")).lower()
        if not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"dataset episode {episode_id!r} has an invalid SHA-256")
        episode_sha256[episode_id] = digest
    identity_payload = {
        "schema": dataset.get("schema_version"),
        "episode_sha256": episode_sha256,
        "split_seed": dataset.get("split_seed"),
        "train": train_ids,
        "validation": validation_ids,
        "instruction": dataset.get("instruction"),
    }
    expected = sha256_json(identity_payload)
    if dataset.get("dataset_identity_sha256") != expected:
        raise ValueError("dataset identity does not match its episode/split contract")


def _validate_dataset_cache_semantics(
    payload: Mapping[str, Any], artifacts: Mapping[str, VerifiedArtifact]
) -> None:
    dataset = json.loads(
        artifacts["dataset_manifest"].path.read_text(encoding="utf-8")
    )
    cache = json.loads(
        artifacts["condition_cache_manifest"].path.read_text(encoding="utf-8")
    )
    attestation = json.loads(
        artifacts["condition_cache_attestation"].path.read_text(encoding="utf-8")
    )
    policy = payload["policy"]
    state = payload["state"]
    action = payload["action"]
    runtime = payload["runtime"]
    pairing = payload["pairing"]

    if dataset.get("schema_version") != REAL_DATASET_SCHEMA:
        raise ValueError("unsupported real dataset schema")
    if dataset.get("verdict") != "REAL_DATASET_CONTRACT_PASS":
        raise ValueError("real dataset conversion contract did not pass")
    _validate_dataset_identity(dataset)
    source = _require_mapping(dataset, "source")
    if source.get("format") != REAL_DATA_SOURCE_FORMAT:
        raise ValueError("real dataset source format differs from the reviewed recorder")
    if source.get("control_pose_channels_used") is not False:
        raise ValueError("real dataset must derive pose actions from measured TCP feedback")
    if source.get("tcp_pose_source") != REAL_DATA_TCP_POSE_SOURCE:
        raise ValueError("real dataset TCP source is not the reviewed rotation-vector field")
    if (
        source.get("observation_action_alignment")
        != REAL_DATA_OBSERVATION_ACTION_ALIGNMENT
    ):
        raise ValueError("real dataset observation/action alignment is not command_t")
    if source.get("gripper_command_source") != REAL_DATA_GRIPPER_COMMAND_SOURCE:
        raise ValueError("real dataset gripper target source/sign differs from deployment")
    if dataset.get("sampling_mode") != REAL_DATA_SAMPLING_MODE:
        raise ValueError("real dataset sampling mode differs from the reviewed native stream")
    if dataset.get("split_seed") != REAL_DATA_SPLIT_SEED:
        raise ValueError("real dataset does not use the predeclared train/validation split")
    if dataset.get("dataset_identity_sha256") != pairing["dataset_identity"]:
        raise ValueError("dataset identity differs from deployment pairing")
    if float(dataset.get("target_hz", -1)) != float(runtime["control_frequency_hz"]):
        raise ValueError("dataset sampling rate and deployment control rate differ")
    if dataset.get("action_horizon") != policy["action_horizon"]:
        raise ValueError("dataset and deployment action horizons differ")
    if dataset.get("execution_horizon") != policy["execution_horizon"]:
        raise ValueError("dataset and deployment execution horizons differ")
    if runtime["instructions"] != [dataset.get("instruction")]:
        raise ValueError("deployment instruction must exactly match the training instruction")

    image_contract = _require_mapping(dataset, "image_contract")
    if image_contract.get("source_order") != ["base_rgb", "wrist_rgb"]:
        raise ValueError("dataset camera order must be [base_rgb, wrist_rgb]")
    if image_contract.get("source_orientation") != "as_captured_no_flip":
        raise ValueError("dataset images must retain their captured orientation")
    if image_contract.get("storage") != REAL_IMAGE_STORAGE:
        raise ValueError("dataset JPEG codec differs from the deployed SimVLA path")
    if image_contract.get("model_preprocessing") != REAL_IMAGE_PREPROCESSING:
        raise ValueError("dataset preprocessing differs from the deployed SimVLA path")

    dataset_state = _require_mapping(dataset, "state_contract")
    if dataset_state.get("representation") != REAL_STATE_REPRESENTATION:
        raise ValueError("dataset state is not TCP rotvec plus opposed fingers")
    if (
        dataset_state.get("condition_updater_rotation_delta")
        != REAL_CONDITION_ROTATION_DELTA
    ):
        raise ValueError("dataset does not declare the reviewed rotvec branch alignment")
    if float(dataset_state.get("gripper_max_opening_m", -1)) != float(
        state["gripper_max_opening_m"]
    ):
        raise ValueError("dataset and deployment gripper state scales differ")

    dataset_action = _require_mapping(dataset, "action_contract")
    if dataset_action.get("representation") != REAL_ACTION_REPRESENTATION:
        raise ValueError("dataset action is not the reviewed local SE(3) delta")
    if dataset_action.get("pose_label_source") != REAL_DATA_POSE_LABEL_SOURCE:
        raise ValueError("dataset pose target timing differs from the reviewed recorder")
    if dataset_action.get("gripper_label_source") != REAL_DATA_GRIPPER_LABEL_SOURCE:
        raise ValueError("dataset gripper target timing differs from the reviewed recorder")
    action_fields = (
        ("translation_scale_m", "translation_scale_m"),
        ("rotation_scale_rad", "rotation_scale_rad"),
        ("clip_abs", "clip_abs"),
    )
    for dataset_key, deployment_key in action_fields:
        if float(dataset_action.get(dataset_key, -1)) != float(action[deployment_key]):
            raise ValueError(f"dataset and deployment {dataset_key} differ")
    if dataset_action.get("model_positive_gripper_means") != action.get(
        "model_positive_gripper_means"
    ):
        raise ValueError("dataset and deployment gripper signs differ")

    norm = _require_mapping(dataset, "norm_stats")
    if norm.get("sha256") != artifacts["norm_stats"].sha256:
        raise ValueError("dataset and deployment normalization statistics differ")
    audit = _require_mapping(dataset, "audit")
    if float(audit.get("pose_clip_transition_fraction", 1.0)) != 0.0:
        raise ValueError("dataset contains clipped Cartesian demonstration targets")
    if float(audit.get("max_pose_roundtrip_matrix_abs", float("inf"))) > 1e-6:
        raise ValueError("dataset local-action roundtrip error exceeds tolerance")
    if int(audit.get("gripper_label_alignment_error_count", -1)) != 0:
        raise ValueError("dataset contains gripper labels that are not command_t")
    gripper = payload["hardware"]["robot"]["gripper"]
    earliest_close = int(audit.get("earliest_gripper_close_frame", -1))
    if earliest_close < int(gripper["startup_force_open_steps"]):
        raise ValueError(
            "startup force-open duration would override demonstrated gripper closure"
        )
    minimum_switch_interval = int(
        audit.get("minimum_gripper_switch_interval", -1)
    )
    if minimum_switch_interval < int(gripper["status_lock_steps"]):
        raise ValueError(
            "gripper crossing lock is longer than a demonstrated status interval"
        )
    training_samples = int(audit.get("training_samples", -1))
    if training_samples < 1:
        raise ValueError("dataset contains no timing-valid H=10 samples")

    if cache.get("schema_version") != REAL_CONDITION_CACHE_SCHEMA:
        raise ValueError("unsupported real Condition cache schema")
    if cache.get("verdict") != "REAL_CONDITION_CACHE_PASS":
        raise ValueError("real Condition cache is incomplete")
    if int(cache.get("count", -1)) != training_samples:
        raise ValueError("Condition cache count differs from dataset training samples")
    if cache.get("shape") != [training_samples, 122, 960]:
        raise ValueError("Condition cache tensor shape differs from deployment policy")
    if cache.get("dtype") != "float32":
        raise ValueError("Condition cache must preserve frozen conditions in FP32")
    if cache.get("dataset_identity_sha256") != pairing["dataset_identity"]:
        raise ValueError("Condition cache was built from a different dataset")
    if cache.get("dataset_manifest_sha256") != artifacts["dataset_manifest"].sha256:
        raise ValueError("Condition cache dataset-manifest checksum differs from deployment")

    official = _require_mapping(cache, "official_base")
    if official.get("model_weights_sha256") != pairing["official_base_model_identity"]:
        raise ValueError("Condition cache was built from a different SimVLA checkpoint")
    if official.get("action_mode") != policy["action_mode"]:
        raise ValueError("Condition cache action mode differs from deployment")
    if int(official.get("action_horizon", -1)) != policy["action_horizon"]:
        raise ValueError("Condition cache action horizon differs from deployment")
    exact_loading = _require_mapping(cache, "exact_loading")
    if exact_loading.get("verdict") != "EXACT_OFFICIAL_INITIALIZATION_PASS":
        raise ValueError("Condition cache lacks exact official initialization proof")
    loading_info = _require_mapping(exact_loading, "loading_info")
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key) not in ([], None):
            raise ValueError(f"Condition cache official loading report contains {key}")
    if exact_loading.get("real_action_overlay") is not None:
        raise ValueError("Condition cache must come from the untouched official VLM path")
    token_layout = _require_mapping(cache, "token_layout")
    valid_mask = token_layout.get("valid_mask")
    if (
        not isinstance(valid_mask, list)
        or not valid_mask
        or any(
            not isinstance(row, list)
            or len(row) != policy["condition_tokens"]
            or not all(value is True for value in row)
            for row in valid_mask
        )
    ):
        raise ValueError(
            "real Generation training/runtime equivalence requires all 122 cached "
            "condition tokens to be source-valid for the fixed task instruction"
        )

    records_sha = str(cache.get("records_sha256", "")).lower()
    if not SHA256_PATTERN.fullmatch(records_sha):
        raise ValueError("Condition cache records SHA-256 is invalid")
    arrays = _require_mapping(cache, "arrays")
    for name in ("condition.npy", "proprio.npy", "action.npy", "complete.npy"):
        specification = _require_mapping(arrays, name)
        if not SHA256_PATTERN.fullmatch(str(specification.get("sha256", "")).lower()):
            raise ValueError(f"Condition cache {name} SHA-256 is invalid")
        if int(specification.get("size_bytes", 0)) < 1:
            raise ValueError(f"Condition cache {name} size is invalid")
    cache_identity_payload = {
        "schema": cache.get("schema_version"),
        "dataset_identity_sha256": cache.get("dataset_identity_sha256"),
        "official_model_weights_sha256": official.get("model_weights_sha256"),
        "records_sha256": records_sha,
        "array_sha256": {
            name: arrays[name].get("sha256")
            for name in (
                "condition.npy",
                "proprio.npy",
                "action.npy",
                "complete.npy",
            )
        },
        "preprocessing": image_contract.get("model_preprocessing"),
    }
    recomputed_cache_identity = sha256_json(cache_identity_payload)
    if cache.get("condition_cache_identity_sha256") != recomputed_cache_identity:
        raise ValueError("Condition cache identity does not match its source contract")
    if recomputed_cache_identity != pairing["condition_cache_identity"]:
        raise ValueError("Condition cache identity differs from deployment pairing")

    if attestation.get("schema_version") != (
        "simvla_real_condition_cache_attestation_v2"
    ):
        raise ValueError("unsupported Condition cache attestation schema")
    if attestation.get("verdict") != "REAL_CONDITION_CACHE_ATTESTATION_PASS":
        raise ValueError("Condition cache recomputation attestation did not pass")
    cache_file_stats = _require_mapping(attestation, "cache_file_stats")
    expected_attested_files = {
        "manifest.json",
        "records.json",
        "condition.npy",
        "proprio.npy",
        "action.npy",
        "complete.npy",
    }
    if set(cache_file_stats) != expected_attested_files:
        raise ValueError("Condition cache attestation file set is incomplete")
    for name in sorted(expected_attested_files):
        file_stat = _require_mapping(cache_file_stats, name)
        if int(file_stat.get("size_bytes", -1)) < 1:
            raise ValueError(f"Condition cache attestation has invalid size for {name}")
        if int(file_stat.get("mtime_ns", -1)) < 0:
            raise ValueError(f"Condition cache attestation has invalid mtime for {name}")
    attestation_checks = {
        "cache_manifest": attestation.get("cache_manifest_sha256")
        == artifacts["condition_cache_manifest"].sha256,
        "condition_array": attestation.get("condition_array_sha256")
        == arrays["condition.npy"].get("sha256"),
        "records": attestation.get("records_sha256") == records_sha,
        "cache_identity": attestation.get("condition_cache_identity_sha256")
        == pairing["condition_cache_identity"],
        "dataset_manifest": attestation.get("dataset_manifest_sha256")
        == artifacts["dataset_manifest"].sha256,
        "dataset_identity": attestation.get("dataset_identity_sha256")
        == pairing["dataset_identity"],
        "official_model": attestation.get("official_model_weights_sha256")
        == pairing["official_base_model_identity"],
        "processor": attestation.get("processor_directory_sha256")
        == artifacts["processor_directory"].sha256,
        "norm_stats": attestation.get("norm_stats_sha256")
        == artifacts["norm_stats"].sha256,
        "selection": attestation.get("selection_strategy")
        == "lower_median_valid_h10_query_per_episode",
    }
    failed_attestation_checks = [
        name for name, passed in attestation_checks.items() if not passed
    ]
    if failed_attestation_checks:
        raise ValueError(
            "Condition cache attestation artifact mismatch: "
            + ", ".join(failed_attestation_checks)
        )
    verifier_path = (
        Path(__file__).resolve().parents[4]
        / "architectures/simvla/adapters/real_world_training/verify_condition_cache.py"
    )
    if attestation.get("verifier_source_sha256") != sha256_file(verifier_path):
        raise ValueError("Condition cache attestation verifier source changed")
    samples = attestation.get("samples")
    if not isinstance(samples, list) or len(samples) != 40:
        raise ValueError("Condition cache attestation must cover all 40 episodes")
    sample_episodes = [str(row.get("episode_id", "")) for row in samples]
    dataset_episodes = [str(row["episode_id"]) for row in dataset["episodes"]]
    if sorted(sample_episodes) != sorted(dataset_episodes) or len(set(sample_episodes)) != 40:
        raise ValueError("Condition cache attestation episode coverage differs from dataset")
    if not all(row.get("bitwise_equal") is True for row in samples):
        raise ValueError("Condition cache attestation contains a non-identical query")
    comparison = _require_mapping(attestation, "comparison")
    if (
        comparison.get("sample_count") != 40
        or comparison.get("exact_equal_count") != 40
        or comparison.get("all_samples_bitwise_equal") is not True
        or comparison.get("max_abs_difference") != 0.0
        or comparison.get("mean_abs_difference") != 0.0
    ):
        raise ValueError("Condition cache attestation is not bitwise exact")
    model_loading = _require_mapping(attestation, "model_loading")
    if (
        model_loading.get("verdict") != "EXACT_OFFICIAL_INITIALIZATION_PASS"
        or model_loading.get("action_transformer_reinitialized") is not False
        or model_loading.get("real_action_overlay") is not None
    ):
        raise ValueError("Condition cache attestation did not use the untouched official VLM")
    loading_info = _require_mapping(model_loading, "loading_info")
    if any(loading_info.get(key) not in ([], None) for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
    )):
        raise ValueError("Condition cache attestation official loading was not exact")
    attestation_identity_payload = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_identity_sha256"
    }
    attestation_identity = sha256_json(attestation_identity_payload)
    if attestation.get("attestation_identity_sha256") != attestation_identity:
        raise ValueError("Condition cache attestation identity is invalid")
    if pairing["condition_cache_attestation_identity"] != attestation_identity:
        raise ValueError("Condition cache attestation identity differs from deployment pairing")


def hardware_configuration_issues(contract: DeploymentContract) -> list[str]:
    hardware = contract.hardware
    robot = hardware["robot"]
    cameras = hardware["cameras"]
    review = contract.payload["safety_review"]
    issues: list[str] = []
    if str(robot["ip"]).startswith("replace-with-"):
        issues.append("robot IP is still a template value")
    for key in ("home_pose_source", "workspace_source"):
        if str(robot[key]).startswith("replace-with-"):
            issues.append(f"robot {key} is still a template value")
    for name in ("exterior", "wrist"):
        if str(cameras[name]["serial"]).startswith("replace-with-"):
            issues.append(f"{name} camera serial is still a template value")
    for key in (
        "hardware_configuration_reviewed",
        "camera_role_mapping_verified",
        "task_home_pose_verified",
        "workspace_bounds_verified",
    ):
        if not bool(review.get(key)):
            issues.append(f"{key} is not true")
    return issues


def require_hardware_configuration(contract: DeploymentContract) -> None:
    issues = hardware_configuration_issues(contract)
    if issues:
        raise PermissionError("Hardware connection rejected: " + "; ".join(issues))


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
    declared_runtime_source = str(
        payload.get("runtime_source_identity_sha256", "")
    ).lower()
    if not SHA256_PATTERN.fullmatch(declared_runtime_source):
        raise ValueError(
            "runtime_source_identity_sha256 must be a 64-character SHA-256"
        )
    observed_runtime_source = runtime_source_identity()["combined_sha256"]
    if declared_runtime_source != observed_runtime_source:
        raise ValueError(
            "Deployment runtime source identity mismatch: "
            f"observed={observed_runtime_source} expected={declared_runtime_source}"
        )

    _validate_policy(payload)
    _validate_state_action(payload)
    _validate_hardware_runtime(payload)
    _validate_pairing(payload)
    safety = _require_mapping(payload, "safety_review")
    for key in (
        "live_authorized",
        "model_preflight_passed",
        "read_only_profile_passed",
        "hardware_configuration_reviewed",
        "camera_role_mapping_verified",
        "task_home_pose_verified",
        "workspace_bounds_verified",
        "control_limits_reviewed",
        "gripper_startup_behavior_reviewed",
        "gripper_no_software_stop_acknowledged",
        "physical_emergency_stop_verified",
        "runtime_timing_reviewed",
        "baseline_bounded_canary_passed",
    ):
        if not isinstance(safety.get(key), bool):
            raise ValueError(f"safety_review.{key} must be boolean")
    artifacts = _verify_artifacts(payload, manifest_path.parent, verify_files=verify_artifacts)
    if verify_artifacts:
        _validate_artifact_pairing(payload["pairing"], artifacts)
        _validate_dataset_cache_semantics(payload, artifacts)
    return DeploymentContract(manifest_path, payload, artifacts)


def require_live_authorization(
    contract: DeploymentContract, *, deployment_method: str
) -> None:
    review = contract.payload["safety_review"]
    failures = []
    failures.extend(hardware_configuration_issues(contract))
    if not contract.live_authorized:
        failures.append("manifest safety_review.live_authorized is false")
    if not bool(review.get("model_preflight_passed")):
        failures.append("model_preflight_passed is not true")
    if not bool(review.get("read_only_profile_passed")):
        failures.append("read_only_profile_passed is not true")
    if not bool(review.get("control_limits_reviewed")):
        failures.append("control_limits_reviewed is not true")
    if not bool(review.get("gripper_startup_behavior_reviewed")):
        failures.append("gripper_startup_behavior_reviewed is not true")
    if not bool(review.get("gripper_no_software_stop_acknowledged")):
        failures.append("gripper_no_software_stop_acknowledged is not true")
    if not bool(review.get("physical_emergency_stop_verified")):
        failures.append("physical_emergency_stop_verified is not true")
    if not bool(review.get("runtime_timing_reviewed")):
        failures.append("runtime_timing_reviewed is not true")
    if deployment_method != "baseline" and not bool(
        review.get("baseline_bounded_canary_passed")
    ):
        failures.append("baseline_bounded_canary_passed is not true")
    tracking = contract.hardware["robot"]["control"]["tracking_error_guard"]
    if not bool(tracking.get("enabled")):
        failures.append("tracking_error_guard.enabled is not true")
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
