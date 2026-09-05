import ast
import argparse
import collections
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision.transforms import InterpolationMode

from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    load_deployment_contract,
    require_hardware_configuration,
    require_live_authorization,
    runtime_source_identity,
    sha256_directory,
    sha256_file,
    sha256_json,
)
from architectures.simvla.adapters.latentloop_real_deploy.controller import (
    SimVLARealController,
    convert_model_action,
    encode_robot_state,
)
from architectures.simvla.adapters.latentloop_real_deploy.hardware import (
    SafeUR5eDeployEnv,
    legacy_deploy,
    rebase_incremental_target_on_actual_tcp,
    tcp_tracking_error,
)
from architectures.simvla.adapters.real_world_training.dataset import (
    build_real_image_transform,
    jpeg_roundtrip,
    resize_with_pad,
)
from architectures.simvla.adapters.real_world_training.build_deployment_bundle import (
    build_bundle,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    resize_with_pad_uint8,
)
from architectures.simvla.adapters.latentloop_real_deploy.source_lock import (
    verify_source_snapshots,
)


ROOT = Path(__file__).resolve().parents[2]


def _manifest(tmp_path: Path) -> Path:
    base = tmp_path / "base_model"
    processor = tmp_path / "processor"
    updater = tmp_path / "updaters"
    stats = tmp_path / "norm_stats"
    for directory in (base, processor, updater, stats):
        directory.mkdir()
    files = {
        "official_base_model_weights": base / "model.safetensors",
        "norm_stats": stats / "real_norm.json",
        "dataset_manifest": stats / "dataset_manifest.json",
        "condition_cache_manifest": stats / "condition_cache_manifest.json",
        "condition_cache_attestation": stats / "condition_cache_attestation.json",
        "real_action_transformer": updater / "real_action_transformer.pt",
        "condition_updater": updater / "condition_updater.pt",
        "generation_updater": updater / "generation_updater.pt",
        "coupled_generation_updater": updater / "coupled_generation_updater.pt",
    }
    for index, path in enumerate(files.values()):
        path.write_bytes(f"artifact-{index}".encode("ascii"))
    (processor / "processor_config.json").write_text("{}", encoding="utf-8")
    official_identity = sha256_file(files["official_base_model_weights"])
    episode_ids = [f"episode_{index:02d}" for index in range(40)]
    train_ids = episode_ids[:32]
    validation_ids = episode_ids[32:]
    instruction = "test instruction"
    episodes = [
        {"episode_id": episode_id, "sha256": f"{index:064x}"}
        for index, episode_id in enumerate(episode_ids)
    ]
    dataset_identity = sha256_json(
        {
            "schema": "simvla_real_hdf5_v3",
            "episode_sha256": {
                item["episode_id"]: item["sha256"] for item in episodes
            },
            "split_seed": 20260904,
            "train": train_ids,
            "validation": validation_ids,
            "instruction": instruction,
        }
    )
    dataset_payload = {
        "schema_version": "simvla_real_hdf5_v3",
        "verdict": "REAL_DATASET_CONTRACT_PASS",
        "dataset_id": "stackcupanddoll",
        "dataset_identity_sha256": dataset_identity,
        "source": {
            "path": "/reviewed/raw/stackcupanddoll",
            "format": "3dflow_teleoperation_pickle",
            "control_pose_channels_used": False,
            "tcp_pose_source": "ee_pos_quat interpreted as xyz+rotation_vector",
            "observation_action_alignment": (
                "frame_t stores observation_t and command_t before env.step(command_t)"
            ),
            "gripper_command_source": (
                "current_frame.control[6] recorded with observation_t before "
                "env.step(command_t); 0=open and 1=close continuous target"
            ),
        },
        "instruction": instruction,
        "target_hz": 15.0,
        "sampling_mode": "native_capture_order_with_timing_valid_windows",
        "action_horizon": 10,
        "execution_horizon": 5,
        "image_contract": {
            "source_order": ["base_rgb", "wrist_rgb"],
            "source_orientation": "as_captured_no_flip",
            "storage": "JPEG quality=95 subsampling=0",
            "model_preprocessing": "resize_with_pad_224_then_bicubic_384_imagenet_norm",
        },
        "state_contract": {
            "representation": "tcp_xyz_rotvec_plus_opposed_finger_positions",
            "condition_updater_rotation_delta": (
                "current rotvec mapped to equivalent 2pi branch nearest previous rotvec"
            ),
            "gripper_max_opening_m": 0.04,
        },
        "action_contract": {
            "representation": (
                "inv(T_current)@T_next local xyz plus xyz_euler plus "
                "continuous gripper target"
            ),
            "pose_label_source": "measured transition observation_t to observation_t+1",
            "gripper_label_source": "1 - 2 * command_t stored in current frame",
            "translation_scale_m": 0.02,
            "rotation_scale_rad": 0.05,
            "clip_abs": 1.0,
            "model_positive_gripper_means": "open",
        },
        "splits": {"train": train_ids, "validation": validation_ids},
        "split_seed": 20260904,
        "episodes": episodes,
        "norm_stats": {"sha256": sha256_file(files["norm_stats"])},
        "audit": {
            "pose_clip_transition_fraction": 0.0,
            "max_pose_roundtrip_matrix_abs": 1e-9,
            "gripper_label_alignment_error_count": 0,
            "earliest_gripper_close_frame": 79,
            "minimum_gripper_switch_interval": 80,
            "training_samples": 40,
        },
    }
    files["dataset_manifest"].write_text(
        json.dumps(dataset_payload), encoding="utf-8"
    )
    records_sha = "a" * 64
    cache_identity = sha256_json(
        {
            "schema": "simvla_real_condition_cache_v2",
            "dataset_identity_sha256": dataset_identity,
            "official_model_weights_sha256": official_identity,
            "records_sha256": records_sha,
            "array_sha256": {
                name: f"{index + 100:064x}"
                for index, name in enumerate(
                    ("condition.npy", "proprio.npy", "action.npy", "complete.npy")
                )
            },
            "preprocessing": "resize_with_pad_224_then_bicubic_384_imagenet_norm",
        }
    )
    files["condition_cache_manifest"].write_text(
        json.dumps(
            {
                "schema_version": "simvla_real_condition_cache_v2",
                "verdict": "REAL_CONDITION_CACHE_PASS",
                "count": 40,
                "shape": [40, 122, 960],
                "dtype": "float32",
                "dataset_identity_sha256": dataset_identity,
                "dataset_manifest_sha256": sha256_file(files["dataset_manifest"]),
                "condition_cache_identity_sha256": cache_identity,
                "records_sha256": records_sha,
                "official_base": {
                    "model_weights_sha256": official_identity,
                    "action_mode": "libero_joint",
                    "action_horizon": 10,
                },
                "exact_loading": {
                    "verdict": "EXACT_OFFICIAL_INITIALIZATION_PASS",
                    "loading_info": {
                        "missing_keys": [],
                        "unexpected_keys": [],
                        "mismatched_keys": [],
                        "error_msgs": [],
                    },
                    "real_action_overlay": None,
                },
                "arrays": {
                    name: {"sha256": f"{index + 100:064x}", "size_bytes": 1}
                    for index, name in enumerate(
                        ("condition.npy", "proprio.npy", "action.npy", "complete.npy")
                    )
                },
                "token_layout": {
                    "valid_mask": [[True] * 122 for _ in range(4)],
                },
            }
        ),
        encoding="utf-8",
    )
    attestation_payload = {
        "schema_version": "simvla_real_condition_cache_attestation_v2",
        "verdict": "REAL_CONDITION_CACHE_ATTESTATION_PASS",
        "cache_manifest_sha256": sha256_file(files["condition_cache_manifest"]),
        "condition_array_sha256": f"{100:064x}",
        "records_sha256": records_sha,
        "condition_cache_identity_sha256": cache_identity,
        "dataset_manifest_sha256": sha256_file(files["dataset_manifest"]),
        "dataset_identity_sha256": dataset_identity,
        "official_model_weights_sha256": official_identity,
        "processor_directory_sha256": sha256_directory(processor),
        "norm_stats_sha256": sha256_file(files["norm_stats"]),
        "verifier_source_sha256": sha256_file(
            ROOT
            / "architectures/simvla/adapters/real_world_training/verify_condition_cache.py"
        ),
        "selection_strategy": "lower_median_valid_h10_query_per_episode",
        "cache_file_stats": {
            name: {"size_bytes": 1, "mtime_ns": 1}
            for name in (
                "manifest.json",
                "records.json",
                "condition.npy",
                "proprio.npy",
                "action.npy",
                "complete.npy",
            )
        },
        "samples": [
            {
                "episode_id": episode_id,
                "split": "train" if episode_id in train_ids else "validation",
                "frame_index": 0,
                "cache_index": index,
                "bitwise_equal": True,
                "max_abs_difference": 0.0,
                "mean_abs_difference": 0.0,
                "cosine_similarity": 1.0,
            }
            for index, episode_id in enumerate(episode_ids)
        ],
        "comparison": {
            "sample_count": 40,
            "exact_equal_count": 40,
            "all_samples_bitwise_equal": True,
            "max_abs_difference": 0.0,
            "mean_abs_difference": 0.0,
            "minimum_cosine_similarity": 1.0,
        },
        "model_loading": {
            "verdict": "EXACT_OFFICIAL_INITIALIZATION_PASS",
            "loading_info": {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
            "action_transformer_reinitialized": False,
            "real_action_overlay": None,
        },
        "environment": {
            "python": "test",
            "torch": "test",
            "cuda_runtime": None,
            "device_type": "cpu",
            "device_name": "cpu",
        },
    }
    attestation_payload["attestation_identity_sha256"] = sha256_json(
        attestation_payload
    )
    files["condition_cache_attestation"].write_text(
        json.dumps(attestation_payload), encoding="utf-8"
    )
    identity = sha256_file(files["real_action_transformer"])
    payload = {
        "schema_version": 4,
        "deployment_id": "test-deployment",
        "simvla_upstream_commit": "32700d0ad8991996e123e4b685abe370ce6e9aab",
        "runtime_source_identity_sha256": runtime_source_identity()[
            "combined_sha256"
        ],
        "artifacts": {
            "official_base_model_directory": {
                "path": str(base),
                "sha256": sha256_directory(base),
            },
            "processor_directory": {
                "path": str(processor),
                "sha256": sha256_directory(processor),
            },
            **{
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in files.items()
            },
        },
        "pairing": {
            "official_base_model_identity": official_identity,
            "real_baseline_identity": identity,
            "norm_stats_identity": sha256_file(files["norm_stats"]),
            "dataset_identity": dataset_identity,
            "condition_cache_identity": cache_identity,
            "condition_cache_attestation_identity": attestation_payload[
                "attestation_identity_sha256"
            ],
            "condition_source_real_baseline_identity": identity,
            "generation_source_real_baseline_identity": identity,
            "coupled_source_real_baseline_identity": identity,
            "coupled_parent_generation_identity": sha256_file(
                files["generation_updater"]
            ),
            "coupled_condition_updater_identity": sha256_file(
                files["condition_updater"]
            ),
            "coupled_generation_identity": sha256_file(
                files["coupled_generation_updater"]
            ),
        },
        "policy": {
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
            "control_protocol": "fresh_h10_execute_r5",
            "generation_condition_coupling": "condition_delta_code",
            "deterministic_action_noise": True,
        },
        "state": {
            "encoding": "opposed_finger_positions",
            "tcp_orientation": "axis_angle_radians",
            "gripper_max_opening_m": 0.04,
            "preflight_robot_state": {
                "pose6d_euler_xyz": [0.4, 0, 0.3, 3.14, 0, 0],
                "tcp_rotvec": [3.14, 0, 0],
                "gripper_open_state": 1.0,
                "gripper_position_normalized": 0.0,
            },
        },
        "action": {
            "representation": "normalized_delta_pose",
            "translation_scale_m": 0.02,
            "rotation_scale_rad": 0.05,
            "clip_abs": 1.0,
            "gripper_representation": "continuous_normalized_position",
            "model_positive_gripper_means": "open",
        },
        "hardware": {
            "robot": {
                "type": "ur5e_robotiq_2f85",
                "ip": "192.0.2.1",
                "home_pose": [3.14, -1.57, 1.57, -1.57, -1.57, -1.57, 0],
                "home_pose_source": "unit-test reviewed fixture",
                "workspace_m": {"min": [0.2, -0.6, 0.05], "max": [0.8, 0.6, 0.8]},
                "workspace_source": "unit-test reviewed fixture",
                "control": {
                    "home_move_duration_s": 3,
                    "home_move_hz": 60,
                    "acceleration": 0.5,
                    "velocity": 0.5,
                    "servoj_time_s": 0.002,
                    "servoj_lookahead_s": 0.2,
                    "servoj_gain": 100,
                    "tracking_error_guard": {
                        "enabled": False,
                        "max_translation_error_m": None,
                        "max_rotation_error_rad": None,
                    },
                },
                "gripper": {
                    "command_mode": "continuous_position_with_crossing_lock",
                    "open_position_threshold": 0.1,
                    "speed": 255,
                    "force": 10,
                    "min_command_delta": 3,
                    "min_command_period_s": 0.05,
                    "startup_force_open_steps": 50,
                    "status_lock_steps": 10,
                },
            },
            "cameras": {
                "model_input_order": ["exterior", "wrist"],
                "exterior": {"serial": "exterior-test"},
                "wrist": {"serial": "wrist-test"},
                "width": 640,
                "height": 480,
                "fps": 60,
            },
        },
        "runtime": {
            "control_frequency_hz": 15,
            "max_steps": 100,
            "warmup_steps": 3,
            "num_rollouts_per_instruction": 1,
            "seed": 42,
            "instructions": [instruction],
            "results_directory": str(tmp_path / "results"),
            "camera_serials_file": str(tmp_path / "camera_serials.json"),
            "save_rollout_media": True,
        },
        "safety_review": {
            "live_authorized": False,
            "model_preflight_passed": False,
            "read_only_profile_passed": False,
            "hardware_configuration_reviewed": False,
            "camera_role_mapping_verified": False,
            "task_home_pose_verified": False,
            "workspace_bounds_verified": False,
            "control_limits_reviewed": False,
            "gripper_startup_behavior_reviewed": False,
            "gripper_no_software_stop_acknowledged": False,
            "physical_emergency_stop_verified": False,
            "runtime_timing_reviewed": False,
            "baseline_bounded_canary_passed": False,
            "approved_by": "",
            "approved_at": "",
        },
    }
    path = tmp_path / "deployment_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_tracked_source_snapshots_are_unchanged():
    result = verify_source_snapshots()
    assert result["verdict"] == "SOURCE_PREFLIGHT_PASS"
    assert result["verified_files"] == 12
    assert result["byte_identical_pairs"] == 0
    assert result["runtime_source_identity_sha256"] == runtime_source_identity()[
        "combined_sha256"
    ]


def test_contract_rejects_runtime_source_identity_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["runtime_source_identity_sha256"] = "f" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime source identity mismatch"):
        load_deployment_contract(manifest)


def test_contract_rejects_unreviewed_raw_channel_semantics(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    dataset_manifest = Path(payload["artifacts"]["dataset_manifest"]["path"])
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    dataset["source"]["tcp_pose_source"] = "control[:6]"
    dataset_manifest.write_text(json.dumps(dataset), encoding="utf-8")
    payload["artifacts"]["dataset_manifest"]["sha256"] = sha256_file(
        dataset_manifest
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="TCP source"):
        load_deployment_contract(manifest)


def test_contract_rejects_legacy_gripper_alignment_dataset(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    dataset_manifest = Path(payload["artifacts"]["dataset_manifest"]["path"])
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    dataset["schema_version"] = "simvla_real_hdf5_v2"
    dataset_manifest.write_text(json.dumps(dataset), encoding="utf-8")
    payload["artifacts"]["dataset_manifest"]["sha256"] = sha256_file(
        dataset_manifest
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported real dataset schema"):
        load_deployment_contract(manifest)


def test_contract_rejects_legacy_condition_cache(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    cache_manifest = Path(payload["artifacts"]["condition_cache_manifest"]["path"])
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    cache["schema_version"] = "simvla_real_condition_cache_v1"
    cache_manifest.write_text(json.dumps(cache), encoding="utf-8")
    payload["artifacts"]["condition_cache_manifest"]["sha256"] = sha256_file(
        cache_manifest
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported real Condition cache schema"):
        load_deployment_contract(manifest)


def test_contract_verifies_every_artifact_and_fixed_policy(tmp_path):
    contract = load_deployment_contract(_manifest(tmp_path))
    assert contract.policy["action_horizon"] == 10
    assert contract.policy["execution_horizon"] == 5
    assert contract.policy["condition_refresh_interval"] == 2
    assert contract.policy["generation_full_evaluations"] == 3
    assert set(contract.artifacts) == {
        "official_base_model_weights",
        "norm_stats",
        "dataset_manifest",
        "condition_cache_manifest",
        "condition_cache_attestation",
        "real_action_transformer",
        "condition_updater",
        "generation_updater",
        "coupled_generation_updater",
        "official_base_model_directory",
        "processor_directory",
    }


def test_contract_rejects_reversed_model_gripper_sign(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["action"]["model_positive_gripper_means"] = "close"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="action.model_positive_gripper_means must be open",
    ):
        load_deployment_contract(manifest)


def test_contract_rejects_non_exact_condition_cache_attestation(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    attestation_path = Path(
        payload["artifacts"]["condition_cache_attestation"]["path"]
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["samples"][0]["bitwise_equal"] = False
    attestation["comparison"]["exact_equal_count"] = 39
    attestation["comparison"]["all_samples_bitwise_equal"] = False
    attestation.pop("attestation_identity_sha256")
    attestation["attestation_identity_sha256"] = sha256_json(attestation)
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    payload["artifacts"]["condition_cache_attestation"]["sha256"] = sha256_file(
        attestation_path
    )
    payload["pairing"]["condition_cache_attestation_identity"] = attestation[
        "attestation_identity_sha256"
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-identical query"):
        load_deployment_contract(manifest)


def test_deployment_bundle_is_relocatable_and_invalidates_authorization(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_manifest = _manifest(source_root)
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_payload["safety_review"].update(
        {
            "live_authorized": True,
            "model_preflight_passed": True,
            "read_only_profile_passed": True,
            "hardware_configuration_reviewed": True,
            "camera_role_mapping_verified": True,
            "task_home_pose_verified": True,
            "workspace_bounds_verified": True,
            "control_limits_reviewed": True,
            "gripper_startup_behavior_reviewed": True,
            "gripper_no_software_stop_acknowledged": True,
            "physical_emergency_stop_verified": True,
            "runtime_timing_reviewed": True,
            "baseline_bounded_canary_passed": True,
            "approved_by": "test",
            "approved_at": "2026-09-05T00:00:00+09:00",
        }
    )
    source_manifest.write_text(json.dumps(source_payload), encoding="utf-8")
    output = tmp_path / "relocated" / "bundle"
    result = build_bundle(
        argparse.Namespace(manifest=str(source_manifest), output=str(output))
    )
    assert result["verdict"] == "REAL_SIMVLA_DEPLOYMENT_BUNDLE_PASS"
    assert not result["live_authorized"]
    contract = load_deployment_contract(output / "deployment_manifest.json")
    assert not contract.live_authorized
    assert all(
        str(spec["path"]).startswith("./")
        for spec in contract.payload["artifacts"].values()
    )
    from architectures.simvla.adapters.latentloop_real_deploy.hardware import (
        build_deploy_config,
    )

    cfg = build_deploy_config(contract)
    assert Path(cfg.results_dir) == output / "runtime_results"
    assert Path(cfg.camera_serials_file) == output / "runtime_results/camera_serials.json"


def test_contract_rejects_artifact_hash_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"]["norm_stats"]["sha256"] = "f" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_deployment_contract(manifest)


def test_example_manifest_is_valid_but_cannot_authorize_live(monkeypatch):
    example = ROOT / "artifacts/simvla/real_world/deployment_manifest.example.json"
    contract = load_deployment_contract(example, verify_artifacts=False)
    assert not contract.live_authorized
    with pytest.raises(PermissionError):
        require_live_authorization(contract, deployment_method="baseline")


@pytest.mark.parametrize("key", ["home_pose", "workspace_m"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_hardware_contract_rejects_nonfinite_motion_bounds(tmp_path, key, value):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    robot = payload["hardware"]["robot"]
    if key == "home_pose":
        robot[key][0] = value
    else:
        robot[key]["min"][0] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite"):
        load_deployment_contract(path, verify_artifacts=False)


def test_hardware_contract_rejects_duplicate_camera_roles(tmp_path):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    cameras = payload["hardware"]["cameras"]
    cameras["wrist"]["serial"] = cameras["exterior"]["serial"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different camera serials"):
        load_deployment_contract(path, verify_artifacts=False)


def test_live_requires_manifest_and_two_environment_confirmations(tmp_path, monkeypatch):
    contract = load_deployment_contract(_manifest(tmp_path))
    with pytest.raises(PermissionError):
        require_live_authorization(contract, deployment_method="baseline")
    contract.payload["safety_review"].update(
        {
            "live_authorized": True,
            "model_preflight_passed": True,
            "read_only_profile_passed": True,
            "hardware_configuration_reviewed": True,
            "camera_role_mapping_verified": True,
            "task_home_pose_verified": True,
            "workspace_bounds_verified": True,
            "control_limits_reviewed": True,
            "gripper_startup_behavior_reviewed": True,
            "gripper_no_software_stop_acknowledged": True,
            "physical_emergency_stop_verified": True,
            "runtime_timing_reviewed": True,
            "approved_by": "reviewer",
            "approved_at": "2026-09-04T00:00:00+09:00",
        }
    )
    contract.payload["hardware"]["robot"]["control"]["tracking_error_guard"].update(
        {
            "enabled": True,
            "max_translation_error_m": 0.01,
            "max_rotation_error_rad": 0.1,
        }
    )
    monkeypatch.setenv("SIMVLA_REAL_LIVE_RUN", "1")
    monkeypatch.setenv("SIMVLA_REAL_DEPLOYMENT_ID", contract.deployment_id)
    require_live_authorization(contract, deployment_method="baseline")

    with pytest.raises(PermissionError, match="baseline_bounded_canary_passed"):
        require_live_authorization(contract, deployment_method="latentloop")
    contract.payload["safety_review"]["baseline_bounded_canary_passed"] = True
    require_live_authorization(contract, deployment_method="latentloop")


def test_hardware_connection_requires_explicit_review(tmp_path):
    contract = load_deployment_contract(_manifest(tmp_path))
    with pytest.raises(PermissionError, match="hardware_configuration_reviewed"):
        require_hardware_configuration(contract)


def test_hardware_connection_requires_independent_role_and_motion_reviews(tmp_path):
    contract = load_deployment_contract(_manifest(tmp_path))
    contract.payload["safety_review"]["hardware_configuration_reviewed"] = True
    with pytest.raises(PermissionError, match="camera_role_mapping_verified"):
        require_hardware_configuration(contract)
    contract.payload["safety_review"]["camera_role_mapping_verified"] = True
    with pytest.raises(PermissionError, match="task_home_pose_verified"):
        require_hardware_configuration(contract)
    contract.payload["safety_review"]["task_home_pose_verified"] = True
    with pytest.raises(PermissionError, match="workspace_bounds_verified"):
        require_hardware_configuration(contract)


def test_real_state_and_action_conventions_are_explicit():
    state = encode_robot_state(
        {
            "pose6d": np.arange(6, dtype=np.float32),
            "tcp_rotvec": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "gripper_open_state": np.asarray([1], dtype=np.float32),
            "gripper_position": np.asarray([0.25], dtype=np.float32),
        },
        "opposed_finger_positions",
        "axis_angle_radians",
        0.04,
    )
    np.testing.assert_allclose(state, [0, 1, 2, 0.1, 0.2, 0.3, 0.03, -0.03])
    pos, rotation, gripper = convert_model_action(
        np.asarray([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0]),
        clip_abs=1.0,
        positive_gripper_means="open",
    )
    np.testing.assert_allclose(pos, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(rotation, [0.4, -0.5, 0.6])
    assert gripper == 1.0


def test_live_hardware_adapter_preserves_tcp_rotation_vector(monkeypatch):
    pose6d_euler = np.asarray(
        [0.45, -0.20, 0.25, 0.25, -0.35, 0.15], dtype=np.float64
    )

    def fake_robot_state(_self):
        return {
            "pose6d": pose6d_euler.copy(),
            "gripper_open_state": np.asarray([1.0], dtype=np.float32),
            "gripper_position": np.asarray([0.0], dtype=np.float32),
        }

    monkeypatch.setattr(legacy_deploy.UR5eDeployEnv, "get_robot_state", fake_robot_state)
    environment = object.__new__(SafeUR5eDeployEnv)
    state = environment.get_robot_state()
    expected = legacy_deploy._pose6d_to_ur_tcp(pose6d_euler)[3:]
    np.testing.assert_allclose(state["tcp_rotvec"], expected, rtol=0.0, atol=1e-7)


def test_emergency_stop_prefers_servo_stop_and_clears_targets():
    calls = []

    class Control:
        def servoStop(self):
            calls.append("servoStop")
            return True

        def stopJ(self, _acceleration):
            calls.append("stopJ")
            return True

    environment = object.__new__(SafeUR5eDeployEnv)
    environment.rtde_ctrl = Control()
    environment.cfg = SimpleNamespace(arm_acceleration=0.5)
    environment._command_lock = threading.RLock()
    environment._motion_abort = threading.Event()
    environment._policy_commands_armed = True
    environment._last_commanded_tcp = np.ones(6)
    environment._upstream_reference_target = np.ones(6)
    report = environment.emergency_stop()
    assert report["stopped"]
    assert calls == ["servoStop"]
    assert environment._last_commanded_tcp is None
    assert environment._upstream_reference_target is None


def test_emergency_stop_falls_back_to_stop_j():
    calls = []

    class Control:
        def servoStop(self):
            calls.append("servoStop")
            raise RuntimeError("servo mode unavailable")

        def stopJ(self, acceleration):
            calls.append(("stopJ", acceleration))
            return True

    environment = object.__new__(SafeUR5eDeployEnv)
    environment.rtde_ctrl = Control()
    environment.cfg = SimpleNamespace(arm_acceleration=0.4)
    environment._command_lock = threading.RLock()
    environment._motion_abort = threading.Event()
    environment._policy_commands_armed = True
    environment._last_commanded_tcp = None
    environment._upstream_reference_target = None
    report = environment.emergency_stop()
    assert report["stopped"]
    assert report["stop_j_fallback_called"]
    assert calls == ["servoStop", ("stopJ", 0.4)]


def test_gui_stop_is_immediate_and_does_not_select_home_move_branch():
    from architectures.simvla.adapters.latentloop_real_deploy.deploy_gui import (
        SimVLADeployGuiApp,
    )

    calls = []

    class Environment:
        def disarm_policy_commands(self):
            calls.append("disarm")

        def emergency_stop(self):
            calls.append("emergency_stop")
            return {"stopped": True}

        def cancel_pending_policy_step(self):
            calls.append("cancel_pending_step")

    app = object.__new__(SimVLADeployGuiApp)
    app.env = Environment()
    app.emergency_stop_reports = []
    app.current_events = {
        name: threading.Event()
        for name in ("stop", "success", "failure", "retry")
    }
    app.set_status = lambda message: calls.append(("status", message))
    app.set_run_state = lambda message, color: calls.append(
        ("state", message, color)
    )

    app.signal_current("stop")

    assert calls[:2] == ["emergency_stop", "cancel_pending_step"]
    assert app.current_events["retry"].is_set()
    assert not app.current_events["stop"].is_set()
    assert app.emergency_stop_reports[0]["reason"] == "operator_stop"


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_gui_outcome_stops_arm_and_preserves_outcome_event(outcome):
    from architectures.simvla.adapters.latentloop_real_deploy.deploy_gui import (
        SimVLADeployGuiApp,
    )

    calls = []

    class Environment:
        def disarm_policy_commands(self):
            calls.append("disarm")

        def emergency_stop(self):
            calls.append("emergency_stop")
            return {"stopped": True}

        def cancel_pending_policy_step(self):
            calls.append("cancel_pending_step")

    app = object.__new__(SimVLADeployGuiApp)
    app.env = Environment()
    app.emergency_stop_reports = []
    app.current_events = {
        name: threading.Event()
        for name in ("stop", "success", "failure", "retry")
    }
    app.set_status = lambda message: calls.append(("status", message))
    app.set_run_state = lambda message, color: calls.append(("state", message, color))

    app.signal_current(outcome)

    assert calls[:2] == ["emergency_stop", "cancel_pending_step"]
    assert app.current_events[outcome].is_set()
    assert not app.current_events["retry"].is_set()
    assert app.emergency_stop_reports[0]["reason"] == f"operator_{outcome}"


def test_gui_close_stops_before_closing_without_parent_home_event():
    from architectures.simvla.adapters.latentloop_real_deploy.deploy_gui import (
        SimVLADeployGuiApp,
    )

    calls = []

    class Environment:
        def disarm_policy_commands(self):
            calls.append("disarm")

        def emergency_stop(self):
            calls.append("emergency_stop")
            return {"stopped": True}

        def close(self):
            calls.append("close")

    class Controller:
        def write_runtime_summary(self):
            calls.append("summary")

    app = object.__new__(SimVLADeployGuiApp)
    app.env = Environment()
    app.simvla_controller = Controller()
    app.emergency_stop_reports = []
    app.current_events = {
        name: threading.Event()
        for name in ("stop", "success", "failure", "retry")
    }
    app.rollout_thread = None
    app.preview = SimpleNamespace(stop=lambda: calls.append("preview_stop"))
    app.save_results = lambda: calls.append("save")
    app.preview_only_cameras = []
    app.root = SimpleNamespace(destroy=lambda: calls.append("destroy"))

    app.on_close()

    assert calls[0] == "emergency_stop"
    assert app.current_events["retry"].is_set()
    assert not app.current_events["stop"].is_set()
    assert calls.index("emergency_stop") < calls.index("close")


def test_live_policy_commands_are_disarmed_until_a_rollout_is_armed():
    environment = object.__new__(SafeUR5eDeployEnv)
    environment._command_lock = threading.RLock()
    environment._motion_abort = threading.Event()
    environment._policy_commands_armed = False
    environment._cancel_next_disarmed_policy_step = False

    with pytest.raises(RuntimeError, match="no live rollout is armed"):
        environment.step(np.zeros(6), 0.0)

    environment.arm_policy_commands()
    assert environment._policy_commands_armed is True
    environment.disarm_policy_commands()
    assert environment._policy_commands_armed is False


def test_operator_cancellation_consumes_one_late_policy_step_without_command():
    from architectures.simvla.adapters.latentloop_real_deploy.hardware import (
        POLICY_STEP_CANCELLED,
    )

    environment = object.__new__(SafeUR5eDeployEnv)
    environment._command_lock = threading.RLock()
    environment._motion_abort = threading.Event()
    environment._policy_commands_armed = True
    environment._cancel_next_disarmed_policy_step = False
    environment.cancel_pending_policy_step()

    result = environment.step(np.zeros(6), 0.0)
    assert result is POLICY_STEP_CANCELLED
    with pytest.raises(RuntimeError, match="no live rollout is armed"):
        environment.step(np.zeros(6), 0.0)


def test_abort_latch_rejects_late_policy_step_without_touching_hardware():
    from architectures.simvla.adapters.latentloop_real_deploy.hardware import (
        POLICY_STEP_CANCELLED,
    )

    environment = object.__new__(SafeUR5eDeployEnv)
    environment._command_lock = threading.RLock()
    environment._motion_abort = threading.Event()
    environment._motion_abort.set()
    environment._policy_commands_armed = True
    environment._cancel_next_disarmed_policy_step = True

    result = environment.step(np.zeros(6), 0.0)

    assert result is POLICY_STEP_CANCELLED
    assert environment._policy_commands_armed is False
    assert environment._cancel_next_disarmed_policy_step is False


def test_reviewed_home_motion_honors_abort_latch_before_next_servo_command():
    calls = []

    class Control:
        def initPeriod(self):
            return "period"

        def servoJ(self, *args):
            calls.append(("servoJ", args[0]))

        def waitPeriod(self, period):
            calls.append(("waitPeriod", period))
            environment._motion_abort.set()

    environment = object.__new__(SafeUR5eDeployEnv)
    environment.cfg = SimpleNamespace(
        home_pose=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
        home_move_duration=1.0,
        home_move_fps=10,
        arm_velocity=0.5,
        arm_acceleration=0.5,
        servoj_time=0.002,
        servoj_lookahead=0.2,
        servoj_gain=100,
    )
    environment.rtde_rec = SimpleNamespace(getActualQ=lambda: [0.0] * 6)
    environment.rtde_ctrl = Control()
    environment.gripper = SimpleNamespace(get_current_position=lambda: 0)
    environment._command_gripper_normalized = lambda value: calls.append(
        ("gripper", value)
    )
    environment._command_lock = threading.RLock()
    environment._motion_abort = threading.Event()
    environment._step_count = 0
    environment._gripper_lock_counter = 0
    environment._last_commanded_tcp = np.ones(6)
    environment._upstream_reference_target = np.ones(6)
    environment.last_tracking_error = {"translation_m": 0.0}

    completed = environment.move_to_home()

    assert completed is False
    assert sum(call[0] == "servoJ" for call in calls) == 1
    assert environment._last_commanded_tcp is None
    assert environment._upstream_reference_target is None


def test_real_gripper_preserves_continuous_targets_and_locks_only_crossings():
    moves = []

    class Gripper:
        def move(self, position, speed, force):
            moves.append((position, speed, force))

    environment = object.__new__(SafeUR5eDeployEnv)
    environment.cfg = SimpleNamespace(
        gripper_force_open_steps=50,
        gripper_status_lock_steps=10,
        gripper_min_delta=1,
        gripper_min_period_s=0.0,
        gripper_speed=100,
        gripper_force=50,
    )
    environment.gripper = Gripper()
    environment._step_count = 50
    environment._gripper_lock_counter = 0
    environment._gripper_status = None
    environment._last_gripper_pos = None
    environment._last_gripper_cmd_time = 0.0

    environment._command_gripper(0.5)
    environment._command_gripper(-0.5)
    environment._command_gripper(0.8)
    environment._command_gripper(-0.8)

    assert moves == [(64, 100, 50), (191, 100, 50), (230, 100, 50)]
    assert environment._gripper_status == "close"
    assert environment._gripper_lock_counter == 8


def test_real_gripper_force_open_period_overrides_early_close_target():
    moves = []
    environment = object.__new__(SafeUR5eDeployEnv)
    environment.cfg = SimpleNamespace(
        gripper_force_open_steps=50,
        gripper_status_lock_steps=10,
        gripper_min_delta=1,
        gripper_min_period_s=0.0,
        gripper_speed=100,
        gripper_force=50,
    )
    environment.gripper = SimpleNamespace(
        move=lambda position, speed, force: moves.append((position, speed, force))
    )
    environment._step_count = 0
    environment._gripper_lock_counter = 0
    environment._gripper_status = None
    environment._last_gripper_pos = None
    environment._last_gripper_cmd_time = 0.0

    environment._command_gripper(-1.0)

    assert moves == [(0, 100, 50)]
    assert environment._gripper_status == "open"


def test_tcp_tracking_error_uses_se3_rotation_distance():
    actual = np.asarray([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
    commanded = np.asarray([0.403, -0.004, 0.3, 0.0, 0.0, np.pi / 2])
    error = tcp_tracking_error(actual, commanded)
    assert error["translation_m"] == pytest.approx(0.005)
    assert error["rotation_rad"] == pytest.approx(np.pi / 2)


def test_local_delta_is_rebased_on_current_actual_tcp():
    previous_requested = np.asarray([0.45, -0.1, 0.25, 0.1, -0.2, 0.3])
    local_delta = legacy_deploy._6d_to_pose(
        np.asarray([0.002, -0.003, 0.001, 0.01, 0.02, -0.01])
    )
    requested = legacy_deploy.pose_to_6d(
        legacy_deploy._6d_to_pose(previous_requested) @ local_delta
    )
    actual_tcp = np.asarray([0.47, -0.08, 0.24, 0.2, -0.1, 0.05])
    rebased = rebase_incremental_target_on_actual_tcp(
        previous_requested, requested, actual_tcp
    )
    expected = legacy_deploy._6d_to_pose(
        legacy_deploy._ur_tcp_to_pose6d(actual_tcp)
    ) @ local_delta
    np.testing.assert_allclose(
        legacy_deploy._6d_to_pose(rebased), expected, atol=1e-8
    )


def test_real_gripper_output_is_saturated_without_masking_pose_overflow():
    _, _, open_gripper = convert_model_action(
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2]),
        clip_abs=1.0,
        positive_gripper_means="open",
    )
    _, _, closed_gripper = convert_model_action(
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4]),
        clip_abs=1.0,
        positive_gripper_means="open",
    )
    assert open_gripper == 1.0
    assert closed_gripper == -1.0

    with pytest.raises(RuntimeError, match="pose action exceeded"):
        convert_model_action(
            np.asarray([1.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            clip_abs=1.0,
            positive_gripper_means="open",
        )


def test_real_image_transform_is_shared_and_bicubic():
    transform = build_real_image_transform(training=False)
    assert transform.transforms[0].interpolation == InterpolationMode.BICUBIC
    assert transform.transforms[0].antialias is True


def test_training_and_runtime_resize_with_pad_are_byte_identical():
    image = np.arange(37 * 61 * 3, dtype=np.uint8).reshape(37, 61, 3)
    training = np.asarray(resize_with_pad(image, 224), dtype=np.uint8)
    runtime = resize_with_pad_uint8(image, 224)
    np.testing.assert_array_equal(training, runtime)


def test_real_jpeg_roundtrip_is_deterministic():
    image = np.arange(31 * 47 * 3, dtype=np.uint8).reshape(31, 47, 3)
    first = jpeg_roundtrip(image)
    second = jpeg_roundtrip(image.copy())
    np.testing.assert_array_equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == np.uint8


def test_controller_applies_training_jpeg_codec_only_on_policy_queries():
    class FakePolicy:
        def __init__(self):
            self.action_queue = collections.deque()
            self.seen = []

        def act(self, exterior, wrist, _state, _instruction):
            self.seen.append((exterior.copy(), wrist.copy()))
            return SimpleNamespace(
                action=np.zeros(7, dtype=np.float32), info={}
            )

    policy = FakePolicy()
    contract = SimpleNamespace(
        state={
            "encoding": "opposed_finger_positions",
            "tcp_orientation": "axis_angle_radians",
            "gripper_max_opening_m": 0.04,
        },
        action={
            "clip_abs": 1.0,
            "model_positive_gripper_means": "open",
        },
    )
    controller = object.__new__(SimVLARealController)
    controller.contract = contract
    controller.policy = policy
    controller.rollout_index = 0
    controller.deployment_method = "baseline"
    controller.step_records = []
    controller.session_dir = None
    image = np.arange(23 * 31 * 3, dtype=np.uint8).reshape(23, 31, 3)
    observation = {
        "color_image": [image, np.flip(image, axis=1).copy()],
        "robot_state": {
            "pose6d": np.zeros(6, dtype=np.float32),
            "tcp_rotvec": np.zeros(3, dtype=np.float32),
            "gripper_open_state": np.ones(1, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
        },
        "language_instruction": "test",
    }
    controller.forward(observation, record_step=False)
    np.testing.assert_array_equal(policy.seen[0][0], jpeg_roundtrip(image))

    policy.action_queue.append(torch.zeros(7))
    controller.forward(observation, record_step=False)
    np.testing.assert_array_equal(policy.seen[1][0], image)


def test_read_only_runtime_has_no_robot_control_or_command_calls():
    source_path = (
        ROOT
        / "architectures/simvla/adapters/latentloop_real_deploy/runtime.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "rtde_control" not in imported
    assert not called_attributes.intersection(
        {"servoJ", "servoStop", "moveJ", "moveL", "move", "activate"}
    )


def test_shell_defaults_to_source_preflight_and_never_defaults_live():
    source = (
        ROOT / "architectures/simvla/wrappers/deploy_latentloop_real.sh"
    ).read_text(encoding="utf-8")
    assert 'mode="${1:-source-preflight}"' in source
    assert "SIMVLA_REAL_LIVE_RUN=1" in source
    assert "SIMVLA_REAL_DEPLOYMENT_ID" in source


def test_latentloop_reset_preserves_generation_latency_counter():
    from architectures.simvla.adapters.latentloop_real_deploy.policy import (
        LatentLoopSimVLARealPolicy,
    )

    policy = object.__new__(LatentLoopSimVLARealPolicy)
    policy.reset()
    assert "generation_loop_ms" in policy.metrics.latencies


def test_condition_only_comparator_is_kc2_ng10():
    from architectures.simvla.adapters.latentloop_real_deploy.policy import (
        ConditionLoopSimVLARealPolicy,
    )

    policy = object.__new__(ConditionLoopSimVLARealPolicy)
    policy.k_c = 2
    policy.n_g = 10
    policy.query_index = 0
    policy.action_queue = collections.deque()
    policy.query_trace = []
    policy.metrics = SimpleNamespace(counters=collections.Counter())
    action_chunk = torch.zeros(1, 10, 7)

    def full_refresh(_batch, *, policy_query_index):
        policy.metrics.counters["num_full_vlm_calls"] += 1
        policy.metrics.counters["num_action_transformer_calls"] += 10
        return torch.zeros(1, 122, 960), action_chunk, policy_query_index

    def condition_update(_batch, *, age, policy_query_index):
        assert age == 1
        policy.metrics.counters["num_condition_updater_calls"] += 1
        policy.metrics.counters["num_action_transformer_calls"] += 10
        return torch.zeros(1, 122, 960), action_chunk, policy_query_index

    policy._full_refresh = full_refresh
    policy._v0_update = condition_update
    for _ in range(4):
        policy._refill_action_queue({})
        policy.action_queue.clear()

    assert policy.metrics.counters["num_policy_queries"] == 4
    assert policy.metrics.counters["num_full_vlm_calls"] == 2
    assert policy.metrics.counters["num_condition_updater_calls"] == 2
    assert policy.metrics.counters["num_action_transformer_calls"] == 40
    assert [item["source"] for item in policy.query_trace] == [
        "full_refresh",
        "condition_update",
        "full_refresh",
        "condition_update",
    ]
