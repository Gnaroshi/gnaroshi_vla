import ast
import json
from pathlib import Path

import numpy as np
import pytest

from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    load_deployment_contract,
    require_live_authorization,
    sha256_directory,
    sha256_file,
)
from architectures.simvla.adapters.latentloop_real_deploy.controller import (
    convert_model_action,
    encode_robot_state,
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
        "real_action_transformer": updater / "real_action_transformer.pt",
        "condition_updater": updater / "condition_updater.pt",
        "generation_updater": updater / "generation_updater.pt",
    }
    for index, path in enumerate(files.values()):
        path.write_bytes(f"artifact-{index}".encode("ascii"))
    (processor / "processor_config.json").write_text("{}", encoding="utf-8")
    identity = "real-simvla-checkpoint-sha256"
    payload = {
        "schema_version": 2,
        "deployment_id": "test-deployment",
        "simvla_upstream_commit": "32700d0ad8991996e123e4b685abe370ce6e9aab",
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
            "official_base_model_identity": "official-test-sha256",
            "real_baseline_identity": identity,
            "condition_source_real_baseline_identity": identity,
            "generation_source_real_baseline_identity": identity,
        },
        "policy": {
            "action_mode": "libero_joint",
            "proprio_dim": 8,
            "action_dim": 7,
            "condition_tokens": 122,
            "condition_dim": 960,
            "model_image_size": 384,
            "client_resize_size": 224,
            "action_horizon": 10,
            "execution_horizon": 5,
            "flow_steps": 10,
            "condition_refresh_interval": 2,
            "generation_full_evaluations": 3,
            "control_protocol": "fresh_h10_execute_r5",
            "generation_condition_coupling": "uncoupled_zero_code",
            "deterministic_action_noise": True,
        },
        "state": {
            "encoding": "opposed_finger_positions",
            "tcp_orientation": "axis_angle_radians",
            "gripper_max_opening_m": 0.04,
            "preflight_vector": [0.4, 0, 0.3, 3.14, 0, 0, 0.04, -0.04],
        },
        "action": {
            "representation": "normalized_delta_pose",
            "translation_scale_m": 0.02,
            "rotation_scale_rad": 0.05,
            "clip_abs": 1.0,
            "model_positive_gripper_means": "open",
        },
        "hardware": {
            "robot": {
                "type": "ur5e_robotiq_2f85",
                "ip": "192.0.2.1",
                "home_pose": [3.14, -1.57, 1.57, -1.57, -1.57, -1.57, 0],
                "workspace_m": {"min": [0.2, -0.6, 0.05], "max": [0.8, 0.6, 0.8]},
                "control": {
                    "home_move_duration_s": 3,
                    "home_move_hz": 60,
                    "acceleration": 0.5,
                    "velocity": 0.5,
                    "servoj_time_s": 0.002,
                    "servoj_lookahead_s": 0.2,
                    "servoj_gain": 100,
                },
                "gripper": {
                    "open_position_threshold": 0.1,
                    "speed": 255,
                    "force": 10,
                    "min_command_delta": 3,
                    "min_command_period_s": 0.05,
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
            "instructions": ["test instruction"],
            "results_directory": str(tmp_path / "results"),
            "camera_serials_file": str(tmp_path / "camera_serials.json"),
            "save_rollout_media": True,
        },
        "safety_review": {
            "live_authorized": False,
            "model_preflight_passed": False,
            "read_only_profile_passed": False,
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
    assert result["byte_identical_pairs"] == 5


def test_contract_verifies_every_artifact_and_fixed_policy(tmp_path):
    contract = load_deployment_contract(_manifest(tmp_path))
    assert contract.policy["action_horizon"] == 10
    assert contract.policy["execution_horizon"] == 5
    assert contract.policy["condition_refresh_interval"] == 2
    assert contract.policy["generation_full_evaluations"] == 3
    assert set(contract.artifacts) == {
        "official_base_model_weights",
        "norm_stats",
        "real_action_transformer",
        "condition_updater",
        "generation_updater",
        "official_base_model_directory",
        "processor_directory",
    }


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
        require_live_authorization(contract)


def test_live_requires_manifest_and_two_environment_confirmations(tmp_path, monkeypatch):
    contract = load_deployment_contract(_manifest(tmp_path))
    with pytest.raises(PermissionError):
        require_live_authorization(contract)
    contract.payload["safety_review"].update(
        {
            "live_authorized": True,
            "model_preflight_passed": True,
            "read_only_profile_passed": True,
            "approved_by": "reviewer",
            "approved_at": "2026-09-04T00:00:00+09:00",
        }
    )
    monkeypatch.setenv("SIMVLA_REAL_LIVE_RUN", "1")
    monkeypatch.setenv("SIMVLA_REAL_DEPLOYMENT_ID", contract.deployment_id)
    require_live_authorization(contract)


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
