"""Receive-only checks must neither require nor grant permission to move."""

import copy
import json
import sys
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from architectures.simvla.adapters.latentloop_real_deploy import cli, runtime
from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    require_live_authorization, require_sensor_configuration,
)


def contract():
    return SimpleNamespace(
        deployment_id="test", live_authorized=False,
        runtime={"instructions": ["test"], "warmup_steps": 0, "control_frequency_hz": 15},
        payload={"safety_review": {"camera_role_mapping_verified": True,
                                   "workspace_bounds_verified": False}},
        hardware={"robot": {"ip": "192.0.2.1", "home_pose_source": "replace-with-home",
                            "workspace_source": "replace-with-workspace",
                            "workspace_m": {"min": [0, 0, 0], "max": [1, 1, 1]},
                            "control": {"tracking_error_guard": {"enabled": False}}},
                  "cameras": {"exterior": {"serial": "A"}, "wrist": {"serial": "B"},
                              "width": 8, "height": 8, "fps": 30}},
    )


def test_sensor_permission_does_not_grant_live_permission():
    c = contract()
    before = copy.deepcopy(c.__dict__)
    require_sensor_configuration(c)
    with pytest.raises(PermissionError, match="Live deployment rejected"):
        require_live_authorization(c, deployment_method="baseline")
    assert c.__dict__ == before
    c.payload["safety_review"]["camera_role_mapping_verified"] = False
    with pytest.raises(PermissionError, match="camera_role_mapping_verified"):
        require_sensor_configuration(c)


def test_failed_second_camera_closes_receive_only_resources(monkeypatch):
    receiver, gripper, exterior = Mock(), Mock(), Mock()
    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(
        RTDEReceiveInterface=Mock(return_value=receiver)))
    monkeypatch.setattr(runtime.legacy_deploy, "RobotiqGripper", Mock(return_value=gripper))
    monkeypatch.setattr(runtime, "InstrumentedRealSenseCamera", Mock(
        side_effect=[exterior, RuntimeError("camera busy")]))
    for key in ("SEER_CAMERA_WIDTH", "SEER_CAMERA_HEIGHT", "SEER_CAMERA_FPS"):
        monkeypatch.setenv(key, "60")
    cfg = SimpleNamespace(robot_ip="192.0.2.1", camera_serial_cache={"exterior": "A", "wrist": "B"})
    with pytest.raises(RuntimeError, match="camera busy"):
        runtime.ReadOnlyDeployEnvironment(cfg)
    exterior.close.assert_called_once()
    receiver.disconnect.assert_called_once()
    gripper.disconnect.assert_called_once()
    gripper.activate.assert_not_called()


@pytest.mark.parametrize("arguments", [
    ["live", "--profile-target-hz", "60"],
    ["live", "--profile-camera-fps", "60"],
    ["read-only-profile", "--profile-target-hz", "nan"],
    ["read-only-profile", "--profile-target-hz", "0"],
    ["read-only-profile", "--profile-camera-fps", "0"],
])
def test_invalid_profile_options_rejected_before_hardware(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["cli", *arguments])
    with pytest.raises(SystemExit) as exc:
        cli.parse_args()
    assert exc.value.code == 2


def test_readonly_60hz_does_not_change_deployment_rate(tmp_path, monkeypatch):
    c = contract()
    before = copy.deepcopy(c.__dict__)
    clock = [0.0]

    def now():
        clock[0] += 0.001
        return clock[0]

    monkeypatch.setattr(runtime.time, "perf_counter", now)
    monkeypatch.setattr(runtime.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))

    class Controller:
        deployment_method = "baseline"
        contract = c
        index = 0
        policy = SimpleNamespace(metrics=SimpleNamespace(counters=Counter()))

        @property
        def needs_policy_query(self):
            return self.index % 5 == 0

        def attach_session_dir(self, output):
            pass

        def reset(self):
            self.index = 0

        def forward(self, *args, **kwargs):
            self.index += 1
            self.policy.metrics.counters = runtime._expected_policy_counters("baseline", self.index)
            return None, None, None, {"record": {"action": [0.0] * 7}}

        def write_runtime_summary(self):
            pass

        def runtime_summary(self):
            return {}

    class Env:
        exterior_camera = SimpleNamespace(last_read_metadata=None)
        wrist_camera = SimpleNamespace(last_read_metadata=None)
        frame = 0

        def get_robot_state(self):
            return {"pose6d": np.array([2., 0., 0., 0., 0., 0.]),
                    "tcp_rotvec": np.zeros(3), "gripper_open_state": np.ones(1),
                    "gripper_position": np.zeros(1), "joint_positions": np.zeros(6)}

        def get_color_images(self):
            self.frame += 1
            for serial, camera in (("A", self.exterior_camera), ("B", self.wrist_camera)):
                camera.last_read_metadata = {
                    "serial": serial, "width": 8, "height": 8, "fps": 60,
                    "format": "rgb8", "frame_number": self.frame,
                    "sensor_timestamp_ms": self.frame * 20.,
                    "host_capture_monotonic_s": self.frame * 0.02,
                }
            return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    out = tmp_path / "profile"
    summary = runtime.run_read_only_profile(controller=Controller(), env=Env(),
        output=out, steps=11, target_hz=60, camera_fps=60)
    assert summary["verdict"] == "READ_ONLY_PROFILE_PASS"
    assert summary["deployment_target_hz"] == 15
    assert summary["profile_target_hz"] == summary["profile_camera_fps"] == 60
    assert 0 < summary["read_only_tick_hz"] <= 60
    assert summary["policy_query_start_hz"] > 0
    assert summary["actual_robot_command_hz"] is None
    assert not summary["robot_command_issued"]
    assert not summary["workspace_validated"]
    assert not summary["live_authorization_granted"]
    assert summary["observed_tcp_xyz_m"]["not_a_safe_workspace_estimate"]
    assert summary["observed_policy_counters"]["num_action_transformer_calls"] == 30
    assert len((out / "read_only_steps.jsonl").read_text().splitlines()) == 11
    assert (out / "first_exterior.png").is_file() and (out / "first_wrist.png").is_file()
    assert json.loads((out / "read_only_summary.json").read_text()) == summary
    assert c.__dict__ == before
    c.payload["safety_review"]["workspace_bounds_verified"] = True
    c.hardware["robot"]["workspace_source"] = "reviewed test fixture"
    with pytest.raises(RuntimeError, match="outside the reviewed workspace"):
        runtime.run_read_only_profile(controller=Controller(), env=Env(),
            output=tmp_path / "bounded", steps=11)
