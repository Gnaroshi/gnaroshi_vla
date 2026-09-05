"""CPU-only deployment orchestration and failure-path regression tests."""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from architectures.simvla.adapters.latentloop_real_deploy import environment
from architectures.simvla.adapters.latentloop_real_deploy import preparation

ROOT = Path(__file__).resolve().parents[2]


def _fake_contract(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"safety_review":{"live_authorized":false}}')
    return SimpleNamespace(
        path=path, deployment_id="test", payload={"runtime_source_identity_sha256": "a" * 64}
    )


@pytest.mark.parametrize("failed_method", [None, "condition_loop"])
def test_prepare_is_sequential_stops_on_error_and_never_authorizes(tmp_path, monkeypatch, failed_method):
    contract = _fake_contract(tmp_path)
    before = contract.path.read_bytes()
    monkeypatch.setattr(preparation, "inspect_environment",
                        lambda **_: {"verdict": "REAL_ENVIRONMENT_PASS", "failures": []})
    monkeypatch.setattr(preparation, "load_deployment_contract", lambda *a, **k: contract)
    monkeypatch.setattr(preparation, "hardware_configuration_issues", lambda _: ["site review required"])
    calls = []

    def child(command, **kwargs):
        method = command[command.index("--method") + 1]
        calls.append(method)
        assert command[command.index("-m") + 1].endswith(".cli")
        assert command[command.index("--device") + 1] == "cpu"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert "live" not in command and "read-only-profile" not in command
        if method == failed_method:
            return SimpleNamespace(returncode=1)
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "artifact_preflight.json").write_text(json.dumps({
            "verdict": "ARTIFACT_PREFLIGHT_PASS", "actions_finite": True,
            "robot_command_issued": False,
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preparation.subprocess, "run", child)
    arguments = dict(manifest=str(contract.path), output=str(tmp_path / "out"),
                     device="cpu", require_gui=False)
    if failed_method:
        with pytest.raises(RuntimeError, match="condition_loop failed"):
            preparation.prepare(**arguments)
        assert calls == ["baseline", "condition_loop"]
    else:
        result = preparation.prepare(**arguments)
        assert result["verdict"] == "MODELS_READY_FOR_SITE_REVIEW"
        assert calls == list(preparation.METHODS)
    report = json.loads((tmp_path / "out/preparation_summary.json").read_text())
    assert not report["live_authorized_by_preparation"]
    assert not report["task_success_verified"]
    assert report["robot_commands_issued"] == 0
    assert contract.path.read_bytes() == before


def test_prepare_environment_failure_precedes_any_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "inspect_environment",
                        lambda **_: {"verdict": "REAL_ENVIRONMENT_FAIL", "failures": ["missing camera SDK"]})
    load = Mock(side_effect=AssertionError("model must not load"))
    monkeypatch.setattr(preparation, "load_deployment_contract", load)
    with pytest.raises(RuntimeError, match="camera SDK"):
        preparation.prepare(manifest="unused", output=str(tmp_path / "out"), device="cpu", require_gui=False)
    load.assert_not_called()
    assert json.loads((tmp_path / "out/preparation_summary.json").read_text())["stage"] == "environment"


def test_environment_headless_check_does_not_run_cuda_or_open_devices(monkeypatch):
    monkeypatch.setattr(environment.importlib.metadata, "version",
                        lambda name: environment.EXPECTED_PACKAGES[name])
    codes = []
    monkeypatch.setattr(environment, "_probe", lambda code: codes.append(code) or {"passed": True})
    monkeypatch.setattr(environment, "_video_probe", lambda: {"passed": True})
    result = environment.inspect_environment(require_cuda=False, require_gui=False)
    assert result["verdict"] == "REAL_ENVIRONMENT_PASS"
    assert len(codes) == 1
    assert "RTDEControlInterface(" not in codes[0] and "pipeline(" not in codes[0]
    assert "Tk()" not in codes[0] and "device='cuda'" not in codes[0]
    assert not result["robot_interfaces_constructed"]


def test_environment_reports_display_failure(monkeypatch):
    monkeypatch.setattr(environment.importlib.metadata, "version",
                        lambda name: environment.EXPECTED_PACKAGES[name])
    monkeypatch.setattr(environment, "_probe",
                        lambda code: {"passed": "Tk()" not in code})
    monkeypatch.setattr(environment, "_video_probe", lambda: {"passed": True})
    result = environment.inspect_environment(require_cuda=False, require_gui=True)
    assert result["verdict"] == "REAL_ENVIRONMENT_FAIL"
    assert result["failures"] == ["tk_display failed"]


def test_gui_display_failure_never_initializes_robot(monkeypatch):
    from architectures.simvla.adapters.latentloop_real_deploy import deploy_gui as gui
    monkeypatch.setattr(gui, "build_deploy_config", lambda _: SimpleNamespace())
    controller = SimpleNamespace(contract=SimpleNamespace(hardware={
        "robot": {"workspace_m": {"min": [0]*3, "max": [1]*3},
                  "control": {"tracking_error_guard": {}}}
    }))
    robot = Mock(side_effect=AssertionError("robot must not initialize"))
    monkeypatch.setattr(gui, "TimedSafeUR5eDeployEnv", robot)
    monkeypatch.setattr(gui.tk, "Tk", Mock(side_effect=gui.tk.TclError("no display")))
    with pytest.raises(gui.tk.TclError, match="no display"):
        gui.run_live_gui(controller=controller)
    robot.assert_not_called()


def test_partial_hardware_constructor_disconnects_existing_resources(monkeypatch):
    from architectures.simvla.adapters.latentloop_real_deploy import hardware
    control, receiver, camera, gripper = Mock(), Mock(), Mock(), Mock()
    control.servoStop.return_value = True

    def failed_init(self, cfg):
        self.rtde_ctrl, self.rtde_rec = control, receiver
        self.exterior_camera, self.gripper = camera, gripper
        raise RuntimeError("wrist camera failed")

    monkeypatch.setattr(hardware.legacy_deploy.UR5eDeployEnv, "__init__", failed_init)
    with pytest.raises(RuntimeError, match="wrist camera failed"):
        hardware.SafeUR5eDeployEnv(SimpleNamespace(arm_acceleration=0.1),
                                   workspace_min=[0]*3, workspace_max=[1]*3)
    assert control.servoStop.called
    control.disconnect.assert_called_once()
    receiver.disconnect.assert_called_once()
    camera.close.assert_called_once()
    gripper.disconnect.assert_called_once()


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_runtime_stationary_code_is_valid_but_nonfinite_is_rejected(monkeypatch, value):
    from architectures.simvla.adapters.latentloop_real_deploy import policy
    cls = policy.LatentLoopSimVLARealPolicy
    instance = object.__new__(cls)
    instance.cached_condition = torch.zeros(1, 2, 4)
    instance.cached_raw_rgb = torch.zeros(1, 2, 3, 2, 2)
    instance.cached_proprio = torch.zeros(1, 8)
    instance.condition_layout = SimpleNamespace(valid_mask=torch.ones(1, 2, dtype=torch.bool),
                                                group_ids=torch.zeros(1, 2, dtype=torch.long))
    instance.native_v0 = object()
    instance._sync = lambda: None
    instance.metrics = SimpleNamespace(counters=Counter(), latencies={})
    instance.condition_change_code_norms = []
    instance._decode = Mock(return_value=(torch.zeros(1, 10, 7), 42))
    monkeypatch.setattr(policy, "condition_update_with_code", lambda *a, **k: SimpleNamespace(
        condition_change_code=torch.full((1, 128), value),
        update=SimpleNamespace(condition=torch.zeros(1, 2, 4))))
    batch = {"raw_rgb": instance.cached_raw_rgb, "proprio": instance.cached_proprio}
    if value == 0.0:
        instance._v0_update(batch, age=1, policy_query_index=1)
        assert instance.condition_change_code_norms == [0.0]
        instance._decode.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="non-finite"):
            instance._v0_update(batch, age=1, policy_query_index=1)
        instance._decode.assert_not_called()


@pytest.mark.parametrize("value", [0.0, float("nan")])
def test_training_stationary_code_matches_runtime_rule(monkeypatch, value):
    from architectures.simvla.adapters.real_world_training import train_coupled_generation as training
    batch = {name: torch.zeros(1, 2, 4) for name in ("previous_condition", "current_condition")}
    batch.update({name: torch.zeros(1, 8) for name in ("previous_proprio", "current_proprio")})
    batch.update({name: torch.zeros(1, 2, 3, 2, 2) for name in ("previous_images", "current_images")})
    batch["current_cache_index"] = torch.tensor([0])
    manifest = {"token_layout": {"valid_mask": [[True, True]], "group_ids": [[0, 0]]}}
    monkeypatch.setattr(training, "condition_update_with_code", lambda *a, **k: SimpleNamespace(
        condition_change_code=torch.full((1, 128), value),
        update=SimpleNamespace(condition=torch.zeros(1, 2, 4))))
    if value == 0.0:
        result = training._coupled_query(object(), batch, manifest, torch.device("cpu"))
        assert result["code_norm"].item() == 0
    else:
        with pytest.raises(RuntimeError, match="non-finite"):
            training._coupled_query(object(), batch, manifest, torch.device("cpu"))


def test_installer_refuses_to_modify_existing_unowned_environment(tmp_path):
    prefix = tmp_path / "existing_env"
    prefix.mkdir()
    sentinel = prefix / "do_not_change"
    sentinel.write_text("untouched")
    result = subprocess.run(["bash", str(ROOT / "architectures/simvla/wrappers/setup_real_deploy_env.sh"),
                             "--install"], env={**os.environ, "SIMVLA_REAL_ENV_INSTALL": "1",
                             "SIMVLA_REAL_ENV_PREFIX": str(prefix), "SIMVLA_REAL_CONDA": "/bin/true"},
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "unowned environment" in result.stderr
    assert sentinel.read_text() == "untouched"


def test_transfer_invalid_bundle_never_contacts_receiver(tmp_path):
    bundle = tmp_path / "invalid"
    bundle.mkdir()
    (bundle / "bundle_inventory.json").write_text('{"verdict":"FAILED"}')
    marker = tmp_path / "ssh_called"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ssh = bindir / "ssh"
    ssh.write_text("#!/bin/sh\ntouch '" + str(marker) + "'\nexit 99\n")
    ssh.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "architectures/simvla/wrappers/publish_real_deploy_bundle.sh"),
         "--bundle", str(bundle), "--send"],
        env={**os.environ, "SIMVLA_REAL_PYTHON": sys.executable,
             "SIMVLA_REAL_TRANSFER_RUN": "1", "PATH": str(bindir) + ":" + os.environ["PATH"]},
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "not completed successfully" in result.stderr
    assert not marker.exists()


def test_artifact_preflight_rejects_nan_even_if_counters_match(tmp_path, monkeypatch):
    from architectures.simvla.adapters.latentloop_real_deploy import cli
    contract = SimpleNamespace(
        hardware={"cameras": {"height": 2, "width": 2}},
        state={"preflight_robot_state": {
            "pose6d_euler_xyz": [0]*6, "tcp_rotvec": [0]*6,
            "gripper_open_state": 1, "gripper_position_normalized": 0}},
        runtime={"instructions": ["test"]},
    )
    controller = SimpleNamespace(
        metrics=None,
        policy=SimpleNamespace(metrics=SimpleNamespace(counters={
            "num_policy_queries": 3, "num_action_queue_steps": 11,
            "num_action_transformer_calls": 30, "num_full_vlm_calls": 3,
            "num_condition_updater_calls": 0})),
        forward=lambda *a, **k: (None, None, None, {"record": {"action": [float("nan")]*7}}),
    )
    monkeypatch.setattr(cli, "_load_controller", lambda _: (contract, controller))
    with pytest.raises(RuntimeError, match="non-finite actions"):
        cli._artifact_preflight(SimpleNamespace(method="baseline", steps=11, output=str(tmp_path / "out")))
    assert not (tmp_path / "out").exists()
