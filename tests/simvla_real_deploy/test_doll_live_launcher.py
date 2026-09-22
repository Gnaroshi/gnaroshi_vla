from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from architectures.simvla.adapters.latentloop_real_deploy.contracts import DeploymentContract, require_live_authorization
from tools.simvla import launch_doll_baseline as launcher

ROOT = Path(__file__).resolve().parents[2]


def test_desktop_discovery_uses_current_session_and_whitelisted_fields(tmp_path):
    authority = tmp_path / "Xauthority"
    authority.write_text("test-only")
    for pid, comm in (("1", "gnome-shell"), ("2", "gnome-session-b"), ("3", "unrelated")):
        process = tmp_path / pid
        process.mkdir()
        (process / "comm").write_text(comm)
        display = ":1" if pid != "3" else ":9"
        (process / "environ").write_bytes(
            f"DISPLAY={display}\0XAUTHORITY={authority}\0UNRELATED_SECRET=do_not_propagate\0".encode())
    assert launcher.desktop_environment({}, tmp_path) == {"DISPLAY": ":1", "XAUTHORITY": str(authority)}


def test_explicit_display_is_not_replaced(tmp_path):
    assert launcher.desktop_environment({"DISPLAY": "localhost:10.0"}, tmp_path) == {"DISPLAY": "localhost:10.0"}


def test_no_desktop_does_not_guess_display_zero(tmp_path):
    with pytest.raises(RuntimeError, match="하나로"):
        launcher.desktop_environment({}, tmp_path)


def test_multiple_desktops_are_not_silently_selected(tmp_path):
    for pid in ("1", "2"):
        process = tmp_path / pid
        process.mkdir()
        (process / "comm").write_text("gnome-shell")
        (process / "environ").write_bytes(f"DISPLAY=:{pid}\0".encode())
    with pytest.raises(RuntimeError, match="하나로"):
        launcher.desktop_environment({}, tmp_path)


def test_other_user_desktop_is_not_selected(tmp_path, monkeypatch):
    process = tmp_path / "1"
    process.mkdir()
    (process / "comm").write_text("gnome-shell")
    (process / "environ").write_bytes(b"DISPLAY=:1\0")
    monkeypatch.setattr(launcher.os, "getuid", lambda: process.stat().st_uid + 1)
    with pytest.raises(RuntimeError):
        launcher.desktop_environment({}, tmp_path)


def test_desktop_probe_failure_is_not_ignored(monkeypatch):
    monkeypatch.setattr(launcher, "desktop_environment", lambda env: {"DISPLAY": ":1"})
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stderr="not authorized"))
    with pytest.raises(RuntimeError, match="not authorized"):
        launcher.checked_desktop_exports({})

def test_operator_can_raise_max_steps_above_old_manifest(evidence):
    contract, _, profile = evidence
    result = launcher.reviewed_payload(contract, profile, [0.2, -0.6, 0.05], [0.8, 0.6, 0.8], [0.04, 0.1], 5000, "DEPLOY")
    assert result["runtime"]["max_steps"] == 5000
    assert result["policy"] == contract.policy



@pytest.fixture
def evidence():
    payload = json.loads((ROOT / "artifacts/simvla/real_world/deployment_manifest.example.json").read_text())
    payload["hardware"]["robot"]["ip"] = "192.168.56.101"
    payload["hardware"]["robot"]["home_pose_source"] = "operator-reviewed test home"
    for role in ("exterior", "wrist"):
        payload["hardware"]["cameras"][role]["serial"] = role + "-test"
    contract = DeploymentContract(Path("/tmp/site.json"), payload, {"model": SimpleNamespace(sha256="hash")})
    metadata = {
        "deployment_method": "baseline", "deployment_id": contract.deployment_id,
        "artifact_sha256": {"model": "hash"},
        "runtime_source_identity_sha256": payload["runtime_source_identity_sha256"],
        "policy_contract": copy.deepcopy(contract.policy),
        "state_contract": copy.deepcopy(contract.state),
        "action_contract": copy.deepcopy(contract.action),
    }
    model = {"deployment": copy.deepcopy(metadata), "actions_finite": True, "verdict": "ARTIFACT_PREFLIGHT_PASS"}
    profile = {
        "controller": metadata, "verdict": "READ_ONLY_PROFILE_PASS",
        "sensor_contract_validated": True, "policy_schedule_validated": True,
        "robot_command_issued": False, "deployment_target_hz": contract.runtime["control_frequency_hz"],
        "observed_tcp_xyz_m": {"min": [0.49, 0.13, 0.33], "max": [0.50, 0.14, 0.34]},
    }
    return contract, model, profile


def test_valid_evidence(evidence):
    launcher.validate_evidence(*evidence)


@pytest.mark.parametrize("field,value", [
    ("deployment_method", "condition_loop"), ("deployment_id", "wrong"),
    ("runtime_source_identity_sha256", "stale"), ("artifact_sha256", {}),
    ("policy_contract", {}), ("action_contract", {}), ("state_contract", {}),
])
def test_stale_evidence_rejected(evidence, field, value):
    contract, model, profile = evidence
    profile["controller"][field] = value
    with pytest.raises(ValueError):
        launcher.validate_evidence(contract, model, profile)


@pytest.mark.parametrize("field,value", [
    ("verdict", "FAIL"), ("policy_schedule_validated", False),
    ("sensor_contract_validated", False), ("robot_command_issued", True),
    ("deployment_target_hz", 999),
])
def test_invalid_profile_rejected(evidence, field, value):
    contract, model, profile = evidence
    profile[field] = value
    with pytest.raises(ValueError):
        launcher.validate_evidence(contract, model, profile)


def test_review_keeps_model_and_control_protocol(evidence, monkeypatch):
    contract, _, profile = evidence
    original = copy.deepcopy(contract.payload)
    # These are test fixtures, not recommended physical limits.
    result = launcher.reviewed_payload(contract, profile, [0.2, -0.6, 0.05], [0.8, 0.6, 0.8], [0.04, 0.1], 700, "DEPLOY")
    assert contract.payload == original
    for field in ("artifacts", "pairing", "policy", "state", "action", "runtime_source_identity_sha256"):
        assert result[field] == original[field]
    assert result["runtime"]["control_frequency_hz"] == original["runtime"]["control_frequency_hz"]
    assert result["runtime"]["num_rollouts_per_instruction"] == 1
    assert result["safety_review"]["baseline_bounded_canary_passed"] is False
    monkeypatch.setenv("SIMVLA_REAL_LIVE_RUN", "1")
    monkeypatch.setenv("SIMVLA_REAL_DEPLOYMENT_ID", contract.deployment_id)
    candidate = DeploymentContract(contract.path, result, contract.artifacts)
    require_live_authorization(candidate, deployment_method="baseline")
    with pytest.raises(PermissionError, match="baseline_bounded_canary"):
        require_live_authorization(candidate, deployment_method="latentloop")


@pytest.mark.parametrize("approval", ["", "yes", "APPROVE"])
def test_no_implicit_approval(evidence, approval):
    contract, _, profile = evidence
    with pytest.raises(PermissionError):
        launcher.reviewed_payload(contract, profile, [0, 0, 0], [1, 1, 1], [0.04, 0.1], 11, approval)
    assert contract.payload["safety_review"]["live_authorized"] is False


@pytest.mark.parametrize("minimum,maximum,tracking", [
    ([0, 0, 0], [0.1, 0.1, 0.1], [0.04, 0.1]),
    ([1, 0, 0], [0, 1, 1], [0.04, 0.1]),
    ([0, 0, 0], [1, 1, 1], [0, 0.1]),
    ([0, 0, 0], [1, 1, 1], [float("inf"), 0.1]),
])
def test_invalid_bounds(evidence, minimum, maximum, tracking):
    contract, _, profile = evidence
    with pytest.raises(ValueError):
        launcher.reviewed_payload(contract, profile, minimum, maximum, tracking, 11, "DEPLOY")


@pytest.mark.parametrize("text", ["", "1 2", "nan 1 2", "1 inf 2", "a b c"])
def test_invalid_numeric_input(text):
    with pytest.raises(ValueError):
        launcher.numbers(text, 3)


def test_numeric_input():
    assert launcher.numbers("0.1, 0.2, 0.3", 3) == [0.1, 0.2, 0.3]


def home_options(**changes):
    return SimpleNamespace(**{
        "control_hz": 60.0, "num_rollouts": 15, "warmup_steps": 3, "camera_fps": 60,
        "home_pose_json": "[3.0502887,-1.6030570,1.8191951,-1.8019783,-1.5417574,-1.6144441,0.0]",
        "home_pose_source": "stackcupanddoll/0511_172010 first joint_positions; gripper open",
        **changes,
    })


def test_home_override_preserves_model_action_and_original_manifest(evidence):
    contract, _, _ = evidence
    original = copy.deepcopy(contract.payload)
    payload = copy.deepcopy(original)
    payload["task_id"] = "stackcupanddoll"
    result = launcher.apply_runtime_options(payload, home_options())
    assert contract.payload == original
    for key in ("policy", "state", "action", "artifacts", "pairing", "runtime_source_identity_sha256"):
        assert result[key] == original[key]
    assert result["hardware"]["robot"]["home_pose"] == json.loads(home_options().home_pose_json)
    audit = result["launch_home_pose"]
    assert audit["manifest_home_pose"] == original["hardware"]["robot"]["home_pose"]
    assert audit["selected_home_pose"] == result["hardware"]["robot"]["home_pose"]
    assert audit["robot_movement_verified"] is False


@pytest.mark.parametrize("home", ["[]", "{}", "[0,0,0,0,0,0]", "[true,0,0,0,0,0,0]",
                                  "[NaN,0,0,0,0,0,0]", "[0,0,0,0,0,0,2]", "[7,0,0,0,0,0,0]"])
def test_invalid_home_is_rejected(evidence, home):
    contract, _, _ = evidence
    contract.payload["task_id"] = "stackcupanddoll"
    with pytest.raises(ValueError, match="home_pose"):
        launcher.apply_runtime_options(contract.payload, home_options(home_pose_json=home))


def test_doll_home_cannot_be_applied_to_another_task(evidence):
    contract, _, _ = evidence
    contract.payload["task_id"] = "cabinet"
    with pytest.raises(ValueError, match="다른 task"):
        launcher.apply_runtime_options(contract.payload, home_options())


def test_home_requires_provenance(evidence):
    contract, _, _ = evidence
    contract.payload["task_id"] = "stackcupanddoll"
    with pytest.raises(ValueError, match="home_pose_source"):
        launcher.apply_runtime_options(contract.payload, home_options(home_pose_source=""))


def test_system_tk_changes_only_child_environment(monkeypatch):
    monkeypatch.setattr(launcher.Path, "is_file", lambda self: True)
    original = {"LD_PRELOAD": "/existing.so", "CUDA_VISIBLE_DEVICES": "0"}
    result = launcher.gui_environment(original, "system")
    assert original == {"LD_PRELOAD": "/existing.so", "CUDA_VISIBLE_DEVICES": "0"}
    assert result["LD_PRELOAD"].endswith(":/existing.so")
    assert "libtcl8.6.so" in result["LD_PRELOAD"] and "libtk8.6.so" in result["LD_PRELOAD"]
    assert result["TCL_LIBRARY"] == "/usr/share/tcltk/tcl8.6"
    assert result["TK_LIBRARY"] == "/usr/share/tcltk/tk8.6"
    assert result["CUDA_VISIBLE_DEVICES"] == "0"
    assert launcher.gui_environment(original, "conda") == original
    monkeypatch.setattr(launcher.Path, "is_file", lambda self: False)
    with pytest.raises(FileNotFoundError, match="conda"):
        launcher.gui_environment(original, "system")


def test_hidden_gui_probe_uses_selected_backend_without_cuda(monkeypatch):
    monkeypatch.setattr(launcher, "desktop_environment", lambda env: {"DISPLAY": ":1"})
    seen = []
    monkeypatch.setattr(launcher, "gui_environment", lambda env, backend: {**env, "TEST_BACKEND": backend})
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: seen.append(k) or SimpleNamespace(returncode=0))
    exports = launcher.checked_desktop_exports({}, "system")
    assert "DISPLAY=:1" in exports
    assert seen[0]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert seen[0]["env"]["TEST_BACKEND"] == "system"


def test_report_selection_does_not_select_other_method(tmp_path):
    for name, status in [("artifact-preflight_baseline", "0"), ("artifact-preflight_baseline_r2", "1"), ("artifact-preflight_baseline_fake", "0")]:
        run = tmp_path / name
        (run / "output").mkdir(parents=True)
        (run / "exit_code.txt").write_text(status)
        (run / "output/result.json").write_text("{}")
    path, _ = launcher.completed_report(tmp_path, "artifact-preflight_baseline", "result.json")
    assert path.parent.parent.name == "artifact-preflight_baseline"


def test_site_profile_requires_local_physical_stop_confirmation(evidence, monkeypatch):
    contract, _, _ = evidence
    preset = {
        "deployment_id": contract.deployment_id,
        "robot": copy.deepcopy(contract.hardware["robot"]),
        "cameras": {**copy.deepcopy(contract.hardware["cameras"]), "fps": 60},
        "provenance": {"source": "unit-test"}, "target_control_hz": 60,
    }
    preset["robot"]["control"].pop("tracking_error_guard")
    for confirmed in (False, True):
        payload = launcher.seer_site_payload(contract, preset, 700, confirmed=confirmed)
        assert payload["runtime"]["control_frequency_hz"] == 60
        assert payload["runtime"]["training_sample_hz"] == 15
        assert payload["safety_review"]["workspace_bounds_verified"] is False
        assert payload["safety_review"]["physical_emergency_stop_verified"] is confirmed
        assert payload["hardware"]["robot"]["control"]["tracking_error_guard"]["enabled"] is False
        for key in ("artifacts", "policy", "action", "state", "pairing"):
            assert payload[key] == contract.payload[key]
        candidate = DeploymentContract(contract.path, payload, contract.artifacts)
        monkeypatch.setenv("SIMVLA_REAL_LIVE_RUN", "1")
        monkeypatch.setenv("SIMVLA_REAL_DEPLOYMENT_ID", contract.deployment_id)
        monkeypatch.setenv("SIMVLA_REAL_SITE_PROFILE", "seer_doll")
        if confirmed:
            require_live_authorization(candidate, deployment_method="baseline")
        else:
            with pytest.raises(PermissionError, match="physical_emergency_stop_verified"):
                require_live_authorization(candidate, deployment_method="baseline")
    candidate.payload["safety_review"]["physical_emergency_stop_verified"] = False
    with pytest.raises(PermissionError, match="physical_emergency_stop_verified"):
        require_live_authorization(candidate, deployment_method="baseline")
    candidate.payload["safety_review"]["physical_emergency_stop_verified"] = True
    monkeypatch.delenv("SIMVLA_REAL_SITE_PROFILE")
    with pytest.raises(PermissionError, match="SITE_PROFILE"):
        require_live_authorization(candidate, deployment_method="baseline")


def test_runtime_revision_only_accepts_reviewed_four_files():
    preset = json.loads(launcher.SITE_PROFILE.read_text())
    revision = preset["runtime_revision"]
    current = launcher.runtime_source_identity()
    contract = SimpleNamespace(payload={"runtime_source_identity_sha256": current["combined_sha256"]})
    previous = launcher.sha256_json(revision["previous_files"])
    launcher.verify_runtime_revision(contract, previous, revision)
    invalid = copy.deepcopy(revision)
    invalid["reviewed_files"][next(iter(invalid["reviewed_files"]))] = "bad"
    with pytest.raises(ValueError):
        launcher.verify_runtime_revision(contract, previous, invalid)
    with pytest.raises(ValueError):
        launcher.verify_runtime_revision(contract, "bad", revision)


@pytest.fixture
def saved_checks(tmp_path, evidence):
    original, model, profile = evidence
    contract = DeploymentContract(tmp_path / "site.json", original.payload, original.artifacts)
    args = SimpleNamespace(log_root=tmp_path / "logs", check=True, refresh_checks=False,
                           control_hz=60, camera_fps=60)
    profile.update(profile_target_hz=60, profile_camera_fps=60)
    first = {f"{role}_camera": {
        "serial": contract.hardware["cameras"][role]["serial"], "fps": 60,
        "width": contract.hardware["cameras"]["width"],
        "height": contract.hardware["cameras"]["height"],
    } for role in ("exterior", "wrist")}
    for stem, name, data in (("artifact-preflight_baseline", "artifact_preflight.json", model),
                             ("read-only-profile_baseline", "read_only_summary.json", profile)):
        output = args.log_root / stem / "output"
        output.mkdir(parents=True)
        (output.parent / "exit_code.txt").write_text("0")
        (output / name).write_text(json.dumps(data))
    (output / "read_only_steps.jsonl").write_text(json.dumps(first) + "\n")
    return contract, args


def test_checks_can_be_reused_without_old_logs_or_model_load(saved_checks, monkeypatch):
    import shutil
    contract, args = saved_checks
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("must not load a model or sensor"))
    first = launcher.obtain_evidence(contract, args, None)
    assert first[0].name == "deployment_checks.json"
    shutil.rmtree(args.log_root)
    assert launcher.obtain_evidence(contract, args, None) == first


@pytest.mark.parametrize("change", ["hardware", "fps", "hz", "environment", "camera", "source"])
def test_changed_checks_do_not_reuse_stale_evidence(saved_checks, monkeypatch, change):
    contract, args = saved_checks
    path, _, _, _ = launcher.obtain_evidence(contract, args, None)
    if change == "hardware": contract.payload["hardware"]["robot"]["home_pose"][0] += 0.1
    elif change == "fps": args.camera_fps = 30
    elif change == "hz": args.control_hz = 15
    elif change == "environment": monkeypatch.setattr(launcher.importlib.metadata, "version", lambda n: "different")
    elif change == "source": contract.payload["runtime_source_identity_sha256"] = "changed"
    else:
        saved = json.loads(path.read_text())
        saved["first_sensor_frame"]["wrist_camera"]["serial"] = "wrong"
        path.write_text(json.dumps(saved))
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("--check must remain read-only"))
    with pytest.raises(ValueError, match="재사용 가능한"):
        launcher.obtain_evidence(contract, args, None)


def test_refresh_failure_does_not_launch_live(saved_checks, monkeypatch):
    import subprocess
    contract, args = saved_checks
    args.check = False
    args.refresh_checks = True
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(9, command)
    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.obtain_evidence(contract, args, None)
    assert len(calls) == 1
    assert "artifact-preflight" in calls[0]
    assert "live" not in calls[0]


def test_explicit_refresh_runs_both_nonmotion_checks(saved_checks, monkeypatch):
    contract, args = saved_checks
    args.check = False
    args.refresh_checks = True
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kw: calls.append(command))
    launcher.obtain_evidence(contract, args, None)
    assert [call[2] for call in calls] == ["artifact-preflight", "read-only-profile"]
    assert all("--method" in call and "baseline" in call for call in calls)
