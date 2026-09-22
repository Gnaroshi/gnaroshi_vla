import copy
import json
import os
from pathlib import Path
import pty
import subprocess

import pytest

from tools.simvla.install_doll_joint import relocation_payload

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "architectures/simvla/wrappers/deploy_ll.sh"


def selection_manifest(tmp_path):
    data = json.loads((ROOT / "artifacts/simvla/real_world/deployment_manifest.example.json").read_text())
    data.update(deployment_id="doll_joint_v1", task_id="stackcupanddoll", enabled_methods=["baseline"])
    for name in ("real_action_transformer", "norm_stats"):
        artifact = tmp_path / name
        artifact.write_text("test fixture only")
        data["artifacts"][name] = {"path": str(artifact), "sha256": "a" * 64}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def manifests():
    old = json.loads((ROOT / "artifacts/simvla/real_world/deployment_manifest.example.json").read_text())
    old["safety_review"]["camera_role_mapping_verified"] = True
    old["safety_review"]["live_authorized"] = True
    new = copy.deepcopy(old)
    new["enabled_methods"] = ["baseline"]
    new["task_id"] = "stackcupanddoll"
    names = {"official_base_model_directory", "official_base_model_weights", "processor_directory",
             "dataset_manifest", "norm_stats", "real_action_transformer"}
    new["artifacts"] = {k: v for k, v in new["artifacts"].items() if k in names}
    new["artifacts"]["real_action_transformer"]["sha256"] = "a" * 64
    new["pairing"]["real_baseline_identity"] = "a" * 64
    new["selection"] = {"old_ours_weights_compatible": False}
    return new, old


def test_install_keeps_policy_and_rejects_stale_approval(manifests):
    new, old = manifests
    before = copy.deepcopy((new, old))
    payload = relocation_payload(new, old, "a" * 64, new["runtime_source_identity_sha256"])
    assert (new, old) == before
    for key in ("policy", "state", "action", "pairing"):
        assert payload[key] == new[key]
    assert payload["enabled_methods"] == ["baseline"]
    assert not payload["safety_review"]["live_authorized"]
    assert not payload["safety_review"]["model_preflight_passed"]
    assert not payload["safety_review"]["read_only_profile_passed"]
    assert payload["safety_review"]["camera_role_mapping_verified"]
    assert "condition_updater" not in payload["artifacts"]
    assert payload["artifacts"]["real_action_transformer"]["path"].startswith("./checkpoints/")


def test_camera_rate_is_not_a_camera_role_review(manifests):
    new, old = manifests
    old["hardware"]["cameras"]["fps"] = 30
    new["hardware"]["cameras"]["fps"] = 60
    result = relocation_payload(new, old, "a" * 64, new["runtime_source_identity_sha256"])
    assert result["hardware"]["cameras"]["fps"] == 60
    assert result["safety_review"]["camera_role_mapping_verified"]
    assert not result["safety_review"]["read_only_profile_passed"]


@pytest.mark.parametrize("change", ["checkpoint", "runtime", "camera", "norm", "action", "ours"])
def test_install_rejects_mismatches(manifests, change):
    new, old = manifests
    digest, runtime = "a" * 64, new["runtime_source_identity_sha256"]
    if change == "checkpoint": digest = "b" * 64
    if change == "runtime": runtime = "b" * 64
    if change == "camera": old["hardware"]["cameras"]["exterior"]["serial"] = "different"
    if change == "norm": old["artifacts"]["norm_stats"]["sha256"] = "b" * 64
    if change == "action": old["action"]["translation_scale_m"] = 0.5
    if change == "ours": new["enabled_methods"].append("condition_loop")
    with pytest.raises(ValueError):
        relocation_payload(new, old, digest, runtime)


@pytest.mark.parametrize("strict,expected", [("0", 0), ("1", 9)])
def test_launcher_failure_keeps_pane_and_records_status(tmp_path, strict, expected):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "commands"
    stub = bin_dir / "python"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$MARKER"\ncase "$*" in *--inspect*) exit 0;; esac\nexit 9\n')
    stub.chmod(0o755)
    manifest = selection_manifest(tmp_path)
    master, slave = pty.openpty()
    try:
        result = subprocess.run(
            ["/bin/bash", str(ENTRY)],
            stdin=slave, capture_output=True, text=True,
            env={**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"],
                 "DISPLAY": ":test", "MARKER": str(marker), "SIMVLA_STRICT_EXIT": strict,
                 "SIMVLA_REAL_PYTHON": str(stub), "SIMVLA_DOLL_MANIFEST": str(manifest),
                 "SIMVLA_REAL_LOG_ROOT": str(tmp_path / "logs")},
        )
    finally:
        os.close(master)
        os.close(slave)
    assert result.returncode == expected
    calls = marker.read_text().splitlines()
    assert len(calls) == 2 and "--inspect" in calls[0]
    assert all("read-only-profile" not in call for call in calls)
    assert "SIMVLA_DEPLOY_COMMAND_FAILED rc=9" in result.stdout
    assert (tmp_path / "logs/launcher.exit_code").read_text().strip() == "9"


def test_missing_desktop_is_explained_and_logged(tmp_path):
    manifest = selection_manifest(tmp_path)
    result = subprocess.run(["/bin/bash", str(ENTRY)],
        input="", text=True, capture_output=True,
        env={**os.environ, "DISPLAY": "", "SIMVLA_REAL_PYTHON": os.sys.executable,
             "SIMVLA_DOLL_MANIFEST": str(manifest), "SIMVLA_REAL_LOG_ROOT": str(tmp_path / "logs"),
             "SIMVLA_STRICT_EXIT": "1"})
    assert result.returncode == 2
    assert "DISPLAY=unset" in result.stdout
    assert "interactive_stdin=no" in (tmp_path / "logs/launcher.log").read_text()


def test_interactive_launcher_recovers_missing_display_before_sensor_check(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "commands"
    python = bin_dir / "python"
    python.write_text('#!/bin/sh\ncase "$*" in\n*--desktop-environment*) printf "export DISPLAY=:1\\nexport XAUTHORITY=/test/auth\\n";;\n*--inspect*) exit 0;;\n*) printf "%s|%s|%s\\n" "$DISPLAY" "$XAUTHORITY" "$*" >> "$MARKER"; exit 9;;\nesac\n')
    python.chmod(0o755)
    bash = bin_dir / "bash"
    bash.write_text('#!/bin/sh\nprintf "%s|%s|%s\\n" "$DISPLAY" "$XAUTHORITY" "$*" >> "$MARKER"\nexit 9\n')
    bash.chmod(0o755)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    master, slave = pty.openpty()
    try:
        result = subprocess.run(
            ["/bin/bash", str(ENTRY)],
            stdin=slave, capture_output=True, text=True,
            env={**os.environ, "DISPLAY": "", "PATH": str(bin_dir) + ":" + os.environ["PATH"],
                 "MARKER": str(marker), "SIMVLA_STRICT_EXIT": "1", "SIMVLA_REAL_PYTHON": str(python),
                 "SIMVLA_DOLL_MANIFEST": str(manifest), "SIMVLA_REAL_LOG_ROOT": str(tmp_path / "logs")})
    finally:
        os.close(master)
        os.close(slave)
    assert result.returncode == 9
    assert "GUI_CONNECTION_PASS DISPLAY=:1" in result.stdout
    calls = marker.read_text().splitlines()
    assert len(calls) == 1 and calls[0].startswith(":1|/test/auth|")
    assert "tools.simvla.launch_doll_baseline" in calls[0]
    assert "read-only-profile" not in calls[0]


def test_single_repository_paths():
    for name in ("deploy_doll_joint_baseline.sh",):
        text = (ROOT / "architectures/simvla/wrappers" / name).read_text()
        assert "gnaroshi_vla_runtime" not in text
        assert '/deploy_ll.sh" --preset ' in text
        assert len(text.splitlines()) == 3
    assert "${root}/runtime" in ENTRY.read_text()
    assert 'ROOT / "runtime"' in (ROOT / "tools/simvla/launch_doll_baseline.py").read_text()


@pytest.mark.parametrize("name", ["basketball", "cabinet", "stack_cups", "fruit", "unknown",
                                 "doll_legacy_baseline", "doll_legacy_ours", "doll_legacy_coupled"])
def test_unavailable_task_stops_without_running_python(tmp_path, name):
    result = subprocess.run(["/bin/bash", str(ENTRY), "--preset", name], capture_output=True, text=True,
        env={**os.environ, "SIMVLA_STRICT_EXIT": "1", "SIMVLA_REAL_PYTHON": "/does/not/exist"})
    assert result.returncode == 2
    assert "SIMVLA_DEPLOY_COMMAND_FAILED" in result.stdout
    assert "Python을 찾을" not in result.stdout


@pytest.mark.parametrize("name", ["deploy_doll_joint_baseline.sh"])
def test_old_command_delegates_to_one_configuration(name):
    result = subprocess.run(["/bin/bash", str(ENTRY.parent / name), "--list"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "doll_joint_baseline:" in result.stdout


@pytest.mark.parametrize("preset,success", [("doll_joint_baseline", True), ("doll_joint_ours", False), ("doll_legacy_baseline", False),
                                          ("doll_baseline", True), ("doll_ours", False)])
def test_real_selection_rejects_wrong_teacher_before_sensors(tmp_path, preset, success):
    manifest = selection_manifest(tmp_path)
    result = subprocess.run(["/bin/bash", str(ENTRY), "--inspect", "--preset", preset, "--max-steps", "9000"],
        capture_output=True, text=True,
        env={**os.environ, "SIMVLA_STRICT_EXIT": "1", "SIMVLA_REAL_PYTHON": os.sys.executable,
             "SIMVLA_DOLL_MANIFEST": str(manifest), "SIMVLA_REAL_LOG_ROOT": str(tmp_path / "logs")})
    assert (result.returncode == 0) == success
    if success:
        assert '"max_steps": 9000' in result.stdout
        assert '"robot_connected": false' in result.stdout
        assert '3.0502887' in result.stdout
    else:
        assert "[1/2]" not in result.stdout


def test_retired_entrypoints_and_presets_are_removed():
    assert not (ENTRY.parent / "deploy_doll_baseline.sh").exists()
    assert not (ENTRY.parent / "deploy_doll_ours.sh").exists()
    result = subprocess.run(["/bin/bash", str(ENTRY), "--list"], capture_output=True, text=True)
    assert "legacy" not in result.stdout
    assert "doll_joint_baseline" in result.stdout


def test_print_config_needs_no_python_assets_or_log_files(tmp_path):
    logs = tmp_path / "logs"
    result = subprocess.run(["/bin/bash", str(ENTRY), "--print-config", "--control-freq", "40",
                             "--max-steps", "1234", "--gui-font-backend", "conda"], capture_output=True, text=True,
        env={**os.environ, "SIMVLA_STRICT_EXIT": "1", "SIMVLA_REAL_PYTHON": "/does/not/exist",
             "SIMVLA_REAL_LOG_ROOT": str(logs)})
    assert result.returncode == 0
    assert "target_control_hz=40" in result.stdout and "max_steps=1234" in result.stdout
    assert "gui_font_backend=conda" in result.stdout and "mode=--live" in result.stdout
    assert not logs.exists()


def test_profile_arguments_reach_read_only_engine(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "engine_command"
    python = bindir / "python"
    python.write_text('#!/bin/sh\nexit 0\n')
    python.chmod(0o755)
    bash = bindir / "bash"
    bash.write_text('#!/bin/sh\nprintf "%s|%s\\n" "$SIMVLA_REAL_CUDA_DEVICE" "$*" > "$MARKER"\n')
    bash.chmod(0o755)
    result = subprocess.run(["/bin/bash", str(ENTRY), "--profile", "--profile-steps", "23",
                            "--control-freq", "40", "--camera-fps", "30", "--cuda-device", "5"],
        capture_output=True, text=True,
        env={**os.environ, "SIMVLA_STRICT_EXIT": "1", "SIMVLA_REAL_PYTHON": str(python),
             "SIMVLA_REAL_LOG_ROOT": str(tmp_path / "logs"), "MARKER": str(marker),
             "PATH": str(bindir) + ":" + os.environ["PATH"]})
    assert result.returncode == 0
    command = marker.read_text()
    assert command.startswith("5|") and "read-only-profile" in command
    assert "--steps 23" in command and "--profile-target-hz 40" in command
    assert "--profile-camera-fps 30" in command


@pytest.mark.parametrize("replacement", ["", '"doll_baseline" "doll_ours"'])
def test_ambiguous_top_level_selection_stops_before_python(tmp_path, replacement):
    script = ENTRY.read_text()
    start = script.index("deploy_presets=(")
    end = script.index(")", start) + 1
    entry = tmp_path / "a/b/c/deploy_ll.sh"
    entry.parent.mkdir(parents=True)
    entry.write_text(script[:start] + f"deploy_presets=({replacement})" + script[end:])
    result = subprocess.run(["/bin/bash", str(entry)], capture_output=True, text=True,
        env={**os.environ, "SIMVLA_STRICT_EXIT": "1", "SIMVLA_REAL_PYTHON": "/does/not/exist"})
    assert result.returncode == 2
    assert "deploy_presets에서 하나만" in result.stdout
