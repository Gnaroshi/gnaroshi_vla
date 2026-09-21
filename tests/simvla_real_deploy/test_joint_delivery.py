import copy
import json
import os
from pathlib import Path
import pty
import subprocess

import pytest

from tools.simvla.install_doll_joint import relocation_payload

ROOT = Path(__file__).resolve().parents[2]


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
def test_sensor_failure_never_reaches_live_gui(tmp_path, strict, expected):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "commands"
    stub = bin_dir / "bash"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$MARKER"\nexit 9\n')
    stub.chmod(0o755)
    master, slave = pty.openpty()
    try:
        result = subprocess.run(
            ["/bin/bash", str(ROOT / "architectures/simvla/wrappers/deploy_doll_joint_baseline.sh")],
            stdin=slave, capture_output=True, text=True,
            env={**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"],
                 "DISPLAY": ":test", "MARKER": str(marker), "SIMVLA_STRICT_EXIT": strict},
        )
    finally:
        os.close(master)
        os.close(slave)
    assert result.returncode == expected
    calls = marker.read_text().splitlines()
    assert len(calls) == 1 and "read-only-profile" in calls[0]
    assert "DOLL_JOINT_COMMAND_FAILED rc=9" in result.stdout
