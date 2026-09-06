from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from architectures.simvla.adapters.latentloop_real_deploy.contracts import DeploymentContract, require_live_authorization
from tools.simvla import launch_doll_baseline as launcher

ROOT = Path(__file__).resolve().parents[2]


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


def test_report_selection_does_not_select_other_method(tmp_path):
    for name, status in [("artifact-preflight_baseline", "0"), ("artifact-preflight_baseline_r2", "1"), ("artifact-preflight_baseline_fake", "0")]:
        run = tmp_path / name
        (run / "output").mkdir(parents=True)
        (run / "exit_code.txt").write_text(status)
        (run / "output/result.json").write_text("{}")
    path, _ = launcher.completed_report(tmp_path, "artifact-preflight_baseline", "result.json")
    assert path.parent.parent.name == "artifact-preflight_baseline"
