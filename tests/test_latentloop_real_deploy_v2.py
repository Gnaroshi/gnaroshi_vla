import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from architectures.seer.adapters.latentloop_real_deploy.controller_v2 import (
    LatentLoopSeerControllerV2,
    TemporalEnsembleRing,
    temporal_ensemble_probability,
    validate_runtime_settings,
)
from architectures.seer.adapters.latentloop_real_deploy.runtime_v2 import (
    ReadOnlyDeployEnvV2,
)
from architectures.seer.adapters.latentloop_real_deploy.deploy_ll_gui_v2 import (
    LatentLoopDeployGuiAppV2,
    execution_mode_from_environment,
)


def test_temporal_ensemble_ring_matches_quadratic_buffer():
    torch.manual_seed(7)
    horizon = 3
    max_steps = 24
    legacy_buffer = torch.zeros(max_steps, max_steps + horizon, 7)
    ring = TemporalEnsembleRing(horizon)

    for timestep in range(16):
        sequence = torch.randn(1, horizon, 7) + 0.25
        if timestep == 5:
            sequence[0, 1, 2] = 0.0
        expected, expected_count = temporal_ensemble_probability(
            sequence, timestep, legacy_buffer, 0.01
        )
        actual, actual_count = ring.probability(sequence, timestep, 0.01)
        assert actual_count == expected_count
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(ring._sequences) <= horizon


def test_read_only_environment_has_no_motion_command_api_calls():
    source = inspect.getsource(ReadOnlyDeployEnvV2)
    forbidden = (
        "RTDEControlInterface",
        ".activate(",
        ".move(",
        "servoJ",
        "move_to_home",
    )
    for token in forbidden:
        assert token not in source


@pytest.mark.parametrize(
    ("method", "control_freq", "query_interval", "rollout_policy", "expected"),
    (
        ("baseline", "15", "1", "full", (15.0, 1, "full")),
        ("baseline", "20", "4", "hold_action", (20.0, 4, "hold_action")),
        ("baseline", "30", "8", "hold_latent", (30.0, 8, "hold_latent")),
        ("latentloop", "20.5", "4", "latentloop", (20.5, 4, "latentloop")),
        ("latentloop", 40, 8.0, "latentloop", (40.0, 8, "latentloop")),
    ),
)
def test_runtime_setting_validation(
    method, control_freq, query_interval, rollout_policy, expected
):
    assert (
        validate_runtime_settings(
            method, control_freq, query_interval, rollout_policy
        )
        == expected
    )


@pytest.mark.parametrize(
    ("method", "control_freq", "query_interval", "rollout_policy"),
    (
        ("baseline", 15, 2, "full"),
        ("baseline", 15, 4, "latentloop"),
        ("latentloop", 15, 4, "hold_action"),
        ("latentloop", 0, 4, "latentloop"),
        ("latentloop", float("inf"), 4, "latentloop"),
        ("latentloop", 15, 0, "latentloop"),
        ("latentloop", 15, 2.5, "latentloop"),
        ("latentloop", 15, True, "latentloop"),
    ),
)
def test_runtime_setting_validation_rejects_invalid_values(
    method, control_freq, query_interval, rollout_policy
):
    with pytest.raises(ValueError):
        validate_runtime_settings(method, control_freq, query_interval, rollout_policy)


def test_controller_runtime_setting_update_changes_schedule_inputs():
    controller = object.__new__(LatentLoopSeerControllerV2)
    controller.deployment_method = "latentloop"
    controller.control_freq = 15.0
    controller.query_interval = 4
    controller.rollout_policy = "latentloop"
    controller.args = SimpleNamespace(lrnode_query_interval=4)
    controller.step_records = []
    controller._rollout_complete = True

    result = controller.configure_runtime_settings(
        control_freq="30", query_interval="8", rollout_policy="latentloop"
    )

    assert result == (30.0, 8, "latentloop")
    assert controller.control_freq == 30.0
    assert controller.query_interval == 8
    assert controller.args.lrnode_query_interval == 8


def test_v2_shell_uses_existing_control_freq_name_and_defaults_to_live():
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "architectures/seer/upstream/scripts/REAL/deploy_ll_gui_v2.sh"
    ).read_text(encoding="utf-8")

    assert "control_hz" not in source
    assert "SEER_CONTROL_HZ" not in source
    assert "--execution-mode" not in source
    assert 'execution_mode="live"' in source
    assert 'export SEER_EXECUTION_MODE="${execution_mode}"' in source
    assert 'export SEER_CONTROL_FREQ="${control_freq}"' in source
    assert '--deployment-control-freq "${control_freq}"' in source


def test_execution_mode_is_owned_by_shell_environment(monkeypatch):
    monkeypatch.setenv("SEER_EXECUTION_MODE", "live")
    assert execution_mode_from_environment() == "live"
    monkeypatch.setenv("SEER_EXECUTION_MODE", "read_only_profile")
    assert execution_mode_from_environment() == "read_only_profile"
    monkeypatch.setenv("SEER_EXECUTION_MODE", "invalid")
    with pytest.raises(ValueError):
        execution_mode_from_environment()


def test_v2_result_metadata_uses_control_freq_and_records_rollout_settings(tmp_path):
    results_file = tmp_path / "deploy_results.json"
    results_file.write_text(
        json.dumps({"metadata": {"control_freq": 15.0}, "results": []}),
        encoding="utf-8",
    )
    app = object.__new__(LatentLoopDeployGuiAppV2)
    app.results_file = str(results_file)
    app.controller = SimpleNamespace(
        control_freq=20.0, query_interval=4, rollout_policy="latentloop"
    )
    app.runtime_settings_by_rollout = {
        "1": {
            "control_freq": 20.0,
            "query_interval": 4,
            "rollout_policy": "latentloop",
        }
    }

    app._normalize_v2_results_metadata()

    payload = json.loads(results_file.read_text(encoding="utf-8"))
    assert payload["metadata"]["control_freq"] == 15.0
    assert payload["metadata"]["query_interval"] == 4
    assert payload["metadata"]["rollout_policy"] == "latentloop"
    assert payload["metadata"]["runtime_settings_by_rollout"] == {
        "1": {
            "control_freq": 20.0,
            "query_interval": 4,
            "rollout_policy": "latentloop",
        }
    }


class _Value:
    def __init__(self, value):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


@pytest.mark.parametrize(
    ("method", "rollout_policy", "requested_k", "expected_k"),
    (
        ("baseline", "full", "99", 1),
        ("baseline", "hold_action", "8", 8),
        ("baseline", "hold_latent", "8", 8),
        ("latentloop", "latentloop", "8", 8),
    ),
)
def test_gui_runtime_settings_apply_to_controller_and_legacy_loop_field(
    method, rollout_policy, requested_k, expected_k
):
    controller = object.__new__(LatentLoopSeerControllerV2)
    controller.deployment_method = method
    controller.control_freq = 15.0
    controller.query_interval = 1 if rollout_policy == "full" else 4
    controller.rollout_policy = rollout_policy
    controller.args = SimpleNamespace(lrnode_query_interval=controller.query_interval)
    controller.step_records = []
    controller._rollout_complete = True

    app = object.__new__(LatentLoopDeployGuiAppV2)
    app.controller = controller
    app.cfg = SimpleNamespace(control_freq=15.0)
    app.runtime_control_freq_var = _Value("30")
    app.runtime_query_interval_var = _Value(requested_k)
    app.runtime_rollout_policy_var = _Value(rollout_policy)
    app.runtime_settings_status_var = _Value("")
    app._rollout_is_active = lambda: False
    app.write_session_files = lambda: None
    app.update_metrics = lambda: None
    app.set_status = lambda _message: None
    app._sync_query_interval_state = lambda: None

    assert app.apply_runtime_settings(show_error=False)
    assert controller.control_freq == 30.0
    assert controller.query_interval == expected_k
    assert controller.rollout_policy == rollout_policy
    assert controller.args.lrnode_query_interval == expected_k
    assert app.cfg.control_freq == 30.0


def test_hold_action_reuses_executed_environment_action_without_head_call():
    controller = object.__new__(LatentLoopSeerControllerV2)
    controller.cached_environment_action = torch.arange(7).numpy().astype("float32")
    controller.cached_age = 0
    controller.hold_action_calls = 0
    controller.hold_action_latency_sum_ms = 0.0

    action, record = controller._hold_action_forward()

    assert action.tolist() == list(range(7))
    assert record["mode"] == "hold_action"
    assert record["action_head_ms"] == 0.0
    assert controller.cached_age == 1
    assert controller.hold_action_calls == 1


def test_hold_latent_decodes_unchanged_cache_with_shared_action_head():
    latent = torch.ones(1, 3, 4)

    class _Model:
        @staticmethod
        def decode_action_from_latent(value):
            assert value is latent
            return torch.ones(1, 3, 6), torch.full((1, 3, 1), 0.75)

    controller = object.__new__(LatentLoopSeerControllerV2)
    controller.model = _Model()
    controller.cached_latent = latent
    controller.cached_age = 0
    controller.hold_latent_calls = 0
    controller.hold_latent_latency_sum_ms = 0.0
    controller._cuda_sync = lambda: None

    sequence, record = controller._hold_latent_forward()

    assert sequence.shape == (1, 3, 7)
    torch.testing.assert_close(sequence[..., :6], torch.ones(1, 3, 6))
    torch.testing.assert_close(sequence[..., 6:], torch.full((1, 3, 1), 0.75))
    assert controller.cached_latent is latent
    assert record["mode"] == "hold_latent"
    assert controller.cached_age == 1
    assert controller.hold_latent_calls == 1


def test_runtime_summary_exposes_60hz_deadline_and_mode_latency():
    controller = object.__new__(LatentLoopSeerControllerV2)
    controller.full_forward_calls = 1
    controller.latentloop_update_calls = 3
    controller.hold_action_calls = 0
    controller.hold_latent_calls = 0
    controller.full_forward_latency_sum_ms = 40.0
    controller.latentloop_latency_sum_ms = 30.0
    controller.hold_action_latency_sum_ms = 0.0
    controller.hold_latent_latency_sum_ms = 0.0
    controller.policy_latency_sum_ms = 70.0
    controller.control_freq = 60.0
    controller.rollout_index = 1
    controller.control_command_monotonic_s = [0.0, 0.04, 0.055, 0.07]
    controller.step_records = [
        {"mode": "full", "policy_ms": 40.0},
        {"mode": "latentloop", "policy_ms": 10.0},
        {"mode": "latentloop", "policy_ms": 10.0},
        {"mode": "latentloop", "policy_ms": 10.0},
    ]
    controller.deployment_metadata = lambda: {"deployment_method": "latentloop"}

    summary = controller.runtime_summary()

    assert summary["control_period_ms"] == pytest.approx(1000.0 / 60.0)
    assert summary["policy_latency_ms"]["full"]["mean"] == 40.0
    assert summary["policy_latency_ms"]["latentloop"]["mean"] == 10.0
    assert summary["policy_deadline_miss_count"] == 1
    assert summary["policy_deadline_miss_rate"] == 0.25
    assert summary["strict_deadline_miss_count"] == 1
