import json

import numpy as np
import pytest
import torch

from architectures.seer.adapters.latentloop_real_deploy.controller import (
    LatentLoopSeerController,
    is_allowed_teacher_missing_key,
    load_artifact_profile,
    remove_ddp_prefix,
    should_use_latentloop,
    temporal_ensemble_probability,
)


def test_remove_ddp_prefix_for_single_gpu_inference():
    tensor = torch.tensor([1.0])
    assert remove_ddp_prefix({"module.lrnode_x": tensor}) == {"lrnode_x": tensor}
    with pytest.raises(ValueError, match="collision"):
        remove_ddp_prefix({"module.x": tensor, "x": tensor})


@pytest.mark.parametrize(
    "key",
    [
        "attention_mask",
        "image_decoder_position_embedding",
        "vision_encoder.blocks.0.attn.qkv.weight",
        "clip_model.token_embedding.weight",
        "lrnode_dynamics.gate_bias",
    ],
)
def test_real_teacher_rebuilt_state_allowlist(key):
    assert is_allowed_teacher_missing_key(key)


def test_real_teacher_missing_trainable_state_is_rejected():
    assert not is_allowed_teacher_missing_key("action_decoder.0.weight")
    assert not is_allowed_teacher_missing_key("transformer_backbone.h.0.attn.c_attn.weight")


@pytest.mark.parametrize(
    ("timestep", "has_cache", "expected"),
    [
        (0, False, False),
        (0, True, False),
        (1, True, True),
        (2, True, True),
        (3, True, True),
        (4, True, False),
    ],
)
def test_k4_schedule(timestep, has_cache, expected):
    assert should_use_latentloop(timestep, 4, has_cache) is expected


def test_k1_schedule_never_uses_latentloop():
    assert should_use_latentloop(0, 1, True) is False
    assert should_use_latentloop(1, 1, True) is False


def test_temporal_ensemble_matches_legacy_probability_domain_rule():
    buffer = torch.zeros(6, 9, 7)
    first = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]] * 3], dtype=torch.float32
    )
    second = torch.tensor(
        [[[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]] * 3], dtype=torch.float32
    )

    action0, count0 = temporal_ensemble_probability(first, 0, buffer, 0.01)
    assert count0 == 1
    assert action0.dtype == torch.float64
    torch.testing.assert_close(action0, first[:, 0].double())

    action1, count1 = temporal_ensemble_probability(second, 1, buffer, 0.01)
    weights = np.exp(-0.01 * np.arange(2))
    weights /= weights.sum()
    expected = first[:, 1].double() * weights[0] + second[:, 0].double() * weights[1]
    assert count1 == 2
    torch.testing.assert_close(action1, expected)


def test_manifest_requires_teacher_specific_adapter_pair(tmp_path):
    manifest = {
        "schema_version": 1,
        "task": "basketball",
        "latentloop_architecture": {"hidden_dim": 256},
        "vit": {"filename": "vit.pth"},
        "teachers": {
            "37": {
                "filename": "teacher_37.pth",
                "adapters": {"39": {"filename": "teacher_37_adapter_39.pth"}},
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    profile = load_artifact_profile(path, teacher_id=37, adapter_id=39)
    assert profile["teacher"]["filename"] == "teacher_37.pth"
    assert profile["adapter"]["filename"] == "teacher_37_adapter_39.pth"
    with pytest.raises(KeyError, match="Teacher 34"):
        load_artifact_profile(path, teacher_id=34, adapter_id=39)

    baseline_profile = load_artifact_profile(path, teacher_id=37, adapter_id=None)
    assert baseline_profile["teacher"]["filename"] == "teacher_37.pth"
    assert baseline_profile["adapter"] is None
    assert baseline_profile["adapter_id"] is None


def test_completed_rollout_warmup_is_not_written_into_previous_summary():
    controller = object.__new__(LatentLoopSeerController)
    controller.history_len = 7
    controller.real_eval_max_steps = 20
    controller.action_pred_steps = 3
    controller.device_id = torch.device("cpu")
    controller.use_ensembling = False
    controller.rollout_index = 1
    controller._rollout_complete = False
    controller.step_records = [{"timestep": 0}]
    controller.control_command_monotonic_s = [1.0]
    writes = []
    controller.write_runtime_summary = lambda: writes.append("write")

    controller.mark_rollout_complete()
    assert controller._rollout_complete is True
    assert writes == ["write"]

    controller.reset()
    assert writes == ["write"]
    assert controller.rollout_index == 2
    assert controller._rollout_complete is False
    assert controller.step_records == []
    assert controller.control_command_monotonic_s == []


def test_only_real_rollout_control_commands_are_recorded():
    controller = object.__new__(LatentLoopSeerController)
    controller.rollout_index = 0
    controller._rollout_complete = False
    controller.control_command_monotonic_s = []
    controller.record_control_command(1.0)
    assert controller.control_command_monotonic_s == []

    controller.rollout_index = 1
    controller.record_control_command(2.0)
    assert controller.control_command_monotonic_s == [2.0]

    controller._rollout_complete = True
    controller.record_control_command(3.0)
    assert controller.control_command_monotonic_s == [2.0]


def test_runtime_summary_reports_measured_command_cadence():
    controller = object.__new__(LatentLoopSeerController)
    controller.full_forward_calls = 2
    controller.latentloop_update_calls = 1
    controller.full_forward_latency_sum_ms = 60.0
    controller.latentloop_latency_sum_ms = 5.0
    controller.policy_latency_sum_ms = 75.0
    controller.control_command_monotonic_s = [1.0, 1.04, 1.10]
    controller.target_control_hz = 20.0
    controller.rollout_index = 1
    controller.deployment_metadata = lambda: {}

    summary = controller.runtime_summary()
    assert summary["measured_control_period_count"] == 2
    assert summary["average_control_period_ms"] == pytest.approx(50.0)
    assert summary["achieved_control_hz"] == pytest.approx(20.0)
    assert summary["strict_deadline_miss_count"] == 1
    assert summary["strict_deadline_miss_rate"] == pytest.approx(0.5)


def test_synthetic_preflight_requires_complete_k4_cycle():
    controller = object.__new__(LatentLoopSeerController)
    controller.deployment_method = "latentloop"
    controller.query_interval = 4
    controller.full_forward_calls = 2
    controller.latentloop_update_calls = 3
    seen = []

    def fake_forward(_observation, include_info, timestep, record_step):
        assert include_info is True
        assert record_step is False
        seen.append(timestep)
        mode = "full" if timestep % 4 == 0 else "latentloop"
        return np.zeros(3), np.zeros(3), -1.0, {
            "mode": mode,
            "cache_age": 0 if mode == "full" else timestep,
        }

    resets = []
    controller.forward = fake_forward
    controller.reset = lambda write_previous: resets.append(write_previous)

    result = controller.run_synthetic_preflight("test instruction")
    assert seen == [0, 1, 2, 3, 4]
    assert result["modes"] == ["full", "latentloop", "latentloop", "latentloop", "full"]
    assert result["all_actions_finite"] is True
    assert resets == [False]


def test_synthetic_preflight_baseline_is_full_only():
    controller = object.__new__(LatentLoopSeerController)
    controller.deployment_method = "baseline"
    controller.query_interval = 1
    controller.full_forward_calls = 2
    controller.latentloop_update_calls = 0

    def fake_forward(_observation, include_info, timestep, record_step):
        assert include_info is True
        assert record_step is False
        return np.zeros(3), np.zeros(3), -1.0, {"mode": "full", "cache_age": 0}

    controller.forward = fake_forward
    controller.reset = lambda write_previous: None

    result = controller.run_synthetic_preflight("test instruction")
    assert result["modes"] == ["full", "full"]
    assert result["full_forward_calls"] == 2
    assert result["latentloop_update_calls"] == 0
