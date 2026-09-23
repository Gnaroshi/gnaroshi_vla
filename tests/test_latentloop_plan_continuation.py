from __future__ import annotations

import json

import numpy as np
import torch

from architectures.seer.adapters.latentloop_plan_continuation.token_alignment import (
    action_token_time_mapping,
    assert_canonical_seer_alignment,
    verified_overlap_pairs,
)
from architectures.seer.adapters.latentloop_plan_continuation.trace_adapter import (
    load_plan_trace_shard,
    save_plan_trace_episode,
)
from methods.latentloop_plan_continuation.action_correction import (
    MatchedActionSpaceCorrection,
    find_action_correction_hidden_dim,
    shift_action_horizon,
)
from methods.latentloop_plan_continuation.anchor_bridge import (
    NonRecurrentAnchorToCurrentBridge,
    find_anchor_bridge_hidden_dim,
)
from methods.latentloop_plan_continuation.cqpc_loss import (
    cqpc_is_enabled,
    cross_query_plan_consistency_loss,
)
from methods.latentloop_plan_continuation.decisions import (
    apply_plan_continuation_decision,
)
from methods.latentloop_plan_continuation.feedback_source import (
    FeedbackFeatureBuffer,
)
from methods.latentloop_plan_continuation.feedback_metrics import (
    clustered_spearman_interval,
    paired_clustered_mean_difference_interval,
)
from methods.latentloop_plan_continuation.overlap_metrics import (
    aligned_overlap_components,
)
from tools.seer.build_plan_continuation_evidence import (
    ASSOCIATION_DRIVERS,
    ASSOCIATION_RESPONSES,
    _matched_baseline_clearly_exceeds,
    _matched_baseline_not_explanation,
)
from tools.seer.analyze_latentloop_plan_continuation import (
    PRIMARY_ASSOCIATION_DRIVERS,
    PRIMARY_ASSOCIATION_RESPONSES,
    _feedback_association_population,
)


def test_exact_token_time_mapping_and_overlap_pairs() -> None:
    mapping = action_token_time_mapping(7, 3)
    assert [item["intended_execution_time"] for item in mapping] == [7, 8, 9]
    assert verified_overlap_pairs(3) == [(1, 0), (2, 1)]
    assert_canonical_seer_alignment(3)


def test_feedback_inference_family_matches_frozen_decision_rule() -> None:
    assert PRIMARY_ASSOCIATION_DRIVERS == ASSOCIATION_DRIVERS
    assert PRIMARY_ASSOCIATION_RESPONSES == ASSOCIATION_RESPONSES


def test_feedback_population_excludes_full_refresh_when_updates_exist() -> None:
    rows = [
        {"mode": "stepwise", "timestep": 1},
        {"mode": "full", "timestep": 4},
        {"mode": "stepwise", "timestep": 5},
    ]
    selected, population = _feedback_association_population(rows)
    assert [row["timestep"] for row in selected] == [1, 5]
    assert population == "intermediate_non_full_transitions"


def test_feedback_population_keeps_full_only_k1_reference() -> None:
    rows = [{"mode": "full", "timestep": 1}]
    selected, population = _feedback_association_population(rows)
    assert selected == rows
    assert population == "full_only_reference_transitions"


def test_overlap_pair_construction_is_component_exact() -> None:
    previous = np.arange(21, dtype=np.float64).reshape(3, 7)
    current = np.stack([previous[1], previous[2], previous[2]])
    rows = aligned_overlap_components(previous, current, [(1, 0), (2, 1)])
    assert len(rows) == 2
    assert all(float(row["all_token_l2"]) == 0.0 for row in rows)


def test_shifted_horizon_repeats_only_unmatched_boundary() -> None:
    horizon = torch.tensor([[[0.0], [1.0], [2.0]]])
    shifted = shift_action_horizon(horizon)
    torch.testing.assert_close(shifted, torch.tensor([[[1.0], [2.0], [2.0]]]))


def test_action_correction_has_separate_arm_and_gripper_logit_heads() -> None:
    module = MatchedActionSpaceCorrection(
        action_pred_steps=3, motion_dim=4, hidden_dim=16, time_dim=4, token_dim=4
    )
    for parameter in module.parameters():
        torch.nn.init.zeros_(parameter)
    arm = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6) / 20.0
    gripper_logit = torch.tensor([[[-2.0], [0.0], [2.0]]])
    output = module(
        shift_action_horizon(arm),
        shift_action_horizon(gripper_logit),
        torch.zeros(1, 4),
        age=1.0,
    )
    assert module.arm_head is not module.gripper_head
    assert output.arm.shape == (1, 3, 6)
    assert output.gripper_logit.shape == (1, 3, 1)
    torch.testing.assert_close(
        output.gripper_probability, torch.sigmoid(output.gripper_logit)
    )
    assert output.gripper_logit.dtype.is_floating_point
    assert set(torch.unique(output.gripper_logit > 0.0).tolist()) <= {False, True}


def test_matched_baseline_predictor_parameter_counts_are_within_tolerance() -> None:
    target = 338_690
    action_hidden, action_count, action_error = find_action_correction_hidden_dim(
        target
    )
    anchor_hidden, anchor_count, anchor_error = find_anchor_bridge_hidden_dim(target)
    assert action_hidden == 486
    assert action_count == 338_507
    assert action_error < 0.001
    assert anchor_hidden == 250
    assert anchor_count == 338_925
    assert anchor_error < 0.001


def test_time_shifted_feature_source_initialization_and_reset() -> None:
    buffer = FeedbackFeatureBuffer("time_shifted")
    first = buffer.select(torch.tensor([[1.0, 2.0]]), current_step=10)
    assert first.source_step == -1
    assert first.initialized_with_zero
    torch.testing.assert_close(first.feature, torch.zeros_like(first.feature))
    second = buffer.select(torch.tensor([[3.0, 4.0]]), current_step=11)
    assert second.source_step == 10
    torch.testing.assert_close(second.feature, torch.tensor([[1.0, 2.0]]))
    buffer.reset()
    after_reset = buffer.select(torch.tensor([[5.0, 6.0]]), current_step=0)
    assert after_reset.source_step == -1
    torch.testing.assert_close(after_reset.feature, torch.zeros_like(after_reset.feature))


def test_current_feature_source_uses_same_step() -> None:
    selection = FeedbackFeatureBuffer("current").select(
        torch.tensor([[1.0]]), current_step=3
    )
    assert selection.source_step == 3
    assert not selection.initialized_with_zero


def test_fixed_rank_clustered_spearman_is_deterministic() -> None:
    rows = []
    for task_id in range(2):
        for episode_id in range(3):
            for step in range(5):
                value = float(task_id * 100 + episode_id * 10 + step)
                rows.append(
                    {
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "x": value,
                        "y": value,
                    }
                )
    first = clustered_spearman_interval(
        rows, "x", "y", iterations=512, seed=7
    )
    second = clustered_spearman_interval(
        rows, "x", "y", iterations=512, seed=7
    )
    assert first == second
    assert np.isclose(first["rho"], 1.0)
    assert np.isclose(first["ci_low"], 1.0)
    assert np.isclose(first["ci_high"], 1.0)
    assert first["iterations"] == 512
    assert first["valid_tasks"] == 2
    assert first["valid_episodes"] == 6
    assert first["bootstrap_method"] == "hierarchical_task_episode_fixed_rank"


def test_secondary_spearman_skips_bootstrap_but_keeps_point_estimate() -> None:
    rows = [
        {"task_id": 0, "episode_id": 0, "x": value, "y": -value}
        for value in range(5)
    ]
    result = clustered_spearman_interval(rows, "x", "y", iterations=0)
    assert np.isclose(result["rho"], -1.0)
    assert np.isnan(result["ci_low"])
    assert np.isnan(result["ci_high"])
    assert result["iterations"] == 0
    assert result["bootstrap_method"] == "not_run"


def test_nonvarying_spearman_has_no_bootstrap_interval() -> None:
    rows = [
        {"task_id": 0, "episode_id": 0, "x": 0.0, "y": float(value)}
        for value in range(5)
    ]
    result = clustered_spearman_interval(rows, "x", "y", iterations=512)
    assert np.isnan(result["rho"])
    assert np.isnan(result["ci_low"])
    assert np.isnan(result["ci_high"])
    assert result["iterations"] == 0
    assert result["bootstrap_method"] == "undefined_nonvarying_input"


def test_vectorized_paired_cluster_bootstrap_preserves_constant_difference() -> None:
    left = {}
    right = {}
    for task_id in range(3):
        for episode_id in range(4):
            key = (task_id, episode_id)
            right[key] = float(task_id + episode_id)
            left[key] = right[key] + 0.25
    result = paired_clustered_mean_difference_interval(
        left, right, iterations=1024, seed=11
    )
    assert np.isclose(result["mean_difference"], 0.25)
    assert np.isclose(result["ci_low"], 0.25)
    assert np.isclose(result["ci_high"], 0.25)
    assert result["paired_episodes"] == 12
    assert result["iterations"] == 1024
    assert result["bootstrap_method"] == "hierarchical_task_episode_multinomial"


def test_anchor_bridge_is_nonrecurrent() -> None:
    torch.manual_seed(4)
    bridge = NonRecurrentAnchorToCurrentBridge(
        latent_dim=8, motion_dim=4, action_pred_steps=3, hidden_dim=16, age_dim=4
    )
    anchor = torch.randn(2, 3, 8)
    feature = torch.randn(2, 4)
    first = bridge(anchor, feature, age=2.0).latent
    irrelevant_previous_prediction = torch.randn_like(first)
    del irrelevant_previous_prediction
    second = bridge(anchor, feature, age=2.0).latent
    torch.testing.assert_close(first, second)


def test_lambda_cqpc_zero_disables_graph_work() -> None:
    assert not cqpc_is_enabled(0.0)
    assert not cqpc_is_enabled(-1.0)
    assert cqpc_is_enabled(1e-8)


def test_cqpc_weighting_and_previous_branch_detach() -> None:
    predicted_arm = torch.zeros(1, 3, 6, requires_grad=True)
    predicted_grip = torch.zeros(1, 3, 1, requires_grad=True)
    previous_arm = torch.ones(1, 3, 6, requires_grad=True)
    previous_grip = torch.ones(1, 3, 1, requires_grad=True)
    teacher_arm = previous_arm.detach().clone()
    teacher_grip = previous_grip.detach().clone()
    output = cross_query_plan_consistency_loss(
        predicted_arm,
        predicted_grip,
        previous_arm,
        previous_grip,
        teacher_arm,
        teacher_grip,
        gamma=2.0,
        lambda_arm=0.5,
        lambda_gripper=0.25,
    )
    assert torch.all((output.teacher_weight >= 0.0) & (output.teacher_weight <= 1.0))
    torch.testing.assert_close(
        output.total,
        0.5 * output.arm_weighted + 0.25 * output.gripper_weighted,
    )
    output.total.backward()
    assert predicted_arm.grad is not None
    assert predicted_grip.grad is not None
    assert previous_arm.grad is None
    assert previous_grip.grad is None


def test_plan_trace_save_reload_round_trip(tmp_path) -> None:
    scalars = [
        {
            "row_id": "dense",
            "paired_group": "g",
            "task_id": 0,
            "episode_id": 0,
            "timestep": 0,
            "mode": "full",
            "cache_age": 0,
            "feature_source_step": 0,
            "full_refresh_flag": 1,
            "primary_raw_change_l1": 0.0,
            "wrist_raw_change_l1": 0.0,
            "proprio_delta_l2": 0.0,
            "u_delta_norm": 0.0,
        }
    ]
    tensors = [
        {
            "raw_action_arm": torch.zeros(3, 6),
            "raw_gripper_logit": torch.zeros(3, 1),
            "raw_gripper_probability": torch.full((3, 1), 0.5),
            "raw_gripper_thresholded": torch.zeros(3, 1),
            "post_ensemble_probability": torch.zeros(7),
            "executed_action": torch.zeros(7),
            "proprio_delta": torch.zeros(8),
        }
    ]
    paths = save_plan_trace_episode(tmp_path, "episode", scalars, tensors, {"success": 1})
    metadata, loaded_scalars, arrays = load_plan_trace_shard(paths["json"])
    assert metadata["schema_version"] == 1
    assert loaded_scalars[0]["row_id"] == "dense"
    np.testing.assert_array_equal(arrays["raw_action_arm"], np.zeros((1, 3, 6)))


def test_decision_rule_supported_inconclusive_and_refuted() -> None:
    required = [f"criterion_{index}" for index in range(1, 7)]
    rule = {
        "required_boolean_fields": required,
        "not_supported_boolean_fields": ["refutation"],
    }
    supported = apply_plan_continuation_decision(
        {**{key: True for key in required}, "refutation": False}, rule
    )
    assert supported["verdict"] == "PLAN_CONTINUATION_SUPPORTED"
    inconclusive = apply_plan_continuation_decision(
        {**{key: key != "criterion_2" for key in required}, "refutation": False},
        rule,
    )
    assert inconclusive["verdict"] == "PLAN_CONTINUATION_INCONCLUSIVE"
    refuted = apply_plan_continuation_decision(
        {**{key: True for key in required}, "refutation": True}, rule
    )
    assert refuted["verdict"] == "PLAN_CONTINUATION_NOT_SUPPORTED"


def test_explicit_refutation_precedes_missing_matched_baselines() -> None:
    rule = {
        "required_boolean_fields": ["phase_a", "matched_action", "matched_anchor"],
        "not_supported_boolean_fields": ["phase_a_refutation"],
    }
    result = apply_plan_continuation_decision(
        {"phase_a": False, "phase_a_refutation": True}, rule
    )
    assert result["verdict"] == "PLAN_CONTINUATION_NOT_SUPPORTED"
    assert result["missing"] == ["matched_action", "matched_anchor"]
    assert result["not_supported_reasons"] == ["phase_a_refutation"]


def test_missing_matched_baselines_remains_inconclusive_without_refutation() -> None:
    rule = {
        "required_boolean_fields": ["phase_a", "matched_action"],
        "not_supported_boolean_fields": ["phase_a_refutation"],
    }
    result = apply_plan_continuation_decision(
        {"phase_a": True, "phase_a_refutation": False}, rule
    )
    assert result["verdict"] == "PLAN_CONTINUATION_INCONCLUSIVE"
    assert result["missing"] == ["matched_action"]


def test_indistinguishable_matched_baseline_is_not_automatic_refutation() -> None:
    sr_interval = {"ci_low": -0.03, "ci_high": 0.04}
    overlap = {
        metric: {"ci_low": -0.02, "ci_high": 0.02}
        for metric in ("translation_l1", "translation_l2", "rotation_l1", "rotation_l2")
    }
    assert not _matched_baseline_not_explanation(sr_interval, overlap)
    assert not _matched_baseline_clearly_exceeds(sr_interval, overlap)
