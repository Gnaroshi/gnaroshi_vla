from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from architectures.simvla.adapters.latentloop.action_equivalent_refresh.checkpoint import (
    MAX_PRIMARY_RISK_HEAD_PARAMETERS,
    load_action_fidelity_checkpoint,
    save_action_fidelity_checkpoint,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.features import (
    SimVLAActionFidelityFeatureConfig,
    build_simvla_action_fidelity_features,
    control_risk_scores,
    feature_names,
    runtime_feature_contract,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.extraction import (
    extract_all_anchor_fidelity_records,
    extract_compact_fidelity_records,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.policy import (
    ActionEquivalentRefreshSimVLAPolicy,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.training import (
    save_compact_action_fidelity_dataset,
    train_compact_action_fidelity_head,
)
from methods.latentloop.modules.action_equivalent_refresh import (
    ActionEquivalentRefreshRouter,
    ActionFidelityHead,
    ActionFidelityPrediction,
    CounterfactualActionTargets,
    ExactCallBudgetCalibration,
    action_fidelity_loss,
    counterfactual_action_targets,
    fit_exact_call_budget_calibration,
    fit_sequential_exact_call_budget_calibration,
    simulate_exact_fraction,
)
from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0UpdateOutput,
)


def _prediction(arm: float, gripper_logit: float) -> ActionFidelityPrediction:
    return ActionFidelityPrediction(
        arm_q90=torch.tensor([arm]),
        direction_q90=torch.tensor([0.1]),
        gripper_mismatch_logit=torch.tensor([gripper_logit]),
    )


def _calibration(threshold: float = 0.75) -> ExactCallBudgetCalibration:
    return ExactCallBudgetCalibration(
        arm_reference=(0.1, 0.2, 0.3, 0.4),
        gripper_reference=(0.1, 0.2, 0.3, 0.4),
        route_threshold=threshold,
        target_exact_fraction=0.25,
        observed_exact_fraction=0.25,
        max_approximate_age=3,
        calibration_queries=4,
    )


def test_counterfactual_targets_use_same_action_units_and_cosine() -> None:
    exact = torch.zeros(2, 5, 7)
    exact[0, :, 0] = 1.0
    exact[0, :, 3] = 1.0
    exact[0, :, 6] = 1.0
    exact[1, :, 0] = 1.0
    exact[1, :, 6] = -1.0
    approximate = exact.clone()
    target = counterfactual_action_targets(
        approximate,
        exact,
        arm_scale=torch.ones(6),
    )
    assert torch.equal(target.arm_normalized_l1, torch.zeros(2))
    assert torch.equal(target.direction_cosine_error, torch.zeros(2))
    assert target.direction_valid.tolist() == [True, True]
    assert torch.equal(target.gripper_mismatch, torch.zeros(2))

    approximate[0, :, 0] = -1.0
    approximate[1, 0, 6] = 1.0
    changed = counterfactual_action_targets(
        approximate,
        exact,
        arm_scale=torch.ones(6),
    )
    assert changed.arm_normalized_l1[0] > 0.0
    assert changed.direction_cosine_error[0] == pytest.approx(1.0)
    assert changed.gripper_mismatch.tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="six positive"):
        counterfactual_action_targets(
            approximate, exact, arm_scale=torch.ones(6) * 0.0
        )


def test_head_is_small_and_all_three_losses_backpropagate() -> None:
    config = SimVLAActionFidelityFeatureConfig()
    head = ActionFidelityHead(config.input_dim)
    prediction = head(torch.randn(4, config.input_dim))
    target = counterfactual_action_targets(
        torch.randn(4, 10, 7),
        torch.randn(4, 10, 7),
        arm_scale=torch.ones(6),
    )
    losses = action_fidelity_loss(prediction, target)
    assert set(losses) == {
        "loss",
        "arm_q90_pinball",
        "direction_q90_pinball",
        "gripper_bce",
    }
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
    assert head.parameter_audit()["total_parameters"] < MAX_PRIMARY_RISK_HEAD_PARAMETERS


def test_simvla_feature_schema_has_no_current_exact_input() -> None:
    torch.manual_seed(3)
    config = SimVLAActionFidelityFeatureConfig()
    batch, tokens, dimension = 2, 7, 960
    update = NativeV0UpdateOutput(
        condition=torch.randn(batch, tokens, dimension),
        residual=torch.randn(batch, tokens, dimension),
        gate=torch.sigmoid(torch.randn(batch, tokens, 1)),
    )
    valid = torch.ones(batch, tokens, dtype=torch.bool)
    groups = torch.arange(tokens).repeat(batch, 1) % config.num_token_groups
    features = build_simvla_action_fidelity_features(
        delta_feature=torch.randn(batch, config.delta_dim),
        update=update,
        valid_mask=valid,
        group_ids=groups,
        previous_action_chunk=torch.randn(batch, 10, 7),
        previous_proprio=torch.randn(batch, 8),
        current_proprio=torch.randn(batch, 8),
        candidate_age=torch.tensor([1, 3]),
        config=config,
    )
    assert features.shape == (batch, config.input_dim)
    assert len(feature_names(config)) == config.input_dim
    contract = runtime_feature_contract(config)
    assert "current exact condition" in contract["forbidden_runtime_inputs"]
    controls = control_risk_scores(
        update=update,
        valid_mask=valid,
        candidate_age=torch.tensor([1, 3]),
    )
    assert set(controls) == {"age_only", "applied_residual_rms", "gate_max"}
    assert controls["age_only"].tolist() == pytest.approx([1 / 3, 1.0])


def test_budget_calibration_is_deterministic_and_includes_forced_refreshes() -> None:
    arm = [0.10, 0.20, 0.90, 0.15, 0.30, 0.80, 0.05, 0.40] * 3
    gripper = [0.10, 0.20, 0.10, 0.80, 0.20, 0.10, 0.05, 0.30] * 3
    starts = [index % 8 == 0 for index in range(len(arm))]
    left = fit_exact_call_budget_calibration(
        arm,
        gripper,
        starts,
        target_exact_fraction=1 / 3,
        max_approximate_age=3,
    )
    right = fit_exact_call_budget_calibration(
        arm,
        gripper,
        starts,
        target_exact_fraction=1 / 3,
        max_approximate_age=3,
    )
    assert left == right
    assert left.observed_exact_fraction == pytest.approx(1 / 3)
    scores = [left.score_values(a, g) for a, g in zip(arm, gripper)]
    fraction, decisions = simulate_exact_fraction(
        scores,
        starts,
        threshold=left.route_threshold,
        max_approximate_age=3,
    )
    assert fraction == left.observed_exact_fraction
    assert len(decisions) == len(starts)


def test_sequential_calibration_uses_row_after_latest_exact_anchor() -> None:
    routing = [
        {"sequence_id": "s0", "anchor_offset": 0, "query_offset": 1},
        {"sequence_id": "s0", "anchor_offset": 0, "query_offset": 2},
        {"sequence_id": "s0", "anchor_offset": 0, "query_offset": 3},
        {"sequence_id": "s0", "anchor_offset": 1, "query_offset": 2},
        {"sequence_id": "s0", "anchor_offset": 1, "query_offset": 3},
        {"sequence_id": "s0", "anchor_offset": 2, "query_offset": 3},
    ]
    arm = [0.9, 0.1, 0.1, 0.8, 0.1, 0.7]
    gripper = [0.1] * len(arm)
    left = fit_sequential_exact_call_budget_calibration(
        arm,
        gripper,
        routing,
        target_exact_fraction=0.5,
    )
    right = fit_sequential_exact_call_budget_calibration(
        arm,
        gripper,
        routing,
        target_exact_fraction=0.5,
    )
    assert left == right
    assert left.observed_exact_fraction == pytest.approx(0.5)
    assert left.calibration_queries == 7


def test_router_forces_episode_start_and_max_age_without_changing_cadence() -> None:
    router = ActionEquivalentRefreshRouter(_calibration(threshold=1.1))
    assert router.candidate_required() is False
    assert router.decide().reason == "episode_start"
    for expected_age in (1, 2, 3):
        assert router.candidate_required() is True
        decision = router.decide(_prediction(0.1, -10.0))
        assert decision.use_exact is False
        assert decision.candidate_age == expected_age
    assert router.candidate_required() is False
    forced = router.decide()
    assert forced.use_exact is True
    assert forced.reason == "max_age"
    contract = router.contract()
    assert contract["changes_policy_query_cadence"] is False
    assert contract["changes_action_execution_horizon"] is False
    assert contract["changes_action_generation_schedule"] is False


def test_router_refreshes_on_predicted_counterfactual_risk() -> None:
    router = ActionEquivalentRefreshRouter(_calibration(threshold=0.75))
    router.decide()
    decision = router.decide(_prediction(0.4, 10.0))
    assert decision.use_exact is True
    assert decision.reason == "counterfactual_risk"
    assert router.approximate_age == 0


def test_checkpoint_round_trip_preserves_calibration_and_freezes_head(
    tmp_path: Path,
) -> None:
    config = SimVLAActionFidelityFeatureConfig()
    head = ActionFidelityHead(config.input_dim)
    path = save_action_fidelity_checkpoint(
        tmp_path / "risk.pt",
        head=head,
        feature_config=config,
        calibration=_calibration(),
        metadata={"split": "episode_disjoint_validation"},
    )
    loaded, loaded_config, calibration, payload = load_action_fidelity_checkpoint(
        path, device="cpu"
    )
    assert loaded_config == config
    assert calibration == _calibration()
    assert payload["metadata"]["split"] == "episode_disjoint_validation"
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    for name, value in head.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])


def test_compact_training_is_episode_disjoint_and_bounded(tmp_path: Path) -> None:
    torch.manual_seed(11)
    config = SimVLAActionFidelityFeatureConfig()

    def write(split: str, prefix: str, rows: int) -> Path:
        starts = torch.zeros(rows, dtype=torch.bool)
        starts[::4] = True
        episode_ids = [f"{prefix}-{index // 4}" for index in range(rows)]
        return save_compact_action_fidelity_dataset(
            tmp_path / f"{split}.pt",
            split=split,
            features=torch.randn(rows, config.input_dim),
            targets=CounterfactualActionTargets(
                arm_normalized_l1=torch.rand(rows),
                direction_cosine_error=torch.rand(rows),
                direction_valid=torch.ones(rows, dtype=torch.bool),
                gripper_mismatch=(torch.arange(rows) % 3 == 0).float(),
            ),
            episode_first_candidates=starts,
            episode_ids=episode_ids,
            feature_config=config,
            source_metadata={"same_noise": True},
        )

    train = write("train", "train", 12)
    validation = write("checkpoint_validation", "validation", 8)
    summary = train_compact_action_fidelity_head(
        train_data=train,
        validation_data=validation,
        output=tmp_path / "run",
        device="cpu",
        max_steps=2,
        batch_size=4,
    )
    assert summary["verdict"] == "ACTION_FIDELITY_HEAD_TRAINING_COMPLETE"
    assert summary["train_validation_episode_overlap"] == 0
    assert Path(summary["checkpoint"]).is_file()
    with pytest.raises(ValueError, match="bounded"):
        train_compact_action_fidelity_head(
            train_data=train,
            validation_data=validation,
            output=tmp_path / "too_long",
            device="cpu",
            max_steps=5_001,
        )


def test_compact_extraction_preserves_sequence_order_and_same_noise() -> None:
    torch.manual_seed(19)
    config = SimVLAActionFidelityFeatureConfig(delta_dim=8)
    adapter = NativeSimVLAV0(
        condition_dim=16,
        delta_dim=8,
        rank_dim=64,
        max_tokens=8,
    ).eval()
    batch, tokens = 2, 5
    calls: list[torch.Tensor] = []

    def decode(condition: torch.Tensor, proprio: torch.Tensor, noise: torch.Tensor):
        calls.append(noise.detach().clone())
        base = condition.mean(dim=(1, 2)).reshape(batch, 1, 1)
        return noise + base + proprio[:, :1].reshape(batch, 1, 1)

    records = extract_compact_fidelity_records(
        condition_adapter=adapter,
        decode_same_noise=decode,
        anchor_condition=torch.randn(batch, tokens, 16),
        exact_conditions=torch.randn(batch, 3, tokens, 16),
        image_sequence=torch.randint(
            0, 256, (batch, 4, 2, 12, 12, 3), dtype=torch.uint8
        ),
        proprio_sequence=torch.randn(batch, 4, 8),
        explicit_noises=torch.randn(batch, 3, 10, 7),
        initial_action_chunk=torch.randn(batch, 10, 7),
        valid_mask=torch.ones(batch, tokens, dtype=torch.bool),
        group_ids=torch.arange(tokens).repeat(batch, 1),
        episode_ids=("episode-a", "episode-b"),
        arm_scale=torch.ones(6),
        feature_config=config,
    )
    assert records.features.shape == (6, config.input_dim)
    assert records.episode_first_candidates.tolist() == [True, False, False] * 2
    assert records.episode_ids == ("episode-a",) * 3 + ("episode-b",) * 3
    assert len(calls) == 6
    for age in range(3):
        assert torch.equal(calls[2 * age], calls[2 * age + 1])


def test_all_anchor_extraction_materializes_reachable_triangular_rows() -> None:
    torch.manual_seed(23)
    config = SimVLAActionFidelityFeatureConfig(delta_dim=8)
    adapter = NativeSimVLAV0(
        condition_dim=16,
        delta_dim=8,
        rank_dim=64,
        max_tokens=8,
    ).eval()
    batch, tokens = 1, 5
    noises = torch.randn(batch, 4, 10, 7)
    calls: list[torch.Tensor] = []

    def decode(condition: torch.Tensor, proprio: torch.Tensor, noise: torch.Tensor):
        calls.append(noise.detach().clone())
        return noise + condition.mean(dim=(1, 2)).reshape(batch, 1, 1)

    records = extract_all_anchor_fidelity_records(
        condition_adapter=adapter,
        decode_same_noise=decode,
        exact_conditions=torch.randn(batch, 4, tokens, 16),
        image_sequence=torch.randint(
            0, 256, (batch, 4, 2, 12, 12, 3), dtype=torch.uint8
        ),
        proprio_sequence=torch.randn(batch, 4, 8),
        explicit_noises=noises,
        valid_mask=torch.ones(batch, tokens, dtype=torch.bool),
        group_ids=torch.arange(tokens).repeat(batch, 1),
        episode_ids=("episode-a",),
        sequence_ids=("sequence-a",),
        arm_scale=torch.ones(6),
        feature_config=config,
    )
    assert records.features.shape == (6, config.input_dim)
    assert [(value["anchor_offset"], value["query_offset"]) for value in records.routing_records] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert len(calls) == 10  # Four exact actions plus six candidates.
    assert torch.equal(calls[1], calls[4])
    assert torch.equal(calls[2], calls[5])
    assert torch.equal(calls[3], calls[6])
    assert torch.equal(calls[2], calls[7])
    assert torch.equal(calls[3], calls[8])
    assert torch.equal(calls[3], calls[9])


def test_policy_source_locks_h10_r5_ng3_and_routes_before_decode() -> None:
    source = inspect.getsource(ActionEquivalentRefreshSimVLAPolicy)
    tree = ast.parse(source)
    assert 'n_g=3' in source
    assert 'k_c=4' in source
    assert 'action_chunk[0, :5]' in source
    assert '"action_horizon": 10' in source
    assert '"execution_horizon": 5' in source
    refill_source = inspect.getsource(
        ActionEquivalentRefreshSimVLAPolicy._refill_action_queue
    )
    candidate_line = refill_source.index("_candidate_update")
    route_line = refill_source.index("self.refresh_router.decide(prediction)")
    decode_line = refill_source.index("_commit_approximate")
    assert candidate_line < route_line < decode_line
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    assert assignments
