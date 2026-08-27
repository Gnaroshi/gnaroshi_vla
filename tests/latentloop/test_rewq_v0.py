from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from architectures.simvla.adapters.latentloop.rewq_v0.checkpoint import (
    load_rewq_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.rewq_v0.data import (
    save_recovery_dataset,
    save_safe_reference,
)
from architectures.simvla.adapters.latentloop.rewq_v0.features import (
    SimVLARecoverabilityFeatureConfig,
    build_simvla_recoverability_features,
    runtime_feature_contract,
)
from architectures.simvla.adapters.latentloop.rewq_v0.training import train_rewq_v0
from architectures.simvla.adapters.latentloop.rewq_v0.policy import RewqV0SimVLAPolicy
from methods.latentloop.modules.native_simvla_v0 import NativeV0UpdateOutput
from methods.latentloop.modules.rewq_v0 import (
    COMPUTE_MODES,
    CONTINUOUS_TARGET_NAMES,
    EVENT_TARGET_NAMES,
    ComputeCostTable,
    NextAnchorRecoveryTargets,
    RecoverabilityHead,
    RecoverabilityRouter,
    RecoveryPrediction,
    RecoverySafetyEnvelope,
    SplitConformalCalibration,
    fit_recovery_safety_envelope,
    fit_split_conformal_calibration,
    next_anchor_recovery_targets,
    recoverability_loss,
)


def _costs() -> ComputeCostTable:
    return ComputeCostTable(
        exact_condition_ms=10.0,
        approximate_condition_ms=2.0,
        generation_ng3_ms=6.0,
        generation_ng2_ms=3.0,
        router_ms=0.5,
        provenance="unit-test measured components",
    )


def _envelope(limit: float = 0.2) -> RecoverySafetyEnvelope:
    return RecoverySafetyEnvelope(
        continuous_limits=(limit,) * len(CONTINUOUS_TARGET_NAMES),
        event_probability_limits=(limit,) * len(EVENT_TARGET_NAMES),
        continuous_quantile=0.95,
        confidence=0.95,
        reference_rows=100,
        reference_row="condition_kc2_ng3",
        provenance="unit-test successful rows",
    )


def _conformal(offset: float = 0.0) -> SplitConformalCalibration:
    return SplitConformalCalibration(
        continuous_offsets=tuple(
            (offset,) * len(CONTINUOUS_TARGET_NAMES) for _ in range(3)
        ),
        event_offsets=tuple((offset,) * len(EVENT_TARGET_NAMES) for _ in range(3)),
        alpha=0.10,
        calibration_rows=300,
        provenance="episode-disjoint unit-test rows",
    )


def _prediction(values: tuple[float, float, float]) -> RecoveryPrediction:
    continuous = torch.tensor(values).reshape(1, 3, 1).expand(
        1, 3, len(CONTINUOUS_TARGET_NAMES)
    )
    events = torch.full((1, 3, len(EVENT_TARGET_NAMES)), -8.0)
    return RecoveryPrediction(continuous_q90=continuous, event_logits=events)


def _random_targets(rows: int) -> NextAnchorRecoveryTargets:
    return NextAnchorRecoveryTargets(
        continuous=torch.rand(rows, 3, len(CONTINUOUS_TARGET_NAMES)) * 0.1,
        continuous_valid=torch.ones(
            rows, 3, len(CONTINUOUS_TARGET_NAMES), dtype=torch.bool
        ),
        events=torch.zeros(rows, 3, len(EVENT_TARGET_NAMES)),
        mode_valid=torch.ones(rows, 3, dtype=torch.bool),
    )


def test_compute_modes_and_costs_include_rejected_candidate_probe() -> None:
    assert [mode.name for mode in COMPUTE_MODES] == [
        "exact_condition_ng3",
        "approx_condition_ng3",
        "exact_condition_ng2",
        "approx_condition_ng2",
    ]
    values = _costs().values(device=torch.device("cpu"), dtype=torch.float32)
    assert values.tolist() == pytest.approx([18.5, 8.5, 15.5, 5.5])
    assert _costs().to_dict()["mode_cost_ms"]["exact_condition_ng3"] == 18.5


def test_next_anchor_targets_are_zero_for_identical_paired_branches() -> None:
    batch, modes, horizon = 2, 3, 10
    reference_action = torch.randn(batch, horizon, 7)
    reference_proprio = torch.randn(batch, 8)
    reference_ee = torch.randn(batch, 6)
    reference_scene = torch.randn(batch, 16)
    reference_gripper = torch.tensor([True, False])
    reference_contact = torch.tensor([False, True])
    targets = next_anchor_recovery_targets(
        candidate_next_action=reference_action.unsqueeze(1).expand(-1, modes, -1, -1),
        reference_next_action=reference_action,
        candidate_post_proprio=reference_proprio.unsqueeze(1).expand(-1, modes, -1),
        reference_post_proprio=reference_proprio,
        candidate_post_ee=reference_ee.unsqueeze(1).expand(-1, modes, -1),
        reference_post_ee=reference_ee,
        candidate_post_scene_feature=reference_scene.unsqueeze(1).expand(-1, modes, -1),
        reference_post_scene_feature=reference_scene,
        candidate_post_gripper_closed=reference_gripper.unsqueeze(1).expand(-1, modes),
        reference_post_gripper_closed=reference_gripper,
        candidate_post_contact=reference_contact.unsqueeze(1).expand(-1, modes),
        reference_post_contact=reference_contact,
        action_scale=torch.ones(6),
        proprio_scale=torch.ones(8),
        ee_scale=torch.ones(6),
    )
    assert torch.allclose(
        targets.continuous, torch.zeros_like(targets.continuous), atol=1e-6
    )
    assert torch.equal(targets.events, torch.zeros_like(targets.events))
    assert targets.mode_valid.all()

    changed_action = reference_action.unsqueeze(1).expand(-1, modes, -1, -1).clone()
    changed_action[:, 2, :, :6] *= -1
    changed_gripper = reference_gripper.unsqueeze(1).expand(-1, modes).clone()
    changed_gripper[:, 2] = ~changed_gripper[:, 2]
    changed = next_anchor_recovery_targets(
        candidate_next_action=changed_action,
        reference_next_action=reference_action,
        candidate_post_proprio=reference_proprio.unsqueeze(1).expand(-1, modes, -1),
        reference_post_proprio=reference_proprio,
        candidate_post_ee=reference_ee.unsqueeze(1).expand(-1, modes, -1),
        reference_post_ee=reference_ee,
        candidate_post_scene_feature=reference_scene.unsqueeze(1).expand(-1, modes, -1),
        reference_post_scene_feature=reference_scene,
        candidate_post_gripper_closed=changed_gripper,
        reference_post_gripper_closed=reference_gripper,
        candidate_post_contact=reference_contact.unsqueeze(1).expand(-1, modes),
        reference_post_contact=reference_contact,
        action_scale=torch.ones(6),
        proprio_scale=torch.ones(8),
        ee_scale=torch.ones(6),
    )
    assert changed.continuous[:, 2, 0].gt(0).all()
    assert changed.continuous[:, 2, 1].gt(1.9).all()
    assert changed.events[:, 2, 0].eq(1).all()


def test_feature_contract_is_gpu_native_and_has_no_teacher_leakage() -> None:
    torch.manual_seed(8)
    config = SimVLARecoverabilityFeatureConfig(delta_dim=16, condition_summary_bins=4)
    batch, tokens, dimension = 2, 9, 32
    update = NativeV0UpdateOutput(
        condition=torch.randn(batch, tokens, dimension),
        residual=torch.randn(batch, tokens, dimension),
        gate=torch.sigmoid(torch.randn(batch, tokens, 1)),
    )
    features = build_simvla_recoverability_features(
        delta_feature=torch.randn(batch, config.delta_dim),
        update=update,
        anchor_condition=torch.randn(batch, tokens, dimension),
        valid_mask=torch.ones(batch, tokens, dtype=torch.bool),
        group_ids=torch.arange(tokens).repeat(batch, 1) % config.num_token_groups,
        previous_action_chunk=torch.randn(batch, 10, 7),
        previous_proprio=torch.randn(batch, 8),
        current_proprio=torch.randn(batch, 8),
        candidate_age=torch.tensor([1, 7]),
        config=config,
    )
    assert features.shape == (batch, config.input_dim)
    contract = runtime_feature_contract(config)
    assert "current exact condition" in contract["forbidden_runtime_inputs"]
    assert "simulator object state" in contract["forbidden_runtime_inputs"]


def test_head_stays_below_100k_and_all_targets_backpropagate() -> None:
    head = RecoverabilityHead(input_dim=363)
    prediction = head(torch.randn(8, 363))
    target = _random_targets(8)
    losses = recoverability_loss(
        prediction,
        target,
        normalization=torch.ones(len(CONTINUOUS_TARGET_NAMES)),
    )
    losses["loss"].backward()
    assert head.parameter_audit()["within_parameter_ceiling"] is True
    assert head.parameter_audit()["total_parameters"] < 100_000
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_safe_envelope_is_locked_to_successful_kc2_ng3_reference() -> None:
    envelope = fit_recovery_safety_envelope(
        torch.rand(100, len(CONTINUOUS_TARGET_NAMES)) * 0.1,
        torch.zeros(100, len(EVENT_TARGET_NAMES)),
        provenance="paired successful seed01/seed02 rows",
    )
    assert envelope.reference_row == "condition_kc2_ng3"
    assert envelope.reference_rows == 100
    assert max(envelope.event_probability_limits) < 0.05
    with pytest.raises(ValueError, match="condition_kc2_ng3"):
        RecoverySafetyEnvelope(
            continuous_limits=(0.1,) * len(CONTINUOUS_TARGET_NAMES),
            event_probability_limits=(0.1,) * len(EVENT_TARGET_NAMES),
            continuous_quantile=0.95,
            confidence=0.95,
            reference_rows=100,
            reference_row="periodic_kc4_ng3",
            provenance="invalid",
        )


def test_split_conformal_offsets_cover_positive_residuals() -> None:
    target = _random_targets(60)
    prediction = RecoveryPrediction(
        continuous_q90=torch.zeros_like(target.continuous),
        event_logits=torch.full_like(target.events, -8.0),
    )
    calibration = fit_split_conformal_calibration(
        prediction,
        target,
        provenance="episode-disjoint validation",
    )
    offsets = torch.tensor(calibration.continuous_offsets)
    assert offsets.shape == (3, len(CONTINUOUS_TARGET_NAMES))
    assert offsets.gt(0).all()


def test_router_picks_cheapest_safe_mode_and_never_bans_kc4_or_kc8_globally() -> None:
    router = RecoverabilityRouter(
        envelope=_envelope(),
        conformal=_conformal(),
        costs=_costs(),
        max_approximate_age=7,
    )
    assessment = router.assess(
        _prediction((0.1, 0.3, 0.15)),
        candidate_age=torch.tensor([3]),
        anchor_available=torch.tensor([True]),
    )
    assert assessment.selected_mode_id.tolist() == [3]
    assert assessment.admissible.tolist() == [[True, True, False, True]]

    # Age 3 corresponds to a K_C=4 candidate; age 7 to K_C=8.  Both remain
    # eligible when their predicted future consequence is inside the envelope.
    for age in (3, 7):
        selected = router.assess(
            _prediction((0.1, 0.3, 0.1)),
            candidate_age=torch.tensor([age]),
            anchor_available=torch.tensor([True]),
        )
        assert selected.admissible[0, 1]
        assert selected.admissible[0, 3]
        assert selected.selected_mode_id.tolist() == [3]

    forced_exact = router.assess(
        _prediction((0.1, 0.1, 0.1)),
        candidate_age=torch.tensor([8]),
        anchor_available=torch.tensor([True]),
    )
    assert not forced_exact.admissible[0, 1]
    assert not forced_exact.admissible[0, 3]
    assert forced_exact.selected_mode_id.tolist() == [0]

    no_anchor = router.assess(
        _prediction((0.1, 0.1, 0.1)),
        candidate_age=torch.tensor([1]),
        anchor_available=torch.tensor([False]),
    )
    assert no_anchor.selected_mode_id.tolist() == [0]


def test_router_scoring_source_has_no_cpu_round_trip() -> None:
    source = inspect.getsource(RecoverabilityRouter.assess)
    for forbidden in (".cpu(", ".tolist(", ".item(", "bisect"):
        assert forbidden not in source
    assert "argmin" in source


def test_policy_keeps_h10_r5_and_saturates_only_frozen_uc_age_embedding() -> None:
    refill = inspect.getsource(RewqV0SimVLAPolicy._refill_action_queue)
    candidate = inspect.getsource(RewqV0SimVLAPolicy._candidate_update)
    assert "action_chunk[0, :5]" in refill
    assert "candidate_decoded_before_routing" in inspect.getsource(
        RewqV0SimVLAPolicy.scientific_contract
    )
    assert "min(int(candidate_age), 3)" in candidate
    assert "assessment.selected_mode_id.item()" in refill
    assert refill.index("_candidate_update") < refill.index("_commit_approximate")


def test_bounded_training_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(31)
    config = SimVLARecoverabilityFeatureConfig(delta_dim=8, condition_summary_bins=2)

    def write(split: str, prefix: str, rows: int) -> Path:
        return save_recovery_dataset(
            tmp_path / f"{split}.pt",
            split=split,
            features=torch.randn(rows, config.input_dim),
            targets=_random_targets(rows),
            candidate_age=(torch.arange(rows) % config.max_age) + 1,
            episode_ids=[f"{prefix}-{index // 3}" for index in range(rows)],
            feature_config=config,
            source_metadata={
                "paired_environment_initialization": True,
                "paired_action_noise": True,
                "candidate_queries": 1,
                "exact_recovery_queries": 1,
                "total_environment_actions": 10,
            },
        )

    train_path = write("train", "train", 60)
    validation_path = write("checkpoint_validation", "validation", 60)
    reference_path = save_safe_reference(
        tmp_path / "safe.pt",
        continuous=torch.rand(60, len(CONTINUOUS_TARGET_NAMES)) * 0.2 + 0.05,
        events=torch.zeros(60, len(EVENT_TARGET_NAMES)),
        episode_ids=[f"safe-{index}" for index in range(60)],
        source_metadata={
            "row_name": "condition_kc2_ng3",
            "success_only": True,
        },
    )
    summary = train_rewq_v0(
        train_data=train_path,
        validation_data=validation_path,
        safe_reference=reference_path,
        output=tmp_path / "run",
        costs=_costs(),
        device="cpu",
        max_steps=2,
        batch_size=16,
    )
    assert summary["verdict"] == "REWQ_V0_RECOVERABILITY_TRAINING_COMPLETE"
    loaded, loaded_config, envelope, conformal, costs, payload = load_rewq_v0_checkpoint(
        summary["checkpoint"], device="cpu"
    )
    assert loaded_config == config
    assert envelope.reference_row == "condition_kc2_ng3"
    assert conformal.calibration_rows == 180
    assert costs == _costs()
    assert payload["frozen_external_modules"] == ["SimVLA", "U_C", "U_G"]
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    with pytest.raises(ValueError, match="bounded"):
        train_rewq_v0(
            train_data=train_path,
            validation_data=validation_path,
            safe_reference=reference_path,
            output=tmp_path / "too_long",
            costs=_costs(),
            device="cpu",
            max_steps=5_001,
        )
