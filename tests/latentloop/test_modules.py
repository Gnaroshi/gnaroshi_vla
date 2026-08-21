"""CPU-only tests for LatentLoop modules and fixed execution notation."""

from __future__ import annotations

import inspect

import torch

from architectures.simvla.adapters.latentloop.condition_adapter import parameter_budget_audit
from methods.latentloop.eval.schedules import QuerySchedule, environment_action_gap
from methods.latentloop.modules import (
    ChunkAwareConditionUpdater,
    ExecutedActionEncoder,
    NonRecurrentConditionPredictor,
    pad_executed_actions,
    shift_action_chunk,
)


def test_r_k_g_and_k1_identity() -> None:
    assert environment_action_gap(4, 1) == 4
    assert environment_action_gap(4, 5) == 20
    assert environment_action_gap(2, 2) == 4
    schedule = QuerySchedule(full_query_interval=1, execution_horizon=5)
    assert all(schedule.is_full_query(index) for index in range(20))
    assert all(schedule.query_age(index) == 0 for index in range(20))


def test_action_padding_mask_and_unexecuted_tokens_do_not_enter_encoder() -> None:
    torch.manual_seed(3)
    actions = torch.randn(2, 5, 7)
    padded = pad_executed_actions(actions, torch.tensor([1, 5]))
    assert padded.actions.shape == (2, 5, 7)
    assert padded.validity_mask.tolist() == [
        [True, False, False, False, False],
        [True, True, True, True, True],
    ]
    encoder = ExecutedActionEncoder().eval()
    first = actions[:1].clone()
    changed = first.clone()
    changed[:, 1:] = 10_000.0
    feature_a = encoder(first, 1, 0.05).feature
    feature_b = encoder(changed, 1, 0.05).feature
    assert torch.equal(feature_a, feature_b)


def test_condition_shape_is_preserved() -> None:
    updater = ChunkAwareConditionUpdater().eval()
    previous = torch.randn(2, 3, 960)
    observation = torch.randn(2, 128)
    action = torch.randn(2, 128)
    output = updater(
        previous,
        observation,
        action,
        execution_horizon=torch.tensor([1, 5]),
        elapsed_time=torch.tensor([0.05, 0.25]),
        query_age=torch.tensor([1, 2]),
    )
    assert output.condition.shape == previous.shape
    assert output.condition.dtype == previous.dtype


def test_nonrecurrent_api_cannot_accept_previous_prediction() -> None:
    signature = inspect.signature(NonRecurrentConditionPredictor.forward)
    assert "previous_condition" not in signature.parameters
    assert "anchor_condition" in signature.parameters
    predictor = NonRecurrentConditionPredictor().eval()
    anchor = torch.randn(1, 2, 960)
    observation = torch.randn(1, 128)
    action = torch.randn(1, 128)
    first = predictor(
        anchor,
        observation,
        action,
        execution_horizon=5,
        elapsed_time=0.25,
        query_age=2,
    ).condition
    second = predictor(
        anchor,
        observation,
        action,
        execution_horizon=5,
        elapsed_time=0.25,
        query_age=2,
    ).condition
    assert torch.equal(first, second)


def test_action_correction_shift_and_mask() -> None:
    chunk = torch.arange(70, dtype=torch.float32).reshape(1, 10, 7)
    shifted = shift_action_chunk(chunk, 5)
    assert shifted.validity_mask.tolist() == [
        [True, True, True, True, True, False, False, False, False, False]
    ]
    assert torch.equal(shifted.actions[0, :5], chunk[0, 5:])
    assert torch.count_nonzero(shifted.actions[0, 5:]) == 0


def test_matched_baseline_parameter_budgets() -> None:
    audit = parameter_budget_audit()
    assert audit["primary_trainable_parameters"] > 0
    for row in audit["variants"].values():
        assert row["parameter_match_pass"]
    assert not audit["variants"]["no_observation"]["parameter_match_required"]
