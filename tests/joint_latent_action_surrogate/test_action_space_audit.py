from __future__ import annotations

import torch

from methods.joint_latent_action_surrogate.action_space_audit import (
    action_from_arm_and_logit,
    build_amended_gate,
    distribution,
    per_sample_l1,
)


def test_action_representation_uses_gripper_probability() -> None:
    arm = torch.zeros(2, 1, 6)
    logit = torch.tensor([[[0.0]], [[2.0]]])
    action = action_from_arm_and_logit(arm, logit)
    assert action.shape == (2, 1, 7)
    torch.testing.assert_close(action[..., 6:], torch.sigmoid(logit))


def test_per_sample_reduction_matches_reference_definition() -> None:
    predicted = torch.tensor([[[1.0] * 7], [[2.0] * 7]])
    target = torch.zeros_like(predicted)
    assert per_sample_l1(predicted, target) == [1.0, 2.0]
    assert distribution([1.0, 2.0])["mean"] == 1.5


def test_amended_gate_separates_decoder_and_teacher_fidelity() -> None:
    metrics = {
        "joint_to_exact_first_token_l1": distribution([0.01, 0.02]),
        "hold_to_exact_first_token_l1": distribution([0.03, 0.04]),
        "joint_to_teacher_first_token_l1": distribution([0.02, 0.03]),
    }
    passed = build_amended_gate(
        metrics=metrics,
        recursive_reference_mean=0.04,
        expected_examples=2,
        split_identity_match=True,
        checkpoint_identity_match=True,
    )
    assert passed["pass"] is True

    failed = build_amended_gate(
        metrics=metrics,
        recursive_reference_mean=0.01,
        expected_examples=2,
        split_identity_match=True,
        checkpoint_identity_match=True,
    )
    assert failed["pass"] is False
    assert (
        failed["checks"]["teacher_fidelity_better_than_recursive_reference"]
        is False
    )
