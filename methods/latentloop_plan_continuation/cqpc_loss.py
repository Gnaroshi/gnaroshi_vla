"""Optional cross-query plan-consistency objective for LatentLoop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class CQPCLossOutput:
    """Raw and weighted CQPC loss terms."""

    total: Tensor
    arm_raw: Tensor
    gripper_raw: Tensor
    arm_weighted: Tensor
    gripper_weighted: Tensor
    teacher_disagreement: Tensor
    teacher_weight: Tensor


def cqpc_is_enabled(weight: float) -> bool:
    """Return whether CQPC should add any graph or decoder work."""

    return float(weight) > 0.0


def _check_shapes(arm: Tensor, grip: Tensor, name: str) -> None:
    if arm.ndim < 3 or arm.shape[-1] != 6:
        raise ValueError(f"{name}_arm must end in [P,6], got {tuple(arm.shape)}")
    if grip.shape != arm.shape[:-1] + (1,):
        raise ValueError(
            f"{name}_grip must match {name}_arm leading/token dimensions, got {tuple(grip.shape)}"
        )
    if arm.shape[-2] < 2:
        raise ValueError("CQPC requires action_pred_steps >= 2")


def cross_query_plan_consistency_loss(
    predicted_current_arm: Tensor,
    predicted_current_gripper: Tensor,
    previous_arm: Tensor,
    previous_gripper: Tensor,
    teacher_current_arm: Tensor,
    teacher_current_gripper: Tensor,
    *,
    gamma: float,
    lambda_arm: float = 1.0,
    lambda_gripper: float = 1.0,
) -> CQPCLossOutput:
    """Compute disagreement-weighted consistency over verified overlap pairs.

    ``previous_*`` is detached inside this function. Gripper values must be
    continuous pre-threshold logits or probabilities, used consistently across
    all four inputs.
    """

    for name, arm, grip in (
        ("predicted_current", predicted_current_arm, predicted_current_gripper),
        ("previous", previous_arm, previous_gripper),
        ("teacher_current", teacher_current_arm, teacher_current_gripper),
    ):
        _check_shapes(arm, grip, name)
    if predicted_current_arm.shape != previous_arm.shape or predicted_current_arm.shape != teacher_current_arm.shape:
        raise ValueError("All CQPC arm horizons must have identical shapes")
    if predicted_current_gripper.shape != previous_gripper.shape or predicted_current_gripper.shape != teacher_current_gripper.shape:
        raise ValueError("All CQPC gripper horizons must have identical shapes")
    if gamma <= 0.0:
        raise ValueError("gamma must be user-calibrated and strictly positive")

    previous_arm_target = previous_arm[..., 1:, :].detach()
    previous_gripper_target = previous_gripper[..., 1:, :].detach()
    current_arm_overlap = predicted_current_arm[..., :-1, :]
    current_gripper_overlap = predicted_current_gripper[..., :-1, :]
    arm_pair = torch.mean(torch.abs(current_arm_overlap - previous_arm_target), dim=-1)
    gripper_pair = F.smooth_l1_loss(
        current_gripper_overlap,
        previous_gripper_target,
        reduction="none",
    ).mean(dim=-1)

    with torch.no_grad():
        teacher_difference = torch.cat(
            (
                teacher_current_arm[..., :-1, :]
                - previous_arm[..., 1:, :].detach(),
                teacher_current_gripper[..., :-1, :]
                - previous_gripper[..., 1:, :].detach(),
            ),
            dim=-1,
        )
        teacher_disagreement = torch.mean(torch.abs(teacher_difference), dim=-1)
        teacher_weight = torch.exp(-float(gamma) * teacher_disagreement).clamp(0.0, 1.0)

    arm_raw = arm_pair.mean()
    gripper_raw = gripper_pair.mean()
    arm_weighted = (teacher_weight * arm_pair).mean()
    gripper_weighted = (teacher_weight * gripper_pair).mean()
    total = float(lambda_arm) * arm_weighted + float(lambda_gripper) * gripper_weighted
    return CQPCLossOutput(
        total=total,
        arm_raw=arm_raw,
        gripper_raw=gripper_raw,
        arm_weighted=arm_weighted,
        gripper_weighted=gripper_weighted,
        teacher_disagreement=teacher_disagreement,
        teacher_weight=teacher_weight,
    )
