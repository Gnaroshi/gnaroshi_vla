"""Losses shared by the two controlled LatentLoop comparison baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ActionCorrectionLossWeights:
    """Explicit calibrated weights; callers must record their provenance."""

    arm: float
    gripper: float
    executed_token: float
    residual_regularization: float


@dataclass(frozen=True)
class NonRecurrentLossWeights:
    """Canonical LatentLoop-compatible latent/action/smooth weights."""

    latent: float
    action: float
    smooth: float


@dataclass(frozen=True)
class LossBundle:
    """Weighted total and named raw terms."""

    total: Tensor
    raw: dict[str, Tensor]
    weights: dict[str, float]


def continuous_gripper_probability(
    *,
    teacher_logit: Tensor | None = None,
    teacher_probability: Tensor | None = None,
) -> Tensor:
    """Return a differentiable pre-threshold teacher probability.

    Exactly one continuous representation must be supplied. Binary thresholded
    actions are intentionally rejected as a primary distillation target.
    """

    if (teacher_logit is None) == (teacher_probability is None):
        raise ValueError("Provide exactly one of teacher_logit or teacher_probability")
    if teacher_logit is not None:
        return torch.sigmoid(teacher_logit.detach())
    assert teacher_probability is not None
    probability = teacher_probability.detach()
    if not probability.dtype.is_floating_point:
        raise TypeError("teacher_probability must be floating point")
    if torch.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("teacher_probability must lie in [0,1]")
    unique = torch.unique(probability)
    if unique.numel() <= 2 and all(
        float(value) in {0.0, 1.0} for value in unique.detach().cpu()
    ):
        raise ValueError("Thresholded binary gripper targets are not allowed")
    return probability


def action_correction_loss(
    *,
    predicted_arm: Tensor,
    predicted_gripper_logit: Tensor,
    teacher_arm: Tensor,
    teacher_gripper_logit: Tensor,
    arm_residual: Tensor,
    gripper_logit_residual: Tensor,
    weights: ActionCorrectionLossWeights,
) -> LossBundle:
    """Compute arm SmoothL1 and soft gripper-logit distillation losses."""

    if predicted_arm.shape != teacher_arm.shape or predicted_arm.shape[-1] != 6:
        raise ValueError("predicted and teacher arm horizons must match [...,P,6]")
    if predicted_gripper_logit.shape != teacher_gripper_logit.shape:
        raise ValueError("predicted and teacher gripper-logit horizons must match")
    soft_gripper = continuous_gripper_probability(
        teacher_logit=teacher_gripper_logit
    )
    raw = {
        "arm": F.smooth_l1_loss(predicted_arm, teacher_arm.detach()),
        "gripper": F.binary_cross_entropy_with_logits(
            predicted_gripper_logit, soft_gripper
        ),
        "executed_token": F.smooth_l1_loss(
            predicted_arm[..., 0, :], teacher_arm.detach()[..., 0, :]
        ),
        "residual_regularization": (
            arm_residual.square().mean()
            + gripper_logit_residual.square().mean()
        ),
    }
    weight_dict = asdict(weights)
    total = (
        weight_dict["arm"] * raw["arm"]
        + weight_dict["gripper"] * raw["gripper"]
        + weight_dict["executed_token"] * raw["executed_token"]
        + weight_dict["residual_regularization"]
        * raw["residual_regularization"]
    )
    return LossBundle(total=total, raw=raw, weights=weight_dict)


def nonrecurrent_latent_loss(
    *,
    predicted_latent: Tensor,
    anchor_latent: Tensor,
    teacher_latent: Tensor,
    predicted_arm: Tensor,
    predicted_gripper_probability: Tensor,
    teacher_arm: Tensor,
    teacher_gripper_probability: Tensor,
    weights: NonRecurrentLossWeights,
) -> LossBundle:
    """Apply the canonical LatentLoop latent/action/smooth loss families."""

    if predicted_latent.shape != teacher_latent.shape:
        raise ValueError("predicted and teacher latent shapes must match")
    if predicted_latent.shape != anchor_latent.shape:
        raise ValueError("predicted and anchor latent shapes must match")
    predicted_action = torch.cat(
        (predicted_arm, predicted_gripper_probability), dim=-1
    )
    teacher_action = torch.cat(
        (teacher_arm.detach(), teacher_gripper_probability.detach()), dim=-1
    )
    if predicted_action.shape != teacher_action.shape:
        raise ValueError("predicted and teacher action horizons must match")
    raw = {
        "latent": F.mse_loss(predicted_latent, teacher_latent.detach()),
        "action": F.l1_loss(predicted_action, teacher_action),
        "smooth": F.mse_loss(
            predicted_latent - anchor_latent.detach(),
            torch.zeros_like(predicted_latent),
        ),
    }
    weight_dict = asdict(weights)
    total = (
        weight_dict["latent"] * raw["latent"]
        + weight_dict["action"] * raw["action"]
        + weight_dict["smooth"] * raw["smooth"]
    )
    return LossBundle(total=total, raw=raw, weights=weight_dict)
