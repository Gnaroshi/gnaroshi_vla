"""Loss helpers for condition-latent DCLD training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class DCLDLossWeights:
    condition_mse: float = 1.0
    condition_cosine: float = 0.05
    action_l1: float = 0.0
    smoothness: float = 0.0


def _zero_like_loss(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_zeros(())


def condition_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def condition_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()


def action_l1(pred_action: torch.Tensor | None, target_action: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    if pred_action is None or target_action is None:
        return _zero_like_loss(ref)
    return F.l1_loss(pred_action, target_action)


def smoothness_loss(pred_sequence: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    if pred_sequence is None or pred_sequence.shape[0] < 2:
        return _zero_like_loss(ref)
    return F.mse_loss(pred_sequence[1:], pred_sequence[:-1])


def hold_vs_pred_metrics(
    hold_condition: torch.Tensor,
    pred_condition: torch.Tensor,
    target_condition: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "hold_mse": condition_mse(hold_condition, target_condition),
        "pred_mse": condition_mse(pred_condition, target_condition),
        "hold_cosine": 1.0 - condition_cosine_loss(hold_condition, target_condition),
        "pred_cosine": 1.0 - condition_cosine_loss(pred_condition, target_condition),
    }


def compute_dcld_losses(
    pred_condition: torch.Tensor,
    target_condition: torch.Tensor,
    *,
    pred_action: torch.Tensor | None = None,
    target_action: torch.Tensor | None = None,
    pred_sequence: torch.Tensor | None = None,
    weights: DCLDLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or DCLDLossWeights()

    losses = {
        "condition_mse": condition_mse(pred_condition, target_condition),
        "condition_cosine": condition_cosine_loss(pred_condition, target_condition),
        "action_l1": action_l1(pred_action, target_action, pred_condition),
        "smoothness": smoothness_loss(pred_sequence, pred_condition),
    }
    total = (
        weights.condition_mse * losses["condition_mse"]
        + weights.condition_cosine * losses["condition_cosine"]
        + weights.action_l1 * losses["action_l1"]
        + weights.smoothness * losses["smoothness"]
    )
    losses["total"] = total
    return losses
