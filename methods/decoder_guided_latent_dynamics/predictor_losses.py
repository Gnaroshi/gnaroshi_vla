"""Losses for observation-conditioned action-delta feasibility experiments."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def action_delta_loss(
    predicted_delta: Tensor,
    target_delta: Tensor,
    switch_logit: Tensor,
    switch_target: Tensor,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    gripper_delta_weight: float = 0.1,
    switch_weight: float = 0.25,
    positive_switch_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted Huber arm loss plus smooth-gripper and switch-classification losses."""
    translation = F.smooth_l1_loss(predicted_delta[:, :3], target_delta[:, :3])
    rotation = F.smooth_l1_loss(predicted_delta[:, 3:6], target_delta[:, 3:6])
    gripper_delta = F.smooth_l1_loss(predicted_delta[:, 6], target_delta[:, 6])
    switch = F.binary_cross_entropy_with_logits(
        switch_logit,
        switch_target,
        pos_weight=torch.as_tensor(
            positive_switch_weight, device=switch_logit.device, dtype=switch_logit.dtype
        ),
    )
    total = (
        translation_weight * translation
        + rotation_weight * rotation
        + gripper_delta_weight * gripper_delta
        + switch_weight * switch
    )
    return total, {
        "translation": translation.detach(),
        "rotation": rotation.detach(),
        "gripper_delta": gripper_delta.detach(),
        "switch": switch.detach(),
    }


def direct_latent_loss(predicted_delta: Tensor, target_delta: Tensor) -> Tensor:
    """Huber loss for a directly predicted flattened latent delta."""
    return F.smooth_l1_loss(predicted_delta, target_delta)
