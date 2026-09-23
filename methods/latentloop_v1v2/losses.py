"""Distillation losses for LatentLoop V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .transition import TransitionOutput


@dataclass(frozen=True)
class V1LossOutput:
    total: Tensor
    raw: Mapping[str, Tensor]
    weighted: Mapping[str, Tensor]


def _continuous_action(decoded: object) -> Tensor:
    if isinstance(decoded, dict):
        return torch.cat((decoded["arm"], decoded["gripper_probability"]), dim=-1)
    if isinstance(decoded, tuple) and len(decoded) == 2:
        return torch.cat(decoded, dim=-1)
    raise TypeError("frozen action generator must return diagnostics dict or (arm, gripper_probability)")


def compute_v1_losses(
    output: TransitionOutput,
    teacher_latent: Tensor,
    anchor_latent: Tensor,
    frozen_action_generator: Callable[[Tensor], object],
    weights: Mapping[str, float],
) -> V1LossOutput:
    """Use detached teacher targets while retaining gradients through frozen readout ops."""

    target_latent = teacher_latent.detach()
    target_action = _continuous_action(frozen_action_generator(target_latent)).detach()
    direct_action = _continuous_action(frozen_action_generator(output.direct))
    composed_action = _continuous_action(frozen_action_generator(output.composed))
    raw = {
        "direct_latent": F.mse_loss(output.direct, target_latent),
        "composed_latent": F.mse_loss(output.composed, target_latent),
        "direct_action": F.l1_loss(direct_action, target_action),
        "composed_action": F.l1_loss(composed_action, target_action),
        # Stop-gradient targets in both directions keep the two estimators from
        # collapsing into one another while still training each path.
        "composition": 0.5 * (
            F.mse_loss(output.composed, output.direct.detach())
            + F.mse_loss(output.direct, output.composed.detach())
        ),
        "smooth": 0.5 * (
            F.mse_loss(output.direct, anchor_latent.detach())
            + F.mse_loss(output.composed, anchor_latent.detach())
        ),
    }
    missing = sorted(set(raw) - set(weights))
    if missing:
        raise ValueError(f"missing frozen loss weights: {missing}")
    weighted = {name: raw[name] * float(weights[name]) for name in raw}
    return V1LossOutput(total=sum(weighted.values()), raw=raw, weighted=weighted)
