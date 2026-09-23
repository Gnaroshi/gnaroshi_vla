"""Raw and weighted losses for the joint surrogate protocol."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class JointLossWeights:
    latent: float
    latent_action: float
    surrogate: float
    executed_token: float
    tail: float
    gripper: float
    residual: float

    def validate(self, *, stage: str) -> None:
        values = vars(self)
        if any(value < 0.0 for value in values.values()):
            raise ValueError("Joint loss weights must be non-negative")
        required = {"surrogate", "executed_token", "tail", "gripper", "residual"}
        if stage == "stage_b":
            required |= {"latent", "latent_action"}
        missing = sorted(name for name in required if values[name] <= 0.0)
        if missing:
            raise ValueError(
                "Predeclared positive loss weights are required for " + ", ".join(missing)
            )


@dataclass(frozen=True)
class JointLossBundle:
    total: Tensor
    raw: dict[str, Tensor]
    weighted: dict[str, Tensor]


def joint_surrogate_loss(
    *,
    predicted_arm: Tensor,
    predicted_gripper_logit: Tensor,
    exact_arm: Tensor,
    exact_gripper_logit: Tensor,
    valid_mask: Tensor,
    residual: Tensor,
    weights: JointLossWeights,
    latent_loss: Tensor,
    latent_action_loss: Tensor,
) -> JointLossBundle:
    exact_arm = exact_arm.detach()
    exact_gripper_logit = exact_gripper_logit.detach()
    predicted = torch.cat((predicted_arm, predicted_gripper_logit), dim=-1)
    exact = torch.cat((exact_arm, exact_gripper_logit), dim=-1)
    invalid = ~valid_mask.bool()
    invalid_full = invalid.expand_as(predicted)
    tail = (
        F.l1_loss(predicted[invalid_full], exact[invalid_full])
        if bool(invalid_full.any())
        else predicted.sum() * 0.0
    )
    raw = {
        "latent": latent_loss,
        "latent_action": latent_action_loss,
        "surrogate": F.l1_loss(predicted, exact),
        "executed_token": F.l1_loss(predicted[..., 0, :], exact[..., 0, :]),
        "tail": tail,
        "gripper": F.l1_loss(predicted_gripper_logit, exact_gripper_logit),
        "residual": residual.pow(2).mean(),
    }
    weighted = {name: raw[name] * getattr(weights, name) for name in raw}
    return JointLossBundle(total=sum(weighted.values()), raw=raw, weighted=weighted)
