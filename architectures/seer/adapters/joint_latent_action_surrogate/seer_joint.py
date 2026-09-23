"""Seer-specific wrapper around architecture-neutral joint modules."""

from __future__ import annotations

from torch import Tensor, nn

from methods.joint_latent_action_surrogate.modules import (
    JointLatentAnchoredActionSurrogate,
    SurrogateOutput,
    WideLatentCapacityControl,
)


class SeerJointLatentActionAdapter(nn.Module):
    """Own only new parameters; canonical encoder/updater remain on SeerAgent."""

    def __init__(
        self,
        *,
        mode: str,
        latent_dim: int,
        motion_dim: int,
        action_pred_steps: int,
        surrogate_hidden_dim: int,
        wide_hidden_dim: int,
    ) -> None:
        super().__init__()
        if mode not in {"joint", "wide"}:
            raise ValueError(f"Unknown joint adapter mode={mode!r}")
        self.mode = mode
        self.action_pred_steps = int(action_pred_steps)
        self.surrogate = (
            JointLatentAnchoredActionSurrogate(
                latent_dim=latent_dim,
                motion_dim=motion_dim,
                action_pred_steps=action_pred_steps,
                hidden_dim=surrogate_hidden_dim,
            )
            if mode == "joint"
            else None
        )
        self.wide_control = (
            WideLatentCapacityControl(
                latent_dim=latent_dim,
                motion_dim=motion_dim,
                action_pred_steps=action_pred_steps,
                hidden_dim=wide_hidden_dim,
            )
            if mode == "wide"
            else None
        )

    def surrogate_forward(
        self,
        *,
        anchor_arm: Tensor,
        anchor_gripper_logit: Tensor,
        anchor_latent: Tensor,
        current_latent: Tensor,
        shared_feature: Tensor,
        elapsed: Tensor | int,
    ) -> SurrogateOutput:
        if self.surrogate is None:
            raise RuntimeError("The wide capacity control has no action surrogate")
        return self.surrogate(
            anchor_arm=anchor_arm,
            anchor_gripper_logit=anchor_gripper_logit,
            anchor_latent=anchor_latent,
            current_latent=current_latent,
            shared_feature=shared_feature,
            elapsed=elapsed,
        )

    def apply_wide_capacity(
        self, latent: Tensor, shared_feature: Tensor, age: Tensor | float
    ) -> Tensor:
        if self.wide_control is None:
            return latent
        return self.wide_control(latent, shared_feature, age)
