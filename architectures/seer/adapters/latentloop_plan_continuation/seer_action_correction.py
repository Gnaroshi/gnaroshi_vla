"""Seer-facing matched action-space correction adapter."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from methods.latentloop_plan_continuation.action_correction import (
    ActionCorrectionOutput,
    MatchedActionSpaceCorrection,
    find_action_correction_hidden_dim,
    shift_action_horizon,
)


@dataclass(frozen=True)
class ParameterMatch:
    """Parameter-count audit against the recurrent LatentLoop adapter."""

    target_parameters: int
    actual_parameters: int
    relative_error: float
    hidden_dim: int


class SeerActionCorrectionAdapter(nn.Module):
    """Own the same-form delta encoder and a full-horizon correction module."""

    def __init__(
        self,
        delta_encoder: nn.Module,
        *,
        target_predictor_parameters: int,
        action_pred_steps: int = 3,
        motion_dim: int = 128,
        hidden_dim: int = 0,
        maximum_relative_error: float = 0.05,
    ) -> None:
        super().__init__()
        self.mode = "action_correction"
        if hidden_dim <= 0:
            hidden_dim, actual, error = find_action_correction_hidden_dim(
                target_predictor_parameters,
                action_pred_steps=action_pred_steps,
                motion_dim=motion_dim,
            )
        else:
            probe = MatchedActionSpaceCorrection(
                action_pred_steps, motion_dim, hidden_dim
            )
            actual = sum(parameter.numel() for parameter in probe.parameters())
            error = abs(actual - target_predictor_parameters) / float(
                target_predictor_parameters
            )
        if error > float(maximum_relative_error):
            raise ValueError(
                "Action correction parameter mismatch exceeds tolerance: "
                f"target={target_predictor_parameters}, actual={actual}, error={error:.4f}"
            )
        self.delta_encoder = delta_encoder
        self.corrector = MatchedActionSpaceCorrection(
            action_pred_steps, motion_dim, hidden_dim
        )
        self.parameter_match = ParameterMatch(
            target_predictor_parameters, actual, error, hidden_dim
        )

    def encode_delta(
        self,
        previous_primary: Tensor,
        previous_wrist: Tensor,
        current_primary: Tensor,
        current_wrist: Tensor,
        previous_proprio: Tensor,
        current_proprio: Tensor,
    ) -> Tensor:
        """Use the exact FastVisualDeltaEncoder input contract."""

        return self.delta_encoder(
            [previous_primary, previous_wrist],
            [current_primary, current_wrist],
            q_key=previous_proprio,
            q_cur=current_proprio,
        )

    def forward_from_feature(
        self,
        previous_arm: Tensor,
        previous_gripper_logit: Tensor,
        u_delta: Tensor,
        age: Tensor | float,
    ) -> ActionCorrectionOutput:
        """Shift the verified horizon then correct all P tokens."""

        shifted_arm = shift_action_horizon(previous_arm)
        shifted_gripper = shift_action_horizon(previous_gripper_logit)
        return self.corrector(shifted_arm, shifted_gripper, u_delta, age)
