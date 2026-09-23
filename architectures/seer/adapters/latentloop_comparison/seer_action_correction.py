"""Seer-facing matched direct action-space correction adapter."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from methods.latentloop_comparison.action_space_correction import (
    ActionCorrectionOutput,
    MatchedActionSpaceCorrection,
    find_action_correction_hidden_dim,
)

from .action_token_alignment import assert_canonical_seer_alignment


@dataclass(frozen=True)
class ParameterMatch:
    """Parameter-count match against LatentLoop's latent updater."""

    target_parameters: int
    actual_parameters: int
    relative_error: float
    hidden_dim: int


class SeerActionCorrectionAdapter(nn.Module):
    """Apply identical observation encoding and correct the full action horizon."""

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
        assert_canonical_seer_alignment(action_pred_steps)
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
        if error > maximum_relative_error:
            raise ValueError(
                "Action-correction parameter mismatch exceeds tolerance: "
                f"target={target_predictor_parameters}, actual={actual}, error={error:.6f}"
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
        """Reuse FastVisualDeltaEncoder's exact two-camera/proprio contract."""

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
        """Align the unshifted cached horizon and correct all P tokens."""

        return self.corrector(
            previous_arm,
            previous_gripper_logit,
            u_delta,
            age,
            inputs_are_shifted=False,
        )
