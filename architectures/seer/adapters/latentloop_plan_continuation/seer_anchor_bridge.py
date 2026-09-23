"""Seer-facing non-recurrent anchor-to-current latent adapter."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from methods.latentloop_plan_continuation.anchor_bridge import (
    AnchorBridgeOutput,
    NonRecurrentAnchorToCurrentBridge,
    find_anchor_bridge_hidden_dim,
)


@dataclass(frozen=True)
class ParameterMatch:
    """Parameter-count audit against LatentLoop dynamics."""

    target_parameters: int
    actual_parameters: int
    relative_error: float
    hidden_dim: int


class SeerAnchorToCurrentAdapter(nn.Module):
    """Predict current action latent directly from a fixed segment anchor."""

    def __init__(
        self,
        delta_encoder: nn.Module,
        *,
        target_predictor_parameters: int,
        latent_dim: int = 384,
        motion_dim: int = 128,
        action_pred_steps: int = 3,
        hidden_dim: int = 0,
        maximum_relative_error: float = 0.05,
    ) -> None:
        super().__init__()
        self.mode = "anchor_bridge"
        if hidden_dim <= 0:
            hidden_dim, actual, error = find_anchor_bridge_hidden_dim(
                target_predictor_parameters,
                latent_dim=latent_dim,
                motion_dim=motion_dim,
                action_pred_steps=action_pred_steps,
            )
        else:
            probe = NonRecurrentAnchorToCurrentBridge(
                latent_dim, motion_dim, action_pred_steps, hidden_dim
            )
            actual = sum(parameter.numel() for parameter in probe.parameters())
            error = abs(actual - target_predictor_parameters) / float(
                target_predictor_parameters
            )
        if error > float(maximum_relative_error):
            raise ValueError(
                "Anchor bridge parameter mismatch exceeds tolerance: "
                f"target={target_predictor_parameters}, actual={actual}, error={error:.4f}"
            )
        self.delta_encoder = delta_encoder
        self.bridge = NonRecurrentAnchorToCurrentBridge(
            latent_dim, motion_dim, action_pred_steps, hidden_dim
        )
        self.parameter_match = ParameterMatch(
            target_predictor_parameters, actual, error, hidden_dim
        )

    def encode_anchor_to_current(
        self,
        anchor_primary: Tensor,
        anchor_wrist: Tensor,
        current_primary: Tensor,
        current_wrist: Tensor,
        anchor_proprio: Tensor,
        current_proprio: Tensor,
    ) -> Tensor:
        """Encode the fixed segment anchor and current observation."""

        return self.delta_encoder(
            [anchor_primary, anchor_wrist],
            [current_primary, current_wrist],
            q_key=anchor_proprio,
            q_cur=current_proprio,
        )

    def forward_from_feature(
        self, anchor_latent: Tensor, feature: Tensor, age: Tensor | float
    ) -> AnchorBridgeOutput:
        """Predict directly without consuming a previous predicted latent."""

        return self.bridge(anchor_latent, feature, age)
