"""Matched nonrecurrent anchor-to-current condition baseline."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from methods.dcld.modules.low_rank_latent_dynamics import (
    LowRankFixedEulerLatentDynamics,
    LowRankLatentDynamicsOutput,
)

from .recurrent_condition_updater import QueryContextFusion, _batch_scalar


@dataclass(frozen=True)
class NonRecurrentConditionOutput:
    """Condition predicted directly from a fixed full-condition anchor."""

    condition: Tensor
    context_feature: Tensor
    dynamics: LowRankLatentDynamicsOutput


class NonRecurrentConditionPredictor(nn.Module):
    """Predict from a full anchor; no previous predicted condition is accepted."""

    def __init__(
        self,
        *,
        condition_dim: int = 960,
        observation_dim: int = 128,
        action_feature_dim: int = 128,
        context_dim: int = 128,
        fusion_hidden_dim: int = 128,
        dynamics_hidden_dim: int = 128,
        rank_dim: int = 64,
        gate_mode: str = "scalar",
        gate_bias: float = -4.0,
        use_post_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.fusion = QueryContextFusion(
            observation_dim=observation_dim,
            action_feature_dim=action_feature_dim,
            output_dim=context_dim,
            hidden_dim=fusion_hidden_dim,
        )
        self.dynamics = LowRankFixedEulerLatentDynamics(
            latent_dim=self.condition_dim,
            delta_dim=context_dim,
            hidden_dim=dynamics_hidden_dim,
            rank_dim=rank_dim,
            gate_mode=gate_mode,
            gate_bias=gate_bias,
            use_post_layernorm=use_post_layernorm,
        )

    def forward(
        self,
        anchor_condition: Tensor,
        anchor_to_current_observation_feature: Tensor,
        executed_action_history_feature: Tensor,
        *,
        execution_horizon: Tensor | int,
        elapsed_time: Tensor | float,
        query_age: Tensor | int,
    ) -> NonRecurrentConditionOutput:
        """Predict current condition using only the fixed anchor and current context."""

        if anchor_condition.ndim != 3 or anchor_condition.shape[-1] != self.condition_dim:
            raise ValueError("anchor_condition has the wrong shape")
        context = self.fusion(
            anchor_to_current_observation_feature,
            executed_action_history_feature,
            execution_horizon=execution_horizon,
            elapsed_time=elapsed_time,
            query_age=query_age,
        ).to(device=anchor_condition.device, dtype=anchor_condition.dtype)
        dynamics = self.dynamics(
            anchor_condition,
            context,
            dt=_batch_scalar(elapsed_time, anchor_condition),
            age=_batch_scalar(query_age, anchor_condition),
        )
        return NonRecurrentConditionOutput(
            condition=dynamics.latent,
            context_feature=context,
            dynamics=dynamics,
        )
