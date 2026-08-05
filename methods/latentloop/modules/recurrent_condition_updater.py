"""Low-rank token-preserving recurrent updater for VLA condition tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from methods.dcld.modules.low_rank_latent_dynamics import (
    LowRankFixedEulerLatentDynamics,
    LowRankLatentDynamicsOutput,
)


def count_trainable_parameters(module: nn.Module) -> int:
    """Count parameters that an optimizer is allowed to update."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _batch_scalar(value: Tensor | float | int, reference: Tensor) -> Tensor:
    batch_size = reference.shape[0]
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    if tensor.shape != (batch_size,):
        raise ValueError(f"Expected scalar or [B], got {tuple(tensor.shape)}")
    return tensor


class _ScalarEmbedding(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, value: Tensor | float | int, reference: Tensor) -> Tensor:
        """Embed one scalar per batch item on the reference device/dtype."""

        return self.network(_batch_scalar(value, reference).unsqueeze(-1))


class QueryContextFusion(nn.Module):
    """Fuse observation, executed-action, R, elapsed-time, and query-age inputs."""

    def __init__(
        self,
        *,
        observation_dim: int = 128,
        action_feature_dim: int = 128,
        scalar_dim: int = 16,
        output_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.output_dim = int(output_dim)
        self.r_embedding = _ScalarEmbedding(scalar_dim)
        self.dt_embedding = _ScalarEmbedding(scalar_dim)
        self.age_embedding = _ScalarEmbedding(scalar_dim)
        input_dim = self.observation_dim + self.action_feature_dim + 3 * scalar_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )

    def forward(
        self,
        observation_feature: Tensor,
        executed_action_feature: Tensor,
        *,
        execution_horizon: Tensor | int,
        elapsed_time: Tensor | float,
        query_age: Tensor | int,
    ) -> Tensor:
        """Return the fixed-width context consumed by condition dynamics."""

        if observation_feature.ndim != 2 or observation_feature.shape[-1] != self.observation_dim:
            raise ValueError("observation_feature has the wrong shape")
        expected_action = (observation_feature.shape[0], self.action_feature_dim)
        if executed_action_feature.shape != expected_action:
            raise ValueError(
                f"executed_action_feature must be {expected_action}, got {tuple(executed_action_feature.shape)}"
            )
        return self.network(
            torch.cat(
                (
                    observation_feature,
                    executed_action_feature,
                    self.r_embedding(execution_horizon, observation_feature),
                    self.dt_embedding(elapsed_time, observation_feature),
                    self.age_embedding(query_age, observation_feature),
                ),
                dim=-1,
            )
        )


@dataclass(frozen=True)
class ChunkAwareConditionOutput:
    """Updated condition and diagnostics from one lightweight query."""

    condition: Tensor
    context_feature: Tensor
    dynamics: LowRankLatentDynamicsOutput


class ChunkAwareConditionUpdater(nn.Module):
    """Recursively update ``[B,T_c,D_c]`` while preserving its token shape."""

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
        self.gate_bias = float(gate_bias)
        self.use_post_layernorm = bool(use_post_layernorm)
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
        previous_condition: Tensor,
        observation_feature: Tensor,
        executed_action_feature: Tensor,
        *,
        execution_horizon: Tensor | int,
        elapsed_time: Tensor | float,
        query_age: Tensor | int,
    ) -> ChunkAwareConditionOutput:
        """Apply one recursive query-boundary update."""

        if previous_condition.ndim != 3 or previous_condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"previous_condition must be [B,T,{self.condition_dim}], got {tuple(previous_condition.shape)}"
            )
        context = self.fusion(
            observation_feature,
            executed_action_feature,
            execution_horizon=execution_horizon,
            elapsed_time=elapsed_time,
            query_age=query_age,
        ).to(device=previous_condition.device, dtype=previous_condition.dtype)
        dynamics = self.dynamics(
            previous_condition,
            context,
            dt=_batch_scalar(elapsed_time, previous_condition),
            age=_batch_scalar(query_age, previous_condition),
        )
        if dynamics.latent.shape != previous_condition.shape:
            raise AssertionError("condition updater changed the condition shape")
        return ChunkAwareConditionOutput(
            condition=dynamics.latent,
            context_feature=context,
            dynamics=dynamics,
        )
