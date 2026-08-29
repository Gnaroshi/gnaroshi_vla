"""Layer/token-shared recurrent transition used by architecture adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _batch_scalar(value: Tensor | float | int, reference: Tensor, name: str) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        result = result.expand(reference.shape[0])
    if result.shape != (reference.shape[0],):
        raise ValueError(f"{name} must be a scalar or [B], got {tuple(result.shape)}")
    return result


class ScalarEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, value: Tensor | float | int, reference: Tensor, name: str) -> Tensor:
        return self.network(_batch_scalar(value, reference, name).unsqueeze(-1))


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.network = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.network(self.norm(value))


@dataclass(frozen=True)
class TransitionConfig:
    state_width: int = 128
    observation_width: int = 128
    action_width: int = 128
    robot_state_dim: int = 32
    robot_width: int = 64
    scalar_width: int = 32
    layer_width: int = 64
    hidden_width: int = 512
    num_blocks: int = 2
    max_layers: int = 32
    gate_bias: float = -4.0


@dataclass(frozen=True)
class TransitionOutput:
    state: Tensor
    delta: Tensor
    gate: Tensor
    context: Tensor


class VariableTimeTransitionCore(nn.Module):
    """Predict a residual update without dense parameters over layer or token axes.

    Inputs are encoded state tokens ``[B,L,S,C]`` and per-prefix-token
    observation features ``[B,S,O]``. Every projection is shared across L and S;
    only a small layer embedding distinguishes transformer layers.
    """

    def __init__(self, config: TransitionConfig = TransitionConfig()) -> None:
        super().__init__()
        self.config = config
        c = config
        self.robot_encoder = nn.Sequential(
            nn.LayerNorm(c.robot_state_dim),
            nn.Linear(c.robot_state_dim, c.robot_width),
            nn.GELU(),
        )
        self.delta_q_embedding = ScalarEmbedding(c.scalar_width)
        self.delta_a_embedding = ScalarEmbedding(c.scalar_width)
        self.age_embedding = ScalarEmbedding(c.scalar_width)
        self.layer_embedding = nn.Embedding(c.max_layers, c.layer_width)
        input_width = (
            c.state_width
            + 4 * c.observation_width
            + c.action_width
            + 2 * c.robot_width
            + 3 * c.scalar_width
            + c.layer_width
        )
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, c.hidden_width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(ResidualMLPBlock(c.hidden_width) for _ in range(c.num_blocks))
        self.delta_head = nn.Linear(c.hidden_width, c.state_width)
        self.gate_head = nn.Linear(c.hidden_width, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, c.gate_bias)

    def forward(
        self,
        previous_state: Tensor,
        current_observation: Tensor,
        previous_observation: Tensor,
        action_feature: Tensor,
        robot_state: Tensor,
        interval_observation: Tensor | None = None,
        robot_history_feature: Tensor | None = None,
        *,
        delta_q: Tensor | int,
        delta_a: Tensor | int,
        full_refresh_age: Tensor | int,
    ) -> TransitionOutput:
        c = self.config
        if previous_state.ndim != 4 or previous_state.shape[-1] != c.state_width:
            raise ValueError("previous_state must be [B,L,S,state_width]")
        batch_size, layers, tokens, _ = previous_state.shape
        expected_obs = (batch_size, tokens, c.observation_width)
        if current_observation.shape != expected_obs or previous_observation.shape != expected_obs:
            raise ValueError(
                f"observation features must both be {expected_obs}; got "
                f"{tuple(current_observation.shape)} and {tuple(previous_observation.shape)}"
            )
        if action_feature.shape != (batch_size, c.action_width):
            raise ValueError("action_feature has the wrong shape")
        if robot_state.shape != (batch_size, c.robot_state_dim):
            raise ValueError("robot_state has the wrong shape")
        if layers > c.max_layers:
            raise ValueError(f"state has {layers} layers, cap is {c.max_layers}")

        robot = self.robot_encoder(robot_state)
        if interval_observation is None:
            interval_observation = current_observation - previous_observation
        if interval_observation.shape != expected_obs:
            raise ValueError("interval_observation has the wrong shape")
        if robot_history_feature is None:
            robot_history_feature = robot
        if robot_history_feature.shape != (batch_size, c.robot_width):
            raise ValueError("robot_history_feature has the wrong shape")
        scalars = torch.cat(
            (
                self.delta_q_embedding(delta_q, previous_state, "delta_q"),
                self.delta_a_embedding(delta_a, previous_state, "delta_a"),
                self.age_embedding(full_refresh_age, previous_state, "full_refresh_age"),
            ),
            dim=-1,
        )
        global_context = torch.cat((action_feature, robot, robot_history_feature, scalars), dim=-1)
        global_context = global_context[:, None, None, :].expand(-1, layers, tokens, -1)

        current = current_observation[:, None, :, :].expand(-1, layers, -1, -1)
        previous = previous_observation[:, None, :, :].expand(-1, layers, -1, -1)
        difference = current - previous
        layer_ids = torch.arange(layers, device=previous_state.device)
        layer_features = self.layer_embedding(layer_ids)[None, :, None, :].expand(
            batch_size, -1, tokens, -1
        )
        context = torch.cat(
            (
                previous_state,
                current,
                previous,
                difference,
                interval_observation[:, None, :, :].expand(-1, layers, -1, -1),
                global_context,
                layer_features,
            ),
            dim=-1,
        )
        hidden = self.input_projection(context)
        for block in self.blocks:
            hidden = block(hidden)
        delta = self.delta_head(hidden)
        gate = torch.sigmoid(self.gate_head(hidden))
        state = previous_state + gate * delta
        return TransitionOutput(state=state, delta=delta, gate=gate, context=hidden)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
