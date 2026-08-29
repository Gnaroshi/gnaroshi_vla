"""Matched action-chunk correction baseline with explicit time alignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .recurrent_condition_updater import _ScalarEmbedding


@dataclass(frozen=True)
class ShiftedActionChunk:
    """Previous unexecuted tail aligned to current chunk prefix positions."""

    actions: Tensor
    validity_mask: Tensor


@dataclass(frozen=True)
class ActionChunkCorrectionOutput:
    """Continuous corrected chunk and separate arm/gripper residuals."""

    action_chunk: Tensor
    arm_residual: Tensor
    gripper_residual: Tensor
    shifted: ShiftedActionChunk


def shift_action_chunk(previous_chunk: Tensor, execution_horizon: Tensor | int) -> ShiftedActionChunk:
    """Map ``A_j[R:H_a]`` to the next chunk prefix and mark valid positions."""

    if previous_chunk.ndim != 3 or previous_chunk.shape[-1] != 7:
        raise ValueError(f"previous_chunk must be [B,H,7], got {tuple(previous_chunk.shape)}")
    batch_size, horizon, _ = previous_chunk.shape
    r = torch.as_tensor(execution_horizon, device=previous_chunk.device, dtype=torch.long)
    if r.ndim == 0:
        r = r.expand(batch_size)
    if r.shape != (batch_size,) or bool((r < 1).any()) or bool((r > horizon).any()):
        raise ValueError("execution_horizon must be scalar/[B] in [1,H]")
    shifted = previous_chunk.new_zeros(previous_chunk.shape)
    mask = torch.zeros((batch_size, horizon), device=previous_chunk.device, dtype=torch.bool)
    for index in range(batch_size):
        start = int(r[index].item())
        valid = horizon - start
        if valid:
            shifted[index, :valid] = previous_chunk[index, start:]
            mask[index, :valid] = True
    return ShiftedActionChunk(actions=shifted, validity_mask=mask)


class MatchedActionChunkCorrection(nn.Module):
    """Predict residuals for all action tokens using matched query context."""

    def __init__(
        self,
        *,
        action_horizon: int = 10,
        observation_dim: int = 128,
        action_feature_dim: int = 128,
        hidden_dim: int = 256,
        scalar_dim: int = 16,
        token_dim: int = 32,
    ) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.observation_dim = int(observation_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.r_embedding = _ScalarEmbedding(scalar_dim)
        self.dt_embedding = _ScalarEmbedding(scalar_dim)
        self.age_embedding = _ScalarEmbedding(scalar_dim)
        self.token_embedding = nn.Embedding(self.action_horizon, token_dim)
        input_dim = 7 + 1 + self.observation_dim + self.action_feature_dim + 3 * scalar_dim + token_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.arm_head = nn.Linear(self.hidden_dim, 6)
        self.gripper_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        previous_action_chunk: Tensor,
        observation_feature: Tensor,
        executed_action_feature: Tensor,
        *,
        execution_horizon: Tensor | int,
        elapsed_time: Tensor | float,
        query_age: Tensor | int,
    ) -> ActionChunkCorrectionOutput:
        """Time-align the previous chunk and predict a continuous residual chunk."""

        if previous_action_chunk.shape[-2:] != (self.action_horizon, 7):
            raise ValueError("previous_action_chunk has the wrong shape")
        batch_size = previous_action_chunk.shape[0]
        if observation_feature.shape != (batch_size, self.observation_dim):
            raise ValueError("observation_feature has the wrong shape")
        if executed_action_feature.shape != (batch_size, self.action_feature_dim):
            raise ValueError("executed_action_feature has the wrong shape")
        shifted = shift_action_chunk(previous_action_chunk, execution_horizon)
        global_feature = torch.cat(
            (
                observation_feature,
                executed_action_feature,
                self.r_embedding(execution_horizon, observation_feature),
                self.dt_embedding(elapsed_time, observation_feature),
                self.age_embedding(query_age, observation_feature),
            ),
            dim=-1,
        )
        expanded = global_feature.unsqueeze(1).expand(-1, self.action_horizon, -1)
        token_ids = torch.arange(self.action_horizon, device=previous_action_chunk.device)
        tokens = self.token_embedding(token_ids).unsqueeze(0).expand(batch_size, -1, -1)
        hidden = self.trunk(
            torch.cat(
                (
                    shifted.actions,
                    shifted.validity_mask.to(previous_action_chunk.dtype).unsqueeze(-1),
                    expanded,
                    tokens,
                ),
                dim=-1,
            )
        )
        arm_residual = self.arm_head(hidden)
        gripper_residual = self.gripper_head(hidden)
        arm = torch.clamp(shifted.actions[..., :6] + arm_residual, -1.0, 1.0)
        gripper = shifted.actions[..., 6:7] + gripper_residual
        return ActionChunkCorrectionOutput(
            action_chunk=torch.cat((arm, gripper), dim=-1),
            arm_residual=arm_residual,
            gripper_residual=gripper_residual,
            shifted=shifted,
        )


def find_action_correction_hidden_dim(
    target_parameters: int,
    *,
    shared_parameters: int = 0,
    action_horizon: int = 10,
    minimum: int = 16,
    maximum: int = 1024,
) -> tuple[int, int, float]:
    """Find the action-correction width closest to a total parameter budget."""

    if target_parameters <= shared_parameters:
        raise ValueError("target_parameters must exceed shared_parameters")
    best: tuple[int, int, int] | None = None
    for hidden_dim in range(int(minimum), int(maximum) + 1):
        module = MatchedActionChunkCorrection(
            action_horizon=action_horizon,
            hidden_dim=hidden_dim,
        )
        count = shared_parameters + sum(parameter.numel() for parameter in module.parameters())
        candidate = (abs(count - target_parameters), hidden_dim, count)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, hidden_dim, count = best
    relative_error = abs(count - target_parameters) / float(target_parameters)
    return hidden_dim, count, relative_error
