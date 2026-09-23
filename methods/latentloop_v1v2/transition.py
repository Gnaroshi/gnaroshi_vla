"""Action-grounded variable-time transition used by LatentLoop V1.

The module predicts the Seer action condition. It does not own an action head.
Both direct and composed paths use the same delta encoder, action conditioner,
and latent dynamics parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _interval_tensor(interval: int | Tensor, batch: int, device: torch.device) -> Tensor:
    value = torch.as_tensor(interval, device=device, dtype=torch.long)
    if value.ndim == 0:
        value = value.expand(batch)
    if value.shape != (batch,):
        raise ValueError(f"interval must be scalar or [B], got {tuple(value.shape)}")
    if bool(((value < 1) | (value > 3)).any()):
        raise ValueError("LatentLoop V1 supports intervals m in {1,2,3}")
    return value


class ExecutedActionSequenceEncoder(nn.Module):
    """Encode only actions that were actually executed between observations."""

    def __init__(self, action_dim: int = 7, max_interval: int = 3, output_dim: int = 64) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_interval = int(max_interval)
        input_dim = self.max_interval * self.action_dim + self.max_interval
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 48),
            nn.SiLU(),
            nn.Linear(48, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, actions: Tensor, interval: int | Tensor) -> Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"executed actions must be [B,M,{self.action_dim}]")
        batch, available, _ = actions.shape
        lengths = _interval_tensor(interval, batch, actions.device)
        if bool((lengths > available).any()):
            raise ValueError("executed-action sequence is shorter than the requested interval")
        positions = torch.arange(self.max_interval, device=actions.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        padded = actions.new_zeros((batch, self.max_interval, self.action_dim))
        copied = min(available, self.max_interval)
        padded[:, :copied] = actions[:, :copied]
        padded = padded * mask.unsqueeze(-1).to(actions.dtype)
        return self.network(torch.cat((padded.flatten(1), mask.to(actions.dtype)), dim=-1))


class ActionGroundedConditioner(nn.Module):
    """Fuse V0 observation change with executed actions and interval identity."""

    def __init__(self, motion_dim: int = 128, action_feature_dim: int = 64) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.action_encoder = ExecutedActionSequenceEncoder(output_dim=action_feature_dim)
        self.interval_embedding = nn.Embedding(4, 16)
        self.residual = nn.Sequential(
            nn.LayerNorm(motion_dim + action_feature_dim + 16),
            nn.Linear(motion_dim + action_feature_dim + 16, motion_dim),
            nn.SiLU(),
            nn.Linear(motion_dim, motion_dim),
        )
        self.output_norm = nn.LayerNorm(motion_dim)

    def forward(self, observation_delta: Tensor, actions: Tensor, interval: int | Tensor) -> Tensor:
        if observation_delta.ndim != 2 or observation_delta.shape[-1] != self.motion_dim:
            raise ValueError("observation_delta must be [B,motion_dim]")
        lengths = _interval_tensor(interval, observation_delta.shape[0], observation_delta.device)
        action_feature = self.action_encoder(actions, lengths)
        interval_feature = self.interval_embedding(lengths)
        residual = self.residual(torch.cat((observation_delta, action_feature, interval_feature), dim=-1))
        return self.output_norm(observation_delta + residual)


@dataclass(frozen=True)
class TransitionOutput:
    direct: Tensor
    composed: Tensor
    direct_feature: Tensor
    composed_features: tuple[Tensor, ...]
    intervals: Tensor


class VariableTimeLatentLoopTransition(nn.Module):
    """Shared-core direct/composed transition over m in {1,2,3}."""

    def __init__(self, delta_encoder: nn.Module, dynamics: nn.Module, motion_dim: int = 128) -> None:
        super().__init__()
        self.delta_encoder = delta_encoder
        self.dynamics = dynamics
        self.conditioner = ActionGroundedConditioner(motion_dim=motion_dim)

    def _delta(
        self,
        primary_a: Tensor,
        wrist_a: Tensor,
        primary_b: Tensor,
        wrist_b: Tensor,
        state_a: Tensor,
        state_b: Tensor,
    ) -> Tensor:
        return self.delta_encoder(
            [primary_a, wrist_a],
            [primary_b, wrist_b],
            q_key=state_a,
            q_cur=state_b,
        )

    def forward_step(
        self,
        *,
        previous_latent: Tensor,
        previous_primary: Tensor,
        previous_wrist: Tensor,
        previous_state: Tensor,
        current_primary: Tensor,
        current_wrist: Tensor,
        current_state: Tensor,
        executed_action: Tensor,
        age: int | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the efficient recurrent path used by fixed-K online evaluation.

        The action is the command executed after ``previous_*`` and before
        ``current_*``. Direct anchor-to-current prediction is intentionally not
        evaluated here; it is reserved for offline composition checks and V2.
        """

        if executed_action.ndim == 2:
            executed_action = executed_action.unsqueeze(1)
        if executed_action.ndim != 3 or executed_action.shape[1:] != (1, 7):
            raise ValueError("executed_action must be [B,7] or [B,1,7]")
        delta = self._delta(
            previous_primary,
            previous_wrist,
            current_primary,
            current_wrist,
            previous_state,
            current_state,
        )
        feature = self.conditioner(delta, executed_action, 1)
        latent = self.dynamics(previous_latent, feature, dt=1.0, age=age)
        return latent, feature

    def forward(
        self,
        *,
        anchor_latent: Tensor,
        primary_sequence: Tensor,
        wrist_sequence: Tensor,
        state_sequence: Tensor,
        executed_actions: Tensor,
        interval: int | Tensor,
    ) -> TransitionOutput:
        """Predict current condition without consuming any future teacher value.

        Sequences contain ordered causal observations tau..tau+m and executed
        actions tau..tau+m-1. The final observation is available at query time.
        """

        if anchor_latent.ndim != 3:
            raise ValueError("anchor_latent must be [B,P,D]")
        batch = anchor_latent.shape[0]
        intervals = _interval_tensor(interval, batch, anchor_latent.device)
        if not bool((intervals == intervals[0]).all()):
            raise ValueError("one training microbatch must use a single interval")
        m = int(intervals[0].item())
        for name, tensor in (
            ("primary_sequence", primary_sequence),
            ("wrist_sequence", wrist_sequence),
            ("state_sequence", state_sequence),
        ):
            if tensor.shape[0] != batch or tensor.shape[1] < m + 1:
                raise ValueError(f"{name} must contain ordered tau..tau+m values")
        if executed_actions.shape[:2] != (batch, m):
            raise ValueError("executed_actions must contain exactly tau..tau+m-1 in order")

        # Encode every causal observation transition once. The direct path
        # summarizes tau..tau+m, while the composed path consumes the same
        # ordered features one step at a time.
        step_deltas = [
            self._delta(
                primary_sequence[:, step - 1], wrist_sequence[:, step - 1],
                primary_sequence[:, step], wrist_sequence[:, step],
                state_sequence[:, step - 1], state_sequence[:, step],
            )
            for step in range(1, m + 1)
        ]
        direct_delta = torch.stack(step_deltas, dim=1).mean(dim=1)
        direct_feature = self.conditioner(direct_delta, executed_actions, intervals)
        direct = self.dynamics(anchor_latent, direct_feature, dt=intervals, age=intervals)

        composed = anchor_latent
        composed_features: list[Tensor] = []
        for step in range(1, m + 1):
            step_delta = step_deltas[step - 1]
            step_feature = self.conditioner(step_delta, executed_actions[:, step - 1 : step], 1)
            composed = self.dynamics(composed, step_feature, dt=1.0, age=float(step))
            composed_features.append(step_feature)
        return TransitionOutput(
            direct=direct,
            composed=composed,
            direct_feature=direct_feature,
            composed_features=tuple(composed_features),
            intervals=intervals,
        )
