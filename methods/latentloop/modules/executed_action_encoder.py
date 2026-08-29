"""Encoder for the action subchunk that was actually sent to the environment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PaddedExecutedActions:
    """Masked fixed-width representation of variable execution horizons."""

    actions: Tensor
    validity_mask: Tensor
    lengths: Tensor


@dataclass(frozen=True)
class ExecutedActionEncoding:
    """Executed-action feature and the exact masked input used to compute it."""

    feature: Tensor
    padded: PaddedExecutedActions
    cumulative_arm_displacement: Tensor
    gripper_switch: Tensor


def _length_tensor(lengths: Tensor | int, batch_size: int, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(lengths, device=device, dtype=torch.long)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    if tensor.shape != (batch_size,):
        raise ValueError(f"lengths must be scalar or [B], got {tuple(tensor.shape)}")
    return tensor


def pad_executed_actions(
    actions: Tensor,
    lengths: Tensor | int,
    *,
    max_actions: int = 5,
    action_dim: int = 7,
) -> PaddedExecutedActions:
    """Pad only the first ``lengths[b]`` executed actions and return a mask."""

    if actions.ndim != 3 or actions.shape[-1] != action_dim:
        raise ValueError(f"actions must be [B,R,{action_dim}], got {tuple(actions.shape)}")
    batch_size, available, _ = actions.shape
    length_tensor = _length_tensor(lengths, batch_size, actions.device)
    if bool((length_tensor < 1).any()) or bool((length_tensor > max_actions).any()):
        raise ValueError(f"executed lengths must be in [1,{max_actions}]")
    if bool((length_tensor > available).any()):
        raise ValueError("an executed length exceeds the supplied action tensor")
    positions = torch.arange(max_actions, device=actions.device).unsqueeze(0)
    mask = positions < length_tensor.unsqueeze(1)
    padded = actions.new_zeros((batch_size, max_actions, action_dim))
    copied = min(available, max_actions)
    padded[:, :copied] = actions[:, :copied]
    padded = padded * mask.unsqueeze(-1).to(actions.dtype)
    return PaddedExecutedActions(actions=padded, validity_mask=mask, lengths=length_tensor)


class ExecutedActionEncoder(nn.Module):
    """Encode final postprocessed executed actions, horizon, and elapsed time."""

    def __init__(
        self,
        *,
        action_dim: int = 7,
        max_actions: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_actions = int(max_actions)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        summary_dim = (
            self.max_actions * self.action_dim
            + self.max_actions
            + 1
            + 1
            + 6
            + 1
        )
        self.network = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )

    def forward(
        self,
        executed_actions: Tensor,
        lengths: Tensor | int,
        elapsed_time: Tensor | float,
    ) -> ExecutedActionEncoding:
        """Encode actions after masking every predicted-but-unexecuted token."""

        padded = pad_executed_actions(
            executed_actions,
            lengths,
            max_actions=self.max_actions,
            action_dim=self.action_dim,
        )
        batch_size = executed_actions.shape[0]
        elapsed = torch.as_tensor(
            elapsed_time,
            device=executed_actions.device,
            dtype=executed_actions.dtype,
        )
        if elapsed.ndim == 0:
            elapsed = elapsed.expand(batch_size)
        if elapsed.shape != (batch_size,):
            raise ValueError("elapsed_time must be scalar or [B]")
        mask_f = padded.validity_mask.to(executed_actions.dtype)
        cumulative_arm = (padded.actions[..., :6] * mask_f.unsqueeze(-1)).sum(dim=1)
        valid_pairs = padded.validity_mask[:, 1:] & padded.validity_mask[:, :-1]
        gripper_sign = padded.actions[..., 6] >= 0
        switches = ((gripper_sign[:, 1:] != gripper_sign[:, :-1]) & valid_pairs).any(dim=1)
        normalized_r = padded.lengths.to(executed_actions.dtype) / float(self.max_actions)
        summary = torch.cat(
            (
                padded.actions.flatten(start_dim=1),
                mask_f,
                normalized_r.unsqueeze(-1),
                elapsed.unsqueeze(-1),
                cumulative_arm,
                switches.to(executed_actions.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        return ExecutedActionEncoding(
            feature=self.network(summary),
            padded=padded,
            cumulative_arm_displacement=cumulative_arm,
            gripper_switch=switches,
        )
