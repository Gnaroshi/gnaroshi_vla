"""Encoding for action subchunks that were actually executed by the robot."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ExecutedActionFeatures:
    """Fixed-width summary and the validated action count."""

    feature: Tensor
    lengths: Tensor


class ExecutedActionEncoder(nn.Module):
    """Encode ordered, postprocessed environment actions with a small GRU.

    The caller must pass the final actions sent to the environment, not raw
    action-expert outputs. Variable sequence lengths support direct transitions
    spanning multiple native execution horizons.
    """

    def __init__(self, action_dim: int = 7, hidden_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.output_projection = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, actions: Tensor, lengths: Tensor | None = None) -> ExecutedActionFeatures:
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must be [B,T,{self.action_dim}], got {tuple(actions.shape)}"
            )
        batch_size, steps, _ = actions.shape
        if steps < 1:
            raise ValueError("at least one executed action is required")
        if not torch.isfinite(actions).all():
            raise ValueError("executed actions contain NaN or Inf")

        if lengths is None:
            lengths = torch.full((batch_size,), steps, dtype=torch.long, device=actions.device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=actions.device)
        if lengths.shape != (batch_size,):
            raise ValueError(f"lengths must be [B], got {tuple(lengths.shape)}")
        if torch.any(lengths < 1) or torch.any(lengths > steps):
            raise ValueError("each executed-action length must be in [1,T]")

        embedded = self.input_projection(actions)
        output, _ = self.gru(embedded)
        index = (lengths - 1).view(batch_size, 1, 1).expand(-1, 1, self.hidden_dim)
        final = output.gather(1, index).squeeze(1)
        mask = torch.arange(steps, device=actions.device)[None, :] < lengths[:, None]
        mean = (output * mask[..., None]).sum(dim=1) / lengths[:, None].to(output.dtype)
        return ExecutedActionFeatures(
            feature=self.output_projection(torch.cat((final, mean), dim=-1)),
            lengths=lengths,
        )
