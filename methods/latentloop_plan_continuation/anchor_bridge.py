"""Non-recurrent segment-anchor to current-latent baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AnchorBridgeOutput:
    """Output and diagnostics from a direct anchor-to-current prediction."""

    latent: Tensor
    residual: Tensor
    gate: Tensor


class NonRecurrentAnchorToCurrentBridge(nn.Module):
    """Predict from a fixed segment anchor without recursive latent input."""

    def __init__(
        self,
        latent_dim: int = 384,
        motion_dim: int = 128,
        action_pred_steps: int = 3,
        hidden_dim: int = 256,
        age_dim: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.motion_dim = int(motion_dim)
        self.action_pred_steps = int(action_pred_steps)
        self.hidden_dim = int(hidden_dim)
        self.anchor_norm = nn.LayerNorm(self.latent_dim)
        self.age_embedding = nn.Sequential(
            nn.Linear(1, age_dim),
            nn.SiLU(),
            nn.Linear(age_dim, age_dim),
        )
        self.token_embedding = nn.Embedding(self.action_pred_steps, self.latent_dim)
        self.dynamics = nn.Sequential(
            nn.Linear(self.latent_dim + self.motion_dim + age_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.motion_dim + age_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, anchor_latent: Tensor, anchor_to_current_feature: Tensor, age: Tensor | float) -> AnchorBridgeOutput:
        """Predict a current latent using only the fixed anchor and current input."""

        if anchor_latent.shape[-2:] != (self.action_pred_steps, self.latent_dim):
            raise ValueError(
                "anchor_latent must end in "
                f"[{self.action_pred_steps},{self.latent_dim}], got {tuple(anchor_latent.shape)}"
            )
        leading = tuple(anchor_latent.shape[:-2])
        if anchor_to_current_feature.shape != leading + (self.motion_dim,):
            raise ValueError(
                "anchor_to_current_feature must have shape "
                f"{leading + (self.motion_dim,)}, got {tuple(anchor_to_current_feature.shape)}"
            )
        age_tensor = torch.as_tensor(
            age, device=anchor_latent.device, dtype=anchor_latent.dtype
        )
        if age_tensor.ndim == 0:
            age_tensor = age_tensor.expand(*leading, 1)
        elif age_tensor.shape == leading:
            age_tensor = age_tensor.unsqueeze(-1)
        else:
            age_tensor = torch.broadcast_to(age_tensor, leading + (1,))
        age_feature = self.age_embedding(age_tensor)
        motion = anchor_to_current_feature
        token_ids = torch.arange(self.action_pred_steps, device=anchor_latent.device)
        token_shape = (1,) * len(leading) + (self.action_pred_steps, self.latent_dim)
        token_feature = self.token_embedding(token_ids).reshape(token_shape).expand_as(anchor_latent)
        global_feature = torch.cat((motion, age_feature), dim=-1)
        expanded_global = global_feature.unsqueeze(-2).expand(
            *leading, self.action_pred_steps, -1
        )
        residual = self.dynamics(
            torch.cat((self.anchor_norm(anchor_latent) + token_feature, expanded_global), dim=-1)
        )
        gate = torch.sigmoid(self.gate(global_feature)).unsqueeze(-2)
        latent = anchor_latent + gate * residual
        return AnchorBridgeOutput(latent=latent, residual=residual, gate=gate)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def find_anchor_bridge_hidden_dim(
    target_parameters: int,
    *,
    latent_dim: int = 384,
    motion_dim: int = 128,
    action_pred_steps: int = 3,
    minimum: int = 16,
    maximum: int = 1024,
) -> tuple[int, int, float]:
    """Return the closest hidden size to a LatentLoop parameter target."""

    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    best: tuple[int, int, int] | None = None
    for hidden_dim in range(int(minimum), int(maximum) + 1):
        count = _parameter_count(
            NonRecurrentAnchorToCurrentBridge(
                latent_dim, motion_dim, action_pred_steps, hidden_dim
            )
        )
        candidate = (abs(count - int(target_parameters)), hidden_dim, count)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, hidden_dim, count = best
    relative_error = abs(count - int(target_parameters)) / float(target_parameters)
    return hidden_dim, count, relative_error
