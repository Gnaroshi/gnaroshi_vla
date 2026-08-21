"""Low-rank fixed-Euler latent dynamics for parameter-matched DCLD."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .time_embedding import sinusoidal_time_embedding


@dataclass
class LowRankLatentDynamicsOutput:
    latent: torch.Tensor
    dr: torch.Tensor
    dc: torch.Tensor
    gate: torch.Tensor
    update: torch.Tensor
    debug: dict[str, torch.Tensor]


def _batch_scalar(
    value: torch.Tensor | float,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.full((batch_size,), float(value), device=device, dtype=dtype)
    else:
        value = value.to(device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch_size)
        elif value.shape[0] != batch_size:
            raise ValueError(f"Expected batch scalar first dim {batch_size}, got {tuple(value.shape)}")
        value = value.reshape(batch_size)
    return value


class LowRankFixedEulerLatentDynamics(nn.Module):
    """One-step fixed Euler update with a low-rank latent vector field."""

    def __init__(
        self,
        latent_dim: int,
        delta_dim: int = 128,
        hidden_dim: int = 128,
        rank_dim: int = 64,
        time_embed_frequencies: int = 4,
        gate_mode: str = "scalar",
        gate_bias: float = -4.0,
        use_post_layernorm: bool = False,
    ) -> None:
        super().__init__()
        if gate_mode not in {"scalar", "token"}:
            raise ValueError(f"Unsupported low-rank gate_mode: {gate_mode}")
        self.latent_dim = int(latent_dim)
        self.delta_dim = int(delta_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank_dim = int(rank_dim)
        self.time_embed_frequencies = int(time_embed_frequencies)
        self.time_dim = 1 + 2 * self.time_embed_frequencies
        self.gate_mode = gate_mode
        self.gate_bias = float(gate_bias)
        self.use_post_layernorm = bool(use_post_layernorm)

        self.down_proj = nn.Linear(self.latent_dim, self.rank_dim)
        self.vector_field = nn.Sequential(
            nn.LayerNorm(self.rank_dim + self.delta_dim + 2 * self.time_dim),
            nn.Linear(self.rank_dim + self.delta_dim + 2 * self.time_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.rank_dim),
        )
        self.up_proj = nn.Linear(self.rank_dim, self.latent_dim)

        if self.gate_mode == "scalar":
            self.gate = nn.Sequential(
                nn.LayerNorm(self.delta_dim + self.time_dim),
                nn.Linear(self.delta_dim + self.time_dim, 1),
            )
        else:
            self.gate = nn.Sequential(
                nn.LayerNorm(self.rank_dim + self.delta_dim + self.time_dim),
                nn.Linear(self.rank_dim + self.delta_dim + self.time_dim, 1),
            )
        self.post_ln = nn.LayerNorm(self.latent_dim) if self.use_post_layernorm else None

    def forward(
        self,
        latent_prev: torch.Tensor,
        delta_feature: torch.Tensor,
        *,
        dt: torch.Tensor | float = 1.0,
        age: torch.Tensor | float = 1.0,
    ) -> LowRankLatentDynamicsOutput:
        if latent_prev.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent last dim {self.latent_dim}, got {latent_prev.shape[-1]}"
            )
        if delta_feature.ndim != 2 or delta_feature.shape[-1] != self.delta_dim:
            raise ValueError(
                f"Expected delta_feature [B,{self.delta_dim}], got {tuple(delta_feature.shape)}"
            )

        batch_size = latent_prev.shape[0]
        original_shape = latent_prev.shape
        token_shape = latent_prev.shape[1:-1]
        token_count = int(torch.tensor(token_shape).prod().item()) if token_shape else 1
        latent_flat = latent_prev.reshape(batch_size, token_count, self.latent_dim)

        delta = delta_feature.to(device=latent_prev.device, dtype=latent_prev.dtype)
        delta_b = delta[:, None, :].expand(batch_size, token_count, self.delta_dim)

        dt_batch = _batch_scalar(dt, batch_size, device=latent_prev.device, dtype=latent_prev.dtype)
        age_batch = _batch_scalar(age, batch_size, device=latent_prev.device, dtype=latent_prev.dtype)
        dt_emb = sinusoidal_time_embedding(
            dt_batch,
            num_frequencies=self.time_embed_frequencies,
            include_value=True,
        ).to(device=latent_prev.device, dtype=latent_prev.dtype)
        age_emb = sinusoidal_time_embedding(
            age_batch,
            num_frequencies=self.time_embed_frequencies,
            include_value=True,
        ).to(device=latent_prev.device, dtype=latent_prev.dtype)
        dt_b = dt_emb[:, None, :].expand(batch_size, token_count, self.time_dim)
        age_b = age_emb[:, None, :].expand(batch_size, token_count, self.time_dim)

        r = self.down_proj(latent_flat)
        dr = self.vector_field(torch.cat([r, delta_b, dt_b, age_b], dim=-1))
        dc = self.up_proj(dr)

        if self.gate_mode == "scalar":
            gate_input = torch.cat([delta, age_emb], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input)[:, None, :] + self.gate_bias)
            gate = gate.expand(batch_size, token_count, 1)
        else:
            gate = torch.sigmoid(self.gate(torch.cat([r, delta_b, age_b], dim=-1)) + self.gate_bias)

        dt_view = dt_batch.reshape(batch_size, 1, 1)
        update = dt_view * gate * dc
        latent_next = latent_flat + update
        if self.post_ln is not None:
            latent_next = self.post_ln(latent_next)
        latent_next = latent_next.reshape(original_shape)

        dc = dc.reshape(original_shape)
        update = update.reshape(original_shape)
        gate_debug = gate.reshape(*original_shape[:-1], 1)
        debug = {
            "delta_norm": delta_feature.detach().norm(dim=-1),
            "dr_norm": dr.detach().flatten(start_dim=1).norm(dim=-1),
            "dc_norm": dc.detach().flatten(start_dim=1).norm(dim=-1),
            "update_norm": update.detach().flatten(start_dim=1).norm(dim=-1),
            "gate_mean": gate.detach().flatten(start_dim=1).mean(dim=-1),
            "gate_min": gate.detach().flatten(start_dim=1).min(dim=-1).values,
            "gate_max": gate.detach().flatten(start_dim=1).max(dim=-1).values,
            "gate_bias": latent_prev.new_full((batch_size,), self.gate_bias).detach(),
        }
        return LowRankLatentDynamicsOutput(
            latent=latent_next,
            dr=dr.reshape(*original_shape[:-1], self.rank_dim),
            dc=dc,
            gate=gate_debug,
            update=update,
            debug=debug,
        )
