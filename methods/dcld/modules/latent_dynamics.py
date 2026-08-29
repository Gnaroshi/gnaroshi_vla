"""Fixed-Euler latent dynamics for DCLD."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .time_embedding import sinusoidal_time_embedding


@dataclass
class LatentDynamicsOutput:
    latent: torch.Tensor
    dz: torch.Tensor
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


class FixedEulerLatentDynamics(nn.Module):
    """One-step fixed Euler update for vector or token latents."""

    def __init__(
        self,
        latent_dim: int,
        delta_dim: int = 512,
        hidden_dim: int = 1024,
        time_embed_frequencies: int = 4,
        gated: bool = True,
        gate_bias: float = -4.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.delta_dim = int(delta_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_embed_frequencies = int(time_embed_frequencies)
        self.time_dim = 1 + 2 * self.time_embed_frequencies
        self.gated = bool(gated)
        self.gate_bias = float(gate_bias)

        input_dim = self.latent_dim + self.delta_dim + self.time_dim
        self.vector_field = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        if self.gated:
            self.gate = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, self.latent_dim),
            )
        else:
            self.gate = None

    def forward(
        self,
        latent_prev: torch.Tensor,
        delta_feature: torch.Tensor,
        *,
        dt: torch.Tensor | float = 1.0,
        age: torch.Tensor | float = 1.0,
    ) -> LatentDynamicsOutput:
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
        delta = delta[:, None, :].expand(batch_size, token_count, self.delta_dim)

        age_batch = _batch_scalar(
            age,
            batch_size,
            device=latent_prev.device,
            dtype=latent_prev.dtype,
        )
        time_emb = sinusoidal_time_embedding(
            age_batch,
            num_frequencies=self.time_embed_frequencies,
            include_value=True,
        ).to(dtype=latent_prev.dtype)
        time_emb = time_emb[:, None, :].expand(batch_size, token_count, self.time_dim)

        inputs = torch.cat([latent_flat, delta, time_emb], dim=-1)
        dz = self.vector_field(inputs)
        if self.gate is not None:
            gate_logits = self.gate(inputs)
            gate = torch.sigmoid(gate_logits + self.gate_bias)
        else:
            gate = torch.ones_like(dz)

        dt_batch = _batch_scalar(
            dt,
            batch_size,
            device=latent_prev.device,
            dtype=latent_prev.dtype,
        )
        dt_view = dt_batch.reshape(batch_size, 1, 1)
        update = dt_view * gate * dz
        latent_next = (latent_flat + update).reshape(original_shape)

        dz = dz.reshape(original_shape)
        gate = gate.reshape(original_shape)
        update = update.reshape(original_shape)
        debug = {
            "delta_norm": delta_feature.detach().norm(dim=-1),
            "dz_norm": dz.detach().flatten(start_dim=1).norm(dim=-1),
            "update_norm": update.detach().flatten(start_dim=1).norm(dim=-1),
            "gate_mean": gate.detach().flatten(start_dim=1).mean(dim=-1),
            "gate_std": gate.detach().flatten(start_dim=1).std(dim=-1, unbiased=False),
            "gate_min": gate.detach().flatten(start_dim=1).min(dim=-1).values,
            "gate_max": gate.detach().flatten(start_dim=1).max(dim=-1).values,
            "gate_bias": latent_prev.new_full((batch_size,), self.gate_bias).detach(),
        }
        return LatentDynamicsOutput(
            latent=latent_next,
            dz=dz,
            gate=gate,
            update=update,
            debug=debug,
        )
