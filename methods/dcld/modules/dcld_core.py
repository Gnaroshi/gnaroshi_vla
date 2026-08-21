"""Architecture-neutral DCLD core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .fast_delta_encoder import DeltaObservation, FastVisualDeltaEncoder
from .latent_dynamics import FixedEulerLatentDynamics, LatentDynamicsOutput
from .low_rank_latent_dynamics import LowRankFixedEulerLatentDynamics


class DCLDMode:
    REAL_DELTA = "real_delta"
    NO_DELTA = "no_delta"
    SHUFFLED_DELTA = "shuffled_delta"
    PROPRIO_ONLY = "proprio_only"
    IMAGE_ONLY = "image_only"

    ALL = {
        REAL_DELTA,
        NO_DELTA,
        SHUFFLED_DELTA,
        PROPRIO_ONLY,
        IMAGE_ONLY,
    }


@dataclass
class DCLDUpdate:
    latent: torch.Tensor
    delta_feature: torch.Tensor
    dynamics: LatentDynamicsOutput
    mode: str
    debug: dict[str, torch.Tensor]


class DCLDCore(nn.Module):
    """Encode observation deltas and update a cached condition latent."""

    def __init__(
        self,
        latent_dim: int,
        delta_dim: int = 512,
        hidden_dim: int = 1024,
        image_size: int = 64,
        image_feature_dim: int = 256,
        proprio_feature_dim: int = 128,
        dynamics_type: str = "dense",
        rank_dim: int = 64,
        gate_mode: str = "dense",
        gate_bias: float = -4.0,
        use_post_layernorm: bool = False,
    ) -> None:
        super().__init__()
        if dynamics_type not in {"dense", "low_rank"}:
            raise ValueError(f"Unsupported DCLD dynamics_type: {dynamics_type}")
        self.latent_dim = int(latent_dim)
        self.delta_dim = int(delta_dim)
        self.hidden_dim = int(hidden_dim)
        self.dynamics_type = dynamics_type
        self.rank_dim = int(rank_dim)
        self.gate_mode = "scalar" if dynamics_type == "low_rank" and gate_mode == "dense" else gate_mode
        self.gate_bias = float(gate_bias)
        self.use_post_layernorm = bool(use_post_layernorm)
        self.delta_encoder = FastVisualDeltaEncoder(
            image_size=image_size,
            image_feature_dim=image_feature_dim,
            proprio_feature_dim=proprio_feature_dim,
            output_dim=delta_dim,
        )
        if self.dynamics_type == "dense":
            self.dynamics = FixedEulerLatentDynamics(
                latent_dim=latent_dim,
                delta_dim=delta_dim,
                hidden_dim=hidden_dim,
                gate_bias=gate_bias,
            )
        else:
            self.dynamics = LowRankFixedEulerLatentDynamics(
                latent_dim=latent_dim,
                delta_dim=delta_dim,
                hidden_dim=hidden_dim,
                rank_dim=rank_dim,
                gate_mode=self.gate_mode,
                gate_bias=gate_bias,
                use_post_layernorm=use_post_layernorm,
            )

    def encode_delta(
        self,
        obs: DeltaObservation,
        *,
        latent_ref: torch.Tensor,
        mode: str = DCLDMode.REAL_DELTA,
    ) -> torch.Tensor:
        if mode not in DCLDMode.ALL:
            raise ValueError(f"Unknown DCLD mode: {mode}")

        batch_size = latent_ref.shape[0]
        device = latent_ref.device
        dtype = latent_ref.dtype

        if mode == DCLDMode.NO_DELTA:
            return torch.zeros(batch_size, self.delta_dim, device=device, dtype=dtype)

        use_images = mode != DCLDMode.PROPRIO_ONLY
        use_proprio = mode != DCLDMode.IMAGE_ONLY
        delta = self.delta_encoder(obs, use_images=use_images, use_proprio=use_proprio)
        delta = delta.to(device=device, dtype=dtype)

        if mode == DCLDMode.SHUFFLED_DELTA and batch_size > 1:
            delta = delta.roll(shifts=1, dims=0)

        return delta

    def update_latent(
        self,
        latent_prev: torch.Tensor,
        obs: DeltaObservation,
        *,
        dt: torch.Tensor | float = 1.0,
        age: torch.Tensor | float = 1.0,
        mode: str = DCLDMode.REAL_DELTA,
    ) -> DCLDUpdate:
        delta_feature = self.encode_delta(obs, latent_ref=latent_prev, mode=mode)
        dynamics = self.dynamics(latent_prev, delta_feature, dt=dt, age=age)
        debug = dict(dynamics.debug)
        debug["delta_feature_mean"] = delta_feature.detach().mean(dim=-1)
        return DCLDUpdate(
            latent=dynamics.latent,
            delta_feature=delta_feature,
            dynamics=dynamics,
            mode=mode,
            debug=debug,
        )
