"""Fast observation-delta encoder for DCLD.

This module is intentionally architecture-neutral. SimVLA-specific observation
packing lives in ``architectures/simvla/adapters/dcld``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


TensorOrImages = torch.Tensor | Sequence[torch.Tensor] | None


@dataclass
class DeltaObservation:
    """Key/current observation pair used to update a cached latent."""

    key_images: TensorOrImages = None
    cur_images: TensorOrImages = None
    key_proprio: torch.Tensor | None = None
    cur_proprio: torch.Tensor | None = None
    age: torch.Tensor | float | None = None
    metadata: dict | None = None


def _as_image_list(images: TensorOrImages) -> list[torch.Tensor]:
    if images is None:
        return []
    if torch.is_tensor(images):
        if images.ndim == 5:
            return [images[:, i] for i in range(images.shape[1])]
        if images.ndim == 4:
            return [images]
        if images.ndim == 3:
            return [images.unsqueeze(0)]
        raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")
    return list(images)


def _channel_first(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor [B,C,H,W] or [B,H,W,C], got {tuple(image.shape)}")
    if image.shape[1] in (1, 3, 4):
        return image
    if image.shape[-1] in (1, 3, 4):
        return image.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Cannot infer image channel dimension from {tuple(image.shape)}")


def _float_image(image: torch.Tensor) -> torch.Tensor:
    image = _channel_first(image)
    if not torch.is_floating_point(image):
        image = image.float().div(255.0)
    else:
        image = image.float()
        if image.detach().max().item() > 2.0:
            image = image.div(255.0)
    if image.shape[1] == 4:
        image = image[:, :3]
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    return image


def _first_tensor(obs: DeltaObservation) -> torch.Tensor | None:
    for item in [obs.key_proprio, obs.cur_proprio]:
        if item is not None:
            return item
    for item in _as_image_list(obs.key_images) + _as_image_list(obs.cur_images):
        return item
    return None


class FastVisualDeltaEncoder(nn.Module):
    """Encode visual/proprio deltas into a compact control feature."""

    def __init__(
        self,
        image_size: int = 64,
        image_feature_dim: int = 256,
        proprio_feature_dim: int = 128,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.image_feature_dim = int(image_feature_dim)
        self.proprio_feature_dim = int(proprio_feature_dim)
        self.output_dim = int(output_dim)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, image_feature_dim),
            nn.GELU(),
        )
        self.proprio_encoder = nn.Sequential(
            nn.LazyLinear(proprio_feature_dim),
            nn.GELU(),
            nn.Linear(proprio_feature_dim, proprio_feature_dim),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(image_feature_dim + proprio_feature_dim),
            nn.Linear(image_feature_dim + proprio_feature_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        obs: DeltaObservation,
        *,
        use_images: bool = True,
        use_proprio: bool = True,
    ) -> torch.Tensor:
        ref = _first_tensor(obs)
        if ref is None:
            raise ValueError("DeltaObservation must contain at least one tensor")

        batch_size = ref.shape[0]
        device = ref.device
        dtype = torch.float32

        image_feature = torch.zeros(
            batch_size,
            self.image_feature_dim,
            device=device,
            dtype=dtype,
        )
        if use_images:
            key_images = _as_image_list(obs.key_images)
            cur_images = _as_image_list(obs.cur_images)
            if len(key_images) != len(cur_images):
                raise ValueError(
                    f"key/cur image count mismatch: {len(key_images)} vs {len(cur_images)}"
                )
            if key_images:
                feats = []
                for key, cur in zip(key_images, cur_images):
                    key = _float_image(key).to(device=device, dtype=dtype)
                    cur = _float_image(cur).to(device=device, dtype=dtype)
                    key = F.interpolate(
                        key,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    cur = F.interpolate(
                        cur,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    delta = cur - key
                    feats.append(self.image_encoder(torch.cat([key, cur, delta], dim=1)))
                image_feature = torch.stack(feats, dim=0).mean(dim=0)

        proprio_feature = torch.zeros(
            batch_size,
            self.proprio_feature_dim,
            device=device,
            dtype=dtype,
        )
        if use_proprio and obs.key_proprio is not None and obs.cur_proprio is not None:
            key_q = obs.key_proprio.to(device=device, dtype=dtype).flatten(start_dim=1)
            cur_q = obs.cur_proprio.to(device=device, dtype=dtype).flatten(start_dim=1)
            proprio_feature = self.proprio_encoder(torch.cat([key_q, cur_q, cur_q - key_q], dim=-1))

        return self.output(torch.cat([image_feature, proprio_feature], dim=-1))
