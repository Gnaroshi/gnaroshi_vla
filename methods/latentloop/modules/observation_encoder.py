"""Compact encoder for changes between consecutive policy-query observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ImageViews = Tensor | Sequence[Tensor]


@dataclass(frozen=True)
class ObservationPair:
    """Previous/current query observations consumed by LatentLoop."""

    previous_images: ImageViews
    current_images: ImageViews
    previous_proprio: Tensor
    current_proprio: Tensor


def _as_views(images: ImageViews) -> list[Tensor]:
    if torch.is_tensor(images):
        if images.ndim == 5:
            return [images[:, index] for index in range(images.shape[1])]
        if images.ndim == 4:
            return [images]
        if images.ndim == 3:
            return [images.unsqueeze(0)]
        raise ValueError(f"Unsupported image shape: {tuple(images.shape)}")
    return list(images)


def _float_chw(image: Tensor) -> Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected a batched image, got {tuple(image.shape)}")
    if image.shape[1] in {1, 3, 4}:
        chw = image
    elif image.shape[-1] in {1, 3, 4}:
        chw = image.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Cannot infer image channels from {tuple(image.shape)}")
    chw = chw.float()
    if chw.numel() and float(chw.detach().max().item()) > 2.0:
        chw = chw / 255.0
    if chw.shape[1] == 4:
        chw = chw[:, :3]
    if chw.shape[1] == 1:
        chw = chw.repeat(1, 3, 1, 1)
    return chw


class ObservationChangeEncoder(nn.Module):
    """Encode adjacent raw RGB/proprio observations without a VLA dependency."""

    def __init__(
        self,
        *,
        proprio_dim: int = 8,
        image_size: int = 64,
        image_feature_dim: int = 64,
        proprio_feature_dim: int = 32,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.proprio_dim = int(proprio_dim)
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
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, self.image_feature_dim),
            nn.GELU(),
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(3 * self.proprio_dim, self.proprio_feature_dim),
            nn.GELU(),
            nn.Linear(self.proprio_feature_dim, self.proprio_feature_dim),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.image_feature_dim + self.proprio_feature_dim),
            nn.Linear(self.image_feature_dim + self.proprio_feature_dim, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, pair: ObservationPair, *, use_observation: bool = True) -> Tensor:
        """Return one feature per batch item; zero only for the no-observation ablation."""

        previous_q = pair.previous_proprio
        current_q = pair.current_proprio
        if previous_q.ndim != 2 or current_q.shape != previous_q.shape:
            raise ValueError("previous/current proprio must have identical [B,Q] shapes")
        if previous_q.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"Expected proprio dim {self.proprio_dim}, got {previous_q.shape[-1]}"
            )
        batch_size = previous_q.shape[0]
        if not use_observation:
            return previous_q.new_zeros((batch_size, self.output_dim), dtype=torch.float32)

        previous_views = _as_views(pair.previous_images)
        current_views = _as_views(pair.current_images)
        if not previous_views or len(previous_views) != len(current_views):
            raise ValueError("previous/current image view counts must match and be nonzero")
        image_features: list[Tensor] = []
        for previous, current in zip(previous_views, current_views):
            previous_f = _float_chw(previous)
            current_f = _float_chw(current).to(previous_f.device)
            if previous_f.shape != current_f.shape:
                raise ValueError("previous/current image shapes must match")
            previous_f = F.interpolate(
                previous_f,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            current_f = F.interpolate(
                current_f,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            image_features.append(
                self.image_encoder(torch.cat((previous_f, current_f, current_f - previous_f), dim=1))
            )
        image_feature = torch.stack(image_features, dim=0).mean(dim=0)
        previous_q = previous_q.to(device=image_feature.device, dtype=image_feature.dtype)
        current_q = current_q.to(device=image_feature.device, dtype=image_feature.dtype)
        proprio_feature = self.proprio_encoder(
            torch.cat((previous_q, current_q, current_q - previous_q), dim=-1)
        )
        return self.output(torch.cat((image_feature, proprio_feature), dim=-1))
