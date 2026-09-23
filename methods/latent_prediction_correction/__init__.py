"""Prediction-correction latent fusion primitives and analysis helpers."""

from .fusion import (
    FILTER_MODES,
    LatentFilterResult,
    ema_latent,
    fixed_latent_fusion,
    select_latent,
)

__all__ = [
    "FILTER_MODES",
    "LatentFilterResult",
    "ema_latent",
    "fixed_latent_fusion",
    "select_latent",
]
