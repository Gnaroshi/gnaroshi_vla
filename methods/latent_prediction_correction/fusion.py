"""Pure latent selection rules for the prediction-correction experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


FILTER_MODES = (
    "off",
    "raw_full",
    "recurrent_prior",
    "fixed_filter",
    "full_latent_ema",
)


@dataclass(frozen=True)
class LatentFilterResult:
    """Selected latent and execution facts for one environment step."""

    latent: torch.Tensor
    mode: str
    used_recurrent_prior: bool
    reused_full_action: bool
    initialized_from_full: bool


def _validate_weight(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _validate_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"Latent shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    if left.dtype != right.dtype:
        raise ValueError(f"Latent dtype mismatch: {left.dtype} vs {right.dtype}")
    if left.device != right.device:
        raise ValueError(f"Latent device mismatch: {left.device} vs {right.device}")


def fixed_latent_fusion(
    z_prior: torch.Tensor,
    z_full: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return (1-alpha) * z_prior + alpha * z_full with exact endpoints."""
    _validate_pair(z_prior, z_full)
    alpha = _validate_weight("alpha", alpha)
    if alpha == 0.0:
        return z_prior
    if alpha == 1.0:
        return z_full
    return torch.lerp(z_prior, z_full, alpha)


def ema_latent(
    z_previous: torch.Tensor,
    z_full: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return a full-latent EMA update with exact beta endpoints."""
    _validate_pair(z_previous, z_full)
    beta = _validate_weight("beta", beta)
    if beta == 0.0:
        return z_previous
    if beta == 1.0:
        return z_full
    return torch.lerp(z_previous, z_full, beta)


def select_latent(
    mode: str,
    z_full: torch.Tensor,
    z_previous: Optional[torch.Tensor] = None,
    z_prior: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> LatentFilterResult:
    """Select the executed latent for one step without decoding an action."""
    if mode not in FILTER_MODES or mode == "off":
        raise ValueError(f"Active filter mode required, got {mode!r}")

    first_step = z_previous is None
    if first_step:
        return LatentFilterResult(z_full, mode, False, True, True)

    _validate_pair(z_previous, z_full)
    if mode == "raw_full":
        return LatentFilterResult(z_full, mode, False, True, False)
    if mode == "full_latent_ema":
        latent = ema_latent(z_previous, z_full, beta)
        return LatentFilterResult(
            latent,
            mode,
            False,
            float(beta) == 1.0,
            False,
        )

    if mode == "fixed_filter":
        alpha = _validate_weight("alpha", alpha)
        if alpha == 1.0:
            return LatentFilterResult(z_full, mode, False, True, False)

    if z_prior is None:
        raise ValueError(f"mode={mode} requires z_prior after episode initialization")
    _validate_pair(z_prior, z_full)
    if mode == "recurrent_prior":
        return LatentFilterResult(z_prior, mode, True, False, False)
    if mode == "fixed_filter":
        latent = fixed_latent_fusion(z_prior, z_full, alpha)
        return LatentFilterResult(
            latent,
            mode,
            float(alpha) < 1.0,
            float(alpha) == 1.0,
            False,
        )
    raise AssertionError(f"Unhandled filter mode: {mode}")
