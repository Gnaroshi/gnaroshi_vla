"""Thin Seer-specific state adapter around architecture-agnostic fusion rules."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from methods.latent_prediction_correction.fusion import (
    FILTER_MODES,
    LatentFilterResult,
    select_latent,
)
from methods.latent_prediction_correction.metrics import (
    LatentPathTracker,
    latent_pair_metrics,
)


class SeerLatentFilterAdapter:
    """Select and audit Seer's cached action latent without trainable state."""

    def __init__(
        self,
        mode: str,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        if mode not in FILTER_MODES or mode == "off":
            raise ValueError(f"Active filter mode required, got {mode!r}")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.mode = mode
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._paths = LatentPathTracker()

    def reset(self) -> None:
        """Reset episode-local metric state."""
        self._paths.reset()

    def requires_recurrent_prior(self, has_previous: bool) -> bool:
        """Return whether this step needs the checkpoint-specific LR-NODE updater."""
        if not has_previous:
            return False
        if self.mode == "recurrent_prior":
            return True
        return self.mode == "fixed_filter" and self.alpha < 1.0

    def select(
        self,
        z_full: torch.Tensor,
        z_previous: Optional[torch.Tensor],
        z_prior: Optional[torch.Tensor],
    ) -> LatentFilterResult:
        """Select the executed latent without running diagnostic reductions."""
        return select_latent(
            mode=self.mode,
            z_full=z_full,
            z_previous=z_previous,
            z_prior=z_prior,
            alpha=self.alpha,
            beta=self.beta,
        )

    def diagnostics(
        self,
        selection: LatentFilterResult,
        z_full: torch.Tensor,
        z_prior: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Compute scalar diagnostics without changing the selected latent."""
        metrics = latent_pair_metrics(
            z_prior=z_prior,
            z_full=z_full,
            z_filter=selection.latent,
            alpha=self.alpha if self.mode == "fixed_filter" else 0.0,
        )
        metrics.update(
            self._paths.update(
                prior=z_prior,
                full=z_full,
                filter=selection.latent,
            )
        )
        return metrics
