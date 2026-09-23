"""Seer adapters for the source-locked LatentLoop comparison baselines."""

from .factory import attach_comparison_adapter
from .seer_action_correction import SeerActionCorrectionAdapter
from .seer_nonrecurrent_latent import SeerNonRecurrentLatentAdapter

__all__ = [
    "attach_comparison_adapter",
    "SeerActionCorrectionAdapter",
    "SeerNonRecurrentLatentAdapter",
]
