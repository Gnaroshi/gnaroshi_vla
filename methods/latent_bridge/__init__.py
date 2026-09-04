"""Architecture-agnostic contracts for Latent Bridge adapters."""

from .contracts import (
    BridgePreset,
    ComputeMatchedTrainingContract,
    TrainingContract,
    bridge_distillation_loss,
    should_full_refresh,
)

__all__ = [
    "BridgePreset",
    "ComputeMatchedTrainingContract",
    "TrainingContract",
    "bridge_distillation_loss",
    "should_full_refresh",
]
