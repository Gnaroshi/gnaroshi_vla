"""Controlled baselines for locating LatentLoop's useful correction state."""

from .action_space_correction import (
    ActionCorrectionCache,
    ActionCorrectionOutput,
    MatchedActionSpaceCorrection,
    ShiftedActionHorizon,
    find_action_correction_hidden_dim,
    shift_action_horizon,
    shift_action_horizon_with_mask,
)
from .nonrecurrent_latent import (
    K4_OFFSETS,
    AnchorBridgeOutput,
    NonRecurrentAnchorState,
    NonRecurrentAnchorToCurrentBridge,
    cyclic_k4_offset,
    find_nonrecurrent_hidden_dim,
)

__all__ = [
    "ActionCorrectionCache",
    "ActionCorrectionOutput",
    "AnchorBridgeOutput",
    "K4_OFFSETS",
    "MatchedActionSpaceCorrection",
    "NonRecurrentAnchorState",
    "NonRecurrentAnchorToCurrentBridge",
    "ShiftedActionHorizon",
    "cyclic_k4_offset",
    "find_action_correction_hidden_dim",
    "find_nonrecurrent_hidden_dim",
    "shift_action_horizon",
    "shift_action_horizon_with_mask",
]
