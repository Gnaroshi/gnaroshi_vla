"""Real-world SimVLA adaptation without modifying the vendored upstream source."""

from .geometry import (
    ACTION_DIM,
    ACTION_HORIZON,
    EXECUTION_HORIZON,
    PROPRIO_DIM,
    RealActionScales,
    encode_opposed_finger_state,
    transition_to_normalized_action,
)

REAL_WORLD_TASK = "stackcupanddoll"

__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "EXECUTION_HORIZON",
    "PROPRIO_DIM",
    "RealActionScales",
    "encode_opposed_finger_state",
    "transition_to_normalized_action",
    "REAL_WORLD_TASK",
]
