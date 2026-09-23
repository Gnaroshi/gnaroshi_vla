"""Build shifted Seer contexts from already-collated LIBERO windows."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ShiftedContexts:
    """Current and one-step-shifted inputs for frozen Seer teacher forwards."""

    primary: Tensor
    wrist: Tensor
    state: Tensor
    text: Tensor
    action: Tensor
    primary_next: Tensor
    wrist_next: Tensor
    state_next: Tensor
    text_next: Tensor
    action_next: Tensor


def build_shifted_contexts(
    primary: Tensor,
    wrist: Tensor,
    state: Tensor,
    text: Tensor,
    action: Tensor,
    sequence_length: int,
) -> ShiftedContexts:
    """Slice ``C_t`` and ``C_{t+1}`` exactly as Seer shifted-context training."""
    tensors = {"primary": primary, "wrist": wrist, "state": state, "text": text, "action": action}
    for name, tensor in tensors.items():
        if tensor.shape[1] < sequence_length + 1:
            raise ValueError(
                f"{name} needs at least {sequence_length + 1} steps, got {tensor.shape}"
            )
    current = slice(0, sequence_length)
    shifted = slice(1, sequence_length + 1)
    return ShiftedContexts(
        primary[:, current],
        wrist[:, current],
        state[:, current],
        text[:, current],
        action[:, current],
        primary[:, shifted],
        wrist[:, shifted],
        state[:, shifted],
        text[:, shifted],
        action[:, shifted],
    )
