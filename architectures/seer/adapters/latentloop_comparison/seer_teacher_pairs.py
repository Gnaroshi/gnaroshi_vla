"""Pure tensor construction for source-locked shifted Seer teacher contexts."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ShiftedContextPair:
    """Anchor/current contexts and observations for one fixed offset."""

    anchor_primary: Tensor
    anchor_wrist: Tensor
    anchor_state: Tensor
    anchor_text: Tensor
    current_primary: Tensor
    current_wrist: Tensor
    current_state: Tensor
    current_text: Tensor
    anchor_observation_index: int
    current_observation_index: int
    offset: int


def build_shifted_context_pair(
    images_primary: Tensor,
    images_wrist: Tensor,
    states: Tensor,
    text_tokens: Tensor,
    *,
    sequence_length: int,
    selected_step: int,
    offset: int,
) -> ShiftedContextPair:
    """Build ``C_tau`` and ``C_{tau+r}`` without changing the dataset."""

    if offset < 1:
        raise ValueError("offset must be at least one")
    if not 0 <= selected_step < sequence_length:
        raise ValueError("selected_step is outside the Seer context")
    required = sequence_length + offset
    tensors = {
        "images_primary": images_primary,
        "images_wrist": images_wrist,
        "states": states,
        "text_tokens": text_tokens,
    }
    short = {name: int(value.shape[1]) for name, value in tensors.items() if value.shape[1] < required}
    if short:
        raise ValueError(f"offset={offset} requires window length {required}; got {short}")
    anchor_index = selected_step
    current_index = selected_step + offset
    return ShiftedContextPair(
        anchor_primary=images_primary[:, :sequence_length],
        anchor_wrist=images_wrist[:, :sequence_length],
        anchor_state=states[:, :sequence_length],
        anchor_text=text_tokens[:, :sequence_length],
        current_primary=images_primary[:, offset:required],
        current_wrist=images_wrist[:, offset:required],
        current_state=states[:, offset:required],
        current_text=text_tokens[:, offset:required],
        anchor_observation_index=anchor_index,
        current_observation_index=current_index,
        offset=offset,
    )
