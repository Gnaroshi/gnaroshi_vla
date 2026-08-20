"""Direct/composed transition utilities for controlled composition consistency."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class CompositionOutput:
    final_state: Tensor
    intermediate_states: tuple[Tensor, ...]


def compose_one_query_updates(
    initial_state: Tensor,
    step_inputs: Sequence[dict[str, Tensor | int]],
    update: Callable[..., Tensor],
) -> CompositionOutput:
    """Apply one shared one-query transition repeatedly in chronological order."""

    state = initial_state
    states: list[Tensor] = []
    for index, kwargs in enumerate(step_inputs, start=1):
        next_state = update(state=state, **kwargs)
        if next_state.shape != state.shape:
            raise ValueError(f"composition step {index} changed state shape")
        state = next_state
        states.append(state)
    if not states:
        raise ValueError("composition requires at least one transition")
    return CompositionOutput(final_state=state, intermediate_states=tuple(states))


def normalized_composition_distance(direct: Tensor, composed: Tensor, eps: float = 1e-6) -> Tensor:
    if direct.shape != composed.shape:
        raise ValueError("direct and composed states must have identical shapes")
    scale = direct.detach().square().mean(dim=tuple(range(1, direct.ndim)), keepdim=True).sqrt().clamp_min(eps)
    normalized = (direct - composed) / scale
    return normalized.square().mean(dim=tuple(range(1, normalized.ndim)))
