"""State machine for current and one-step time-shifted feedback features."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class FeedbackSelection:
    """Feature consumed now and the environment step that produced it."""

    feature: Tensor
    source_step: int
    initialized_with_zero: bool


class FeedbackFeatureBuffer:
    """Stage features within one episode without permitting cross-episode reuse."""

    def __init__(self, mode: str = "current") -> None:
        if mode not in {"current", "time_shifted"}:
            raise ValueError(f"Unknown feedback source mode: {mode}")
        self.mode = mode
        self._staged_feature: Tensor | None = None
        self._staged_step: int | None = None

    def reset(self) -> None:
        """Clear all state at an episode boundary or full semantic refresh."""

        self._staged_feature = None
        self._staged_step = None

    def select(self, current_feature: Tensor, current_step: int) -> FeedbackSelection:
        """Select the consumed feature, then stage the current feature.

        ``time_shifted`` consumes the previously staged feature. Its first
        intermediate step consumes an explicit zero tensor and reports source
        step ``-1``. The current feature is always staged after selection.
        """

        if self.mode == "current":
            return FeedbackSelection(current_feature, int(current_step), False)
        if self._staged_feature is None:
            selected = current_feature.detach().clone().zero_()
            source_step = -1
            initialized = True
        else:
            selected = self._staged_feature
            source_step = int(self._staged_step)
            initialized = False
        self._staged_feature = current_feature.detach()
        self._staged_step = int(current_step)
        return FeedbackSelection(selected, source_step, initialized)
