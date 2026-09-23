"""Predeclared feedback schedules for LatentLoop segment execution.

The notation used by this module is fixed for the paper protocol:

``o_t = (I_t^p, I_t^w, q_t)``
``C_t = (o_{t-H+1}, ..., o_t, ell)``
``z_t^F = F_phi(C_t)``
``u_t = E_eta(o_{t-1}, o_t)``
``z_t^L = U_theta(z_{t-1}^L, u_t, r_t)``
``a_t = A_psi(z_t)``

The feedback mask controls whether ``u_t`` is observation-conditioned. It does
not control updater frequency: the updater runs at every intermediate offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


SUPPORTED_SEGMENT_LENGTHS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10)
FEEDBACK_SCHEDULES: Tuple[str, ...] = ("dense", "alternate", "none")


def validate_segment_length(segment_length: int) -> int:
    """Validate and return a predeclared segment length ``L``."""

    value = int(segment_length)
    if value not in SUPPORTED_SEGMENT_LENGTHS:
        raise ValueError(
            f"Unsupported segment_length={value}; expected one of "
            f"{SUPPORTED_SEGMENT_LENGTHS}"
        )
    return value


def validate_feedback_schedule(schedule: str) -> str:
    """Validate and normalize a named feedback schedule."""

    value = str(schedule).strip().lower()
    if value not in FEEDBACK_SCHEDULES:
        raise ValueError(
            f"Unsupported feedback_schedule={value!r}; expected one of "
            f"{FEEDBACK_SCHEDULES}"
        )
    return value


def feedback_mask(segment_length: int, schedule: str) -> Tuple[int, ...]:
    """Return ``m_i`` for intermediate offsets ``i=1,...,L-1``.

    ``dense`` enables every offset, ``alternate`` enables odd offsets, and
    ``none`` disables every offset. ``L=1`` has no intermediate offsets and
    therefore returns an empty mask.
    """

    length = validate_segment_length(segment_length)
    name = validate_feedback_schedule(schedule)
    offsets = range(1, length)
    if name == "dense":
        return tuple(1 for _ in offsets)
    if name == "alternate":
        return tuple(1 if offset % 2 == 1 else 0 for offset in offsets)
    return tuple(0 for _ in offsets)


def actual_feedback_density(mask: Sequence[int]) -> Optional[float]:
    """Return ``rho = sum_i m_i / (L-1)`` or ``None`` for ``L=1``."""

    values = tuple(int(value) for value in mask)
    if any(value not in (0, 1) for value in values):
        raise ValueError(f"Feedback mask must be binary, got {values}")
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def feedback_enabled(segment_length: int, schedule: str, segment_offset: int) -> bool:
    """Return whether current observation conditioning is active at an offset."""

    length = validate_segment_length(segment_length)
    offset = int(segment_offset)
    if offset <= 0 or offset >= length:
        raise ValueError(
            f"segment_offset must be in [1, {length - 1}] for L={length}; "
            f"got {offset}"
        )
    return bool(feedback_mask(length, schedule)[offset - 1])


@dataclass(frozen=True)
class FeedbackPlan:
    """Immutable feedback schedule for one segment length."""

    segment_length: int
    schedule: str
    mask: Tuple[int, ...]
    density: Optional[float]

    def enabled_at(self, segment_offset: int) -> bool:
        """Return the binary feedback decision at one intermediate offset."""

        return feedback_enabled(self.segment_length, self.schedule, segment_offset)


def build_feedback_plan(segment_length: int, schedule: str) -> FeedbackPlan:
    """Construct a validated feedback plan and its actual density."""

    length = validate_segment_length(segment_length)
    name = validate_feedback_schedule(schedule)
    mask = feedback_mask(length, name)
    return FeedbackPlan(
        segment_length=length,
        schedule=name,
        mask=mask,
        density=actual_feedback_density(mask),
    )
