"""LatentLoop segment-length and feedback-density experiment utilities."""

from .decision import (
    apply_commitment_feedback_decision,
    apply_pi05_port_readiness,
)
from .feedback_schedule import (
    FEEDBACK_SCHEDULES,
    SUPPORTED_SEGMENT_LENGTHS,
    FeedbackPlan,
    actual_feedback_density,
    build_feedback_plan,
    feedback_enabled,
    feedback_mask,
)

__all__ = [
    "FEEDBACK_SCHEDULES",
    "SUPPORTED_SEGMENT_LENGTHS",
    "FeedbackPlan",
    "actual_feedback_density",
    "apply_commitment_feedback_decision",
    "apply_pi05_port_readiness",
    "build_feedback_plan",
    "feedback_enabled",
    "feedback_mask",
]
