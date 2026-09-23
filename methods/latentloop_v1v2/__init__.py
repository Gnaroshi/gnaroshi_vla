"""Source-locked LatentLoop V1/V2 research primitives."""

from .losses import V1LossOutput, compute_v1_losses
from .protocol import Operation, OperationCounters, operation_for_step, periodic_schedule
from .scheduler import (
    AdaptiveRefreshScheduler,
    DefectNormalizer,
    RefreshDecision,
    SchedulerState,
)
from .selection import select_v1_budget
from .splits import assert_disjoint_split_manifests
from .transition import (
    TransitionOutput,
    VariableTimeLatentLoopTransition,
    count_trainable_parameters,
)

__all__ = [
    "AdaptiveRefreshScheduler",
    "DefectNormalizer",
    "RefreshDecision",
    "SchedulerState",
    "TransitionOutput",
    "V1LossOutput",
    "VariableTimeLatentLoopTransition",
    "Operation",
    "OperationCounters",
    "assert_disjoint_split_manifests",
    "compute_v1_losses",
    "count_trainable_parameters",
    "operation_for_step",
    "periodic_schedule",
    "select_v1_budget",
]
