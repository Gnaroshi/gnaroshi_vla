"""Architecture-neutral components for variable-time LatentLoop."""

from .action_grounding import ExecutedActionEncoder
from .budget_calibration import BudgetCalibration, BudgetCalibrator, MonotonicBinnedCalibrator
from .composition import CompositionOutput, compose_one_query_updates, normalized_composition_distance
from .decisions import RefreshDecision, RefreshStats
from .defect import DefectValidity, evaluate_defect_validity, normalized_latent_defect
from .transition import TransitionConfig, TransitionOutput, VariableTimeTransitionCore

__all__ = [
    "BudgetCalibration",
    "BudgetCalibrator",
    "CompositionOutput",
    "DefectValidity",
    "ExecutedActionEncoder",
    "MonotonicBinnedCalibrator",
    "RefreshDecision",
    "RefreshStats",
    "TransitionConfig",
    "TransitionOutput",
    "VariableTimeTransitionCore",
    "compose_one_query_updates",
    "evaluate_defect_validity",
    "normalized_composition_distance",
    "normalized_latent_defect",
]
