"""Small, architecture-neutral LatentLoop modules."""

from .action_equivalent_refresh import (
    ActionEquivalentRefreshRouter,
    ActionFidelityHead,
    ActionFidelityPrediction,
    CounterfactualActionTargets,
    ExactCallBudgetCalibration,
    RefreshDecision,
    action_fidelity_loss,
    counterfactual_action_targets,
    fit_exact_call_budget_calibration,
    simulate_exact_fraction,
)
from .action_chunk_correction import (
    ActionChunkCorrectionOutput,
    MatchedActionChunkCorrection,
    ShiftedActionChunk,
    find_action_correction_hidden_dim,
    shift_action_chunk,
)
from .executed_action_encoder import (
    ExecutedActionEncoding,
    ExecutedActionEncoder,
    PaddedExecutedActions,
    pad_executed_actions,
)
from .nonrecurrent_condition_predictor import (
    NonRecurrentConditionOutput,
    NonRecurrentConditionPredictor,
)
from .observation_encoder import ObservationChangeEncoder, ObservationPair
from .recurrent_condition_updater import (
    ChunkAwareConditionOutput,
    ChunkAwareConditionUpdater,
    QueryContextFusion,
    count_trainable_parameters,
)

__all__ = [
    "ActionChunkCorrectionOutput",
    "ActionEquivalentRefreshRouter",
    "ActionFidelityHead",
    "ActionFidelityPrediction",
    "ChunkAwareConditionOutput",
    "ChunkAwareConditionUpdater",
    "CounterfactualActionTargets",
    "ExactCallBudgetCalibration",
    "ExecutedActionEncoding",
    "ExecutedActionEncoder",
    "MatchedActionChunkCorrection",
    "NonRecurrentConditionOutput",
    "NonRecurrentConditionPredictor",
    "ObservationChangeEncoder",
    "ObservationPair",
    "PaddedExecutedActions",
    "QueryContextFusion",
    "RefreshDecision",
    "ShiftedActionChunk",
    "action_fidelity_loss",
    "counterfactual_action_targets",
    "count_trainable_parameters",
    "find_action_correction_hidden_dim",
    "fit_exact_call_budget_calibration",
    "pad_executed_actions",
    "shift_action_chunk",
    "simulate_exact_fraction",
]
