"""Small, architecture-neutral LatentLoop modules."""

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
    "ChunkAwareConditionOutput",
    "ChunkAwareConditionUpdater",
    "ExecutedActionEncoding",
    "ExecutedActionEncoder",
    "MatchedActionChunkCorrection",
    "NonRecurrentConditionOutput",
    "NonRecurrentConditionPredictor",
    "ObservationChangeEncoder",
    "ObservationPair",
    "PaddedExecutedActions",
    "QueryContextFusion",
    "ShiftedActionChunk",
    "count_trainable_parameters",
    "find_action_correction_hidden_dim",
    "pad_executed_actions",
    "shift_action_chunk",
]
