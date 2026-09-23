"""Native-R5 exact-q2 condition-regeneration method contracts."""

from .dataset import (
    EXACT_Q2_DATASET_SCHEMA,
    ExactQ2TupleDataset,
    build_exact_q2_index,
    collate_exact_q2,
    validate_exact_q2_pair,
)
from .decisions import evaluate_exact_q2_offline_gate, evaluate_short_budget_gate
from .direct_model import DirectExactQ2
from .losses import ExactQ2LossWeights, compute_exact_q2_losses
from .recurrent_model import ExactQ2ModelConfig, RecurrentExactQ2
from .validation import select_validation_checkpoint

__all__ = [
    "EXACT_Q2_DATASET_SCHEMA",
    "DirectExactQ2",
    "ExactQ2LossWeights",
    "ExactQ2ModelConfig",
    "ExactQ2TupleDataset",
    "RecurrentExactQ2",
    "build_exact_q2_index",
    "collate_exact_q2",
    "compute_exact_q2_losses",
    "evaluate_exact_q2_offline_gate",
    "evaluate_short_budget_gate",
    "select_validation_checkpoint",
    "validate_exact_q2_pair",
]
