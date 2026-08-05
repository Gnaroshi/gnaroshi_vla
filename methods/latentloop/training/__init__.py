"""Cache, loss, and on-policy training utilities for LatentLoop."""

from .losses import (
    LatentLoopLossWeights,
    LossScaleAccumulator,
    compute_t1_losses,
    normalized_condition_mse,
)
from .on_policy_dataset import (
    OnPolicyRecordDataset,
    validate_on_policy_cache,
    validate_on_policy_record,
)
from .query_cache_dataset import (
    QUERY_CACHE_SCHEMA_VERSION,
    QueryCacheDataset,
    QueryCacheShardWriter,
    collate_query_records,
    deterministic_episode_split_indices,
    merge_query_cache_parts,
    validate_query_cache,
    validate_query_record,
)
from .sampling import DeterministicStepBatchSampler

__all__ = [
    "LatentLoopLossWeights",
    "LossScaleAccumulator",
    "OnPolicyRecordDataset",
    "QUERY_CACHE_SCHEMA_VERSION",
    "QueryCacheDataset",
    "QueryCacheShardWriter",
    "DeterministicStepBatchSampler",
    "collate_query_records",
    "deterministic_episode_split_indices",
    "merge_query_cache_parts",
    "compute_t1_losses",
    "normalized_condition_mse",
    "validate_on_policy_record",
    "validate_on_policy_cache",
    "validate_query_cache",
    "validate_query_record",
]
