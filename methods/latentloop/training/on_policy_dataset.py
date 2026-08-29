"""On-policy recursive-distillation record validation and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .query_cache_dataset import (
    QueryCacheDataset,
    iter_query_records,
    validate_query_cache,
    validate_query_record,
)


ON_POLICY_REQUIRED_KEYS = (
    "predicted_condition",
    "rollout_depth",
    "rollout_episode_id",
    "adapter_checkpoint",
)


def validate_on_policy_record(record: Mapping[str, Any]) -> list[str]:
    """Validate a query record plus the predicted-condition provenance."""

    errors = validate_query_record(record)
    errors.extend(f"missing key: {key}" for key in ON_POLICY_REQUIRED_KEYS if key not in record)
    if "rollout_depth" in record and int(record["rollout_depth"]) not in {1, 2, 3}:
        errors.append("rollout_depth must be 1, 2, or 3")
    if "predicted_condition" in record and "full_condition" in record:
        if tuple(record["predicted_condition"].shape) != tuple(record["full_condition"].shape):
            errors.append("predicted_condition shape does not match full_condition")
    return errors


class OnPolicyRecordDataset(QueryCacheDataset):
    """Query-cache dataset that enforces recursive rollout fields on read."""

    def __init__(self, cache_dir: str | Path, *, maximum_rollout_depth: int) -> None:
        super().__init__(cache_dir)
        self.maximum_rollout_depth = int(maximum_rollout_depth)
        if self.maximum_rollout_depth not in {1, 2, 3}:
            raise ValueError("maximum_rollout_depth must be 1, 2, or 3")

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = super().__getitem__(index)
        errors = validate_on_policy_record(record)
        if errors:
            raise ValueError("Invalid on-policy record: " + "; ".join(errors))
        if int(record["rollout_depth"]) > self.maximum_rollout_depth:
            raise ValueError("record rollout_depth exceeds the configured gate")
        return record


def validate_on_policy_cache(
    cache_dir: str | Path,
    *,
    maximum_rollout_depth: int,
) -> dict[str, Any]:
    """Validate base query semantics and every T2 predicted-condition field."""

    base = validate_query_cache(cache_dir)
    errors = list(base["errors"])
    depths: dict[int, int] = {}
    for record in iter_query_records(cache_dir):
        record_errors = validate_on_policy_record(record)
        depth = int(record.get("rollout_depth", -1))
        depths[depth] = depths.get(depth, 0) + 1
        if depth > maximum_rollout_depth:
            record_errors.append(
                f"rollout_depth={depth} exceeds maximum={maximum_rollout_depth}"
            )
        errors.extend(
            f"task={record.get('task_id')} episode={record.get('episode_id')} "
            f"query={record.get('query_index')}: {error}"
            for error in record_errors
        )
    return {
        **base,
        "cache_kind": "on_policy_recursive_distillation",
        "maximum_rollout_depth": int(maximum_rollout_depth),
        "rollout_depth_counts": depths,
        "errors": errors,
        "passed": not errors,
    }
