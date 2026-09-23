"""Serializable cache metadata and invariant checks for the three-level policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schedules import ExecutionLevel, HierarchicalSchedule


HYBRID_TRACE_SCHEMA_VERSION = "simvla_hierarchical_correction_trace_v1"
REQUIRED_TRACE_FIELDS = (
    "policy_query_index",
    "execution_level",
    "execution_level_name",
    "query_age",
    "condition_cache_source",
    "action_chunk_cache_source",
    "executed_action_source",
    "last_full_refresh_query",
    "last_action_regeneration_query",
    "condition_cache_revision",
    "action_chunk_cache_revision",
    "condition_cache_hash",
    "action_chunk_cache_hash",
    "previous_action_chunk_cache_hash",
    "action_correction_input_chunk_hash",
    "action_correction_residual",
    "condition_cache_drift",
    "flow_noise_hash",
    "full_condition_called",
    "condition_updater_called",
    "action_transformer_called",
    "action_correction_called",
)


def validate_trace_record(record: dict[str, Any]) -> list[str]:
    """Return schema and mutually-exclusive route violations for one query."""

    errors = [f"missing field: {name}" for name in REQUIRED_TRACE_FIELDS if name not in record]
    if errors:
        return errors
    level = ExecutionLevel(int(record["execution_level"]))
    calls = (
        bool(record["full_condition_called"]),
        bool(record["condition_updater_called"]),
        bool(record["action_transformer_called"]),
        bool(record["action_correction_called"]),
    )
    expected = {
        ExecutionLevel.FULL_REFRESH: (True, False, True, False),
        ExecutionLevel.CONDITION_REGENERATION: (False, True, True, False),
        ExecutionLevel.ACTION_CORRECTION: (False, True, False, True),
    }[level]
    if calls != expected:
        errors.append(f"Level {int(level)} call tuple {calls} != {expected}")
    if level is ExecutionLevel.ACTION_CORRECTION:
        if record["flow_noise_hash"] is not None:
            errors.append("Level 0 must not have flow noise")
        if record["action_correction_input_chunk_hash"] != record["previous_action_chunk_cache_hash"]:
            errors.append("Level 0 did not consume the latest action cache")
    else:
        if not record["flow_noise_hash"]:
            errors.append(f"Level {int(level)} requires a flow-noise hash")
    return errors


@dataclass
class HierarchicalPolicyState:
    """Track cache provenance without owning architecture-specific tensors."""

    schedule: HierarchicalSchedule
    next_query_index: int = 0
    initialized: bool = False
    condition_cache_source: str | None = None
    action_chunk_cache_source: str | None = None
    query_age: int = 0
    last_full_refresh_query: int | None = None
    last_action_regeneration_query: int | None = None
    condition_cache_revision: int = 0
    action_chunk_cache_revision: int = 0
    condition_cache_hash: str | None = None
    action_chunk_cache_hash: str | None = None
    counters: dict[str, int] = field(default_factory=dict)

    def _increment(self, name: str) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1

    def reset(self) -> None:
        self.next_query_index = 0
        self.initialized = False
        self.condition_cache_source = None
        self.action_chunk_cache_source = None
        self.query_age = 0
        self.last_full_refresh_query = None
        self.last_action_regeneration_query = None
        self.condition_cache_revision = 0
        self.action_chunk_cache_revision = 0
        self.condition_cache_hash = None
        self.action_chunk_cache_hash = None
        self.counters = {}

    def apply_query(
        self,
        *,
        policy_query_index: int,
        level: ExecutionLevel,
        condition_cache_hash: str,
        action_chunk_cache_hash: str,
        flow_noise_hash: str | None,
        action_correction_input_chunk_hash: str | None = None,
        action_correction_residual: dict[str, float] | None = None,
        condition_cache_drift: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Validate one transition, update metadata, and return its trace record."""

        index = int(policy_query_index)
        level = ExecutionLevel(level)
        if index != self.next_query_index:
            raise AssertionError(
                f"non-contiguous query index: expected {self.next_query_index}, got {index}"
            )
        expected = self.schedule.level(index)
        if level is not expected:
            raise AssertionError(f"schedule requires Level {int(expected)}, got {int(level)}")
        if not condition_cache_hash or not action_chunk_cache_hash:
            raise ValueError("condition and action cache hashes must be nonempty")

        previous_action_hash = self.action_chunk_cache_hash
        if level is ExecutionLevel.FULL_REFRESH:
            condition_source = "full_condition"
            action_source = "full_action_transformer"
            calls = (True, False, True, False)
            self.query_age = 0
            self.last_full_refresh_query = index
            self.last_action_regeneration_query = index
            if flow_noise_hash is None:
                raise AssertionError("Level 2 requires an explicit flow-noise hash")
        elif level is ExecutionLevel.CONDITION_REGENERATION:
            if not self.initialized:
                raise AssertionError("Level 1 cannot run before Level 2 initialization")
            condition_source = "recurrent_condition_update"
            action_source = "updated_condition_action_transformer"
            calls = (False, True, True, False)
            self.query_age += 1
            self.last_action_regeneration_query = index
            if flow_noise_hash is None:
                raise AssertionError("Level 1 requires an explicit flow-noise hash")
            if action_correction_input_chunk_hash is not None:
                raise AssertionError("Level 1 must replace, not correct, the action cache")
        else:
            if not self.initialized or previous_action_hash is None:
                raise AssertionError("Level 0 cannot run before Level 2 initialization")
            condition_source = "recurrent_condition_update"
            action_source = "shifted_action_chunk_correction"
            calls = (False, True, False, True)
            self.query_age += 1
            if flow_noise_hash is not None:
                raise AssertionError("Level 0 must not create flow noise")
            if action_correction_input_chunk_hash != previous_action_hash:
                raise AssertionError(
                    "Level 0 correction input is stale or differs from the current action cache"
                )

        expected_age = self.schedule.query_age(index)
        if self.query_age != expected_age:
            raise AssertionError(
                f"query age mismatch: state={self.query_age}, schedule={expected_age}"
            )
        full_called, updater_called, transformer_called, correction_called = calls
        self.initialized = True
        self.condition_cache_source = condition_source
        self.action_chunk_cache_source = action_source
        self.condition_cache_hash = condition_cache_hash
        self.action_chunk_cache_hash = action_chunk_cache_hash
        self.condition_cache_revision += 1
        self.action_chunk_cache_revision += 1
        self.next_query_index += 1
        self._increment("num_policy_queries")
        self._increment(f"num_level{int(level)}_queries")
        if full_called:
            self._increment("num_full_vlm_calls")
        if updater_called:
            self._increment("num_condition_updater_calls")
        if transformer_called:
            self._increment("num_action_transformer_decodes")
        if correction_called:
            self._increment("num_action_correction_calls")

        return {
            "trace_schema_version": HYBRID_TRACE_SCHEMA_VERSION,
            "policy_query_index": index,
            "execution_level": int(level),
            "execution_level_name": level.name,
            "query_age": self.query_age,
            "condition_cache_source": condition_source,
            "action_chunk_cache_source": action_source,
            "executed_action_source": action_source,
            "last_full_refresh_query": self.last_full_refresh_query,
            "last_action_regeneration_query": self.last_action_regeneration_query,
            "condition_cache_revision": self.condition_cache_revision,
            "action_chunk_cache_revision": self.action_chunk_cache_revision,
            "condition_cache_hash": condition_cache_hash,
            "action_chunk_cache_hash": action_chunk_cache_hash,
            "previous_action_chunk_cache_hash": previous_action_hash,
            "action_correction_input_chunk_hash": action_correction_input_chunk_hash,
            "action_correction_residual": action_correction_residual,
            "condition_cache_drift": condition_cache_drift,
            "flow_noise_hash": flow_noise_hash,
            "full_condition_called": full_called,
            "condition_updater_called": updater_called,
            "action_transformer_called": transformer_called,
            "action_correction_called": correction_called,
        }

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schedule"] = {
            "full_refresh_interval": self.schedule.full_refresh_interval,
            "action_regeneration_interval": self.schedule.action_regeneration_interval,
            "execution_horizon": self.schedule.execution_horizon,
        }
        return payload
