"""Tensor-owning cache for the SimVLA hierarchical correction adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from architectures.simvla.adapters.latentloop.query_cache_state import (
    RecursiveQueryCache,
    SimVLAQueryObservation,
    tensor_hash,
)
from methods.hierarchical_correction.schedules import ExecutionLevel


@dataclass
class SimVLAHybridCache:
    """Keep recurrent condition and corrected/regenerated action caches synchronized."""

    condition: RecursiveQueryCache = field(default_factory=RecursiveQueryCache)
    action_chunk: Tensor | None = None
    action_chunk_source: str | None = None
    action_chunk_revision: int = 0
    last_action_regeneration_query: int | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.condition.reset()
        self.action_chunk = None
        self.action_chunk_source = None
        self.action_chunk_revision = 0
        self.last_action_regeneration_query = None
        self.trace = []

    def full_refresh(
        self,
        condition: Tensor,
        action_chunk: Tensor,
        observation: SimVLAQueryObservation,
        *,
        policy_query_index: int,
    ) -> None:
        self.condition.full_refresh(condition, observation)
        self.action_chunk = action_chunk.detach().clone()
        self.action_chunk_source = "full_action_transformer"
        self.action_chunk_revision += 1
        self.last_action_regeneration_query = int(policy_query_index)
        self.trace.append(
            {
                "event": "level2_full_refresh",
                "policy_query_index": int(policy_query_index),
                "condition_hash": tensor_hash(condition),
                "action_chunk_hash": tensor_hash(action_chunk),
                "action_chunk_revision": self.action_chunk_revision,
            }
        )

    def record_executed_subchunk(self, actions_sent_to_env: Tensor) -> None:
        self.condition.record_executed_subchunk(actions_sent_to_env)

    def lightweight_inputs(
        self,
        current_observation: SimVLAQueryObservation,
    ) -> dict[str, Any]:
        if self.action_chunk is None:
            raise RuntimeError("lightweight transition requires an action-chunk cache")
        return {
            **self.condition.lightweight_transition_inputs(current_observation),
            "previous_action_chunk": self.action_chunk,
            "previous_action_chunk_hash": tensor_hash(self.action_chunk),
            "previous_action_chunk_source": self.action_chunk_source,
            "previous_action_chunk_revision": self.action_chunk_revision,
            "last_action_regeneration_query": self.last_action_regeneration_query,
        }

    def commit_lightweight(
        self,
        *,
        level: ExecutionLevel,
        condition: Tensor,
        action_chunk: Tensor,
        observation: SimVLAQueryObservation,
        policy_query_index: int,
    ) -> dict[str, Any]:
        if level not in {
            ExecutionLevel.ACTION_CORRECTION,
            ExecutionLevel.CONDITION_REGENERATION,
        }:
            raise ValueError("commit_lightweight accepts only Level 0 or Level 1")
        previous_action_hash = tensor_hash(self.action_chunk) if self.action_chunk is not None else None
        self.condition.commit_lightweight_update(condition, observation)
        self.action_chunk = action_chunk.detach().clone()
        self.action_chunk_source = (
            "updated_condition_action_transformer"
            if level is ExecutionLevel.CONDITION_REGENERATION
            else "shifted_action_chunk_correction"
        )
        self.action_chunk_revision += 1
        if level is ExecutionLevel.CONDITION_REGENERATION:
            self.last_action_regeneration_query = int(policy_query_index)
        event = {
            "event": f"level{int(level)}_lightweight_commit",
            "policy_query_index": int(policy_query_index),
            "query_age": self.condition.query_age,
            "condition_hash": tensor_hash(condition),
            "previous_action_chunk_hash": previous_action_hash,
            "action_chunk_hash": tensor_hash(action_chunk),
            "action_chunk_source": self.action_chunk_source,
            "action_chunk_revision": self.action_chunk_revision,
            "last_action_regeneration_query": self.last_action_regeneration_query,
            "condition_cache_advanced": True,
            "action_cache_replaced": True,
        }
        self.trace.append(event)
        return event

    @property
    def cached_condition(self) -> Tensor | None:
        return self.condition.condition
