"""Lightweight recursive query-boundary cache state for SimVLA LatentLoop."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


def tensor_hash(tensor: Tensor) -> str:
    """Hash tensor dtype, shape, and bytes for rollout trace parity."""

    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SimVLAQueryObservation:
    """Raw two-view RGB and 8D proprio at one policy-query boundary."""

    raw_rgb: Tensor
    proprio: Tensor

    def clone_detached(self) -> "SimVLAQueryObservation":
        """Return cache-owned tensors that cannot alias a caller buffer."""

        return SimVLAQueryObservation(
            raw_rgb=self.raw_rgb.detach().clone(),
            proprio=self.proprio.detach().clone(),
        )

    def hashes(self) -> dict[str, str]:
        """Hash trace fields used to prove cache progression."""

        return {"raw_rgb": tensor_hash(self.raw_rgb), "proprio": tensor_hash(self.proprio)}


@dataclass
class RecursiveQueryCache:
    """Recursive condition cache with explicit query-boundary transitions."""

    condition: Tensor | None = None
    previous_query_observation: SimVLAQueryObservation | None = None
    full_anchor_condition: Tensor | None = None
    full_anchor_observation: SimVLAQueryObservation | None = None
    last_executed_subchunk: Tensor | None = None
    executed_subchunks_since_anchor: list[Tensor] = field(default_factory=list)
    query_age: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        """Clear every episode-local condition, observation, and action cache."""

        self.condition = None
        self.previous_query_observation = None
        self.full_anchor_condition = None
        self.full_anchor_observation = None
        self.last_executed_subchunk = None
        self.executed_subchunks_since_anchor = []
        self.query_age = 0
        self.trace = []

    def full_refresh(self, condition: Tensor, observation: SimVLAQueryObservation) -> None:
        """Start a segment from a full SimVLA condition at the current query."""

        cached_observation = observation.clone_detached()
        self.condition = condition.detach().clone()
        self.previous_query_observation = cached_observation
        self.full_anchor_condition = condition.detach().clone()
        self.full_anchor_observation = cached_observation.clone_detached()
        self.last_executed_subchunk = None
        self.executed_subchunks_since_anchor = []
        self.query_age = 0
        self.trace.append(
            {
                "event": "full_refresh",
                "query_age": 0,
                "observation_hashes": cached_observation.hashes(),
            }
        )

    def record_executed_subchunk(self, actions_sent_to_env: Tensor) -> None:
        """Record only final postprocessed actions actually sent after a query."""

        if self.condition is None:
            raise RuntimeError("cannot record executed actions before a query")
        if actions_sent_to_env.ndim != 3 or actions_sent_to_env.shape[-1] != 7:
            raise ValueError("actions_sent_to_env must be [B,R,7]")
        executed = actions_sent_to_env.detach().clone()
        self.last_executed_subchunk = executed
        self.executed_subchunks_since_anchor.append(executed)
        self.trace.append(
            {
                "event": "executed_subchunk",
                "query_age": self.query_age,
                "shape": list(executed.shape),
                "hash": tensor_hash(executed),
            }
        )

    def lightweight_transition_inputs(
        self,
        current_observation: SimVLAQueryObservation,
    ) -> dict[str, Any]:
        """Return adjacent recursive and fixed-anchor inputs for the next update."""

        if (
            self.condition is None
            or self.previous_query_observation is None
            or self.full_anchor_condition is None
            or self.full_anchor_observation is None
            or self.last_executed_subchunk is None
        ):
            raise RuntimeError("lightweight transition requires full cache plus executed subchunk")
        return {
            "previous_condition": self.condition,
            "previous_query_observation": self.previous_query_observation,
            "current_query_observation": current_observation,
            "executed_subchunk": self.last_executed_subchunk,
            "anchor_condition": self.full_anchor_condition,
            "anchor_observation": self.full_anchor_observation,
            "executed_subchunks_since_anchor": tuple(self.executed_subchunks_since_anchor),
            "next_query_age": self.query_age + 1,
        }

    def commit_lightweight_update(
        self,
        updated_condition: Tensor,
        current_observation: SimVLAQueryObservation,
    ) -> None:
        """Advance both condition and observation caches after every lightweight query."""

        if self.condition is None or self.previous_query_observation is None:
            raise RuntimeError("cannot commit a lightweight update before full refresh")
        previous_hashes = self.previous_query_observation.hashes()
        current_cached = current_observation.clone_detached()
        self.condition = updated_condition.detach().clone()
        self.previous_query_observation = current_cached
        self.last_executed_subchunk = None
        self.query_age += 1
        self.trace.append(
            {
                "event": "lightweight_commit",
                "query_age": self.query_age,
                "previous_observation_hashes": previous_hashes,
                "current_observation_hashes": current_cached.hashes(),
                "observation_cache_advanced": previous_hashes != current_cached.hashes(),
            }
        )

    def assert_ready_for_next_lightweight_update(self) -> None:
        """Assert the previous query's execution has been recorded exactly once."""

        if self.last_executed_subchunk is None:
            raise AssertionError("missing executed subchunk from the previous policy query")
