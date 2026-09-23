"""Fixed nested schedules for full, regenerated, and corrected action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ExecutionLevel(IntEnum):
    """Mutually exclusive action-production routes at a policy query."""

    ACTION_CORRECTION = 0
    CONDITION_REGENERATION = 1
    FULL_REFRESH = 2

    @property
    def source(self) -> str:
        return {
            self.ACTION_CORRECTION: "action_correction",
            self.CONDITION_REGENERATION: "condition_regeneration",
            self.FULL_REFRESH: "full_refresh",
        }[self]


@dataclass(frozen=True)
class HierarchicalSchedule:
    """Periodic full refreshes with nested action-transformer regeneration."""

    full_refresh_interval: int
    action_regeneration_interval: int
    execution_horizon: int = 1

    def __post_init__(self) -> None:
        k_f = int(self.full_refresh_interval)
        k_g = int(self.action_regeneration_interval)
        if k_f < 1:
            raise ValueError("K_F must be positive")
        if k_g < 1 or k_g > k_f:
            raise ValueError("K_G must satisfy 1 <= K_G <= K_F")
        if k_f % k_g:
            raise ValueError("K_G must divide K_F for a nested fixed schedule")
        if int(self.execution_horizon) not in {1, 2, 5}:
            raise ValueError("execution_horizon must be 1, 2, or 5")

    @property
    def is_pure_condition_endpoint(self) -> bool:
        return self.action_regeneration_interval == 1

    @property
    def is_pure_action_endpoint(self) -> bool:
        return self.action_regeneration_interval == self.full_refresh_interval

    def level(self, policy_query_index: int) -> ExecutionLevel:
        """Return Level 2, 1, or 0 for a zero-based policy query."""

        index = int(policy_query_index)
        if index < 0:
            raise ValueError("policy_query_index must be nonnegative")
        if index % self.full_refresh_interval == 0:
            return ExecutionLevel.FULL_REFRESH
        if index % self.action_regeneration_interval == 0:
            return ExecutionLevel.CONDITION_REGENERATION
        return ExecutionLevel.ACTION_CORRECTION

    def query_age(self, policy_query_index: int) -> int:
        """Return lightweight-query age since the latest full refresh."""

        index = int(policy_query_index)
        if index < 0:
            raise ValueError("policy_query_index must be nonnegative")
        return index % self.full_refresh_interval

    def last_full_refresh(self, policy_query_index: int) -> int:
        index = int(policy_query_index)
        return index - self.query_age(index)

    def levels(self, count: int, *, start: int = 0) -> tuple[ExecutionLevel, ...]:
        if count < 0 or start < 0:
            raise ValueError("count and start must be nonnegative")
        return tuple(self.level(index) for index in range(start, start + count))

    def expected_calls(self, query_count: int) -> dict[str, int]:
        """Return exact route call counts for the first ``query_count`` queries."""

        levels = self.levels(query_count)
        full = levels.count(ExecutionLevel.FULL_REFRESH)
        regenerated = levels.count(ExecutionLevel.CONDITION_REGENERATION)
        corrected = levels.count(ExecutionLevel.ACTION_CORRECTION)
        return {
            "num_policy_queries": int(query_count),
            "num_level2_queries": full,
            "num_level1_queries": regenerated,
            "num_level0_queries": corrected,
            "num_full_vlm_calls": full,
            "num_condition_updater_calls": regenerated + corrected,
            "num_action_transformer_decodes": full + regenerated,
            "num_action_correction_calls": corrected,
        }
