"""Execution-horizon and full-condition-query schedules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionProtocol(str, Enum):
    """Predeclared SimVLA execution protocols."""

    ENVIRONMENT_STEP = "A_R1"
    NATIVE_CHUNK = "B_R5"
    MICRO_CHUNK = "C_R2"

    @property
    def execution_horizon(self) -> int:
        """Number of action tokens executed between policy queries."""

        return {
            self.ENVIRONMENT_STEP: 1,
            self.NATIVE_CHUNK: 5,
            self.MICRO_CHUNK: 2,
        }[self]


def environment_action_gap(full_query_interval: int, execution_horizon: int) -> int:
    """Return ``G = K * R`` with strict positive-input validation."""

    if int(full_query_interval) < 1 or int(execution_horizon) < 1:
        raise ValueError("K and R must be positive")
    return int(full_query_interval) * int(execution_horizon)


@dataclass(frozen=True)
class QuerySchedule:
    """Periodic full-condition schedule measured in policy queries."""

    full_query_interval: int
    execution_horizon: int

    def __post_init__(self) -> None:
        if self.full_query_interval < 1:
            raise ValueError("full_query_interval must be positive")
        if self.execution_horizon not in {1, 2, 5}:
            raise ValueError("execution_horizon must be 1, 2, or 5")

    @property
    def environment_action_gap(self) -> int:
        """Environment actions between periodic full-condition queries."""

        return environment_action_gap(self.full_query_interval, self.execution_horizon)

    def is_full_query(self, policy_query_index: int) -> bool:
        """Return true for the initial query and each K-th query thereafter."""

        if policy_query_index < 0:
            raise ValueError("policy_query_index must be nonnegative")
        return policy_query_index % self.full_query_interval == 0

    def query_age(self, policy_query_index: int) -> int:
        """Lightweight updates since the most recent full refresh."""

        if policy_query_index < 0:
            raise ValueError("policy_query_index must be nonnegative")
        return policy_query_index % self.full_query_interval
