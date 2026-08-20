"""Refresh decisions and accounting shared by online runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class RefreshDecision(IntEnum):
    SEQUENTIAL = 0
    DIRECT_REANCHOR = 1
    FULL_PREFIX = 2


@dataclass
class RefreshStats:
    total_queries: int = 0
    sequential_calls: int = 0
    direct_reanchors: int = 0
    full_prefix_calls: int = 0
    executed_actions: int = 0

    def record(self, decision: RefreshDecision, executed_actions: int) -> None:
        self.total_queries += 1
        self.executed_actions += int(executed_actions)
        if decision is RefreshDecision.SEQUENTIAL:
            self.sequential_calls += 1
        elif decision is RefreshDecision.DIRECT_REANCHOR:
            self.direct_reanchors += 1
        elif decision is RefreshDecision.FULL_PREFIX:
            self.full_prefix_calls += 1
        else:
            raise ValueError(f"unknown refresh decision: {decision}")

    @property
    def full_prefix_ratio(self) -> float:
        return self.full_prefix_calls / self.total_queries if self.total_queries else 0.0

    @property
    def mean_k_q(self) -> float:
        return self.total_queries / self.full_prefix_calls if self.full_prefix_calls else float("inf")

    @property
    def mean_k_a(self) -> float:
        return self.executed_actions / self.full_prefix_calls if self.full_prefix_calls else float("inf")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_queries": self.total_queries,
            "sequential_calls": self.sequential_calls,
            "direct_reanchors": self.direct_reanchors,
            "full_prefix_calls": self.full_prefix_calls,
            "executed_actions": self.executed_actions,
            "full_prefix_ratio": self.full_prefix_ratio,
            "mean_k_q": self.mean_k_q,
            "mean_k_a": self.mean_k_a,
        }
