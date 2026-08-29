"""Three-level V2 state machine with forced sequential/full safety ages."""

from __future__ import annotations

from dataclasses import dataclass

from methods.variable_time_latentloop.decisions import RefreshDecision


@dataclass
class V2ScheduleState:
    m_seq: int
    m_full: int
    last_full_query: int | None = None
    last_sequential_anchor_query: int | None = None
    policy_queries: int = 0
    executed_actions: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.m_seq <= 3:
            raise ValueError("M_seq must be in [1,3] for the K_q=4 transition family")
        if not 2 <= self.m_full <= 4:
            raise ValueError("M_full must be in [2,4] so direct delta_q never exceeds three")

    def reset(self) -> None:
        self.last_full_query = None
        self.last_sequential_anchor_query = None
        self.policy_queries = 0
        self.executed_actions = 0

    def forced_decision(self, query_index: int) -> RefreshDecision | None:
        if self.last_full_query is None or query_index == 0:
            return RefreshDecision.FULL_PREFIX
        full_age = int(query_index) - self.last_full_query
        if full_age >= self.m_full:
            return RefreshDecision.FULL_PREFIX
        sequential_anchor = (
            self.last_sequential_anchor_query
            if self.last_sequential_anchor_query is not None
            else self.last_full_query
        )
        if int(query_index) - sequential_anchor >= self.m_seq:
            return RefreshDecision.DIRECT_REANCHOR
        return None

    def commit(self, decision: RefreshDecision, query_index: int, executed_actions: int) -> None:
        self.policy_queries += 1
        self.executed_actions += int(executed_actions)
        if decision is RefreshDecision.FULL_PREFIX:
            self.last_full_query = int(query_index)
            self.last_sequential_anchor_query = int(query_index)
        elif decision is RefreshDecision.DIRECT_REANCHOR:
            self.last_sequential_anchor_query = int(query_index)

    @property
    def full_refresh_age(self) -> int:
        if self.last_full_query is None:
            return 0
        return max(0, self.policy_queries - 1 - self.last_full_query)

    @property
    def sequential_age(self) -> int:
        anchor = self.last_sequential_anchor_query
        if anchor is None:
            return 0
        return max(0, self.policy_queries - 1 - anchor)
