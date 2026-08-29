"""Schedulers deciding when to refresh the full VLA condition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkipDecision:
    step: int
    refresh: bool
    age: int
    reason: str


class PeriodicSkipScheduler:
    """Refresh every ``refresh_every`` policy steps."""

    def __init__(self, refresh_every: int) -> None:
        if refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
        self.refresh_every = int(refresh_every)
        self._last_refresh_step = -1

    def decision(self, step: int) -> SkipDecision:
        if step < 0:
            raise ValueError("step must be non-negative")
        refresh = step == 0 or step % self.refresh_every == 0
        if refresh:
            age = 0
            self._last_refresh_step = step
            reason = "periodic_refresh"
        else:
            age = step - self._last_refresh_step
            reason = "dcld_update"
        return SkipDecision(step=step, refresh=refresh, age=age, reason=reason)


class QueryReductionScheduler:
    """Approximate query-reduction scheduler over a fixed interval.

    ``query_reduction=0.2`` means roughly 20 percent fewer full refreshes than
    refreshing every step.
    """

    def __init__(self, query_reduction: float) -> None:
        if query_reduction < 0.0 or query_reduction >= 1.0:
            raise ValueError("query_reduction must be in [0, 1)")
        self.query_reduction = float(query_reduction)
        self.keep_fraction = 1.0 - self.query_reduction
        self._refresh_count = 0

    def decision(self, step: int) -> SkipDecision:
        if step < 0:
            raise ValueError("step must be non-negative")
        desired_refreshes = int((step + 1) * self.keep_fraction + 1e-9)
        refresh = step == 0 or self._refresh_count < desired_refreshes
        if refresh:
            self._refresh_count += 1
            return SkipDecision(step=step, refresh=True, age=0, reason="qred_refresh")
        return SkipDecision(step=step, refresh=False, age=1, reason="qred_dcld_update")
