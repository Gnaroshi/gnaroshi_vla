"""V2 budget-calibrated error-controlled full-prefix refresh."""

from __future__ import annotations

from dataclasses import dataclass

from methods.variable_time_latentloop.budget_calibration import BudgetCalibration
from methods.variable_time_latentloop.decisions import RefreshDecision, RefreshStats

from .v2_schedule_state import V2ScheduleState


@dataclass(frozen=True)
class DynamicDecision:
    decision: RefreshDecision
    defect_score: float
    predicted_error: float


class BudgetedDynamicPolicy:
    """Frozen threshold policy; fitting belongs only to a disjoint calibration set."""

    def __init__(
        self,
        calibration: BudgetCalibration,
        execution_horizon: int = 5,
        *,
        m_seq: int,
        m_full: int,
    ) -> None:
        self.calibration = calibration
        self.execution_horizon = int(execution_horizon)
        self.stats = RefreshStats()
        self.full_query_indices: list[int] = []
        self.schedule = V2ScheduleState(m_seq=m_seq, m_full=m_full)

    def reset(self) -> None:
        self.stats = RefreshStats()
        self.full_query_indices = []
        self.schedule.reset()

    def forced_decision(self, query_index: int) -> RefreshDecision | None:
        return self.schedule.forced_decision(query_index)

    def decide(
        self,
        defect_score: float,
        query_index: int,
        *,
        forced: RefreshDecision | None = None,
        executed_actions: int | None = None,
    ) -> DynamicDecision:
        predicted = float(self.calibration.calibrator.predict([defect_score])[0])
        decision = forced if forced is not None else self.calibration.decide(defect_score)
        actual_actions = self.execution_horizon if executed_actions is None else int(executed_actions)
        self.stats.record(decision, actual_actions)
        self.schedule.commit(decision, query_index, actual_actions)
        if decision is RefreshDecision.FULL_PREFIX:
            self.full_query_indices.append(int(query_index))
        return DynamicDecision(decision=decision, defect_score=float(defect_score), predicted_error=predicted)

    def record_full(self, query_index: int, *, executed_actions: int | None = None) -> None:
        actual_actions = self.execution_horizon if executed_actions is None else int(executed_actions)
        self.stats.record(RefreshDecision.FULL_PREFIX, actual_actions)
        self.schedule.commit(RefreshDecision.FULL_PREFIX, query_index, actual_actions)
        self.full_query_indices.append(int(query_index))

    def interval_quantiles(self) -> dict[str, float]:
        if len(self.full_query_indices) < 2:
            return {"p10_k_q": float("nan"), "p50_k_q": float("nan"), "p90_k_q": float("nan")}
        import numpy as np

        intervals = np.diff(self.full_query_indices)
        return {
            "p10_k_q": float(np.quantile(intervals, 0.10)),
            "p50_k_q": float(np.quantile(intervals, 0.50)),
            "p90_k_q": float(np.quantile(intervals, 0.90)),
        }
