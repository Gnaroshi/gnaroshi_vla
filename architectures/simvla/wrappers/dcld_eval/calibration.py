"""Calibration diagnostics for SimVLA DCLD wrapper baselines.

Purpose:
    Keep K=1 parity and baseline calibration helpers discoverable apart from
    rollout execution.

Inputs/outputs:
    Exports helpers that write QRED20 K=1 plans and action-diff reports.

Official-match scope:
    Official-vs-wrapper saved-batch diagnostics must pass before claiming the
    wrapper action path matches official SimVLA at policy-query boundaries.

DCLD scope:
    K-sweep results should be interpreted only after K=1 full path calibration
    is understood.

Caveat:
    Benchmark-level success rates may still differ because rollout stochasticity
    and evaluation sample size are not removed by saved-batch parity.
"""

from __future__ import annotations

from .rollout_runner import qred20_k1_equivalence_plan, write_k1_action_diff_report, write_qred20_k1_equivalence_plan

__all__ = ["qred20_k1_equivalence_plan", "write_k1_action_diff_report", "write_qred20_k1_equivalence_plan"]
