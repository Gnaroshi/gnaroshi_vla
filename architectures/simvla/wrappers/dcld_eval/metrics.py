"""Metrics and counters for SimVLA DCLD evaluation.

Purpose:
    Document the counters and latency fields needed for efficiency claims.

Inputs/outputs:
    The runtime metrics class and helper functions are re-exported from
    `rollout_runner` during this behavior-preserving refactor.

Official-match scope:
    Full VLM call counts and policy-query counts are required to distinguish
    official-style full baselines from DCLD-reduced rows.

DCLD scope:
    DCLD efficiency claims rely on FastEncoder, DCLD update, action transformer,
    and full VLM counters.

Caveat:
    HZUP20Q records intended control Hz in metrics but does not alter MuJoCo
    control timing.
"""

from __future__ import annotations

from .rollout_runner import LATENCY_FIELDS, REQUIRED_COUNTERS, RealEvalMetrics, latency_stats, l2_stats, miss_rate

__all__ = ["LATENCY_FIELDS", "REQUIRED_COUNTERS", "RealEvalMetrics", "latency_stats", "l2_stats", "miss_rate"]
