"""Statistics for LatentLoop segment-grid evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Return a two-sided Wilson score interval for a Bernoulli proportion."""

    n = int(total)
    k = int(successes)
    if n <= 0:
        return (0.0, 0.0)
    if k < 0 or k > n:
        raise ValueError(f"successes must be in [0, total], got {k}/{n}")
    proportion = float(k) / float(n)
    denominator = 1.0 + (z * z) / n
    center = (proportion + (z * z) / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / n
            + (z * z) / (4.0 * n * n)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def distribution_profile(values: Iterable[float]) -> Dict[str, float]:
    """Return count, mean, and p50/p95/p99 for finite scalar values."""

    finite_values = []
    for value in values:
        scalar = float(value)
        if np.isfinite(scalar):
            finite_values.append(scalar)
    array = np.asarray(finite_values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def paired_flip_counts(
    baseline: Mapping[Tuple[object, ...], int],
    candidate: Mapping[Tuple[object, ...], int],
) -> Dict[str, object]:
    """Count paired success/failure transitions over matching episode keys."""

    common = sorted(set(baseline).intersection(candidate), key=str)
    counts: Dict[str, object] = {
        "paired_count": len(common),
        "both_success": 0,
        "both_failure": 0,
        "baseline_success_candidate_failure": 0,
        "baseline_failure_candidate_success": 0,
    }
    for key in common:
        base = int(baseline[key])
        cand = int(candidate[key])
        if base not in (0, 1) or cand not in (0, 1):
            raise ValueError(f"Paired outcomes must be binary for key={key}")
        if base == 1 and cand == 1:
            counts["both_success"] = int(counts["both_success"]) + 1
        elif base == 0 and cand == 0:
            counts["both_failure"] = int(counts["both_failure"]) + 1
        elif base == 1:
            counts["baseline_success_candidate_failure"] = (
                int(counts["baseline_success_candidate_failure"]) + 1
            )
        else:
            counts["baseline_failure_candidate_success"] = (
                int(counts["baseline_failure_candidate_success"]) + 1
            )
    counts["net_success_gain"] = (
        int(counts["baseline_failure_candidate_success"])
        - int(counts["baseline_success_candidate_failure"])
    )
    counts["missing_from_candidate"] = len(set(baseline) - set(candidate))
    counts["missing_from_baseline"] = len(set(candidate) - set(baseline))
    return counts


def paired_hierarchical_bootstrap_interval(
    pairs: Sequence[Mapping[str, object]],
    *,
    iterations: int = 10_000,
    seed: int = 20260729,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap paired SR differences by resampling tasks and episodes.

    Each row must contain ``task_id``, ``baseline_success``, and
    ``candidate_success``. Tasks are sampled with replacement, then paired
    episodes are sampled with replacement inside each sampled task.
    """

    grouped: Dict[object, List[float]] = defaultdict(list)
    for row in pairs:
        grouped[row["task_id"]].append(
            float(row["candidate_success"]) - float(row["baseline_success"])
        )
    task_ids = list(grouped)
    if not task_ids:
        return {
            "paired_count": 0,
            "mean_difference": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "iterations": int(iterations),
        }
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sampled_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
        sampled_differences: List[float] = []
        for task_id in sampled_tasks:
            task_values = np.asarray(grouped[task_id], dtype=np.float64)
            sampled_differences.extend(
                rng.choice(task_values, size=task_values.size, replace=True).tolist()
            )
        estimates[index] = float(np.mean(sampled_differences))
    observed = np.concatenate(
        [np.asarray(grouped[task_id], dtype=np.float64) for task_id in task_ids]
    )
    return {
        "paired_count": int(observed.size),
        "mean_difference": float(observed.mean()),
        "ci_low": float(np.percentile(estimates, 100.0 * alpha / 2.0)),
        "ci_high": float(np.percentile(estimates, 100.0 * (1.0 - alpha / 2.0))),
        "iterations": int(iterations),
    }


def episode_key(row: Mapping[str, object]) -> Tuple[object, ...]:
    """Return the canonical within-checkpoint pairing key for an episode row."""

    return (
        row.get("task_id"),
        row.get("episode_id"),
        row.get("seed"),
    )
