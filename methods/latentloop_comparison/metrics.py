"""CPU-only offline and paired online comparison metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Return the required mean and tail percentiles for finite values."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    quantiles = np.quantile(array, [0.50, 0.90, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(array.max()),
    }


def paired_outcome_counts(
    left: Mapping[tuple[int, ...], int],
    right: Mapping[tuple[int, ...], int],
) -> dict[str, int]:
    """Count paired success/failure transitions on identical episode keys."""

    if set(left) != set(right):
        raise RuntimeError("Paired outcomes require identical episode keys")
    counts = {
        "both_success": 0,
        "both_failure": 0,
        "right_fail_to_left_success": 0,
        "right_success_to_left_fail": 0,
    }
    for key in sorted(left):
        pair = (int(left[key]), int(right[key]))
        if pair == (1, 1):
            counts["both_success"] += 1
        elif pair == (0, 0):
            counts["both_failure"] += 1
        elif pair == (1, 0):
            counts["right_fail_to_left_success"] += 1
        else:
            counts["right_success_to_left_fail"] += 1
    return counts


def hierarchical_paired_sr_interval(
    left: Mapping[tuple[int, ...], int],
    right: Mapping[tuple[int, ...], int],
    *,
    iterations: int = 10_000,
    seed: int = 20260805,
) -> dict[str, float | int | str]:
    """Bootstrap paired SR differences by task, then episode within task."""

    if set(left) != set(right):
        raise RuntimeError("Paired bootstrap requires identical episode keys")
    by_task: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in sorted(left):
        by_task[int(key[0])].append((int(left[key]), int(right[key])))
    tasks = sorted(by_task)
    if not tasks or iterations <= 0:
        raise ValueError("At least one task and one bootstrap iteration are required")
    observed = np.mean(
        [left[key] - right[key] for key in sorted(left)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        task_draw = rng.choice(tasks, size=len(tasks), replace=True)
        differences: list[float] = []
        for task_id in task_draw:
            task_pairs = by_task[int(task_id)]
            episode_draw = rng.integers(0, len(task_pairs), size=len(task_pairs))
            differences.extend(
                task_pairs[item][0] - task_pairs[item][1] for item in episode_draw
            )
        samples[index] = np.mean(differences, dtype=np.float64)
    return {
        "difference": float(observed),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "iterations": int(iterations),
        "seed": int(seed),
        "paired_episodes": len(left),
        "tasks": len(tasks),
        "method": "paired_task_hierarchical_bootstrap",
    }


def task_success_rates(
    outcomes: Mapping[tuple[int, ...], int]
) -> dict[str, dict[str, float | int]]:
    """Compute task-wise success counts and rates."""

    grouped: dict[int, list[int]] = defaultdict(list)
    for key, value in outcomes.items():
        grouped[int(key[0])].append(int(value))
    return {
        str(task_id): {
            "successes": int(sum(values)),
            "episodes": len(values),
            "sr": float(np.mean(values)),
        }
        for task_id, values in sorted(grouped.items())
    }


def common_success_completion_steps(
    left_success: Mapping[tuple[int, ...], int],
    right_success: Mapping[tuple[int, ...], int],
    left_steps: Mapping[tuple[int, ...], float],
    right_steps: Mapping[tuple[int, ...], float],
) -> dict[str, Any]:
    """Compare completion length only where both methods succeed."""

    keys = [
        key
        for key in sorted(set(left_success) & set(right_success))
        if left_success[key] and right_success[key]
    ]
    differences = [left_steps[key] - right_steps[key] for key in keys]
    return {
        "common_success_episodes": len(keys),
        "left_mean_steps": float(np.mean([left_steps[key] for key in keys]))
        if keys
        else float("nan"),
        "right_mean_steps": float(np.mean([right_steps[key] for key in keys]))
        if keys
        else float("nan"),
        "left_minus_right_mean_steps": float(np.mean(differences))
        if differences
        else float("nan"),
    }
