"""Metrics for task quality, action stability, and correction provenance."""

from __future__ import annotations

import collections
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from methods.latentloop.eval.metrics import action_diagnostics, distribution_summary


def hierarchical_action_diagnostics(
    actions: Tensor,
    *,
    chunk_boundaries: Tensor | None = None,
) -> dict[str, float]:
    """Extend existing action diagnostics with rapid binary gripper reversals."""

    result = action_diagnostics(actions, chunk_boundaries=chunk_boundaries)
    signs = actions[:, 6] >= 0 if actions.numel() else torch.empty(0, dtype=torch.bool)
    if signs.numel() >= 3:
        reversals = ((signs[2:] == signs[:-2]) & (signs[1:-1] != signs[:-2])).sum()
    else:
        reversals = torch.zeros((), dtype=torch.long, device=actions.device)
    result["gripper_reversals"] = float(reversals.item())
    result["finite_action_fraction"] = (
        float(torch.isfinite(actions).float().mean().item()) if actions.numel() else 1.0
    )
    if signs.numel():
        positive_fraction = float(signs.float().mean().item())
        result["gripper_positive_fraction"] = positive_fraction
        result["gripper_collapsed"] = float(positive_fraction in {0.0, 1.0})
    else:
        result["gripper_positive_fraction"] = 0.0
        result["gripper_collapsed"] = 0.0
    return result


def correction_residuals_by_age(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    """Summarize scalar correction residual diagnostics by full-query age."""

    return trace_metrics_by_age(records, field="action_correction_residual")


def trace_metrics_by_age(
    records: Iterable[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize one mapping-valued trace field by full-query age."""

    values: dict[int, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for record in records:
        metrics = record.get(field)
        if not isinstance(metrics, Mapping):
            continue
        age = int(record["query_age"])
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                values[age][str(name)].append(float(value))
    return {
        str(age): {
            name: distribution_summary(samples)
            for name, samples in sorted(metrics.items())
        }
        for age, metrics in sorted(values.items())
    }


def task_hierarchical_paired_ci(
    baseline: Mapping[tuple[int, int], bool],
    candidate: Mapping[tuple[int, int], bool],
    *,
    seed: int,
    samples: int = 10_000,
) -> list[float | None]:
    """Bootstrap tasks and then paired episodes, reporting percentage points."""

    tasks = sorted({task for task, episode in baseline if (task, episode) in candidate})
    if not tasks:
        return [None, None]
    by_task: dict[int, list[float]] = {}
    for task in tasks:
        episodes = sorted(
            episode
            for candidate_task, episode in baseline
            if candidate_task == task and (task, episode) in candidate
        )
        by_task[task] = [
            float(candidate[(task, episode)]) - float(baseline[(task, episode)])
            for episode in episodes
        ]
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        means = [
            float(np.mean(rng.choice(by_task[int(task)], size=len(by_task[int(task)]), replace=True)))
            for task in sampled_tasks
        ]
        estimates.append(100.0 * float(np.mean(means)))
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def paired_outcome_summary(
    baseline: Mapping[tuple[int, int], bool],
    candidate: Mapping[tuple[int, int], bool],
    *,
    seed: int,
) -> dict[str, Any]:
    """Return paired flips, difference, and task-hierarchical confidence interval."""

    keys = sorted(set(baseline) & set(candidate))
    flips: collections.Counter[str] = collections.Counter()
    labels = {
        (False, False): "both_fail",
        (False, True): "fail_to_success",
        (True, False): "success_to_fail",
        (True, True): "both_success",
    }
    for key in keys:
        flips[labels[(bool(baseline[key]), bool(candidate[key]))]] += 1
    difference = (
        100.0
        * float(np.mean([float(candidate[key]) - float(baseline[key]) for key in keys]))
        if keys
        else None
    )
    return {
        "pairs": len(keys),
        "candidate_minus_baseline_pp": difference,
        "task_hierarchical_paired_ci95_pp": task_hierarchical_paired_ci(
            baseline,
            candidate,
            seed=seed,
        ),
        "paired_flips": dict(flips),
    }
