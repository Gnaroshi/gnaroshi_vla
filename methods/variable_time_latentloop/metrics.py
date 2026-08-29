"""Action, efficiency, and paired-outcome metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch import Tensor


def action_error_components(predicted: Tensor, target: Tensor, executed_horizon: int = 5) -> dict[str, Tensor]:
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] < 7:
        raise ValueError("actions must be aligned [B,H,D>=7] tensors")
    if not 1 <= executed_horizon <= predicted.shape[1]:
        raise ValueError("executed_horizon is outside the action chunk")
    delta = predicted - target
    prefix = delta[:, :executed_horizon]
    return {
        "chunk_mse": delta[..., :7].square().mean(dim=(1, 2)),
        "executed_mse": prefix[..., :7].square().mean(dim=(1, 2)),
        "translation_mse": prefix[..., :3].square().mean(dim=(1, 2)),
        "rotation_mse": prefix[..., 3:6].square().mean(dim=(1, 2)),
        "gripper_mse": prefix[..., 6].square().mean(dim=1),
    }


def action_second_difference(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions)
    if actions.ndim < 2 or actions.shape[-2] < 3:
        return np.empty((*actions.shape[:-2], 0, actions.shape[-1]), dtype=actions.dtype)
    return actions[..., 2:, :] - 2.0 * actions[..., 1:-1, :] + actions[..., :-2, :]


def paired_flip_counts(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    baseline = np.asarray(baseline, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if baseline.shape != candidate.shape:
        raise ValueError("paired outcomes must have identical shapes")
    return {
        "both_success": int(np.sum(baseline & candidate)),
        "baseline_only": int(np.sum(baseline & ~candidate)),
        "candidate_only": int(np.sum(~baseline & candidate)),
        "both_failure": int(np.sum(~baseline & ~candidate)),
    }


def hierarchical_paired_bootstrap(
    rows: list[dict[str, object]],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap candidate-baseline SR difference over suite/task/episode."""

    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["suite"])][str(row["task"])].append(
            (float(row["baseline_success"]), float(row["candidate_success"]))
        )
    if not grouped:
        raise ValueError("no paired rows")
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    suites = sorted(grouped)
    for sample_index in range(samples):
        sampled_suites = rng.choice(suites, size=len(suites), replace=True)
        values: list[float] = []
        for suite in sampled_suites:
            tasks = sorted(grouped[suite])
            sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
            for task in sampled_tasks:
                episodes = grouped[suite][task]
                indices = rng.integers(0, len(episodes), size=len(episodes))
                values.extend(episodes[index][1] - episodes[index][0] for index in indices)
        differences[sample_index] = np.mean(values)
    return {
        "mean_difference": float(np.mean(differences)),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "samples": int(samples),
        "seed": int(seed),
    }
