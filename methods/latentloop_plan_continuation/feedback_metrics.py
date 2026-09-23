"""Feedback-correction statistics with episode/task clustered uncertainty."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks for ties without requiring SciPy."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Compute Spearman's rho after jointly filtering non-finite pairs."""

    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if x_array.shape != y_array.shape:
        raise ValueError(f"Spearman inputs must match, got {x_array.shape} and {y_array.shape}")
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if int(valid.sum()) < 3:
        return float("nan")
    x_rank = _rankdata(x_array[valid])
    y_rank = _rankdata(y_array[valid])
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(x_rank, y_rank) / denominator)


def binned_summary(
    driver: Iterable[float], response: Iterable[float], *, bins: int = 4
) -> list[dict[str, float | int]]:
    """Summarize response magnitude in driver quantile bins."""

    x = np.asarray(list(driver), dtype=np.float64)
    y = np.asarray(list(response), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Binned inputs must match, got {x.shape} and {y.shape}")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        return []
    edges = np.quantile(x, np.linspace(0.0, 1.0, int(bins) + 1))
    assignments = np.searchsorted(edges[1:-1], x, side="right")
    rows: list[dict[str, float | int]] = []
    for bin_index in range(int(bins)):
        selected = assignments == bin_index
        rows.append(
            {
                "bin": bin_index,
                "count": int(selected.sum()),
                "driver_min": float(np.min(x[selected])) if selected.any() else float("nan"),
                "driver_max": float(np.max(x[selected])) if selected.any() else float("nan"),
                "driver_mean": float(np.mean(x[selected])) if selected.any() else float("nan"),
                "response_mean": float(np.mean(y[selected])) if selected.any() else float("nan"),
                "response_median": float(np.median(y[selected])) if selected.any() else float("nan"),
            }
        )
    return rows


def clustered_spearman_interval(
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    y_key: str,
    *,
    task_key: str = "task_id",
    episode_key: str = "episode_id",
    iterations: int = 2000,
    seed: int = 20260803,
) -> dict[str, float | int | str]:
    """Bootstrap Spearman rho by resampling tasks then episodes within tasks.

    The point estimate is the ordinary timestep-level Spearman correlation. For
    uncertainty, ranks are computed once on the full valid sample and treated as
    fixed pseudo-observations. Each episode is then represented by sufficient
    statistics of those ranks. A hierarchical task/episode bootstrap resamples
    only the episode statistics, avoiding repeated materialization and sorting of
    tens of thousands of timestep records.

    This is a fixed-rank clustered bootstrap: it preserves the declared task and
    episode sampling units while making large trace analyses computationally
    tractable. It is not the much more expensive bootstrap that recomputes ranks
    after every cluster resample.
    """

    clusters: dict[object, dict[object, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        try:
            x_value = float(row[x_key])
            y_value = float(row[y_key])
            task = row[task_key]
            episode = row[episode_key]
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(x_value) and np.isfinite(y_value):
            index = len(x_values)
            x_values.append(x_value)
            y_values.append(y_value)
            clusters[task][episode].append(index)
    tasks = sorted(clusters, key=str)
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    point = spearman_correlation(x_array, y_array)
    episode_count = sum(len(episodes) for episodes in clusters.values())
    if not np.isfinite(point):
        return {
            "rho": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "iterations": 0,
            "valid_records": int(x_array.size),
            "valid_episodes": int(episode_count),
            "valid_tasks": int(len(tasks)),
            "bootstrap_method": "undefined_nonvarying_input",
            "rank_reference": "full_valid_sample",
        }
    if not tasks or iterations <= 0:
        return {
            "rho": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "iterations": 0,
            "valid_records": int(x_array.size),
            "valid_episodes": int(episode_count),
            "valid_tasks": int(len(tasks)),
            "bootstrap_method": "not_run",
            "rank_reference": "full_valid_sample",
        }

    x_rank = _rankdata(x_array)
    y_rank = _rankdata(y_array)
    task_episode_statistics: list[np.ndarray] = []
    for task in tasks:
        episode_statistics = []
        for episode in sorted(clusters[task], key=str):
            indices = np.asarray(clusters[task][episode], dtype=np.int64)
            episode_x = x_rank[indices]
            episode_y = y_rank[indices]
            episode_statistics.append(
                np.asarray(
                    [
                        indices.size,
                        episode_x.sum(),
                        episode_y.sum(),
                        np.dot(episode_x, episode_x),
                        np.dot(episode_y, episode_y),
                        np.dot(episode_x, episode_y),
                    ],
                    dtype=np.float64,
                )
            )
        task_episode_statistics.append(np.stack(episode_statistics))

    # Resampling task counts is equivalent to drawing ordered tasks and then
    # discarding order. If a task is selected c times, its combined episode
    # draws follow Multinomial(c * episode_count, uniform probabilities).
    rng = np.random.default_rng(seed)
    iteration_count = int(iterations)
    task_count = len(tasks)
    sampled_task_counts = rng.multinomial(
        task_count,
        np.full(task_count, 1.0 / task_count, dtype=np.float64),
        size=iteration_count,
    )
    totals = np.zeros((iteration_count, 6), dtype=np.float64)
    for task_index, episode_statistics in enumerate(task_episode_statistics):
        task_multiplicities = sampled_task_counts[:, task_index]
        episodes_in_task = int(episode_statistics.shape[0])
        probabilities = np.full(
            episodes_in_task, 1.0 / episodes_in_task, dtype=np.float64
        )
        for multiplicity in range(1, task_count + 1):
            selected_iterations = np.flatnonzero(
                task_multiplicities == multiplicity
            )
            if selected_iterations.size == 0:
                continue
            episode_draw_counts = rng.multinomial(
                multiplicity * episodes_in_task,
                probabilities,
                size=int(selected_iterations.size),
            )
            totals[selected_iterations] += episode_draw_counts @ episode_statistics

    counts = totals[:, 0]
    covariance = totals[:, 5] - totals[:, 1] * totals[:, 2] / counts
    x_variance = totals[:, 3] - totals[:, 1] ** 2 / counts
    y_variance = totals[:, 4] - totals[:, 2] ** 2 / counts
    denominator = np.sqrt(np.maximum(x_variance, 0.0) * np.maximum(y_variance, 0.0))
    valid = np.isfinite(covariance) & np.isfinite(denominator) & (denominator > 0.0)
    samples = covariance[valid] / denominator[valid]
    return {
        "rho": point,
        "ci_low": float(np.percentile(samples, 2.5)) if samples.size else float("nan"),
        "ci_high": float(np.percentile(samples, 97.5)) if samples.size else float("nan"),
        "iterations": int(samples.size),
        "valid_records": int(x_array.size),
        "valid_episodes": int(episode_count),
        "valid_tasks": int(task_count),
        "bootstrap_method": "hierarchical_task_episode_fixed_rank",
        "rank_reference": "full_valid_sample",
    }


def paired_clustered_mean_difference_interval(
    left: Mapping[tuple[object, object], float],
    right: Mapping[tuple[object, object], float],
    *,
    iterations: int = 10000,
    seed: int = 20260803,
) -> dict[str, float | int | str]:
    """Bootstrap ``left-right`` episode means by task and paired episode."""

    common = sorted(set(left) & set(right), key=lambda item: (str(item[0]), str(item[1])))
    by_task: dict[object, list[tuple[object, object]]] = defaultdict(list)
    for key in common:
        by_task[key[0]].append(key)
    tasks = sorted(by_task, key=str)
    differences = np.asarray([float(left[key]) - float(right[key]) for key in common])
    point = float(np.mean(differences)) if differences.size else float("nan")
    if not tasks or iterations <= 0:
        return {
            "mean_difference": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "paired_episodes": len(common),
            "iterations": 0,
            "bootstrap_method": "not_run",
        }

    rng = np.random.default_rng(seed)
    iteration_count = int(iterations)
    task_count = len(tasks)
    sampled_task_counts = rng.multinomial(
        task_count,
        np.full(task_count, 1.0 / task_count, dtype=np.float64),
        size=iteration_count,
    )
    sampled_sums = np.zeros(iteration_count, dtype=np.float64)
    sampled_counts = np.zeros(iteration_count, dtype=np.int64)
    for task_index, task in enumerate(tasks):
        task_values = np.asarray(
            [float(left[key]) - float(right[key]) for key in by_task[task]],
            dtype=np.float64,
        )
        episodes_in_task = int(task_values.size)
        probabilities = np.full(
            episodes_in_task, 1.0 / episodes_in_task, dtype=np.float64
        )
        task_multiplicities = sampled_task_counts[:, task_index]
        for multiplicity in range(1, task_count + 1):
            selected_iterations = np.flatnonzero(
                task_multiplicities == multiplicity
            )
            if selected_iterations.size == 0:
                continue
            episode_draw_counts = rng.multinomial(
                multiplicity * episodes_in_task,
                probabilities,
                size=int(selected_iterations.size),
            )
            sampled_sums[selected_iterations] += episode_draw_counts @ task_values
            sampled_counts[selected_iterations] += multiplicity * episodes_in_task
    samples = sampled_sums / sampled_counts
    return {
        "mean_difference": point,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "paired_episodes": int(len(common)),
        "iterations": int(samples.size),
        "bootstrap_method": "hierarchical_task_episode_multinomial",
    }
