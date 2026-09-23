"""Metrics for action-horizon overlap across consecutive policy queries."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)


def _as_horizon(value: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[-1] != 7:
        raise ValueError(f"{name} must have shape [P, 7], got {array.shape}")
    return array


def aligned_overlap_components(
    previous_horizon: np.ndarray | Sequence[Sequence[float]],
    current_horizon: np.ndarray | Sequence[Sequence[float]],
    pairs: Sequence[tuple[int, int]],
    *,
    previous_gripper_logit: np.ndarray | None = None,
    current_gripper_logit: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Return per-pair component errors for verified token-time pairs.

    Each pair is ``(previous_token, current_token)``. Arm L1 values are means
    over their three axes while L2 values are Euclidean norms. Gripper
    probability is action dimension six. Logit errors are emitted when both
    pre-sigmoid arrays are supplied.
    """

    previous = _as_horizon(previous_horizon, "previous_horizon")
    current = _as_horizon(current_horizon, "current_horizon")
    previous_logits = None if previous_gripper_logit is None else np.asarray(
        previous_gripper_logit, dtype=np.float64
    ).reshape(previous.shape[0], -1)
    current_logits = None if current_gripper_logit is None else np.asarray(
        current_gripper_logit, dtype=np.float64
    ).reshape(current.shape[0], -1)

    rows: list[dict[str, float | int]] = []
    for previous_token, current_token in pairs:
        if not 0 <= previous_token < previous.shape[0]:
            raise IndexError(f"previous token {previous_token} is outside P={previous.shape[0]}")
        if not 0 <= current_token < current.shape[0]:
            raise IndexError(f"current token {current_token} is outside P={current.shape[0]}")
        difference = current[current_token] - previous[previous_token]
        translation = difference[:3]
        rotation = difference[3:6]
        grip_probability = float(abs(difference[6]))
        row: dict[str, float | int] = {
            "previous_token": int(previous_token),
            "current_token": int(current_token),
            "translation_l1": float(np.mean(np.abs(translation))),
            "translation_l2": float(np.linalg.norm(translation)),
            "rotation_l1": float(np.mean(np.abs(rotation))),
            "rotation_l2": float(np.linalg.norm(rotation)),
            "gripper_probability_abs": grip_probability,
            "gripper_threshold_disagreement": float(
                (current[current_token, 6] > 0.5)
                != (previous[previous_token, 6] > 0.5)
            ),
            "all_token_l1": float(np.mean(np.abs(difference))),
            "all_token_l2": float(np.linalg.norm(difference)),
        }
        if previous_logits is not None and current_logits is not None:
            row["gripper_logit_abs"] = float(
                abs(current_logits[current_token, 0] - previous_logits[previous_token, 0])
            )
        else:
            row["gripper_logit_abs"] = float("nan")
        rows.append(row)
    return rows


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    """Summarize finite values with the paper protocol percentiles."""

    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    summary: dict[str, float | int] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
    }
    for percentile in PERCENTILES:
        summary[f"p{percentile}"] = (
            float(np.percentile(finite, percentile)) if finite.size else float("nan")
        )
    return summary


def summarize_metric_rows(
    rows: Sequence[Mapping[str, float | int]], metric_names: Sequence[str]
) -> dict[str, dict[str, float | int]]:
    """Summarize named scalar fields from overlap rows."""

    return {
        metric: summarize_values(float(row[metric]) for row in rows if metric in row)
        for metric in metric_names
    }
