"""Latent-defect computation and validation statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def normalized_latent_defect(
    sequential: Tensor,
    direct: Tensor,
    *,
    scale: Tensor | float | None = None,
    projection: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Return one normalized direct/composed discrepancy per batch item."""

    if sequential.shape != direct.shape:
        raise ValueError("sequential and direct states must have identical shapes")
    difference = sequential - direct
    if projection is not None:
        if projection.ndim != 2 or projection.shape[0] != difference.shape[-1]:
            raise ValueError("projection must be [state_width, projected_width]")
        difference = difference @ projection.to(device=difference.device, dtype=difference.dtype)
    if scale is None:
        denominator = direct.detach().square().mean(dim=tuple(range(1, direct.ndim))).sqrt()
    else:
        denominator = torch.as_tensor(scale, device=difference.device, dtype=difference.dtype)
        if denominator.ndim == 0:
            denominator = denominator.expand(difference.shape[0])
    numerator = difference.square().mean(dim=tuple(range(1, difference.ndim))).sqrt()
    return numerator / denominator.clamp_min(eps)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("Spearman inputs must be same-length vectors with at least two values")
    rx, ry = _average_ranks(x), _average_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("AUROC inputs must be same-length vectors")
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    ranks = _average_ranks(scores)
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def monotonic_bin_means(scores: np.ndarray, errors: np.ndarray, bins: int = 10) -> list[float]:
    order = np.argsort(scores)
    chunks = np.array_split(order, min(bins, len(order)))
    return [float(np.mean(errors[index])) for index in chunks if len(index)]


@dataclass(frozen=True)
class DefectValidity:
    spearman: float
    auroc: float
    bin_means: tuple[float, ...]
    monotonic_fraction: float
    baseline_aurocs: dict[str, float]
    high_error_threshold: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "spearman": self.spearman,
            "auroc": self.auroc,
            "bin_means": list(self.bin_means),
            "monotonic_fraction": self.monotonic_fraction,
            "baseline_aurocs": self.baseline_aurocs,
            "high_error_threshold": self.high_error_threshold,
            "passed": self.passed,
        }


def evaluate_defect_validity(
    defect: np.ndarray,
    action_error: np.ndarray,
    baselines: dict[str, np.ndarray],
    *,
    high_error_quantile: float = 0.9,
    minimum_auroc: float = 0.70,
    high_error_threshold: float | None = None,
) -> DefectValidity:
    defect = np.asarray(defect, dtype=np.float64)
    action_error = np.asarray(action_error, dtype=np.float64)
    if defect.shape != action_error.shape or len(defect) < 10:
        raise ValueError("defect audit requires at least ten aligned samples")
    threshold = (
        float(high_error_threshold)
        if high_error_threshold is not None
        else float(np.quantile(action_error, high_error_quantile))
    )
    labels = (action_error >= threshold).astype(np.int64)
    bin_means = monotonic_bin_means(defect, action_error)
    monotonic_pairs = [b >= a for a, b in zip(bin_means, bin_means[1:], strict=False)]
    monotonic_fraction = float(np.mean(monotonic_pairs)) if monotonic_pairs else 0.0
    auroc = binary_auroc(defect, labels)
    baseline_aurocs = {name: binary_auroc(np.asarray(value), labels) for name, value in baselines.items()}
    best_baseline = max((value for value in baseline_aurocs.values() if np.isfinite(value)), default=0.0)
    correlation = spearman_correlation(defect, action_error)
    passed = bool(
        correlation > 0.0
        and auroc >= minimum_auroc
        and auroc > best_baseline
        and monotonic_fraction >= 0.75
    )
    return DefectValidity(
        spearman=correlation,
        auroc=auroc,
        bin_means=tuple(bin_means),
        monotonic_fraction=monotonic_fraction,
        baseline_aurocs=baseline_aurocs,
        high_error_threshold=threshold,
        passed=passed,
    )
