"""Monotonic error calibration and fixed-budget three-level scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .decisions import RefreshDecision


@dataclass(frozen=True)
class MonotonicBinnedCalibrator:
    boundaries: tuple[float, ...]
    values: tuple[float, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MonotonicBinnedCalibrator":
        return cls(
            boundaries=tuple(float(value) for value in payload["boundaries"]),  # type: ignore[index]
            values=tuple(float(value) for value in payload["values"]),  # type: ignore[index]
        )

    @classmethod
    def fit(cls, scores: np.ndarray, errors: np.ndarray, bins: int = 32) -> "MonotonicBinnedCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        errors = np.asarray(errors, dtype=np.float64)
        if scores.shape != errors.shape or scores.ndim != 1 or len(scores) < 2:
            raise ValueError("calibration inputs must be aligned one-dimensional vectors")
        order = np.argsort(scores)
        groups = np.array_split(order, min(bins, len(order)))
        boundaries = [float(np.max(scores[group])) for group in groups if len(group)]
        means = [float(np.mean(errors[group])) for group in groups if len(group)]

        # Pool-adjacent-violators isotonic regression over bin means.
        blocks: list[dict[str, float | int]] = []
        for index, (boundary, value, group) in enumerate(zip(boundaries, means, groups, strict=False)):
            blocks.append({"start": index, "end": index, "weight": len(group), "value": value, "boundary": boundary})
            while len(blocks) >= 2 and float(blocks[-2]["value"]) > float(blocks[-1]["value"]):
                right = blocks.pop()
                left = blocks.pop()
                weight = int(left["weight"]) + int(right["weight"])
                value = (
                    float(left["value"]) * int(left["weight"])
                    + float(right["value"]) * int(right["weight"])
                ) / weight
                blocks.append(
                    {
                        "start": int(left["start"]),
                        "end": int(right["end"]),
                        "weight": weight,
                        "value": value,
                        "boundary": float(right["boundary"]),
                    }
                )
        fitted = np.empty(len(boundaries), dtype=np.float64)
        for block in blocks:
            fitted[int(block["start"]) : int(block["end"]) + 1] = float(block["value"])
        return cls(boundaries=tuple(boundaries), values=tuple(float(value) for value in fitted))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        indices = np.searchsorted(np.asarray(self.boundaries), scores, side="left")
        indices = np.clip(indices, 0, len(self.values) - 1)
        return np.asarray(self.values)[indices]


@dataclass(frozen=True)
class BudgetCalibration:
    low_threshold: float
    high_threshold: float
    target_full_prefix_ratio: float
    validation_full_prefix_ratio: float
    validation_direct_ratio: float
    validation_selected_error: float
    calibrator: MonotonicBinnedCalibrator

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BudgetCalibration":
        return cls(
            low_threshold=float(payload["low_threshold"]),
            high_threshold=float(payload["high_threshold"]),
            target_full_prefix_ratio=float(payload["target_full_prefix_ratio"]),
            validation_full_prefix_ratio=float(payload["validation_full_prefix_ratio"]),
            validation_direct_ratio=float(payload["validation_direct_ratio"]),
            validation_selected_error=float(payload["validation_selected_error"]),
            calibrator=MonotonicBinnedCalibrator.from_dict(payload["calibrator"]),  # type: ignore[arg-type]
        )

    def decide(self, defect_score: float) -> RefreshDecision:
        predicted_error = float(self.calibrator.predict(np.asarray([defect_score]))[0])
        if predicted_error >= self.high_threshold:
            return RefreshDecision.FULL_PREFIX
        if predicted_error >= self.low_threshold:
            return RefreshDecision.DIRECT_REANCHOR
        return RefreshDecision.SEQUENTIAL

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["calibrator"] = asdict(self.calibrator)
        return result


class BudgetCalibrator:
    """Select validation-only thresholds under a full-prefix call budget."""

    def __init__(self, target_full_prefix_ratio: float = 0.25, bins: int = 32) -> None:
        if not 0.0 < target_full_prefix_ratio < 1.0:
            raise ValueError("target_full_prefix_ratio must be in (0,1)")
        self.target = float(target_full_prefix_ratio)
        self.bins = int(bins)

    def fit(
        self,
        defect: np.ndarray,
        sequential_error: np.ndarray,
        direct_error: np.ndarray,
    ) -> BudgetCalibration:
        defect = np.asarray(defect, dtype=np.float64)
        sequential_error = np.asarray(sequential_error, dtype=np.float64)
        direct_error = np.asarray(direct_error, dtype=np.float64)
        if defect.shape != sequential_error.shape or defect.shape != direct_error.shape:
            raise ValueError("calibration arrays must have identical shapes")
        calibrator = MonotonicBinnedCalibrator.fit(defect, sequential_error, self.bins)
        predicted = calibrator.predict(defect)

        # Select a threshold whose runtime predicate is exactly reproducible.
        # Never report a tie-broken mask that ``predicted >= high`` cannot enact.
        threshold_candidates = np.unique(predicted)
        feasible = [
            (float(np.mean(predicted >= threshold)), float(threshold))
            for threshold in threshold_candidates
            if float(np.mean(predicted >= threshold)) <= self.target
        ]
        if feasible:
            _, high = max(feasible, key=lambda item: (item[0], -item[1]))
        else:
            high = float(np.nextafter(np.max(predicted), np.inf))
        full_mask = predicted >= high

        candidates = np.unique(predicted[~full_mask])
        if not len(candidates):
            candidates = np.asarray([high])
        best: tuple[float, float, np.ndarray] | None = None
        for low in candidates:
            direct_mask = (~full_mask) & (predicted >= low)
            selected = np.where(full_mask, 0.0, np.where(direct_mask, direct_error, sequential_error))
            objective = float(np.mean(selected))
            candidate = (objective, float(low), direct_mask)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        objective, low, direct_mask = best
        return BudgetCalibration(
            low_threshold=low,
            high_threshold=high,
            target_full_prefix_ratio=self.target,
            validation_full_prefix_ratio=float(np.mean(full_mask)),
            validation_direct_ratio=float(np.mean(direct_mask)),
            validation_selected_error=objective,
            calibrator=calibrator,
        )
