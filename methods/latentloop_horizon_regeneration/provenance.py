"""Action-horizon lineage and level-specific runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schedule import ExecutionLevel, ExecutionMode


PROVENANCE_LABELS = (
    "full_origin",
    "shifted_corrected",
    "synthesized",
    "regenerated",
)
_INDEX = {name: index for index, name in enumerate(PROVENANCE_LABELS)}


class HorizonProvenance:
    """Track one categorical lineage distribution for every action token."""

    def __init__(self, action_pred_steps: int = 3):
        self.action_pred_steps = int(action_pred_steps)
        if self.action_pred_steps < 1:
            raise ValueError("action_pred_steps must be positive")
        self.weights: np.ndarray | None = None
        self.generation = -1
        self.fully_synthetic_horizons_prevented = 0

    @property
    def initialized(self) -> bool:
        return self.weights is not None

    def _one_hot(self, label: str) -> np.ndarray:
        values = np.zeros((self.action_pred_steps, len(PROVENANCE_LABELS)), dtype=np.float64)
        values[:, _INDEX[label]] = 1.0
        return values

    def reset_full(self) -> np.ndarray:
        self.weights = self._one_hot("full_origin")
        self.generation += 1
        return self.weights.copy()

    def reset_regenerated(self) -> np.ndarray:
        if self.would_be_fully_synthetic_after_correction():
            self.fully_synthetic_horizons_prevented += 1
        self.weights = self._one_hot("regenerated")
        self.generation += 1
        return self.weights.copy()

    def would_be_fully_synthetic_after_correction(self) -> bool:
        if self.weights is None:
            return False
        if self.action_pred_steps == 1:
            return True
        candidate = np.zeros_like(self.weights)
        previous_synthetic = self.weights[1:, _INDEX["synthesized"]] >= 1.0
        candidate[:-1, _INDEX["synthesized"]] = previous_synthetic.astype(np.float64)
        candidate[:-1, _INDEX["shifted_corrected"]] = 1.0 - candidate[
            :-1, _INDEX["synthesized"]
        ]
        candidate[-1, _INDEX["synthesized"]] = 1.0
        return bool(np.all(candidate[:, _INDEX["synthesized"]] == 1.0))

    def correct(self) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("provenance must be initialized by a complete horizon")
        previous = self.weights
        updated = np.zeros_like(previous)
        if self.action_pred_steps > 1:
            inherited_synthetic = previous[1:, _INDEX["synthesized"]] >= 1.0
            updated[:-1, _INDEX["synthesized"]] = inherited_synthetic.astype(np.float64)
            updated[:-1, _INDEX["shifted_corrected"]] = 1.0 - updated[
                :-1, _INDEX["synthesized"]
            ]
        updated[-1, _INDEX["synthesized"]] = 1.0
        self.weights = updated
        return updated.copy()

    def snapshot(self) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("provenance is not initialized")
        if not np.allclose(self.weights.sum(axis=-1), 1.0):
            raise RuntimeError("each token provenance distribution must sum to one")
        return self.weights.copy()


@dataclass(frozen=True)
class LevelCallCounts:
    full_seer: int
    action_head: int
    latent_updater: int
    action_correction: int


def assert_level_call_contract(
    level: int,
    counts: LevelCallCounts,
    *,
    mode: str,
) -> None:
    level = ExecutionLevel(level)
    mode = ExecutionMode(mode)
    if level == ExecutionLevel.FULL_SEER:
        if counts != LevelCallCounts(1, 1, 0, 0):
            raise RuntimeError(
                "Level 2 requires exactly one full Seer call and one action-head "
                f"call with no skip modules, got {counts}"
            )
        return
    if level == ExecutionLevel.HORIZON_REGENERATION:
        if counts != LevelCallCounts(0, 1, 1, 0):
            raise RuntimeError(f"Level 1 contract violated: {counts}")
        return
    expected = (
        LevelCallCounts(0, 0, 0, 1)
        if mode == ExecutionMode.PURE_ACTION_CORRECTION
        else LevelCallCounts(0, 0, 1, 1)
    )
    if counts != expected:
        raise RuntimeError(f"Level 0 contract violated for mode={mode.value}: {counts}")


def weighted_provenance(
    provenance_rows: Iterable[np.ndarray], weights: Iterable[float]
) -> np.ndarray:
    rows = np.asarray(list(provenance_rows), dtype=np.float64)
    values = np.asarray(list(weights), dtype=np.float64)
    if rows.ndim != 2 or rows.shape[-1] != len(PROVENANCE_LABELS):
        raise ValueError("provenance rows must have shape [N,4]")
    if values.shape != (rows.shape[0],):
        raise ValueError("weights must align with provenance rows")
    if rows.shape[0] == 0:
        raise ValueError("at least one provenance row is required")
    values = values / values.sum()
    result = (rows * values[:, None]).sum(axis=0)
    return result / result.sum()
