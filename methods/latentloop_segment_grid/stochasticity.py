"""Array-only helpers for the same-input full-Seer stochasticity sanity."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable

import numpy as np


def maximum_repeat_range(values: Iterable[np.ndarray]) -> float:
    """Return the largest element-wise max-minus-min range across repeats."""

    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if not arrays:
        return 0.0
    reference_shape = arrays[0].shape
    if any(array.shape != reference_shape for array in arrays):
        raise ValueError("All repeated outputs must have the same shape")
    stacked = np.stack(arrays, axis=0)
    return float(np.max(np.max(stacked, axis=0) - np.min(stacked, axis=0)))


def array_fingerprint(array: np.ndarray) -> str:
    """Return a SHA-256 fingerprint including dtype, shape, and bytes."""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def summarize_repeated_outputs(
    latents: Iterable[np.ndarray],
    raw_actions: Iterable[np.ndarray],
    executed_actions: Iterable[np.ndarray],
) -> Dict[str, float]:
    """Summarize repeated identical-input outputs with maximum differences."""

    return {
        "max_latent_difference": maximum_repeat_range(latents),
        "max_raw_action_difference": maximum_repeat_range(raw_actions),
        "max_executed_action_difference": maximum_repeat_range(executed_actions),
    }
