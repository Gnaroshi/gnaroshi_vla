"""Policy row planning and mode naming for SimVLA DCLD eval.

Purpose:
    Expose mode/row planning APIs used by dry-runs and command wrappers.

Inputs/outputs:
    The public functions return dictionaries written to planned-row JSON/CSV
    artifacts.

Official-match scope:
    `full` mode with K=1 is the wrapper's closest official-style full SimVLA
    path, but wrapper rows are not themselves official upstream baselines.

DCLD scope:
    DCLD-specific rows include `stepwise_dcld`, `no_delta`, `shuffled_delta`,
    `proprio_only`, and `image_only`.

Caveat:
    Implementations are re-exported from `rollout_runner` in this preservation
    refactor so before/after dry-run rows remain exactly identical.
"""

from __future__ import annotations

from .rollout_runner import base_row, planned_rows

__all__ = ["base_row", "planned_rows"]
