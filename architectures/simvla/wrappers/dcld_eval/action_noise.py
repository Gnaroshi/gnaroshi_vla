"""Explicit action-flow noise utilities for SimVLA DCLD eval.

Purpose:
    Centralize deterministic action-noise seed construction so baseline/ours
    parity rows can share the same initial action-flow noise without hidden RNG
    coupling.

Inputs/outputs:
    The exported `stable_seed` function maps structured key parts to an integer
    seed. Runtime initial-noise tensors are still created by the action adapter
    in `rollout_runner`.

Official-match scope:
    Official SimVLA samples noise inside `generate_actions`. Wrapper parity
    diagnostics make the corresponding noise explicit and logged.

DCLD scope:
    Not DCLD-specific. It controls stochasticity for both full and DCLD rows.

Caveat:
    The seed key intentionally must not include row name for K=1 baseline/ours
    parity rows.
"""

from __future__ import annotations

from .rollout_runner import stable_seed

__all__ = ["stable_seed"]
