#!/usr/bin/env python3
"""Thin CLI entrypoint for SimVLA DCLD LIBERO eval.

Purpose:
    Keep the historical command path stable while delegating implementation to
    `architectures.simvla.wrappers.dcld_eval`.

Inputs/outputs:
    Accepts the same CLI arguments as the previous monolithic wrapper and writes
    the same dry-run/evaluation artifacts.

Official-match scope:
    Full-mode K=1 rows are intended to preserve official-style SimVLA action
    queue semantics; official upstream baselines are still the upstream
    server/client evaluation path.

DCLD scope:
    This file is not DCLD-specific logic. DCLD policy modes, rollout execution,
    calibration, and metrics live under `wrappers/dcld_eval/`.

Caveat:
    This is a behavior-preserving entrypoint refactor. Algorithm semantics are
    intentionally unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architectures.simvla.wrappers.dcld_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
