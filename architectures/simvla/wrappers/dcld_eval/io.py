"""I/O helpers for SimVLA DCLD eval artifacts.

Purpose:
    Centralize JSON, text, JSONL, environment, and command metadata helpers for
    evaluation runs and dry-runs.

Inputs/outputs:
    Helpers write structured artifacts under the caller's output directory.

Official-match scope:
    Environment and git snapshots are required to make official-baseline
    calibration auditable.

DCLD scope:
    Applies to both full baseline rows and DCLD rows.

Caveat:
    Generated results, videos, caches, and checkpoints should remain untracked.
"""

from __future__ import annotations

from .rollout_runner import append_jsonl, collect_environment_metadata, command_output, write_json, write_text

__all__ = ["append_jsonl", "collect_environment_metadata", "command_output", "write_json", "write_text"]
