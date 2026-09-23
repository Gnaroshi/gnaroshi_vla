#!/usr/bin/env python3
"""Verify K1 outcome and executed-action parity for comparison adapters."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _episode_outcomes(root: Path) -> dict[tuple[int, int, int], int]:
    return {
        (int(row["task_id"]), int(row["episode_id"]), int(row["seed"])): int(
            float(row["success"])
        )
        for row in _rows(_find_one(root, "eval_episode_metrics.csv"))
    }


def _step_actions(root: Path) -> dict[tuple[int, int, int], np.ndarray]:
    episode_csv = _find_one(root, "eval_episode_metrics.csv")
    step_dir = episode_csv.parent / "eval_step_logs"
    result: dict[tuple[int, int, int], np.ndarray] = {}
    for path in sorted(step_dir.glob("*.csv")):
        for row in _rows(path):
            key = int(row["task_id"]), int(row["episode_id"]), int(row["timestep"])
            if key in result:
                raise RuntimeError(f"Duplicate step key in {step_dir}: {key}")
            result[key] = np.asarray(
                [float(row[f"action_{index}"]) for index in range(7)],
                dtype=np.float64,
            )
    if not result:
        raise RuntimeError(f"No step actions under {step_dir}")
    return result


def compare_rows(baseline_root: Path, candidate_root: Path, tolerance: float) -> dict[str, Any]:
    """Compare paired episodes and executed actions for one K1 candidate."""

    baseline_outcomes = _episode_outcomes(baseline_root)
    candidate_outcomes = _episode_outcomes(candidate_root)
    episode_keys_identical = set(baseline_outcomes) == set(candidate_outcomes)
    outcome_parity = episode_keys_identical and baseline_outcomes == candidate_outcomes
    baseline_actions = _step_actions(baseline_root)
    candidate_actions = _step_actions(candidate_root)
    action_keys_identical = set(baseline_actions) == set(candidate_actions)
    common = sorted(set(baseline_actions) & set(candidate_actions))
    maximum = max(
        (
            float(np.max(np.abs(baseline_actions[key] - candidate_actions[key])))
            for key in common
        ),
        default=float("inf"),
    )
    action_parity = action_keys_identical and bool(common) and maximum <= tolerance
    return {
        "pass": bool(outcome_parity and action_parity),
        "episode_keys_identical": episode_keys_identical,
        "outcome_parity": outcome_parity,
        "action_keys_identical": action_keys_identical,
        "paired_action_steps": len(common),
        "max_executed_action_difference": maximum,
        "action_tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--action-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    comparisons: dict[str, Any] = {}
    for specification in args.candidate:
        name, separator, raw_path = specification.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Expected NAME=PATH, got {specification!r}")
        comparisons[name] = compare_rows(
            args.baseline_root.resolve(),
            Path(raw_path).resolve(),
            float(args.action_tolerance),
        )
    result = {
        "schema_version": 1,
        "protocol": "seer_latentloop_comparison_k1_parity_v1",
        "baseline_root": str(args.baseline_root.resolve()),
        "comparisons": comparisons,
        "pass": all(row["pass"] for row in comparisons.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["pass"]:
        raise SystemExit(f"K1 parity failed: {result}")
    print(f"[VERIFY][OK] K1 comparison parity: {args.output}")


if __name__ == "__main__":
    main()
