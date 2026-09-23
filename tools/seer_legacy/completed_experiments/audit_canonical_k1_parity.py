#!/usr/bin/env python3
"""Fail-closed K1 parity audit for Full Seer versus loaded LatentLoop."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def find_one(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(paths)}")
    return paths[0]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def episode_outcomes(root: Path) -> dict[tuple[int, int, int], int]:
    return {
        (int(row["task_id"]), int(row["episode_id"]), int(row["seed"])): int(float(row["success"]))
        for row in rows(find_one(root, "eval_episode_metrics.csv"))
    }


def step_records(root: Path) -> dict[tuple[int, int, int], dict[str, str]]:
    step_dir = find_one(root, "eval_episode_metrics.csv").parent / "eval_step_logs"
    result: dict[tuple[int, int, int], dict[str, str]] = {}
    for path in sorted(step_dir.glob("*.csv")):
        for row in rows(path):
            key = int(row["task_id"]), int(row["episode_id"]), int(row["timestep"])
            if key in result:
                raise RuntimeError(f"duplicate step key: {key}")
            result[key] = row
    if not result:
        raise RuntimeError(f"no step logs under {step_dir}")
    return result


def action(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[f"action_{index}"]) for index in range(7)], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    baseline_outcomes = episode_outcomes(args.baseline_root)
    candidate_outcomes = episode_outcomes(args.candidate_root)
    baseline_steps = step_records(args.baseline_root)
    candidate_steps = step_records(args.candidate_root)
    episode_keys_equal = set(baseline_outcomes) == set(candidate_outcomes)
    action_keys_equal = set(baseline_steps) == set(candidate_steps)
    common = sorted(set(baseline_steps) & set(candidate_steps))
    maximum = max(
        (float(np.max(np.abs(action(baseline_steps[key]) - action(candidate_steps[key])))) for key in common),
        default=float("inf"),
    )
    candidate_updates = sum(int(float(row.get("lrnode_update_called", "0") or 0)) for row in candidate_steps.values())
    candidate_full = all(int(float(row.get("full_forward_called", "0") or 0)) == 1 for row in candidate_steps.values())
    ensemble_flags = all(
        baseline_steps[key].get("temporal_ensemble_enabled") == candidate_steps[key].get("temporal_ensemble_enabled") == "1"
        for key in common
    )
    ensemble_counts = all(
        baseline_steps[key].get("ensemble_candidate_count") == candidate_steps[key].get("ensemble_candidate_count")
        for key in common
    )
    checks = {
        "episode_keys_identical": episode_keys_equal,
        "episode_outcomes_identical": episode_keys_equal and baseline_outcomes == candidate_outcomes,
        "action_keys_identical": action_keys_equal and bool(common),
        "executed_action_max_abs_difference_le_1e6": maximum <= args.tolerance,
        "updater_calls_zero": candidate_updates == 0,
        "candidate_uses_full_action_path_every_step": candidate_full,
        "temporal_ensemble_enabled_both": ensemble_flags,
        "temporal_ensemble_candidate_counts_identical": ensemble_counts,
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "verdict": "K1_PARITY_PASS" if passed else "K1_PARITY_FAIL",
        "checks": checks,
        "episode_count": len(baseline_outcomes),
        "paired_action_steps": len(common),
        "max_executed_action_difference": maximum,
        "raw_action_comparison": {
            "available_in_canonical_step_log": False,
            "reason": "canonical source logs post-ensemble executed action only; source was not modified for this gate",
        },
        "candidate_updater_calls": candidate_updates,
        "hidden_rng_shift_detected": not (checks["episode_outcomes_identical"] and checks["executed_action_max_abs_difference_le_1e6"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["verdict"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
