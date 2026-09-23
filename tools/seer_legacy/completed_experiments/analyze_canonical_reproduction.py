#!/usr/bin/env python3
"""Paired canonical V0 reproduction statistics and frozen verdict inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def find_one(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(paths)}")
    return paths[0]


def load(path: Path) -> dict[tuple[int, int, int], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["task_id"]), int(row["episode_id"]), int(row["seed"])): int(float(row["success"]))
            for row in csv.DictReader(handle)
        }


def task_rates(values: dict[tuple[int, int, int], int]) -> dict[int, float]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for key, value in values.items():
        grouped[key[0]].append(value)
    return {task: float(np.mean(items)) for task, items in grouped.items()}


def hierarchical_ci(base: dict[tuple[int, int, int], int], ours: dict[tuple[int, int, int], int], iterations: int, seed: int) -> tuple[float, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for key in sorted(base):
        grouped[key[0]].append(float(ours[key] - base[key]))
    tasks = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        task_draw = rng.choice(tasks, size=len(tasks), replace=True)
        task_means = []
        for task in task_draw:
            values = np.asarray(grouped[int(task)], dtype=np.float64)
            task_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        samples[index] = float(np.mean(task_means))
    low, high = np.quantile(samples, [0.025, 0.975]) * 100.0
    return float(low), float(high)


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--v0-root", type=Path, required=True)
    parser.add_argument("--canonical-v0-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = load(find_one(args.baseline_root, "eval_episode_metrics.csv"))
    ours = load(find_one(args.v0_root, "eval_episode_metrics.csv"))
    reference = load(args.canonical_v0_episodes)
    if set(base) != set(ours) or set(ours) != set(reference) or len(base) != 200:
        raise RuntimeError("baseline, V0, and canonical reference must share the exact 200 episode keys")
    low, high = hierarchical_ci(base, ours, args.bootstrap_iterations, args.bootstrap_seed)
    base_success = sum(base.values())
    ours_success = sum(ours.values())
    b = sum(base[key] == 1 and ours[key] == 0 for key in base)
    c = sum(base[key] == 0 and ours[key] == 1 for key in base)
    ours_task = task_rates(ours)
    reference_task = task_rates(reference)
    regressions = [task for task in sorted(ours_task) if reference_task[task] - ours_task[task] > 0.20 + 1e-12]
    summary = json.loads(find_one(args.v0_root, "eval_summary.json").read_text(encoding="utf-8"))
    query = summary.get("query_reduction", {})
    reduction = float(query.get("full_query_reduction_ratio", summary.get("full_query_reduction_ratio", -1)))
    payload = {
        "schema_version": 1,
        "episode_keys_identical": True,
        "baseline": {"successes": base_success, "episodes": 200, "sr": base_success / 200},
        "v0": {"successes": ours_success, "episodes": 200, "sr": ours_success / 200, "full_query_reduction": reduction},
        "paired_flips": {"baseline_only_success": b, "v0_only_success": c},
        "paired_task_hierarchical_bootstrap_ci_pp": {"low": low, "high": high, "iterations": args.bootstrap_iterations, "seed": args.bootstrap_seed},
        "exact_mcnemar_p": exact_mcnemar(b, c),
        "tasks_regressing_over_20pp_from_canonical": regressions,
        "task_regression_count": len(regressions),
        "canonical_episode_outcomes_identical": ours == reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
