#!/usr/bin/env python3
"""V2 target-K4 non-inferiority, budget, and matched-random gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def find_one(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(paths)}")
    return paths[0]


def load_outcomes(root: Path) -> dict[tuple[int, int, int], int]:
    with find_one(root, "eval_episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return {(int(r["task_id"]), int(r["episode_id"]), int(r["seed"])): int(float(r["success"])) for r in csv.DictReader(handle)}


def task_rates(rows: dict[tuple[int, int, int], int]) -> dict[int, float]:
    values: dict[int, list[int]] = defaultdict(list)
    for key, success in rows.items():
        values[key[0]].append(success)
    return {task: float(np.mean(success)) for task, success in values.items()}


def bootstrap_low(base: dict[tuple[int, int, int], int], ours: dict[tuple[int, int, int], int]) -> float:
    values: dict[int, list[float]] = defaultdict(list)
    for key in base:
        values[key[0]].append(float(ours[key] - base[key]))
    tasks = sorted(values)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(10000):
        means = []
        for task in rng.choice(tasks, len(tasks), replace=True):
            row = np.asarray(values[int(task)])
            means.append(float(rng.choice(row, len(row), replace=True).mean()))
        samples.append(float(np.mean(means)))
    return float(np.quantile(samples, 0.025) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    v1, v2, random = map(load_outcomes, (args.v1_root, args.v2_root, args.random_root))
    if set(v1) != set(v2) or set(v2) != set(random) or len(v1) != 200:
        raise RuntimeError("V1, V2, and matched-random rows require identical 200 episode keys")
    s1 = json.loads(find_one(args.v1_root, "eval_summary.json").read_text())
    s2 = json.loads(find_one(args.v2_root, "eval_summary.json").read_text())
    sr = json.loads(find_one(args.random_root, "eval_summary.json").read_text())
    v2_ops, random_ops = s2["v2_scheduler"], sr["v2_scheduler"]
    effective_k = float(v2_ops["effective_k"])
    v1_task, v2_task = task_rates(v1), task_rates(v2)
    regressions = [task for task in v1_task if v1_task[task] - v2_task[task] > 0.20 + 1e-12]
    recoveries = sum(v1[key] == 0 and v2[key] == 1 for key in v1)
    losses = sum(v1[key] == 1 and v2[key] == 0 for key in v1)
    improvements = {
        "success_rate": sum(v2.values()) > sum(v1.values()),
        "paired_failure_recovery": recoveries > losses,
        "p95_reference_action_error_proxy": float(v2_ops["p95_reference_action_error"]) < float(s1["v1_metrics"]["p95_reference_action_error"]),
        "long_interval_stability": float(v2_ops["long_interval_failure_rate"]) < float(s1["v1_metrics"]["long_interval_failure_rate"]),
    }
    checks = {
        "effective_k_3p8_to_4p2": 3.8 <= effective_k <= 4.2,
        "paired_noninferiority_ci_low_above_minus_1pp": bootstrap_low(v1, v2) > -1.0,
        "matched_random_full_call_count": int(v2_ops["full_calls"]) == int(random_ops["full_calls"]),
        "at_least_one_improvement": any(improvements.values()),
        "at_most_two_large_task_regressions": len(regressions) <= 2,
        "no_gripper_catastrophe": not bool(v2_ops["gripper_catastrophe"]),
        "all_scheduler_costs_counted": bool(v2_ops["operation_costs_complete"]),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "verdict": "V2_TARGET_K4_PASS" if passed else "V2_TARGET_K4_FAIL",
        "checks": checks,
        "improvements": improvements,
        "v1_successes": sum(v1.values()),
        "v2_successes": sum(v2.values()),
        "random_successes": sum(random.values()),
        "paired_ci_low_pp": bootstrap_low(v1, v2),
        "paired_flips": {"recoveries": recoveries, "losses": losses},
        "effective_k": effective_k,
        "tasks_regressing_over_20pp": regressions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(payload["verdict"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
