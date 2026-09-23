#!/usr/bin/env python3
"""Paired V1 fixed-K4 non-inferiority and efficiency gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def find_one(root: Path, name: str) -> Path:
    found = sorted(root.rglob(name))
    if len(found) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(found)}")
    return found[0]


def outcomes(root: Path) -> dict[tuple[int, int, int], int]:
    with find_one(root, "eval_episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return {(int(r["task_id"]), int(r["episode_id"]), int(r["seed"])): int(float(r["success"])) for r in csv.DictReader(handle)}


def task_rates(values: dict[tuple[int, int, int], int]) -> dict[int, float]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for key, value in values.items():
        grouped[key[0]].append(value)
    return {key: float(np.mean(rows)) for key, rows in grouped.items()}


def ci_low(base: dict[tuple[int, int, int], int], ours: dict[tuple[int, int, int], int]) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for key in base:
        grouped[key[0]].append(float(ours[key] - base[key]))
    tasks = sorted(grouped)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(10000):
        means = []
        for task in rng.choice(tasks, size=len(tasks), replace=True):
            values = np.asarray(grouped[int(task)])
            means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        samples.append(float(np.mean(means)))
    return float(np.quantile(samples, 0.025) * 100.0)


def nested(data: dict, *paths: tuple[str, ...], default: float = 0.0) -> float:
    for path in paths:
        value = data
        try:
            for key in path:
                value = value[key]
            return float(value)
        except (KeyError, TypeError):
            pass
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    v0, v1 = outcomes(args.v0_root), outcomes(args.v1_root)
    if set(v0) != set(v1) or len(v0) != 200:
        raise RuntimeError("V0 and V1 must use the exact same 200 episode keys")
    v0_summary = json.loads(find_one(args.v0_root, "eval_summary.json").read_text())
    v1_summary = json.loads(find_one(args.v1_root, "eval_summary.json").read_text())
    v0_task, v1_task = task_rates(v0), task_rates(v1)
    regressions = [task for task in v0_task if v0_task[task] - v1_task[task] > 0.20 + 1e-12]
    reduction = nested(v1_summary, ("query_reduction", "full_query_reduction_ratio"))
    v0_latency = nested(v0_summary, ("lrnode", "avg_policy_step_latency_sec"), ("avg_policy_step_latency_sec",))
    v1_latency = nested(v1_summary, ("lrnode", "avg_policy_step_latency_sec"), ("avg_policy_step_latency_sec",))
    v0_gripper = nested(v0_summary, ("action_smoothness", "gripper_switch_rate"))
    v1_gripper = nested(v1_summary, ("action_smoothness", "gripper_switch_rate"))
    v0_second = nested(v0_summary, ("action_smoothness", "action_jerk_l2_p95"))
    v1_second = nested(v1_summary, ("action_smoothness", "action_jerk_l2_p95"))
    checks = {
        "paired_noninferiority_ci_low_above_minus_2pp": ci_low(v0, v1) > -2.0,
        "query_reduction_74_to_76pct": 0.74 <= reduction <= 0.76,
        "policy_latency_within_10pct": v0_latency > 0 and v1_latency <= 1.10 * v0_latency,
        "at_most_two_large_task_regressions": len(regressions) <= 2,
        "no_gripper_catastrophe": v1_gripper <= max(2.0 * v0_gripper, v0_gripper + 0.05),
        "no_normalized_second_difference_catastrophe": v1_second <= max(2.0 * v0_second, v0_second + 1e-6),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "verdict": "V1_ONLINE_PASS" if passed else "V1_ONLINE_FAIL",
        "checks": checks,
        "v0_successes": sum(v0.values()),
        "v1_successes": sum(v1.values()),
        "paired_ci_low_pp": ci_low(v0, v1),
        "tasks_regressing_over_20pp": regressions,
        "full_query_reduction": reduction,
        "mean_policy_latency_ratio": v1_latency / v0_latency if v0_latency else None,
        "terminology": {"action_jerk_l2_p95": "normalized action-space second difference; not physical jerk"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(payload["verdict"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
