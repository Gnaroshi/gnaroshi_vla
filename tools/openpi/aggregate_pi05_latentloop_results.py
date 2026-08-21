#!/usr/bin/env python3
"""Aggregate a 2,000-episode row without overstating baseline noise pairing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _common import DEFAULT_EVALUATION, refuse_nonempty_output
from methods.variable_time_latentloop.metrics import hierarchical_paired_bootstrap, paired_flip_counts


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--row-name", required=True)
    parser.add_argument("--variant", choices=("hold", "latent_bridge", "v0", "v1", "v2"), required=True)
    parser.add_argument("--baseline-root", default=str(DEFAULT_EVALUATION))
    parser.add_argument("--adapter-parameters", type=int, default=0)
    parser.add_argument("--target-k-q", type=float, default=4.0)
    parser.add_argument("--baseline-episode-wall-seconds", type=float)
    parser.add_argument("--latent-bridge-speedup", type=float)
    args = parser.parse_args()
    candidate_root = Path(args.candidate_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    output = refuse_nonempty_output(args.output)
    comparison_rows = []
    suite_results = []
    efficiency = []

    for suite in SUITES:
        candidate_path = candidate_root / suite / "client" / "episode_outcomes.csv"
        baseline_path = baseline_root / suite / "episode_outcomes.csv"
        candidate = list(csv.DictReader(candidate_path.open(encoding="utf-8")))
        baseline = list(csv.DictReader(baseline_path.open(encoding="utf-8")))
        if len(candidate) != 500 or len(baseline) != 500:
            raise RuntimeError(f"{suite} must contain exactly 500 candidate and baseline episodes")
        baseline_by_key = {(row["task"], int(row["trial"])): row for row in baseline}
        for row in candidate:
            key = (row["task"], int(row["trial"]))
            if key not in baseline_by_key:
                raise RuntimeError(f"candidate episode has no completed-baseline identity match: {suite} {key}")
            reference = baseline_by_key[key]
            comparison_rows.append(
                {
                    "suite": suite,
                    "task": row["task"],
                    "trial": int(row["trial"]),
                    "baseline_success": _bool(reference["success"]),
                    "candidate_success": _bool(row["success"]),
                }
            )
        candidate_successes = sum(_bool(row["success"]) for row in candidate)
        baseline_successes = sum(_bool(row["success"]) for row in baseline)
        suite_results.append(
            {
                "suite": suite,
                "candidate_successes": candidate_successes,
                "baseline_successes": baseline_successes,
                "episodes": 500,
                "candidate_success_rate": candidate_successes / 500,
                "baseline_success_rate": baseline_successes / 500,
                "difference_pp": (candidate_successes - baseline_successes) / 5.0,
            }
        )
        efficiency.append(
            json.loads((candidate_root / suite / "client" / "summary.json").read_text(encoding="utf-8"))[
                "efficiency"
            ]
        )

    baseline_values = np.asarray([row["baseline_success"] for row in comparison_rows], dtype=bool)
    candidate_values = np.asarray([row["candidate_success"] for row in comparison_rows], dtype=bool)
    bootstrap = hierarchical_paired_bootstrap(comparison_rows, samples=10_000, seed=20260820)
    total_queries = sum(item["policy_queries"] for item in efficiency)
    total_full = sum(item["full_prefix_calls"] for item in efficiency)
    total_executed = sum(item["actually_executed_actions"] for item in efficiency)
    candidate_rate = float(np.mean(candidate_values))
    baseline_rate = float(np.mean(baseline_values))
    episode_wall = sum(
        float(row["episode_wall_seconds"])
        for suite in SUITES
        for row in csv.DictReader(
            (candidate_root / suite / "client" / "episode_outcomes.csv").open(encoding="utf-8")
        )
    )
    baseline_speedup = (
        (args.baseline_episode_wall_seconds * 2000.0) / episode_wall
        if args.baseline_episode_wall_seconds is not None
        else None
    )
    successful_episode_lengths = [
        int(row["episode_steps"])
        for suite in SUITES
        for row in csv.DictReader(
            (candidate_root / suite / "client" / "episode_outcomes.csv").open(encoding="utf-8")
        )
        if _bool(row["success"])
    ]
    action_metrics = {}
    for key in (
        "translation_second_difference_mean",
        "rotation_second_difference_mean",
        "gripper_switches",
        "gripper_short_reversals",
    ):
        values = [
            float(row[key])
            for suite in SUITES
            for row in csv.DictReader(
                (candidate_root / suite / "client" / "episode_outcomes.csv").open(encoding="utf-8")
            )
            if row.get(key) not in (None, "", "None")
        ]
        action_metrics[key] = float(np.mean(values)) if values else None
    observed_ratio = total_full / total_queries
    target_ratio = 1.0 / args.target_k_q
    ratio_tolerance = 0.025 if args.target_k_q == 4.0 else max(0.0125, 0.1 * target_ratio)
    criteria = {
        "complete_2000_episode_row": len(comparison_rows) == 2000,
        "full_prefix_ratio_matches_declared_target": abs(observed_ratio - target_ratio) <= ratio_tolerance,
    }
    if args.variant in {"v0", "v1", "v2"}:
        criteria["adapter_parameter_cap"] = args.adapter_parameters <= 19_000_000
    if args.variant == "v0":
        criteria.update(
            {
                "noninferiority_lower_bound_above_minus_1pp": bootstrap["lower_95"] > -0.01,
                "no_suite_drop_over_2pp": min(row["difference_pp"] for row in suite_results) >= -2.0,
                "speedup_at_least_1_5_or_bridge": bool(
                    (baseline_speedup is not None and baseline_speedup >= 1.5)
                    or (args.latent_bridge_speedup is not None and baseline_speedup is not None
                        and baseline_speedup >= args.latent_bridge_speedup)
                ),
            }
        )
    payload = {
        "row_name": args.row_name,
        "variant": args.variant,
        "complete": len(comparison_rows) == 2000,
        "episodes": len(comparison_rows),
        "candidate_success_rate": candidate_rate,
        "completed_local_baseline_success_rate": baseline_rate,
        "difference_pp": 100.0 * (candidate_rate - baseline_rate),
        "suite_results": suite_results,
        "episode_identity_aligned_flips": paired_flip_counts(baseline_values, candidate_values),
        "successful_episode_length": {
            "count": len(successful_episode_lengths),
            "mean": float(np.mean(successful_episode_lengths)),
            "p50": float(np.quantile(successful_episode_lengths, 0.5)),
            "p95": float(np.quantile(successful_episode_lengths, 0.95)),
        },
        "action_metrics": action_metrics,
        "hierarchical_bootstrap": bootstrap,
        "pairing_boundary": (
            "Initial-state episode identities are aligned. The completed baseline did not persist explicit per-query "
            "flow-noise keys, so this is not an exact same-noise paired comparison."
        ),
        "efficiency": {
            "full_prefix_calls": total_full,
            "policy_queries": total_queries,
            "full_prefix_call_ratio": total_full / total_queries,
            "target_k_q": args.target_k_q,
            "target_k_a": 5.0 * args.target_k_q,
            "actual_mean_k_q": total_queries / total_full,
            "actually_executed_actions": total_executed,
            "actual_mean_k_a": total_executed / total_full,
            "candidate_total_episode_wall_seconds": episode_wall,
            "speedup_vs_completed_baseline": baseline_speedup,
            "speedup_limitation": (
                None if baseline_speedup is not None else
                "Completed baseline episode wall times were not persisted; net speedup needs an approved instrumented control or Latent Bridge comparison."
            ),
            "peak_vram_bytes": max(item["peak_vram_bytes"] for item in efficiency),
            "adapter_parameters": args.adapter_parameters,
        },
        "predeclared_criteria": criteria,
        "row_passed_available_criteria": all(criteria.values()),
        "cross_row_criteria_pending": (
            "V1 and V2 non-inferiority, composition, matched-budget scheduler, and comparative speed criteria "
            "require their declared reference rows and are not inferred from one row."
        ),
    }
    with (output / "paired_episode_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    (output / "aggregate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "aggregate_report.md").write_text(
        "\n".join(
            (
                f"# {args.row_name}",
                "",
                f"- Success: `{int(candidate_values.sum())}/2000 ({100 * candidate_rate:.2f}%)`",
                f"- Completed local baseline: `{int(baseline_values.sum())}/2000 ({100 * baseline_rate:.2f}%)`",
                f"- Difference: `{100 * (candidate_rate - baseline_rate):+.2f} pp`",
                f"- Actual K_q: `{total_queries / total_full:.4f}`",
                f"- Actual K_a: `{total_executed / total_full:.4f}`",
                f"- Full-prefix ratio: `{total_full / total_queries:.6f}`",
                f"- Declared target K_q/K_a: `{args.target_k_q:g}/{5 * args.target_k_q:g}`",
                f"- Extra parameters: `{args.adapter_parameters}`",
                "",
                "Episode initial-state keys are aligned, but exact flow-noise pairing to the completed baseline cannot be claimed.",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
