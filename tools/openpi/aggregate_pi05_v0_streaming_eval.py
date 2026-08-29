#!/usr/bin/env python3
"""Aggregate paired four-suite original/V0 evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ROWS = ("original", "v0")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean {value!r}")


def load_outcomes(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            key = (raw["suite"], int(raw["task_id"]), int(raw["trial"]))
            if key in rows:
                raise RuntimeError(f"duplicate episode {key}")
            rows[key] = {**raw, "success": parse_bool(raw["success"])}
    return rows


def aggregate_efficiency(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    names = set()
    for summary in summaries:
        names.update(summary["efficiency"]["operation_call_matrix"])
    operations = {
        name: sum(
            int(summary["efficiency"]["operation_call_matrix"].get(name, 0))
            for summary in summaries
        )
        for name in sorted(names)
    }
    queries = sum(int(summary["efficiency"]["policy_queries"]) for summary in summaries)
    executed = sum(
        int(summary["efficiency"]["actually_executed_actions"]) for summary in summaries
    )
    full = operations.get("full_prefix_refreshes", 0)
    infer_total_ms = 0.0
    infer_count = 0
    for summary in summaries:
        latency = summary["efficiency"]["latency_ms"]["infer_ms"]
        if latency["mean"] is not None:
            infer_total_ms += float(latency["mean"]) * int(latency["count"])
            infer_count += int(latency["count"])
    return {
        "policy_queries": queries,
        "actually_executed_actions": executed,
        "operation_call_matrix": operations,
        "full_prefix_call_ratio": full / queries if queries else None,
        "actual_mean_k_q": queries / full if full else None,
        "actual_mean_k_a": executed / full if full else None,
        "mean_server_infer_ms_per_query": infer_total_ms / infer_count
        if infer_count
        else None,
        "peak_vram_bytes": max(
            int(summary["efficiency"]["peak_vram_bytes"]) for summary in summaries
        ),
        "sum_suite_elapsed_seconds": sum(
            float(summary["elapsed_seconds"]) for summary in summaries
        ),
    }


def aggregate(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    summaries: dict[str, list[dict[str, Any]]] = {row: [] for row in ROWS}
    outcomes: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {
        row: {} for row in ROWS
    }
    manifest_hashes = set()
    source_lock_ids = set()
    checkpoint_steps = set()
    harness_hashes = set()
    for suite in SUITES:
        preflight = load_json(root / suite / "protocol" / "evaluation_preflight.json")
        checkpoint_steps.add(int(preflight["training"]["checkpoint_step"]))
        harness_hashes.add(preflight["evaluation_harness"]["combined_sha256"])
        for row in ROWS:
            row_root = root / suite / row
            summary = load_json(row_root / "summary.json")
            if not summary.get("complete") or int(summary.get("rollouts", -1)) != 500:
                raise RuntimeError(f"incomplete {suite}/{row} result")
            if summary.get("method_label") != row:
                raise RuntimeError(f"method label mismatch in {suite}/{row}")
            summaries[row].append(summary)
            manifest_hashes.add(summary["final_evaluation_manifest_sha256"])
            source_lock_ids.add(summary["source_lock_id"])
            loaded = load_outcomes(row_root / "episode_outcomes.csv")
            if len(loaded) != 500:
                raise RuntimeError(f"expected 500 outcomes in {suite}/{row}")
            outcomes[row].update(loaded)

    if len(manifest_hashes) != 1 or len(source_lock_ids) != 1:
        raise RuntimeError("suite shards do not share one manifest/source lock")
    if checkpoint_steps != {7000}:
        raise RuntimeError(
            f"primary evaluation did not consistently use best step 7000: {checkpoint_steps}"
        )
    if len(harness_hashes) != 1:
        raise RuntimeError("suite shards used different evaluation harnesses")
    if (
        outcomes["original"].keys() != outcomes["v0"].keys()
        or len(outcomes["v0"]) != 2000
    ):
        raise RuntimeError("paired original/V0 episode identities differ")

    paired_rows = []
    contingency = {
        "original_success_v0_success": 0,
        "original_success_v0_failure": 0,
        "original_failure_v0_success": 0,
        "original_failure_v0_failure": 0,
    }
    for key in sorted(outcomes["original"]):
        original_success = bool(outcomes["original"][key]["success"])
        v0_success = bool(outcomes["v0"][key]["success"])
        label = (
            f"original_{'success' if original_success else 'failure'}_"
            f"v0_{'success' if v0_success else 'failure'}"
        )
        contingency[label] += 1
        paired_rows.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "trial": key[2],
                "original_success": original_success,
                "v0_success": v0_success,
                "success_delta": int(v0_success) - int(original_success),
            }
        )

    per_task_rows = []
    for suite in SUITES:
        for task_id in range(10):
            selected = [
                row
                for row in paired_rows
                if row["suite"] == suite and row["task_id"] == task_id
            ]
            original_successes = sum(int(row["original_success"]) for row in selected)
            v0_successes = sum(int(row["v0_success"]) for row in selected)
            per_task_rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "episodes": len(selected),
                    "original_successes": original_successes,
                    "original_success_rate": original_successes / len(selected),
                    "v0_successes": v0_successes,
                    "v0_success_rate": v0_successes / len(selected),
                    "success_rate_delta": (v0_successes - original_successes)
                    / len(selected),
                }
            )

    row_results = {}
    for row in ROWS:
        row_summaries = summaries[row]
        successes = sum(int(summary["successes"]) for summary in row_summaries)
        row_results[row] = {
            "episodes": 2000,
            "successes": successes,
            "micro_success_rate": successes / 2000,
            "macro_four_suite_success_rate": sum(
                float(summary["success_rate"]) for summary in row_summaries
            )
            / 4,
            "suite_summaries": [
                {
                    "suite": summary["suite"],
                    "successes": int(summary["successes"]),
                    "episodes": int(summary["rollouts"]),
                    "success_rate": float(summary["success_rate"]),
                    "elapsed_seconds": float(summary["elapsed_seconds"]),
                }
                for summary in row_summaries
            ],
            "efficiency": aggregate_efficiency(row_summaries),
        }

    result = {
        "V0_STREAMING_PAIRED_EVALUATION_COMPLETE": True,
        "protocol": {
            "suites": list(SUITES),
            "tasks_per_suite": 10,
            "trials_per_task": 50,
            "episodes_per_row": 2000,
            "seed": 7,
            "execution_horizon_r": 5,
            "action_horizon_h": 10,
            "v0_k_q": 4,
            "v0_k_a": 20,
            "policy_noise": "explicit_query_keyed_sha256_v2",
            "paired_episode_manifest_sha256": next(iter(manifest_hashes)),
            "source_lock_id": next(iter(source_lock_ids)),
            "evaluation_harness_sha256": next(iter(harness_hashes)),
        },
        "checkpoint": {
            "selection": "heldout_checkpoint_validation_minimum",
            "step": 7000,
            "training_budget_steps": 10000,
        },
        "rows": row_results,
        "paired": {
            "contingency": contingency,
            "success_rate_delta_v0_minus_original": (
                row_results["v0"]["micro_success_rate"]
                - row_results["original"]["micro_success_rate"]
            ),
        },
    }

    with (root / "paired_episode_outcomes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    with (root / "per_task_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_task_rows[0]))
        writer.writeheader()
        writer.writerows(per_task_rows)
    (root / "combined_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
