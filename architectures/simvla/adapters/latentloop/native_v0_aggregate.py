"""Merge two task shards and issue the native V0 LIBERO-Long verdict."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from architectures.simvla.adapters.latentloop.native_v0_runtime import write_json


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def merge_row(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    row_root = Path(args.row_root).expanduser().resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    shards = sorted(row_root.glob("shard_rank*_tasks_*_*"))
    if len(shards) != 2:
        raise RuntimeError(f"expected two task shards, found {len(shards)}")
    summaries = [json.loads((shard / "shard_summary.json").read_text(encoding="utf-8")) for shard in shards]
    if {item["rank"] for item in summaries} != {0, 1}:
        raise RuntimeError("shard ranks must be exactly 0 and 1")
    if any(item["manifest_sha256"] != manifest["manifest_sha256"] for item in summaries):
        raise RuntimeError("shard/manifest hash mismatch")
    parameter_counts = {int(item["v0_module_parameters"]) for item in summaries}
    if len(parameter_counts) != 1:
        raise RuntimeError("shards disagree on V0 module parameter count")
    rows = [row for shard in shards for row in _read_csv(shard / "episode_metrics.csv")]
    expected_keys = {(task, trial) for task in range(10) for trial in range(50)}
    observed_keys = {(int(row["task_id"]), int(row["trial_id"])) for row in rows}
    if len(rows) != 500 or observed_keys != expected_keys:
        raise RuntimeError(f"row is not exact 10x50: rows={len(rows)} keys={len(observed_keys)}")
    row_names = {row["row"] for row in rows}
    if len(row_names) != 1:
        raise RuntimeError(f"merged shards disagree on row name: {row_names}")
    row_name = row_names.pop()
    rows.sort(key=lambda row: (int(row["task_id"]), int(row["trial_id"])))
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    latency_records = []
    query_records = []
    for shard in shards:
        with (shard / "latency_records.jsonl").open(encoding="utf-8") as handle:
            latency_records.extend(json.loads(line) for line in handle if line.strip())
        with (shard / "query_metrics.jsonl").open(encoding="utf-8") as handle:
            query_records.extend(json.loads(line) for line in handle if line.strip())
    query_records.sort(
        key=lambda row: (
            int(row["task_id"]),
            int(row["trial_id"]),
            int(row["policy_query_index"]),
        )
    )
    with (output / "query_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for record in query_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    latency_fields = (
        "synchronized_policy_ms_per_executed_action",
        "VLM_encoder_ms",
        "condition_updater_ms",
        "action_transformer_ms",
    )
    latency_summary = {
        field: _distribution(
            value
            for record in latency_records
            for value in record.get(field, [])
        )
        for field in latency_fields
        if any(record.get(field) for record in latency_records)
    }
    successes = sum(_as_bool(row["success"]) for row in rows)
    per_task = {
        str(task): {
            "successes": sum(
                _as_bool(row["success"]) for row in rows if int(row["task_id"]) == task
            ),
            "episodes": 50,
        }
        for task in range(10)
    }
    for value in per_task.values():
        value["success_rate"] = value["successes"] / 50.0
    summary = {
        "verdict": "LIBERO_LONG_ROW_500_COMPLETE",
        "row": row_name,
        "episodes": 500,
        "successes": successes,
        "success_rate": successes / 500.0,
        "per_task": per_task,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_combined_sha256": manifest["source_combined_sha256"],
        "selected_physical_gpu_ids": manifest["selected_physical_gpu_ids"],
        "v0_module_parameters": parameter_counts.pop(),
        "full_vlm_calls": sum(int(row["num_full_vlm_calls"]) for row in rows),
        "condition_updater_calls": sum(int(row["num_condition_updater_calls"]) for row in rows),
        "action_transformer_flow_iterations": sum(int(row["num_action_transformer_flow_iterations"]) for row in rows),
        "action_transformer_decodes": sum(int(row["num_action_transformer_decodes"]) for row in rows),
        "fallback_full_calls": sum(int(row["fallback_full_calls"]) for row in rows),
        "effective_k": sum(int(row["num_policy_queries"]) for row in rows) / max(sum(int(row["num_full_vlm_calls"]) for row in rows), 1),
        "episode_length": _distribution(float(row["episode_length"]) for row in rows),
        "latency": latency_summary,
        "peak_vram_bytes": max(int(item["peak_vram_bytes"]) for item in summaries),
        "action": {
            field: _distribution(float(row[field]) for row in rows)
            for field in ("normalized_second_difference", "short_reversal", "switch_disagreement")
        },
        "episode_metrics_csv": str(output / "episode_metrics.csv"),
        "query_metrics_jsonl": str(output / "query_metrics.jsonl"),
    }
    write_json(output / "row_summary.json", summary)
    return summary


def _paired_ci(differences: np.ndarray, seed: int, samples: int = 20_000) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(samples, differences.size))
    estimates = differences[indices].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def paired_decision(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.v0_summary).read_text(encoding="utf-8"))
    if baseline["row"] != "baseline_k1" or candidate["row"] != "native_v0_k4":
        raise ValueError("paired decision requires baseline_k1 and native_v0_k4")
    if baseline["manifest_sha256"] != candidate["manifest_sha256"]:
        raise RuntimeError("rows use different episode manifests")
    if baseline["source_combined_sha256"] != candidate["source_combined_sha256"]:
        raise RuntimeError("rows use different source locks")
    baseline_rows = _read_csv(Path(baseline["episode_metrics_csv"]))
    candidate_rows = _read_csv(Path(candidate["episode_metrics_csv"]))
    baseline_map = {(int(row["task_id"]), int(row["trial_id"])): row for row in baseline_rows}
    candidate_map = {(int(row["task_id"]), int(row["trial_id"])): row for row in candidate_rows}
    if set(baseline_map) != set(candidate_map) or len(baseline_map) != 500:
        raise RuntimeError("rows are not paired over the same 500 episodes")
    for key in baseline_map:
        if (
            baseline_map[key]["environment_seed"] != candidate_map[key]["environment_seed"]
            or baseline_map[key]["init_state_index"] != candidate_map[key]["init_state_index"]
        ):
            raise RuntimeError(f"paired environment contract differs at {key}")
    def query_map(summary: dict[str, Any]) -> dict[tuple[int, int, int], int]:
        path = Path(summary["query_metrics_jsonl"])
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return {
            (int(row["task_id"]), int(row["trial_id"]), int(row["policy_query_index"])):
            int(row["action_noise_seed"])
            for row in rows
        }

    baseline_queries = query_map(baseline)
    candidate_queries = query_map(candidate)
    common_query_keys = set(baseline_queries) & set(candidate_queries)
    noise_seed_mismatches = [
        key
        for key in sorted(common_query_keys)
        if baseline_queries[key] != candidate_queries[key]
    ]
    if noise_seed_mismatches:
        raise RuntimeError(
            f"paired explicit flow-noise seeds differ for {len(noise_seed_mismatches)} queries"
        )
    ordered = sorted(baseline_map)
    baseline_success = np.asarray([_as_bool(baseline_map[key]["success"]) for key in ordered], dtype=np.float64)
    candidate_success = np.asarray([_as_bool(candidate_map[key]["success"]) for key in ordered], dtype=np.float64)
    difference = candidate_success - baseline_success
    paired_flips = {
        "baseline_fail_v0_success": int(((baseline_success == 0) & (candidate_success == 1)).sum()),
        "baseline_success_v0_fail": int(((baseline_success == 1) & (candidate_success == 0)).sum()),
        "both_success": int(((baseline_success == 1) & (candidate_success == 1)).sum()),
        "both_fail": int(((baseline_success == 0) & (candidate_success == 0)).sum()),
    }
    task_drops_pp = {
        str(task): 100.0 * (
            baseline["per_task"][str(task)]["success_rate"]
            - candidate["per_task"][str(task)]["success_rate"]
        )
        for task in range(10)
    }
    vlm_reduction = 1.0 - candidate["full_vlm_calls"] / max(baseline["full_vlm_calls"], 1)
    baseline_latency = baseline["latency"]["synchronized_policy_ms_per_executed_action"]["mean"]
    candidate_latency = candidate["latency"]["synchronized_policy_ms_per_executed_action"]["mean"]
    latency_reduction = 1.0 - candidate_latency / max(baseline_latency, 1e-12)
    sr_drop_pp = 100.0 * (baseline["success_rate"] - candidate["success_rate"])
    continuity_limits = {
        "normalized_second_difference": max(
            2.0 * baseline["action"]["normalized_second_difference"]["p95"],
            baseline["action"]["normalized_second_difference"]["p95"] + 0.5,
        ),
        "short_reversal": max(
            2.0 * baseline["action"]["short_reversal"]["p95"],
            baseline["action"]["short_reversal"]["p95"] + 0.1,
        ),
        "switch_disagreement": max(
            2.0 * baseline["action"]["switch_disagreement"]["p95"],
            baseline["action"]["switch_disagreement"]["p95"] + 0.1,
        ),
    }
    continuity_ok = all(
        candidate["action"][name]["p95"] <= limit
        for name, limit in continuity_limits.items()
    )
    checks = {
        "vlm_call_reduction_approximately_75_percent": 0.70 <= vlm_reduction <= 0.80,
        "fallback_full_calls_zero": candidate["fallback_full_calls"] == 0,
        "success_rate_drop_at_most_2pp": sr_drop_pp <= 2.0,
        "no_task_drop_over_10pp": max(task_drops_pp.values()) <= 10.0,
        "policy_latency_reduction_at_least_15_percent": latency_reduction >= 0.15,
        "action_continuity_no_catastrophe": continuity_ok,
        "paired_explicit_flow_noise_matches": not noise_seed_mismatches,
    }
    passed = all(checks.values())
    result = {
        "verdict": "LIBERO_LONG_EXPANSION_PASS" if passed else "LIBERO_LONG_EXPANSION_FAIL",
        "manifest_sha256": baseline["manifest_sha256"],
        "source_combined_sha256": baseline["source_combined_sha256"],
        "baseline": {"successes": baseline["successes"], "episodes": 500, "success_rate": baseline["success_rate"]},
        "native_v0_k4": {"successes": candidate["successes"], "episodes": 500, "success_rate": candidate["success_rate"]},
        "success_rate_difference_v0_minus_baseline_pp": float(100.0 * difference.mean()),
        "paired_95_percent_ci_pp": [100.0 * value for value in _paired_ci(difference, args.bootstrap_seed)],
        "paired_flips": paired_flips,
        "paired_flow_noise": {
            "common_query_count": len(common_query_keys),
            "seed_mismatch_count": len(noise_seed_mismatches),
        },
        "per_task_baseline_minus_v0_pp": task_drops_pp,
        "vlm_call_reduction": vlm_reduction,
        "policy_latency_reduction": latency_reduction,
        "checks": checks,
        "action_continuity_p95_limits": continuity_limits,
        "other_suite_evaluation_enabled": passed,
        "rank96_enabled": False,
        "k2_online_run": False,
        "previous_100_episode_rows_used": False,
    }
    write_json(output / "simvla_v0_long_decision.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge = subparsers.add_parser("merge-row")
    merge.add_argument("--row-root", required=True)
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(handler=merge_row)
    paired = subparsers.add_parser("paired-decision")
    paired.add_argument("--baseline-summary", required=True)
    paired.add_argument("--v0-summary", required=True)
    paired.add_argument("--output", required=True)
    paired.add_argument("--bootstrap-seed", type=int, default=20260822)
    paired.set_defaults(handler=paired_decision)
    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
