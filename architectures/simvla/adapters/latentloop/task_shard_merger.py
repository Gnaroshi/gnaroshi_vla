"""Merge disjoint-task LatentLoop online shards without duplicating query traces."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any

from methods.latentloop.eval import distribution_summary

from .online_evaluator import _paired_summary
from .source_lock import require_empty_output, sha256_file


PROTOCOL_A_ROWS = (
    "full_k1",
    "old_observation_only_k2",
    "chunk_aware_latentloop_k2",
    "nonrecurrent_condition_k2",
    "action_chunk_correction_k2",
    "hold_condition_k2",
    "no_observation_k2",
    "old_observation_only_k4",
    "chunk_aware_latentloop_k4",
    "nonrecurrent_condition_k4",
    "action_chunk_correction_k4",
    "hold_condition_k4",
    "no_observation_k4",
)
ACTION_DIAGNOSTIC_FIELDS = (
    "translation_second_difference",
    "rotation_second_difference",
    "gripper_switches",
    "chunk_boundary_discontinuity",
    "within_chunk_action_variation",
)
CONFIG_INVARIANTS = (
    "matrix",
    "checkpoint",
    "smolvlm_model_path",
    "norm_stats",
    "suite",
    "execution_horizon",
    "num_trials",
    "max_env_actions",
    "num_wait_steps",
    "flow_steps",
    "client_resize_size",
    "image_size",
    "resolution",
    "seed",
    "action_noise_seed_base",
    "bootstrap_seed",
    "task_order",
    "teacher_tracking",
    "rows",
    "environment_action_gap_by_row",
)
SOURCE_INVARIANTS = (
    "root_commit",
    "simvla_upstream_commit",
    "checkpoint",
    "norm_stats_sha256",
    "conda_env",
    "python",
    "python_executable",
    "torch",
    "torch_cuda",
    "packages",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0"}:
        raise ValueError(f"invalid boolean value in episode CSV: {value!r}")
    return normalized in {"true", "1"}


def _invariant(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: payload.get(field) for field in fields}


def _aggregate_distributions(
    shard_rows: list[tuple[str, dict[str, Any]]],
    section: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    names = sorted(
        {
            name
            for _, row in shard_rows
            for name in row.get(section, {})
        }
    )
    aggregate: dict[str, dict[str, Any]] = {}
    by_shard: dict[str, dict[str, Any]] = {}
    for shard_name, row in shard_rows:
        by_shard[shard_name] = row.get(section, {})
    for name in names:
        count = 0
        weighted_sum = 0.0
        maxima: list[float] = []
        for _, row in shard_rows:
            summary = row.get(section, {}).get(name, {})
            candidate_count = int(summary.get("count", 0) or 0)
            candidate_mean = summary.get("mean")
            candidate_max = summary.get("max")
            count += candidate_count
            if candidate_count and candidate_mean is not None:
                weighted_sum += candidate_count * float(candidate_mean)
            if candidate_max is not None:
                maxima.append(float(candidate_max))
        aggregate[name] = {
            "count": count,
            "mean": weighted_sum / count if count else None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": max(maxima) if maxima else None,
            "quantile_note": "Exact quantiles remain in latency_ms_by_shard.",
        }
    return aggregate, by_shard


def merge_task_shards(
    *,
    shard_dirs: list[str | Path],
    output_dir: str | Path,
    expected_task_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Validate and merge a disjoint union of task-complete online shards."""

    if len(shard_dirs) < 2:
        raise ValueError("at least two task shards are required")
    shards: list[dict[str, Any]] = []
    seen_tasks: set[int] = set()
    reference_config: dict[str, Any] | None = None
    reference_source: dict[str, Any] | None = None
    csv_fields: list[str] | None = None
    merged_episode_rows: list[dict[str, str]] = []

    for raw_path in shard_dirs:
        path = Path(raw_path).expanduser().resolve()
        required = (
            "online_summary.json",
            "eval_config.json",
            "episode_metrics.csv",
            "query_trace.jsonl",
            "source_lock.json",
        )
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete shard {path}: missing {missing}")
        summary = _read_json(path / "online_summary.json")
        config = _read_json(path / "eval_config.json")
        source = _read_json(path / "source_lock.json")
        task_ids = tuple(int(value) for value in config.get("resolved_task_ids", []))
        if not task_ids:
            raise ValueError(f"shard {path} has no resolved_task_ids")
        overlap = seen_tasks & set(task_ids)
        if overlap:
            raise ValueError(f"task IDs overlap across shards: {sorted(overlap)}")
        seen_tasks.update(task_ids)
        if summary.get("matrix") != "protocol_a_screening":
            raise ValueError(f"shard {path} is not protocol_a_screening")
        if int(config.get("execution_horizon", -1)) != 1:
            raise ValueError(f"shard {path} is not execution horizon R=1")
        if tuple(summary.get("task_ids", [])) != task_ids:
            raise ValueError(f"summary/config task IDs disagree in {path}")
        if set(summary.get("rows", {})) != set(PROTOCOL_A_ROWS):
            raise ValueError(f"shard {path} does not contain the fixed 13 Protocol A rows")
        expected_episodes = len(task_ids) * int(config.get("num_trials", 0))
        if int(summary.get("episodes_per_row", -1)) != expected_episodes:
            raise ValueError(f"unexpected episodes_per_row in {path}")

        config_axis = _invariant(config, CONFIG_INVARIANTS)
        source_axis = _invariant(source, SOURCE_INVARIANTS)
        if reference_config is None:
            reference_config = config_axis
            reference_source = source_axis
        elif config_axis != reference_config:
            raise ValueError(f"configuration axis mismatch in {path}")
        elif source_axis != reference_source:
            raise ValueError(f"source axis mismatch in {path}")

        with (path / "episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"missing CSV header in {path}")
            if csv_fields is None:
                csv_fields = list(reader.fieldnames)
            elif list(reader.fieldnames) != csv_fields:
                raise ValueError(f"episode CSV fields differ in {path}")
            shard_episode_rows = list(reader)
        expected_csv_rows = len(PROTOCOL_A_ROWS) * expected_episodes
        if len(shard_episode_rows) != expected_csv_rows:
            raise ValueError(
                f"shard {path} has {len(shard_episode_rows)} episode rows; "
                f"expected {expected_csv_rows}"
            )
        for record in shard_episode_rows:
            if int(record["task_id"]) not in task_ids:
                raise ValueError(f"episode task outside shard assignment in {path}")
        merged_episode_rows.extend(shard_episode_rows)
        shards.append(
            {
                "name": path.name,
                "path": path,
                "task_ids": task_ids,
                "summary": summary,
                "config": config,
                "source": source,
            }
        )

    if seen_tasks != set(expected_task_ids):
        raise ValueError(
            f"task union is {sorted(seen_tasks)}; expected {sorted(expected_task_ids)}"
        )
    if len(set(expected_task_ids)) != len(expected_task_ids):
        raise ValueError("expected task IDs must be unique")
    assert csv_fields is not None
    assert reference_config is not None
    assert reference_source is not None

    expected_pairs = {
        (row, task_id, episode)
        for row in PROTOCOL_A_ROWS
        for task_id in expected_task_ids
        for episode in range(int(reference_config["num_trials"]))
    }
    actual_pairs = {
        (record["row"], int(record["task_id"]), int(record["episode"]))
        for record in merged_episode_rows
    }
    if len(actual_pairs) != len(merged_episode_rows) or actual_pairs != expected_pairs:
        raise ValueError("merged episode rows are not an exact row/task/trial matrix")

    output = require_empty_output(output_dir)
    episode_csv = output / "episode_metrics.csv"
    merged_episode_rows.sort(
        key=lambda record: (
            PROTOCOL_A_ROWS.index(record["row"]),
            expected_task_ids.index(int(record["task_id"])),
            int(record["episode"]),
        )
    )
    with episode_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(merged_episode_rows)

    outcomes: dict[str, dict[tuple[int, int], bool]] = {
        row: {} for row in PROTOCOL_A_ROWS
    }
    diagnostics: dict[str, dict[str, list[float]]] = {
        row: collections.defaultdict(list) for row in PROTOCOL_A_ROWS
    }
    for record in merged_episode_rows:
        row = record["row"]
        key = (int(record["task_id"]), int(record["episode"]))
        outcomes[row][key] = _as_bool(record["success"])
        for field in ACTION_DIAGNOSTIC_FIELDS:
            diagnostics[row][field].append(float(record[field]))

    row_summaries: dict[str, Any] = {}
    for row_name in PROTOCOL_A_ROWS:
        shard_rows = [
            (shard["name"], shard["summary"]["rows"][row_name]) for shard in shards
        ]
        counters: collections.Counter[str] = collections.Counter()
        for _, row in shard_rows:
            counters.update({name: int(value) for name, value in row.get("counters", {}).items()})
        policy_queries = int(counters["num_policy_queries"])
        full_calls = int(counters["num_full_vlm_calls"])
        env_actions = int(counters["num_env_steps"])
        latency_ms, latency_ms_by_shard = _aggregate_distributions(shard_rows, "latency_ms")
        policy_distribution = latency_ms.get("policy_total_ms", {})
        first_row = shard_rows[0][1]
        row_summaries[row_name] = {
            "successes": sum(outcomes[row_name].values()),
            "episodes": len(outcomes[row_name]),
            "success_rate": sum(outcomes[row_name].values()) / len(outcomes[row_name]),
            "task_wise_success": {
                str(task_id): sum(
                    value
                    for (candidate_task, _), value in outcomes[row_name].items()
                    if candidate_task == task_id
                )
                / int(reference_config["num_trials"])
                for task_id in expected_task_ids
            },
            "prediction_horizon": first_row["prediction_horizon"],
            "execution_horizon": first_row["execution_horizon"],
            "full_condition_interval": first_row["full_condition_interval"],
            "full_condition_environment_action_gap": first_row[
                "full_condition_environment_action_gap"
            ],
            "mean_executed_chunk_length": env_actions / max(policy_queries, 1),
            "full_condition_reduction_per_policy_query": 1.0
            - full_calls / max(policy_queries, 1),
            "full_condition_reduction_per_environment_step": 1.0
            - full_calls
            / max(env_actions / float(first_row["execution_horizon"]), 1.0),
            "full_condition_calls_per_environment_step": full_calls / max(env_actions, 1),
            "amortized_policy_ms_per_environment_action": policy_distribution.get("mean"),
            "amortized_policy_ms_per_environment_action_distribution": policy_distribution,
            "counters": dict(counters),
            "latency_ms": latency_ms,
            "latency_ms_by_shard": latency_ms_by_shard,
            "action_diagnostics": {
                name: distribution_summary(values)
                for name, values in diagnostics[row_name].items()
            },
            "condition_action_tracking_by_query_age": {},
            "condition_action_tracking_by_query_age_by_shard": {
                shard_name: row.get("condition_action_tracking_by_query_age", {})
                for shard_name, row in shard_rows
            },
            "teacher_tracking_enabled": all(
                bool(row.get("teacher_tracking_enabled")) for _, row in shard_rows
            ),
            "teacher_tracking_excluded_from_operational_latency": True,
        }

    full_outcomes = outcomes["full_k1"]
    paired_vs_full = {
        row: _paired_summary(full_outcomes, outcomes[row], seed=int(reference_config["bootstrap_seed"]))
        for row in PROTOCOL_A_ROWS
        if row != "full_k1"
    }
    paired_between_rows: dict[str, dict[str, Any]] = {}
    for candidate in ("chunk_aware_latentloop_k2", "chunk_aware_latentloop_k4"):
        paired_between_rows[candidate] = {
            baseline: _paired_summary(
                outcomes[baseline],
                outcomes[candidate],
                seed=int(reference_config["bootstrap_seed"]),
            )
            for baseline in PROTOCOL_A_ROWS
            if baseline != candidate
        }

    query_manifest = {
        "storage_policy": "Query traces remain in each immutable shard; they are not duplicated.",
        "shards": [
            {
                "name": shard["name"],
                "task_ids": list(shard["task_ids"]),
                "query_trace_jsonl": str(shard["path"] / "query_trace.jsonl"),
                "query_trace_sha256": sha256_file(shard["path"] / "query_trace.jsonl"),
                "query_trace_size_bytes": (shard["path"] / "query_trace.jsonl").stat().st_size,
            }
            for shard in shards
        ],
    }
    query_manifest_path = output / "query_trace_shards.json"
    _write_json(query_manifest_path, query_manifest)

    summary = {
        "matrix": "protocol_a_screening",
        "suite": reference_config["suite"],
        "task_ids": list(expected_task_ids),
        "episodes_per_row": len(expected_task_ids) * int(reference_config["num_trials"]),
        "rows": row_summaries,
        "paired_vs_full": paired_vs_full,
        "paired_between_rows": paired_between_rows,
        "k1_parity": None,
        "episode_metrics_csv": str(episode_csv),
        "query_trace_jsonl": None,
        "query_trace_shards_json": str(query_manifest_path),
        "merge_notes": {
            "task_shards_are_disjoint": True,
            "task_union_complete": True,
            "exact_episode_matrix_validated": True,
            "latency_quantiles": (
                "Per-shard quantiles are retained exactly. Only count/weighted mean/max are "
                "combined because raw latency samples are not duplicated."
            ),
            "videos": "Video paths remain in episode_metrics.csv and point into shard outputs.",
        },
    }
    _write_json(output / "online_summary.json", summary)
    _write_json(
        output / "eval_config.json",
        {
            **reference_config,
            "output": str(output),
            "task_ids": list(expected_task_ids),
            "resolved_task_ids": list(expected_task_ids),
            "task_shards": [str(shard["path"]) for shard in shards],
        },
    )
    shard_manifest = {
        "expected_task_ids": list(expected_task_ids),
        "rows": list(PROTOCOL_A_ROWS),
        "episodes_per_row": summary["episodes_per_row"],
        "shards": [
            {
                "name": shard["name"],
                "path": str(shard["path"]),
                "task_ids": list(shard["task_ids"]),
                "online_summary_sha256": sha256_file(shard["path"] / "online_summary.json"),
                "episode_metrics_sha256": sha256_file(shard["path"] / "episode_metrics.csv"),
                "source_lock_sha256": sha256_file(shard["path"] / "source_lock.json"),
            }
            for shard in shards
        ],
    }
    _write_json(output / "shard_manifest.json", shard_manifest)
    _write_json(
        output / "source_lock.json",
        {
            "merge_type": "disjoint_task_union_v1",
            "source_invariants": reference_source,
            "shard_source_locks": [
                {
                    "path": str(shard["path"] / "source_lock.json"),
                    "sha256": sha256_file(shard["path"] / "source_lock.json"),
                }
                for shard in shards
            ],
        },
    )
    return summary


def _parse_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-task-ids", type=_parse_ids, default=tuple(range(9, -1, -1)))
    args = parser.parse_args()
    summary = merge_task_shards(
        shard_dirs=args.shard,
        output_dir=args.output,
        expected_task_ids=args.expected_task_ids,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "rows": len(summary["rows"]),
                "episodes_per_row": summary["episodes_per_row"],
                "task_ids": summary["task_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
