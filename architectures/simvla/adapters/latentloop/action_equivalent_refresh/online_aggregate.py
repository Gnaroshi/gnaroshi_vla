"""Merge selective-refresh shards and compare identical Long-500 episodes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from architectures.simvla.adapters.latentloop.action_equivalent_refresh.online_evaluator import (
    EPISODE_SCHEMA,
    ROW,
    SHARD_VERDICT,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    atomic_write_json,
    exact_mcnemar,
    load_json,
    validate_manifest_identity,
)


FINAL_VERDICT = "ACTION_EQUIVALENT_REFRESH_ONLINE_COMPLETE"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _int(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name, 0)
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no", "", "none"}:
        return 0
    return int(float(text))


def _float(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if value in {None, "", "None"}:
        return float("nan")
    return float(value)


def _episode_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return _int(row, "task_id"), _int(row, "trial_id")


def _expected_episode_keys(manifest: Mapping[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(item["task_id"]), int(item["trial_id"]))
        for item in manifest["episodes"]
    }


def _summarize(row_name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    queries = sum(_int(row, "num_policy_queries") for row in rows)
    full_vlm = sum(_int(row, "num_full_vlm_calls") for row in rows)
    condition = sum(_int(row, "num_condition_updater_calls") for row in rows)
    full_action = sum(
        _int(
            row,
            (
                "num_full_action_transformer_evaluations"
                if row.get("num_full_action_transformer_evaluations") not in {None, ""}
                else "num_action_transformer_flow_iterations"
            ),
        )
        for row in rows
    )
    generation = sum(
        _int(
            row,
            (
                "num_generation_loop_updates"
                if row.get("num_generation_loop_updates") not in {None, ""}
                else "num_generation_decoder_only_steps"
            ),
        )
        for row in rows
    )
    actions = sum(_int(row, "episode_length") for row in rows)
    policy_seconds = sum(
        (
            _float(row, "policy_wall_time_seconds")
            if row.get("policy_wall_time_seconds") not in {None, ""}
            else _float(row, "latency_per_executed_action_ms")
            * _int(row, "episode_length")
            / 1000.0
        )
        for row in rows
    )
    success = sum(_int(row, "success") for row in rows)
    return {
        "row": row_name,
        "episodes": len(rows),
        "successes": success,
        "success_rate": float(success / max(1, len(rows))),
        "total_policy_queries": queries,
        "total_full_vlm_calls": full_vlm,
        "total_condition_updater_calls": condition,
        "total_full_action_transformer_evaluations": full_action,
        "total_generation_loop_updates": generation,
        "total_integration_updates": full_action + generation,
        "total_executed_actions": actions,
        "observed_exact_fraction": float(full_vlm / max(1, queries)),
        "effective_k_c": float(queries / max(1, full_vlm)),
        "full_action_transformer_evaluations_per_query": float(
            full_action / max(1, queries)
        ),
        "generation_loop_updates_per_query": float(generation / max(1, queries)),
        "latency_per_policy_query_ms": float(
            policy_seconds * 1000.0 / max(1, queries)
        ),
        "latency_per_executed_action_ms": float(
            policy_seconds * 1000.0 / max(1, actions)
        ),
        "policy_wall_time_seconds": float(policy_seconds),
    }


def _load_control(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_keys: set[tuple[int, int]],
    accepted_rows: set[str],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    summary = load_json(root / "row_summary.json")
    row_name = str(summary.get("row"))
    if row_name not in accepted_rows:
        raise RuntimeError(f"unexpected control row {row_name} under {root}")
    if summary.get("manifest_sha256") != expected_manifest_sha256:
        raise RuntimeError(f"control manifest mismatch under {root}")
    direct = root / "episode_metrics.csv"
    csv_paths = [direct] if direct.is_file() else sorted(
        root.glob("shard_rank*_tasks_*/episode_metrics.csv")
    )
    if not csv_paths:
        raise RuntimeError(f"control episode table is missing under {root}")
    rows = [row for path in csv_paths for row in _read_csv(path)]
    keys = [_episode_key(row) for row in rows]
    if len(rows) != 500 or len(set(keys)) != 500 or set(keys) != expected_keys:
        raise RuntimeError(f"control is not the identical Long-500 axis: {root}")
    return row_name, rows, summary


def _paired_comparison(
    candidate: Mapping[tuple[int, int], Mapping[str, Any]],
    control: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    if set(candidate) != set(control):
        raise RuntimeError("paired comparison episode identities differ")
    both = candidate_only = control_only = neither = 0
    for key in candidate:
        left = _int(candidate[key], "success")
        right = _int(control[key], "success")
        both += int(left == 1 and right == 1)
        candidate_only += int(left == 1 and right == 0)
        control_only += int(left == 0 and right == 1)
        neither += int(left == 0 and right == 0)
    return {
        "both_success": both,
        "candidate_only_success": candidate_only,
        "control_only_success": control_only,
        "both_failure": neither,
        "candidate_minus_control_successes": candidate_only - control_only,
        "mcnemar_exact_two_sided_p": exact_mcnemar(control_only, candidate_only),
    }


def _relative_reduction(candidate: float, control: float) -> float:
    return float(1.0 - candidate / control) if control else 0.0


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest)
    report = validate_manifest_identity(
        manifest, expected_manifest_sha256=args.expected_manifest_sha256
    )
    if report["verdict"] != "EPISODE_MANIFEST_PASS":
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    expected_keys = _expected_episode_keys(manifest)
    if len(expected_keys) != 500:
        raise RuntimeError("immutable manifest does not contain 500 unique episodes")

    payloads: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    task_owners: dict[int, int] = {}
    for shard_index, raw in enumerate(args.shards):
        shard = Path(raw).expanduser().resolve()
        summary = load_json(shard / "shard_summary.json")
        if summary.get("verdict") != SHARD_VERDICT:
            raise RuntimeError(f"incomplete shard: {shard}")
        if summary.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError(f"shard manifest mismatch: {shard}")
        source_hashes.add(str(summary.get("source_combined_sha256")))
        for task_id in summary["task_ids"]:
            task = int(task_id)
            if task in task_owners:
                raise RuntimeError(f"task {task} appears in multiple shards")
            task_owners[task] = shard_index
        shard_summaries.append(summary)
        for path in sorted((shard / "episodes").glob("*.json")):
            payload = load_json(path)
            if payload.get("schema_version") != EPISODE_SCHEMA:
                raise RuntimeError(f"episode schema mismatch: {path}")
            if payload.get("verdict") != "ACTION_EQUIVALENT_REFRESH_EPISODE_COMPLETE":
                raise RuntimeError(f"episode is incomplete: {path}")
            if payload.get("manifest_sha256") != manifest["manifest_sha256"]:
                raise RuntimeError(f"episode manifest mismatch: {path}")
            if payload.get("metrics", {}).get("counter_gate") != (
                "ACTION_EQUIVALENT_REFRESH_COUNTER_PASS"
            ):
                raise RuntimeError(f"episode counter gate failed: {path}")
            payloads.append(payload)
    if set(task_owners) != set(range(10)):
        raise RuntimeError(f"shards do not cover tasks 0..9 exactly: {task_owners}")
    if len(source_hashes) != 1:
        raise RuntimeError("shards were not produced from one locked source")

    candidate_rows = [dict(payload["metrics"]) for payload in payloads]
    candidate_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in candidate_rows:
        key = _episode_key(row)
        if key in candidate_by_key:
            raise RuntimeError(f"duplicate candidate episode: {key}")
        candidate_by_key[key] = row
    if set(candidate_by_key) != expected_keys:
        missing = sorted(expected_keys - set(candidate_by_key))
        extra = sorted(set(candidate_by_key) - expected_keys)
        raise RuntimeError(
            f"candidate does not exactly match Long-500; missing={missing[:5]} extra={extra[:5]}"
        )
    candidate_rows.sort(key=_episode_key)
    decisions: list[dict[str, Any]] = []
    for payload in sorted(payloads, key=lambda item: _episode_key(item["metrics"])):
        base = {
            "task_id": _int(payload["metrics"], "task_id"),
            "trial_id": _int(payload["metrics"], "trial_id"),
        }
        decisions.extend({**base, **decision} for decision in payload["route_decisions"])
    _write_csv_atomic(output / "episode_metrics.csv", candidate_rows)
    _write_jsonl_atomic(output / "route_decisions.jsonl", decisions)

    controls = {
        "full_nfe10": _load_control(
            Path(args.full_control).expanduser().resolve(),
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_keys=expected_keys,
            accepted_rows={"full_nfe10", "baseline_k1"},
        ),
        "generation_ng3": _load_control(
            Path(args.generation_control).expanduser().resolve(),
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_keys=expected_keys,
            accepted_rows={"generation_ng3"},
        ),
        "periodic_kc3_ng3": _load_control(
            Path(args.periodic_kc3_control).expanduser().resolve(),
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_keys=expected_keys,
            accepted_rows={"condition_kc3_ng3"},
        ),
    }
    if args.periodic_kc4_control:
        controls["periodic_kc4_ng3"] = _load_control(
            Path(args.periodic_kc4_control).expanduser().resolve(),
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_keys=expected_keys,
            accepted_rows={"condition_kc4_ng3"},
        )

    candidate_summary = _summarize(ROW, candidate_rows)
    control_summaries: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    efficiency: dict[str, Any] = {}
    for label, (recorded_name, rows, recorded_summary) in controls.items():
        summary = _summarize(recorded_name, rows)
        control_summaries[label] = {
            **summary,
            "recorded_summary": recorded_summary,
        }
        by_key = {_episode_key(row): row for row in rows}
        paired[label] = _paired_comparison(candidate_by_key, by_key)
        efficiency[label] = {
            "full_vlm_call_reduction": _relative_reduction(
                candidate_summary["total_full_vlm_calls"],
                summary["total_full_vlm_calls"],
            ),
            "full_action_transformer_evaluation_reduction": _relative_reduction(
                candidate_summary["total_full_action_transformer_evaluations"],
                summary["total_full_action_transformer_evaluations"],
            ),
            "latency_per_action_reduction": _relative_reduction(
                candidate_summary["latency_per_executed_action_ms"],
                summary["latency_per_executed_action_ms"],
            ),
            "success_rate_delta": float(
                candidate_summary["success_rate"] - summary["success_rate"]
            ),
        }

    task_rows: list[dict[str, Any]] = []
    for task_id in range(10):
        candidate_task = [
            row for row in candidate_rows if _int(row, "task_id") == task_id
        ]
        item: dict[str, Any] = {
            "task_id": task_id,
            "candidate_successes": sum(_int(row, "success") for row in candidate_task),
            "candidate_trials": len(candidate_task),
            "candidate_success_rate": float(
                np.mean([_int(row, "success") for row in candidate_task])
            ),
        }
        for label, (_, rows, _) in controls.items():
            task_control = [row for row in rows if _int(row, "task_id") == task_id]
            item[f"{label}_successes"] = sum(
                _int(row, "success") for row in task_control
            )
            item[f"{label}_success_rate"] = float(
                np.mean([_int(row, "success") for row in task_control])
            )
        task_rows.append(item)
    _write_csv_atomic(output / "per_task_comparison.csv", task_rows)

    comparison = {
        "verdict": FINAL_VERDICT,
        "classification": args.classification,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_combined_sha256": next(iter(source_hashes)),
        "candidate": candidate_summary,
        "controls": control_summaries,
        "paired_outcomes": paired,
        "efficiency": efficiency,
        "matched_budget": {
            "declared_target_exact_fraction": 1.0 / 3.0,
            "candidate_observed_exact_fraction": candidate_summary[
                "observed_exact_fraction"
            ],
            "periodic_kc3_observed_exact_fraction": control_summaries[
                "periodic_kc3_ng3"
            ]["observed_exact_fraction"],
            "candidate_minus_periodic_exact_fraction": float(
                candidate_summary["observed_exact_fraction"]
                - control_summaries["periodic_kc3_ng3"]["observed_exact_fraction"]
            ),
        },
        "shards": shard_summaries,
        "route_decisions": len(decisions),
        "interpretation": (
            "Completion verdict only. Statistical and efficiency values are reported "
            "without an undeclared automatic success threshold."
        ),
    }
    atomic_write_json(output / "online_comparison_summary.json", comparison)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--classification",
        choices=("SD1_HOST_LOCAL_EGL_LONG500", "RB2_HOST_LOCAL_EGL_LONG500"),
        default="SD1_HOST_LOCAL_EGL_LONG500",
    )
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--full-control", required=True)
    parser.add_argument("--generation-control", required=True)
    parser.add_argument("--periodic-kc3-control", required=True)
    parser.add_argument("--periodic-kc4-control", default="")
    return parser


def main() -> int:
    result = aggregate(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
