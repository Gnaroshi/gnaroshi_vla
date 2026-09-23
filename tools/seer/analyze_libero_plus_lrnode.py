#!/usr/bin/env python3
"""Analyze paired LIBERO-Plus baseline/K=1/K=4 Seer results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROW_NAMES = ("baseline_k1", "ours_k1", "ours_k4")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern!r} under {root}, found {len(matches)}")
    return matches[0]


def load_row(result_root: Path, name: str) -> dict[str, Any]:
    root = result_root / name
    summary_path = _find_one(root, "analysis/eval_summary.json")
    task_path = _find_one(root, "analysis/libero_plus_task_results.csv")
    category_path = _find_one(root, "analysis/libero_plus_category_results.csv")
    return {
        "name": name,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "tasks": _read_csv(task_path),
        "categories": _read_csv(category_path),
    }


def outcome_map(row: Mapping[str, Any]) -> dict[tuple[int, str], int]:
    return {
        (int(item["task_id"]), item["task_name"]): int(item["result"])
        for item in row["tasks"]
    }


def paired_outcomes(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_map = outcome_map(left)
    right_map = outcome_map(right)
    keys_match = set(left_map) == set(right_map)
    shared = sorted(set(left_map) & set(right_map))
    valid = [key for key in shared if left_map[key] in (0, 1) and right_map[key] in (0, 1)]
    fail_to_success = sum(left_map[key] == 0 and right_map[key] == 1 for key in valid)
    success_to_fail = sum(left_map[key] == 1 and right_map[key] == 0 for key in valid)
    left_success = sum(left_map[key] == 1 for key in valid)
    right_success = sum(right_map[key] == 1 for key in valid)
    return {
        "left": left["name"],
        "right": right["name"],
        "keys_match": keys_match,
        "shared_tasks": len(shared),
        "paired_valid_tasks": len(valid),
        "left_skips": sum(value not in (0, 1) for value in left_map.values()),
        "right_skips": sum(value not in (0, 1) for value in right_map.values()),
        "both_success": sum(left_map[key] == 1 and right_map[key] == 1 for key in valid),
        "both_fail": sum(left_map[key] == 0 and right_map[key] == 0 for key in valid),
        "fail_to_success": fail_to_success,
        "success_to_fail": success_to_fail,
        "net_flip": fail_to_success - success_to_fail,
        "left_success_rate": float(left_success) / len(valid) if valid else 0.0,
        "right_success_rate": float(right_success) / len(valid) if valid else 0.0,
        "success_rate_delta": float(right_success - left_success) / len(valid) if valid else 0.0,
        "exact_outcome_parity": keys_match and all(left_map[key] == right_map[key] for key in shared),
    }


def row_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    summary = row["summary"]
    query = summary.get("query_reduction", {})
    lrnode = summary.get("lrnode", {})
    plus = summary.get("libero_plus", {})
    return {
        "row": row["name"],
        "success_rate": float(summary.get("success_rate", 0.0)),
        "evaluated_tasks": int(plus.get("evaluated_task_count", len(row["tasks"]))),
        "valid_tasks": sum(int(item["result"]) in (0, 1) for item in row["tasks"]),
        "skipped_tasks": sum(int(item["result"]) not in (0, 1) for item in row["tasks"]),
        "full_forward_calls": int(query.get("num_full_forward_calls", 0)),
        "lrnode_update_calls": int(query.get("num_lrnode_update_calls", 0)),
        "full_query_reduction_ratio": float(query.get("full_query_reduction_ratio", 0.0)),
        "effective_query_interval": float(query.get("effective_query_interval", 0.0)),
        "avg_full_forward_ms": 1000.0 * float(lrnode.get("avg_full_forward_latency_sec", 0.0)),
        "avg_lrnode_ms": 1000.0 * float(lrnode.get("avg_lrnode_latency_sec", 0.0)),
        "avg_policy_step_ms": 1000.0 * float(lrnode.get("avg_policy_step_latency_sec", 0.0)),
        "renderer": summary.get("renderer_backend", {}).get("effective_backend"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def analyze(result_root: Path) -> dict[str, Any]:
    rows = {name: load_row(result_root, name) for name in ROW_NAMES}
    return {
        "schema_version": 1,
        "benchmark": "LIBERO-Plus",
        "suite": "libero_10",
        "rows": [row_metrics(rows[name]) for name in ROW_NAMES],
        "baseline_vs_ours_k1": paired_outcomes(rows["baseline_k1"], rows["ours_k1"]),
        "baseline_vs_ours_k4": paired_outcomes(rows["baseline_k1"], rows["ours_k4"]),
    }


def write_report(result_root: Path, payload: Mapping[str, Any]) -> None:
    output_dir = result_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "libero_plus_lrnode_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "libero_plus_lrnode_rows.csv", payload["rows"])
    rows = {item["row"]: item for item in payload["rows"]}
    k1 = payload["baseline_vs_ours_k1"]
    k4 = payload["baseline_vs_ours_k4"]
    lines = [
        "# Seer/LR-NODE LIBERO-Plus libero_10 report",
        "",
        "All rows use the same ordered task set, one initial state per task, and no Plus training.",
        "",
        "| Row | SR | Valid / skipped | Full calls | LR-NODE calls | Full-query reduction | Policy ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ROW_NAMES:
        item = rows[name]
        lines.append(
            f"| {name} | {100 * item['success_rate']:.2f}% | "
            f"{item['valid_tasks']} / {item['skipped_tasks']} | "
            f"{item['full_forward_calls']} | {item['lrnode_update_calls']} | "
            f"{100 * item['full_query_reduction_ratio']:.2f}% | "
            f"{item['avg_policy_step_ms']:.3f} |"
        )
    lines.extend([
        "", "## Paired controls", "",
        f"- Baseline vs ours K=1 exact task-outcome parity: `{k1['exact_outcome_parity']}`.",
        f"- Ours K=4 minus baseline paired SR: `{100 * k4['success_rate_delta']:+.2f} pp`.",
        f"- K=4 flips: fail->success `{k4['fail_to_success']}`, success->fail `{k4['success_to_fail']}`, net `{k4['net_flip']}`.",
        "", "Skipped tasks are reported separately and are not counted as failures.",
    ])
    (output_dir / "libero_plus_lrnode_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--require-k1-parity", action="store_true")
    args = parser.parse_args()
    payload = analyze(args.result_root)
    write_report(args.result_root, payload)
    print(json.dumps(payload, indent=2))
    if args.require_k1_parity and not payload["baseline_vs_ours_k1"]["exact_outcome_parity"]:
        print("[VERIFY][FAIL] Baseline and adapter-loaded K=1 task outcomes differ")
        return 2
    print(f"[VERIFY][OK] analysis saved under {args.result_root / 'analysis'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
