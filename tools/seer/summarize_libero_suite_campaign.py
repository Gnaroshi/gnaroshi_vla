#!/usr/bin/env python3
"""Aggregate baseline/K=1-parity/K=4 results for the cross-suite campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _single_summary(root: Path, label: str) -> tuple[Path, dict]:
    paths = sorted(root.glob("*/analysis/eval_summary.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label} summary under {root}, found {len(paths)}")
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def _row(suite: str, method: str, path: Path, payload: dict) -> dict:
    lrnode = payload.get("lrnode", {})
    query = payload.get("query_reduction", {})
    tasks = payload.get("task_results", [])
    episodes = sum(int(item.get("num_episodes", 0)) for item in tasks)
    return {
        "suite": suite,
        "method": method,
        "success_rate_pct": 100.0 * float(payload.get("success_rate", 0.0)),
        "episodes": episodes,
        "query_interval": int(lrnode.get("query_interval", 1)),
        "full_forward_calls": int(query.get("num_full_forward_calls", 0)),
        "lrnode_update_calls": int(query.get("num_lrnode_update_calls", 0)),
        "full_query_reduction_pct": 100.0
        * float(query.get("full_query_reduction_ratio", 0.0)),
        "avg_policy_step_ms": 1000.0
        * float(lrnode.get("avg_policy_step_latency_sec", 0.0)),
        "avg_full_forward_ms": 1000.0
        * float(lrnode.get("avg_full_forward_latency_sec", 0.0)),
        "avg_lrnode_ms": 1000.0
        * float(lrnode.get("avg_lrnode_latency_sec", 0.0)),
        "summary_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root).resolve()
    rows = []
    missing = []
    for suite in args.suites:
        variants = (
            ("baseline_k1", campaign_root / "eval" / suite / "baseline"),
            ("adapter_loaded_k1", campaign_root / "eval" / suite / "adapter_k1"),
            ("latentloop_k4", campaign_root / "eval" / suite / "latentloop_k4"),
        )
        for method, root in variants:
            try:
                path, payload = _single_summary(root, f"{suite}/{method}")
            except RuntimeError as exc:
                missing.append(str(exc))
                continue
            rows.append(_row(suite, method, path, payload))

    if missing and not args.allow_incomplete:
        raise RuntimeError("\n".join(missing))

    analysis_dir = campaign_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "campaign_root": str(campaign_root),
        "rows": rows,
        "missing": missing,
    }
    (analysis_dir / "suite_matrix_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(rows[0]) if rows else ["suite", "method"]
    with (analysis_dir / "suite_matrix_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("suite            method              SR      episodes  K  query reduction")
    print("---------------- ------------------- ------- --------- -- ---------------")
    for row in rows:
        print(
            f"{row['suite']:<16} {row['method']:<19} "
            f"{row['success_rate_pct']:>6.2f}% {row['episodes']:>9} "
            f"{row['query_interval']:>2} {row['full_query_reduction_pct']:>14.2f}%"
        )
    if missing:
        print(f"[SUMMARY] incomplete rows={len(missing)}")
    print(f"[SUMMARY] {analysis_dir / 'suite_matrix_summary.json'}")


if __name__ == "__main__":
    main()
