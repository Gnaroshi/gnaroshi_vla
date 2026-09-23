#!/usr/bin/env python3
"""Validate and serialize one canonical V0 evaluation row."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-root", type=Path, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-query-interval", type=int, required=True)
    parser.add_argument("--expected-lrnode-enabled", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary_path = find_one(args.row_root, "eval_summary.json")
    episode_path = find_one(args.row_root, "eval_episode_metrics.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with episode_path.open(newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    if len(episodes) != args.expected_episodes:
        raise RuntimeError(f"expected {args.expected_episodes} episodes, found {len(episodes)}")
    lrnode = summary.get("lrnode", {})
    enabled = int(bool(lrnode.get("enabled", False)))
    interval = int(lrnode.get("query_interval", 1))
    if enabled != args.expected_lrnode_enabled or interval != args.expected_query_interval:
        raise RuntimeError(f"row mode mismatch: enabled={enabled}, K={interval}")
    payload = {
        "schema_version": 1,
        "status": "CANONICAL_EVAL_ROW_COMPLETE",
        "row_id": args.row_id,
        "row_root": str(args.row_root.resolve()),
        "episode_count": len(episodes),
        "successes": sum(int(float(row["success"])) for row in episodes),
        "success_rate": float(summary["success_rate"]),
        "query_interval": interval,
        "lrnode_enabled": bool(enabled),
        "eval_summary": str(summary_path.resolve()),
        "episode_metrics": str(episode_path.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["status"])


if __name__ == "__main__":
    main()
