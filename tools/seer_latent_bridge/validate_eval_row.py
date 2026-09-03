#!/usr/bin/env python3
"""Fail-closed validation for one distributed LIBERO evaluation row."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def validate_eval_row(
    root: str | Path,
    *,
    seed: int,
    episodes_per_task: int,
    num_tasks: int,
    renderer: str,
) -> dict:
    root = Path(root)
    analysis = root / "analysis"
    summary_path = analysis / "eval_summary.json"
    episodes_path = analysis / "eval_episode_metrics.csv"
    latency_path = analysis / "eval_latency_profile.json"
    for path in (summary_path, episodes_path, latency_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty evaluation artifact: {path}")

    with episodes_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_count = episodes_per_task * num_tasks
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} episode rows, found {len(rows)}")
    identities = [(int(row["task_id"]), int(row["episode_id"])) for row in rows]
    expected = {
        (task_id, episode_id)
        for task_id in range(num_tasks)
        for episode_id in range(episodes_per_task)
    }
    if set(identities) != expected or len(identities) != len(set(identities)):
        raise RuntimeError("evaluation task/episode identities are missing, duplicated, or unexpected")
    if {int(row["seed"]) for row in rows} != {seed}:
        raise RuntimeError("evaluation row contains an unexpected execution seed")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime = summary.get("lrnode", {})
    if runtime.get("renderer_backend") != renderer:
        raise RuntimeError(
            "renderer mismatch: "
            f"expected={renderer}, actual={runtime.get('renderer_backend')!r}"
        )
    renderer_metadata = summary.get("environment", {}).get("renderer", {})
    for field in ("requested_backend", "effective_backend"):
        if renderer_metadata.get(field) != renderer:
            raise RuntimeError(
                f"renderer metadata {field} mismatch: "
                f"expected={renderer}, actual={renderer_metadata.get(field)!r}"
            )
    if renderer == "egl":
        if renderer_metadata.get("actual_context_verified") is not True:
            raise RuntimeError("EGL evaluation lacks a verified active GL context")
        if renderer_metadata.get("software_renderer") is not False:
            raise RuntimeError("EGL evaluation used or failed to exclude a software renderer")
        for field in ("actual_gl_vendor", "actual_gl_renderer", "actual_gl_version"):
            if not renderer_metadata.get(field):
                raise RuntimeError(f"EGL evaluation lacks {field}")
    summary_episodes = sum(int(item.get("num_episodes", 0)) for item in summary["task_results"])
    if summary_episodes != expected_count:
        raise RuntimeError(
            f"summary contains {summary_episodes} episodes, expected {expected_count}"
        )
    successes = sum(int(row["success"]) for row in rows)
    observed_rate = successes / expected_count
    if abs(float(summary["success_rate"]) - observed_rate) > 1e-12:
        raise RuntimeError("summary success rate does not match episode CSV")

    payload = {
        "status": "SEER_LATENT_BRIDGE_EVAL_ROW_COMPLETE",
        "seed": seed,
        "renderer": renderer,
        "num_tasks": num_tasks,
        "episodes_per_task": episodes_per_task,
        "episodes": expected_count,
        "successes": successes,
        "success_rate": observed_rate,
        "renderer_metadata": renderer_metadata,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--renderer", choices=("egl", "osmesa"), required=True)
    parser.add_argument("--write-complete", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    payload = validate_eval_row(
        root,
        seed=args.seed,
        episodes_per_task=args.episodes_per_task,
        num_tasks=args.num_tasks,
        renderer=args.renderer,
    )
    if args.write_complete:
        _atomic_text(root / "EVAL_COMPLETE", json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
