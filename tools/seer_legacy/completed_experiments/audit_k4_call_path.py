#!/usr/bin/env python3
"""Verify the canonical periodic Full/Update K4 runtime path."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def find_one(root: Path, name: str) -> Path:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(paths)}")
    return paths[0]


def integer(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key, "0") or 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audited-episodes", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads(find_one(args.row_root, "eval_summary.json").read_text(encoding="utf-8"))
    episode_csv = find_one(args.row_root, "eval_episode_metrics.csv")
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for path in sorted((episode_csv.parent / "eval_step_logs").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped[(integer(row, "task_id"), integer(row, "episode_id"))].append(row)
    episodes = sorted(grouped)[: args.audited_episodes]
    if len(episodes) != args.audited_episodes:
        raise RuntimeError(f"expected at least {args.audited_episodes} episodes")
    schedule_ok = True
    fresh_feedback_ok = True
    ensemble_ok = True
    evidence = []
    expected_full = [1, 0, 0, 0, 1]
    expected_update = [0, 1, 1, 1, 0]
    for key in episodes:
        ordered = sorted(grouped[key], key=lambda row: integer(row, "timestep"))[:5]
        full = [integer(row, "full_forward_called") for row in ordered]
        update = [integer(row, "lrnode_update_called") for row in ordered]
        fast = [integer(row, "fast_encoder_called") for row in ordered]
        schedule_ok &= full == expected_full and update == expected_update and fast == expected_update
        for row in ordered[1:4]:
            fresh_feedback_ok &= all(
                integer(row, field) == 1
                for field in (
                    "fast_encoder_called",
                    "observation_conditioned_update_called",
                    "observation_cache_advanced",
                    "action_head_called",
                )
            )
        ensemble_ok &= all(integer(row, "temporal_ensemble_enabled") == 1 for row in ordered)
        evidence.append({"episode": key, "full": full, "update": update, "fast_encoder": fast})

    source = (args.repo_root / "architectures/seer/upstream/utils/eval_utils_libero.py").read_text(encoding="utf-8")
    static_inputs_ok = all(
        token in source
        for token in (
            "key_image_primary=self.lrnode_cached_image_primary[:, 0]",
            "key_image_wrist=self.lrnode_cached_image_wrist[:, 0]",
            "cur_image_primary=image_x[:, 0]",
            "cur_image_wrist=gripper[:, 0]",
            "q_key=self.lrnode_cached_state[:, 0]",
            "q_cur=state[:, 0]",
        )
    )
    query = summary.get("query_reduction", {})
    reduction = float(query.get("full_query_reduction_ratio", summary.get("full_query_reduction_ratio", -1)))
    fallback = int(query.get("num_fallback_full_calls", 0))
    full_calls = int(query.get("num_full_forward_calls", summary.get("num_full_forward_calls", 0)))
    updates = int(query.get("num_lrnode_update_calls", summary.get("num_lrnode_update_calls", 0)))
    fast_calls = int(query.get("num_fast_encoder_calls", summary.get("num_fast_encoder_calls", 0)))
    action_calls = int(query.get("num_total_action_head_calls", summary.get("num_total_action_head_calls", 0)))
    env_steps = int(query.get("num_env_steps", summary.get("num_env_steps", 0)))
    checks = {
        "full_update_update_update_schedule": schedule_ok,
        "fresh_observation_update_runtime_markers": fresh_feedback_ok,
        "primary_wrist_proprio_callsite_verified": static_inputs_ok,
        "fallback_full_calls_zero": fallback == 0,
        "full_query_reduction_approximately_75pct": 0.74 <= reduction <= 0.76,
        "exact_call_accounting": env_steps > 0 and full_calls + updates == env_steps and updates == fast_calls and action_calls == env_steps,
        "temporal_ensemble_enabled": ensemble_ok,
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "verdict": "K4_CALL_PATH_PASS" if passed else "K4_CALL_PATH_FAIL",
        "checks": checks,
        "audited_episode_schedules": evidence,
        "counters": {
            "env_steps": env_steps,
            "full_calls": full_calls,
            "update_calls": updates,
            "fast_encoder_calls": fast_calls,
            "action_generator_calls": action_calls,
            "fallback_full_calls": fallback,
            "full_query_reduction": reduction,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["verdict"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
