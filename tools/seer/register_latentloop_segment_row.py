#!/usr/bin/env python3
"""Register and validate one completed LatentLoop segment-grid row."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from methods.latentloop_segment_grid.feedback_schedule import build_feedback_plan
from methods.latentloop_segment_grid.serialization import atomic_write_json, read_json


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_step_logs(step_log_dir: Path) -> List[Dict[str, str]]:
    """Read every per-step trace produced for one evaluation row."""

    rows: List[Dict[str, str]] = []
    if not step_log_dir.is_dir():
        return rows
    for path in sorted(step_log_dir.glob("*.csv")):
        rows.extend(_read_csv(path))
    return rows


def _as_int(row: Mapping[str, object], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in (None, ""):
        return int(default)
    return int(float(value))


def validate_segment_step_contract(
    step_rows: Sequence[Mapping[str, object]],
    *,
    segment_length: int,
    feedback_schedule: str,
) -> List[str]:
    """Validate the full-refresh, updater, feedback, and cache-write topology."""

    failures: List[str] = []
    if not step_rows:
        return ["per_step_logs_missing"]
    plan = build_feedback_plan(segment_length, feedback_schedule)
    for row in step_rows:
        timestep = _as_int(row, "timestep", -1)
        offset = _as_int(row, "segment_offset", -1)
        expected_offset = timestep % segment_length
        if offset != expected_offset:
            failures.append("segment_offset_mismatch")
            break
        expected_full = int(offset == 0)
        if _as_int(row, "full_forward_called") != expected_full:
            failures.append("full_forward_schedule_mismatch")
            break
        if _as_int(row, "lrnode_update_called") != 1 - expected_full:
            failures.append("updater_schedule_mismatch")
            break
        if _as_int(row, "observation_cache_advanced") != 1:
            failures.append("observation_cache_not_advanced")
            break
        if expected_full:
            if row.get("feedback_mask", "") not in ("", None):
                failures.append("full_step_has_feedback_mask")
                break
            continue
        expected_feedback = int(plan.mask[offset - 1])
        if _as_int(row, "feedback_mask", -1) != expected_feedback:
            failures.append("feedback_mask_mismatch")
            break
        if _as_int(row, "observation_conditioned_update_called") != expected_feedback:
            failures.append("observation_conditioned_counter_mismatch")
            break
        if _as_int(row, "zero_feature_update_called") != 1 - expected_feedback:
            failures.append("zero_feature_counter_mismatch")
            break
        if expected_feedback == 0 and _as_int(row, "fast_encoder_called") != 0:
            failures.append("masked_step_called_observation_encoder")
            break
    return failures


def _git_state(repo_root: Path) -> Dict[str, object]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit_sha": sha, "dirty": bool(status), "status_porcelain": status}


def build_row_record(args: argparse.Namespace) -> Dict[str, object]:
    """Validate one row directory and return its serialized registry record."""

    row_root = args.row_root.resolve()
    summaries = sorted(row_root.glob("*/analysis/eval_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            f"Expected exactly one eval_summary.json under {row_root}, got {summaries}"
        )
    summary_path = summaries[0]
    analysis_dir = summary_path.parent
    episode_path = analysis_dir / "eval_episode_metrics.csv"
    latency_path = analysis_dir / "eval_latency_profile.json"
    if not episode_path.is_file() or not latency_path.is_file():
        raise FileNotFoundError(
            f"Missing episode/latency artifacts in {analysis_dir}"
        )

    summary = read_json(summary_path)
    episodes = _read_csv(episode_path)
    expected_episodes = int(args.num_tasks) * int(args.episodes_per_task)
    if len(episodes) != expected_episodes:
        raise RuntimeError(
            f"Episode count mismatch for {args.row_id}: "
            f"expected={expected_episodes}, actual={len(episodes)}"
        )
    renderer = summary.get("renderer_backend", {})
    if args.require_osmesa:
        if renderer.get("effective_backend") != "osmesa":
            raise RuntimeError(f"Canonical row is not OSMesa: {renderer}")
        if not renderer.get("all_ranks_actual_context_verified", False):
            raise RuntimeError(
                "Canonical row did not verify the actual OSMesa context on every rank"
            )

    length = int(args.segment_length)
    plan = (
        None
        if args.feedback_schedule == "not_applicable"
        else build_feedback_plan(length, args.feedback_schedule)
    )
    lrnode = summary.get("lrnode", {})
    grid = lrnode.get("segment_grid", {})
    query_reduction = summary.get("query_reduction", {})
    step_log_dir = analysis_dir / "eval_step_logs"
    step_rows = _read_step_logs(step_log_dir)
    failures: List[str] = []
    if int(summary.get("lrnode_query_interval", 1)) != length:
        failures.append("query_interval_mismatch")
    if args.baseline_kind == "full_replanning":
        if length != 1 or bool(lrnode.get("eval_skip_full_forward", False)):
            failures.append("full_replanning_contract_failed")
    elif args.baseline_kind == "adapter_k1_parity":
        if length != 1 or bool(lrnode.get("eval_skip_full_forward", False)):
            failures.append("adapter_k1_contract_failed")
    elif args.baseline_kind in {
        "dense_latentloop",
        "alternate_latentloop",
        "no_observation_latent_dynamics",
    }:
        if not bool(grid.get("enabled", False)):
            failures.append("segment_grid_not_enabled")
        if grid.get("feedback_schedule") != args.feedback_schedule:
            failures.append("feedback_schedule_mismatch")
        updater_calls = int(query_reduction.get("num_lrnode_update_calls", 0))
        observed_calls = int(
            grid.get("observation_conditioned_updater_calls", 0)
        )
        zero_calls = int(grid.get("zero_feature_updater_calls", 0))
        if updater_calls != observed_calls + zero_calls:
            failures.append("feedback_counter_partition_failed")
        if args.feedback_schedule == "dense" and zero_calls != 0:
            failures.append("dense_schedule_has_zero_feature_calls")
        if args.feedback_schedule == "none" and observed_calls != 0:
            failures.append("none_schedule_has_observation_calls")
        if any(not row.get("feedback_mask_json", "") for row in episodes):
            failures.append("episode_feedback_mask_missing")
        failures.extend(
            validate_segment_step_contract(
                step_rows,
                segment_length=length,
                feedback_schedule=args.feedback_schedule,
            )
        )
        cache_advances = int(grid.get("observation_cache_advance_calls", 0))
        policy_steps = int(summary.get("num_env_steps", 0))
        if cache_advances != policy_steps:
            failures.append("aggregate_observation_cache_count_mismatch")
        actual_density = grid.get("actual_feedback_density")
        expected_actual_density = (
            float(observed_calls) / float(updater_calls) if updater_calls else None
        )
        if (
            actual_density is None
            or expected_actual_density is None
            or abs(float(actual_density) - expected_actual_density) > 1e-12
        ):
            failures.append("actual_feedback_density_mismatch")
    elif args.baseline_kind == "hold_latent":
        if int(query_reduction.get("num_hold_latent_steps", 0)) <= 0:
            failures.append("hold_latent_steps_missing")
    elif args.baseline_kind == "hold_action":
        if int(query_reduction.get("num_hold_action_steps", 0)) <= 0:
            failures.append("hold_action_steps_missing")
    elif args.baseline_kind == "predicted_horizon_replay":
        if length != int(args.action_pred_steps) + 1:
            failures.append("predicted_horizon_exceeds_canonical_tokens")
        if int(query_reduction.get("num_chunk_token_steps", 0)) <= 0:
            failures.append("predicted_horizon_steps_missing")

    if failures:
        raise RuntimeError(f"Row contract failed for {args.row_id}: {failures}")

    record = {
        "schema_version": 1,
        "row_id": args.row_id,
        "stage": args.stage,
        "checkpoint_id": int(args.checkpoint_id),
        "adapter_id": int(args.adapter_id),
        "checkpoint_profile": args.checkpoint_profile,
        "checkpoint_source": args.checkpoint_source,
        "baseline_checkpoint_path": args.baseline_checkpoint_path,
        "baseline_checkpoint_sha256": args.baseline_checkpoint_sha256,
        "adapter_checkpoint_path": args.adapter_checkpoint_path,
        "adapter_checkpoint_sha256": args.adapter_checkpoint_sha256,
        "segment_length": length,
        "feedback_schedule": args.feedback_schedule,
        "planned_feedback_mask": [] if plan is None else list(plan.mask),
        "planned_feedback_density": None if plan is None else plan.density,
        "baseline_kind": args.baseline_kind,
        "ablation_mode": args.ablation_mode,
        "num_tasks": int(args.num_tasks),
        "episodes_per_task": int(args.episodes_per_task),
        "expected_episodes": expected_episodes,
        "actual_episodes": len(episodes),
        "action_pred_steps": int(args.action_pred_steps),
        "summary_path": str(summary_path),
        "episode_metrics_path": str(episode_path),
        "latency_profile_path": str(latency_path),
        "step_log_dir": str(step_log_dir),
        "result_root": str(row_root),
        "renderer_backend": renderer,
        "validation": {"pass": True, "failures": []},
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--row-root", type=Path, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--checkpoint-id", type=int, required=True)
    parser.add_argument("--adapter-id", type=int, required=True)
    parser.add_argument("--checkpoint-profile", required=True)
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--baseline-checkpoint-path", required=True)
    parser.add_argument("--baseline-checkpoint-sha256", required=True)
    parser.add_argument("--adapter-checkpoint-path", required=True)
    parser.add_argument("--adapter-checkpoint-sha256", required=True)
    parser.add_argument("--segment-length", type=int, required=True)
    parser.add_argument(
        "--feedback-schedule",
        choices=["dense", "alternate", "none", "not_applicable"],
        required=True,
    )
    parser.add_argument("--baseline-kind", required=True)
    parser.add_argument("--ablation-mode", required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--action-pred-steps", type=int, default=3)
    parser.add_argument("--require-osmesa", action="store_true")
    args = parser.parse_args()

    record = build_row_record(args)
    row_path = args.row_root / "segment_grid_row.json"
    atomic_write_json(row_path, record, refuse_overwrite=True)

    campaign_path = args.campaign_root / "segment_grid_campaign.json"
    if campaign_path.exists():
        campaign = read_json(campaign_path)
    else:
        campaign = {
            "schema_version": 1,
            "experiment": "latentloop_segment_length_feedback_density",
            "git_state": _git_state(args.repo_root.resolve()),
            "rows": [],
        }
    if any(row.get("row_id") == args.row_id for row in campaign["rows"]):
        raise RuntimeError(f"Duplicate row_id={args.row_id} in {campaign_path}")
    campaign["rows"].append(record)
    campaign["rows"] = sorted(campaign["rows"], key=lambda row: row["row_id"])
    atomic_write_json(campaign_path, campaign)
    print(f"[VERIFY][OK] row registry: {row_path}")
    print(f"[VERIFY][OK] campaign registry: {campaign_path}")


if __name__ == "__main__":
    main()
