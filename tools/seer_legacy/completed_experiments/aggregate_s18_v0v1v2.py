#!/usr/bin/env python3
"""Aggregate only completed, provenance-bearing V0/V1/V2 gate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def maybe(path: Path) -> object | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def maybe_one(root: Path, name: str) -> object | None:
    paths = sorted(root.rglob(name)) if root.is_dir() else []
    if len(paths) > 1:
        raise RuntimeError(f"expected at most one {name} under {root}, found {len(paths)}")
    return maybe(paths[0]) if paths else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gates = args.campaign_root / "gates"
    main_rows = {
        "full_seer_k1": maybe_one(args.campaign_root / "v0/reproduction/full_k1", "eval_summary.json"),
        "v0_fixed_k4": maybe_one(args.campaign_root / "v0/reproduction/v0_k4", "eval_summary.json"),
        "v1_fixed_k4": maybe_one(args.campaign_root / "v1/eval/fixed_k4_200", "eval_summary.json"),
        "v2_target_k4": maybe_one(args.campaign_root / "v2/eval/target_k4_200", "eval_summary.json"),
        "random_matched_budget": maybe_one(args.campaign_root / "v2/eval/random_matched_budget_200", "eval_summary.json"),
    }
    if any(value is None for value in main_rows.values()):
        raise RuntimeError("all five teacher33 main comparison rows must be complete before aggregation")
    payload = {
        "schema_version": 1,
        "teacher_lineage": "sd1_local_scratch_teacher33",
        "teacher35_results_merged": False,
        "canonical_reproduction": maybe(gates / "canonical_reproduction_decision.json"),
        "v1_budget_selection": maybe(args.campaign_root / "v1/selection/v1_budget_selection.json"),
        "v1_offline": maybe(args.campaign_root / "v1/gates/v1_offline_gate.json"),
        "v1_online": maybe(args.campaign_root / "v1/gates/v1_online_gate.json"),
        "v2_defect": maybe(args.campaign_root / "v2/gates/defect_signal.json"),
        "v2_threshold": maybe(args.campaign_root / "v2/calibration/v2_threshold_lock.json"),
        "v2_online": maybe(args.campaign_root / "v2/gates/v2_online_gate.json"),
        "teacher33_main_rows": main_rows,
        "teacher33_training_efficiency": {
            "e20_validation": maybe(args.campaign_root / "v1/train/e20/validation_metrics.json"),
            "e40_validation": maybe(args.campaign_root / "v1/train/e40/validation_metrics.json"),
            "budget_selection": maybe(args.campaign_root / "v1/selection/v1_budget_selection.json"),
        },
        "teacher35_external_replication": {
            "merged_into_teacher33_rows": False,
            "role": "independent-teacher appendix training-efficiency evidence only",
            "freeze_record": "teacher35_historical_freeze.md",
        },
        "metric_groups": {
            "task": ["overall_sr", "task_sr", "paired_flips", "paired_task_hierarchical_ci", "exact_mcnemar", "successful_episode_length"],
            "inference_efficiency": ["full_calls", "transition_calls", "direct_transition_calls", "direct_reanchors", "action_generator_calls", "effective_k", "full_query_reduction", "component_latency", "policy_latency", "complete_env_step_latency"],
            "training_efficiency": ["optimizer_steps", "effective_batch", "epoch_update_budget", "wall_clock", "e20_e40_compute_ratio"],
            "action_quality": ["translation_normalized_second_difference", "rotation_normalized_second_difference", "gripper_switches", "short_reversals"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
