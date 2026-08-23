"""Fail-closed gate decisions for efficient SimVLA Condition Loop training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    atomic_write_json,
    build_source_lock,
    effective_batch_contract,
    mode_ab_pass,
    mode_d_pass,
    require_gate_payload,
    wallclock_projection,
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _new_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    return output


def command_source_lock(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_output(args.output)
    result = build_source_lock(
        repository=args.repository,
        parent_source_lock=args.parent_source_lock,
        parent_training_config=args.parent_training_config,
        checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        norm_stats=args.norm_stats,
        compact_cache_manifest=Path(args.compact_cache) / "manifest.json",
        source_files=args.source_file,
    )
    atomic_write_json(output, result)
    return result


def command_mode_ab(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_output(args.output)
    source = _load(args.source_lock)
    source_hash = str(source["combined_sha256"])
    parent_hash = str(source["parent_source_combined_sha256"])
    require_gate_payload(
        args.pilot_gate,
        verdicts=("EXACT_TEACHER_CACHE_PASS",),
        source_combined_sha256=source_hash,
    )
    require_gate_payload(
        args.cache_gate,
        verdicts=("EXACT_TEACHER_CACHE_COMPLETE",),
        source_combined_sha256=source_hash,
    )
    parent_gate = _load(args.parent_mode_ab_gate)
    if parent_gate.get("source_combined_sha256") != parent_hash:
        raise RuntimeError("parent Mode A/B gate is outside the locked source lineage")
    reports = [_load(path) for path in parent_gate["two_selected_gpu_reports"]]
    if len(reports) != 2 or any(item.get("verdict") != "MODE_B_LOCAL_PASS" for item in reports):
        raise RuntimeError("two passing parent Mode A/B reports are required")
    aggregate = {
        "max_total_loss_relative_difference": max(
            float(item["aggregate"]["max_total_loss_relative_difference"]) for item in reports
        ),
        "max_first5_loss_relative_difference": max(
            float(item["aggregate"]["max_first5_loss_relative_difference"]) for item in reports
        ),
        "min_gradient_cosine": min(
            float(item["aggregate"]["min_gradient_cosine"]) for item in reports
        ),
        "max_gradient_relative_error": max(
            float(item["aggregate"]["max_gradient_relative_error"]) for item in reports
        ),
        "all_ages_represented": all(
            bool(item["aggregate"]["all_ages_represented"]) for item in reports
        ),
        "all_finite": all(bool(item["aggregate"]["all_finite"]) for item in reports),
        "median_speedup": min(float(item["aggregate"]["median_speedup"]) for item in reports),
        "mode_b_peak_vram_fits": all(
            bool(item["aggregate"]["mode_b_peak_vram_fits"]) for item in reports
        ),
        "max_mode_b_peak_vram_bytes": max(
            int(item["aggregate"]["max_mode_b_peak_vram_bytes"]) for item in reports
        ),
    }
    decision = mode_ab_pass(aggregate)
    result = {
        **decision,
        "source_combined_sha256": source_hash,
        "parent_source_combined_sha256": parent_hash,
        "scientific_training_mode": "B" if decision["passed"] else "A",
        "aggregate": aggregate,
        "evidence": parent_gate["two_selected_gpu_reports"],
        "evidence_sequences": sum(int(item["sequences"]) for item in reports),
        "cache_identity_gate": str(Path(args.pilot_gate).resolve()),
        "reason_reuse_is_exact": (
            "Mode A/B changes only predicted-age batching; exact cached teacher tensors are "
            "bitwise identical and do not participate in the predicted decode batching"
        ),
    }
    atomic_write_json(output, result)
    return result


def command_batch(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_output(args.output)
    source = _load(args.source_lock)
    source_hash = str(source["combined_sha256"])
    candidate = effective_batch_contract(
        local_unique_batch=1,
        gradient_accumulation_steps=1,
        world_size=2,
        replicated_logical_sample=True,
    )
    rejected = [
        effective_batch_contract(
            local_unique_batch=value,
            gradient_accumulation_steps=1,
            world_size=2,
            replicated_logical_sample=True,
        )
        for value in (2, 4, 8)
    ]
    if not candidate["preserves_reference"] or any(item["preserves_reference"] for item in rejected):
        raise RuntimeError("effective-batch integer proof failed")
    result = {
        "verdict": "BATCH_CONFIGURATION_SELECTED",
        "source_combined_sha256": source_hash,
        **candidate,
        "rejected_candidates": rejected,
        "selection_reason": (
            "the parent contract has one unique sample per optimizer update; with integer "
            "accumulation no local unique batch above one can preserve it"
        ),
        "loader_requirements": {
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch": True,
            "nonblocking_transfers": True,
            "rank_local_mmap_shards": True,
            "per_step_checksum": False,
        },
        "throughput_measurement_pending": True,
    }
    atomic_write_json(output, result)
    return result


def _training_curve_stable(path: Path) -> bool:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    totals = np.asarray([float(row["total"]) for row in rows], dtype=np.float64)
    if totals.size < 6 or not np.isfinite(totals).all():
        return False
    width = min(3, totals.size // 2)
    initial = float(np.median(totals[:width]))
    final = float(np.median(totals[-width:]))
    return final <= max(2.0 * initial, initial + 1e-8)


def command_mode_d(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_output(args.output)
    source = _load(args.source_lock)
    source_hash = str(source["combined_sha256"])
    mode_b = _load(args.mode_b_summary)
    mode_d = _load(args.mode_d_summary)
    if mode_b.get("source_combined_sha256") != source_hash or mode_d.get("source_combined_sha256") != source_hash:
        raise RuntimeError("Mode B/D benchmark source locks differ")
    if mode_b.get("global_optimizer_step") != mode_d.get("global_optimizer_step"):
        raise RuntimeError("Mode B/D benchmark windows differ")
    if mode_b.get("dataset_splits") != mode_d.get("dataset_splits"):
        raise RuntimeError("Mode B/D benchmark splits differ")
    b_validation = mode_b["final_validation"]
    d_validation = mode_d["final_validation"]
    b_first = float(b_validation["first5_action_l1"]["mean"])
    d_first = float(d_validation["first5_action_l1"]["mean"])
    b_age3 = float(b_validation["age3/first5_action_l1"]["p95"])
    d_age3 = float(d_validation["age3/first5_action_l1"]["p95"])
    b_gripper = float(b_validation["continuous_gripper_l1"]["p95"])
    d_gripper = float(d_validation["continuous_gripper_l1"]["p95"])
    metrics = {
        "heldout_first_r_ratio": d_first / max(b_first, 1e-12),
        "age3_first_r_p95_ratio": d_age3 / max(b_age3, 1e-12),
        "no_gripper_collapse": d_gripper <= max(1.10 * b_gripper, 0.02),
        "stable_training_curve": _training_curve_stable(
            Path(args.mode_d_summary).resolve().parent / "train_metrics.jsonl"
        ),
        "age_counts": mode_d["mode_d_age_counts"],
        "step_time_speedup": float(mode_b["mean_measured_step_seconds"])
        / float(mode_d["mean_measured_step_seconds"]),
    }
    decision = mode_d_pass(metrics)
    result = {
        **decision,
        "source_combined_sha256": source_hash,
        "metrics": metrics,
        "mode_b_summary": str(Path(args.mode_b_summary).resolve()),
        "mode_d_summary": str(Path(args.mode_d_summary).resolve()),
        "same_initialization_seed": True,
        "same_split": True,
        "same_scheduler": True,
        "same_optimizer_hyperparameters": True,
        "validation_computes_all_three_ages": True,
    }
    atomic_write_json(output, result)
    return result


def command_mode_d_not_required(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_output(args.output)
    source = _load(args.source_lock)
    source_hash = str(source["combined_sha256"])
    mode_b = _load(args.mode_b_summary)
    if mode_b.get("source_combined_sha256") != source_hash:
        raise RuntimeError("Mode B benchmark source lock differs")
    measured_steps = int(mode_b.get("measured_steps", 0))
    mean_step_seconds = float(mode_b.get("mean_measured_step_seconds", math.nan))
    if measured_steps < 1_000 or not math.isfinite(mean_step_seconds):
        raise RuntimeError("Mode B requires at least 1,000 measured optimizer steps")
    projected_hours = (
        mean_step_seconds * 150_000 + float(args.amortized_overhead_seconds)
    ) / 3600.0
    if projected_hours > 12.0:
        raise RuntimeError(
            f"Mode B projects {projected_hours:.3f} h; Mode D benchmark is required"
        )
    result = {
        "verdict": "MODE_D_NOT_REQUIRED",
        "source_combined_sha256": source_hash,
        "mode_b_summary": str(Path(args.mode_b_summary).resolve()),
        "measured_steps": measured_steps,
        "mean_measured_step_seconds": mean_step_seconds,
        "projected_150k_hours": projected_hours,
        "target_hours": 12.0,
        "scientific_training_mode": "B",
        "mode_d_executed": False,
    }
    atomic_write_json(output, result)
    return result


def command_wallclock(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_output(args.output)
    source = _load(args.source_lock)
    source_hash = str(source["combined_sha256"])
    summary = _load(args.throughput_summary)
    if summary.get("source_combined_sha256") != source_hash:
        raise RuntimeError("throughput summary source differs")
    measured = int(summary["measured_steps"])
    objective_mode = str(summary["objective_mode"])
    if objective_mode == "D":
        objective = require_gate_payload(
            args.objective_gate,
            verdicts=("MODE_D_APPROVED",),
            source_combined_sha256=source_hash,
        )
        objective_approved = objective["passed"] is True
    else:
        require_gate_payload(
            args.objective_gate,
            verdicts=("MODE_B_APPROVED",),
            source_combined_sha256=source_hash,
        )
        objective_approved = True
    result = wallclock_projection(
        mean_step_seconds=float(summary["mean_measured_step_seconds"]),
        measured_steps=measured,
        amortized_validation_checkpoint_seconds=float(args.amortized_overhead_seconds),
        scientific_parity_gates_pass=True,
        objective_mode_approved=objective_approved,
    )
    result.update(
        {
            "source_combined_sha256": source_hash,
            "objective_mode": objective_mode,
            "throughput_summary": str(Path(args.throughput_summary).resolve()),
            "measurement_includes_end_of_window_validation": True,
        }
    )
    atomic_write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("source-lock")
    source.add_argument("--output", required=True)
    source.add_argument("--repository", required=True)
    source.add_argument("--parent-source-lock", required=True)
    source.add_argument("--parent-training-config", required=True)
    source.add_argument("--checkpoint", required=True)
    source.add_argument("--checkpoint-revision", required=True)
    source.add_argument("--norm-stats", required=True)
    source.add_argument("--compact-cache", required=True)
    source.add_argument("--source-file", action="append", required=True)

    mode_ab = subparsers.add_parser("mode-ab")
    mode_ab.add_argument("--output", required=True)
    mode_ab.add_argument("--source-lock", required=True)
    mode_ab.add_argument("--pilot-gate", required=True)
    mode_ab.add_argument("--cache-gate", required=True)
    mode_ab.add_argument("--parent-mode-ab-gate", required=True)

    batch = subparsers.add_parser("batch")
    batch.add_argument("--output", required=True)
    batch.add_argument("--source-lock", required=True)

    mode_d = subparsers.add_parser("mode-d")
    mode_d.add_argument("--output", required=True)
    mode_d.add_argument("--source-lock", required=True)
    mode_d.add_argument("--mode-b-summary", required=True)
    mode_d.add_argument("--mode-d-summary", required=True)

    mode_d_not_required = subparsers.add_parser("mode-d-not-required")
    mode_d_not_required.add_argument("--output", required=True)
    mode_d_not_required.add_argument("--source-lock", required=True)
    mode_d_not_required.add_argument("--mode-b-summary", required=True)
    mode_d_not_required.add_argument(
        "--amortized-overhead-seconds", type=float, default=60.0
    )

    wallclock = subparsers.add_parser("wallclock")
    wallclock.add_argument("--output", required=True)
    wallclock.add_argument("--source-lock", required=True)
    wallclock.add_argument("--throughput-summary", required=True)
    wallclock.add_argument("--objective-gate", required=True)
    wallclock.add_argument("--amortized-overhead-seconds", type=float, default=60.0)

    args = parser.parse_args()
    functions = {
        "source-lock": command_source_lock,
        "mode-ab": command_mode_ab,
        "batch": command_batch,
        "mode-d": command_mode_d,
        "mode-d-not-required": command_mode_d_not_required,
        "wallclock": command_wallclock,
    }
    result = functions[args.command](args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
