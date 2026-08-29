#!/usr/bin/env python3
"""Verify evaluation of a validation-selected V0 checkpoint from any fixed budget."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

import verify_pi05_v0_streaming_eval as base
from architectures.openpi.adapters.latentloop.cache_contract_v2 import SUITES


ROOT = Path(__file__).resolve().parents[2]
ADDITIVE_EVALUATION_FILES = (
    "architectures/openpi/wrappers/run_pi05_v0_training_budget_eval_1gpu.sh",
    "tools/openpi/serve_pi05_v0_training_budget.py",
    "tools/openpi/verify_pi05_v0_training_budget_eval.py",
)


def _validation_record(path: Path, step: int) -> dict[str, Any]:
    matches = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record.get("step", -1)) == step:
                matches.append(record)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one held-out validation record for step {step}, found {len(matches)}"
        )
    return matches[0]


def verify_training_budget_run(
    run_summary_path: str | Path,
    adapter_checkpoint_path: str | Path,
    source_lock_id: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    base_checkpoint: str | Path,
    expected_training_steps: int,
) -> dict[str, Any]:
    if expected_training_steps <= 0:
        raise ValueError("expected_training_steps must be positive")

    run_summary_path = Path(run_summary_path).resolve()
    adapter_checkpoint_path = Path(adapter_checkpoint_path).resolve()
    run_dir = run_summary_path.parent
    summary = base.load_json(run_summary_path)
    config = base.load_json(run_dir / "config.json")

    if summary.get("V0_TRAIN_COMPLETE") is not True or summary.get("complete") is not True:
        raise RuntimeError("V0 streaming training did not complete")
    if summary.get("source_lock_id") != source_lock_id:
        raise RuntimeError("V0 run summary belongs to another source lock")
    if summary.get("variant") != "v0" or summary.get("training_source_mode") != "online_frozen_teacher_rolling_v0":
        raise RuntimeError("training run is not cacheless rolling V0")
    if int(summary.get("steps", -1)) != expected_training_steps:
        raise RuntimeError(
            "training budget mismatch: "
            f"expected {expected_training_steps}, observed {summary.get('steps')}"
        )
    if int(config.get("trainer", {}).get("max_steps", -1)) != expected_training_steps:
        raise RuntimeError("training config max_steps differs from the declared budget")

    persistent_bytes = summary.get("streaming_source_statistics", {}).get(
        "persistent_teacher_tensor_bytes"
    )
    if int(persistent_bytes if persistent_bytes is not None else -1) != 0:
        raise RuntimeError("V0 run unexpectedly persisted teacher tensors")
    if not adapter_checkpoint_path.is_relative_to((run_dir / "checkpoints").resolve()):
        raise RuntimeError("adapter checkpoint is not owned by the declared V0 run")
    if adapter_checkpoint_path.name != "best.pt":
        raise RuntimeError("primary evaluation requires validation-selected best.pt")

    payload = torch.load(adapter_checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config", {})
    provenance = checkpoint_config.get("provenance", {})
    if checkpoint_config.get("adapter_type") != "openpi_variable_time_latentloop":
        raise RuntimeError("adapter checkpoint has an unexpected type")
    if provenance.get("source_lock_id") != source_lock_id:
        raise RuntimeError("adapter checkpoint belongs to another source lock")
    if provenance.get("final_manifest_id") != manifest["manifest_id"]:
        raise RuntimeError("adapter checkpoint final-manifest id mismatch")
    if provenance.get("final_manifest_sha256") != manifest_sha256:
        raise RuntimeError("adapter checkpoint final-manifest hash mismatch")
    if Path(provenance.get("teacher_checkpoint", "")).resolve() != Path(base_checkpoint).resolve():
        raise RuntimeError("adapter teacher/base checkpoint mismatch")
    if provenance.get("training_source_mode") != "online_frozen_teacher_rolling_v0":
        raise RuntimeError("adapter was not produced by cacheless rolling V0 training")

    best_step = int(summary["best_step"])
    if best_step <= 0 or best_step > expected_training_steps:
        raise RuntimeError("validation-selected step lies outside the training budget")
    if int(payload.get("step", -1)) != best_step:
        raise RuntimeError("best.pt step differs from run_summary best_step")
    best_metric = float(summary["best_validation_first_r_action_mse"])
    checkpoint_metric = float(payload.get("validation", {}).get("recursive_first_r", math.nan))
    if not math.isclose(best_metric, checkpoint_metric, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("best.pt validation metric differs from run_summary")
    validation_record = _validation_record(run_dir / "validation.jsonl", best_step)
    logged_metric = float(validation_record["validation"]["recursive_first_r"])
    if not math.isclose(best_metric, logged_metric, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("best.pt metric differs from the held-out validation log")

    parameter_count = sum(int(tensor.numel()) for tensor in payload.get("adapter", {}).values())
    if parameter_count != int(summary["adapter_trainable_parameters"]):
        raise RuntimeError("adapter parameter count differs from run_summary")
    if config.get("provenance", {}).get("source_lock_id") != source_lock_id:
        raise RuntimeError("training config belongs to another source lock")
    if config.get("trainer", {}).get("action_execution_mode") != summary.get(
        "action_execution_mode"
    ):
        raise RuntimeError("training action execution mode is inconsistent")

    return {
        "V0_TRAINING_BUDGET_CHECKPOINT_PASS": True,
        "run_summary": str(run_summary_path),
        "run_summary_sha256": base.sha256_file(run_summary_path),
        "training_config_sha256": base.sha256_file(run_dir / "config.json"),
        "adapter_checkpoint": str(adapter_checkpoint_path),
        "adapter_checkpoint_sha256": base.sha256_file(adapter_checkpoint_path),
        "checkpoint_selection": "heldout_checkpoint_validation_minimum",
        "checkpoint_step": best_step,
        "training_steps": int(summary["steps"]),
        "validation_first_r_action_mse": best_metric,
        "adapter_trainable_parameters": parameter_count,
        "action_execution_mode": summary["action_execution_mode"],
        "training_source_id": summary["training_source_id"],
        "training_source_mode": summary["training_source_mode"],
        "persistent_teacher_tensor_bytes": 0,
    }


def evaluation_harness_manifest() -> dict[str, Any]:
    names = tuple(base.EVALUATION_FILES) + ADDITIVE_EVALUATION_FILES
    files = {relative: base.sha256_file(ROOT / relative) for relative in names}
    return {"files": files, "combined_sha256": base.canonical_hash(files)}


def verify_evaluation_inputs(
    *,
    source_lock: str | Path,
    checkpoint: str | Path,
    norm_stats: str | Path,
    final_manifest: str | Path,
    suite: str,
    k1_tensor_report: str | Path,
    k1_episode_gate: str | Path,
    freeze_gate: str | Path,
    training_run_summary: str | Path,
    adapter_checkpoint: str | Path,
    expected_training_steps: int,
) -> dict[str, Any]:
    if suite not in SUITES:
        raise ValueError(f"unknown LIBERO suite {suite!r}")
    lock, source_result = base.verify_source_lock(source_lock, checkpoint, norm_stats)
    source_lock_id = lock["source_lock_id"]
    manifest, manifest_result = base.verify_manifest(final_manifest, source_lock_id, suite)
    gates = {
        "k1_tensor": base.require_gate(
            k1_tensor_report,
            source_lock_id,
            ("REAL_KV_ROUNDTRIP_PASS", "K1_ACTION_PARITY_PASS"),
        ),
        "k1_episode": base.require_gate(
            k1_episode_gate,
            source_lock_id,
            ("REAL_KV_ROUNDTRIP_PASS", "K1_ACTION_PARITY_PASS", "K1_EPISODE_PARITY_PASS"),
        ),
        "freeze": base.require_gate(freeze_gate, source_lock_id, ("BASE_FREEZE_PASS",)),
    }
    training = verify_training_budget_run(
        training_run_summary,
        adapter_checkpoint,
        source_lock_id,
        manifest,
        manifest_result["sha256"],
        checkpoint,
        expected_training_steps,
    )
    return {
        "V0_TRAINING_BUDGET_EVALUATION_PREFLIGHT_PASS": True,
        "source_lock_id": source_lock_id,
        "suite": suite,
        "paired_rows": ["original", "v0"],
        "evaluation_order": ["v0", "original"],
        "source_verification": source_result,
        "manifest": manifest_result,
        "gates": gates,
        "training": training,
        "evaluation_harness": evaluation_harness_manifest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--final-manifest", required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--k1-tensor-report", required=True)
    parser.add_argument("--k1-episode-gate", required=True)
    parser.add_argument("--freeze-gate", required=True)
    parser.add_argument("--training-run-summary", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--expected-training-steps", type=int, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = verify_evaluation_inputs(
        source_lock=args.source_lock,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        final_manifest=args.final_manifest,
        suite=args.suite,
        k1_tensor_report=args.k1_tensor_report,
        k1_episode_gate=args.k1_episode_gate,
        freeze_gate=args.freeze_gate,
        training_run_summary=args.training_run_summary,
        adapter_checkpoint=args.adapter_checkpoint,
        expected_training_steps=args.expected_training_steps,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
