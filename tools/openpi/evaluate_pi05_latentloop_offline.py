#!/usr/bin/env python3
"""Held-out same-noise offline gates for V0/V1 and defect calibration data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _common import DEFAULT_CHECKPOINT, load_json, load_local_policy, require_run
from architectures.openpi.adapters.latentloop.cache_dataset import LatentLoopSequenceDataset
from architectures.openpi.adapters.latentloop.losses import normalized_kv_loss
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook
from architectures.openpi.adapters.latentloop.serialization import (
    load_adapter_checkpoint,
    prefix_embedding_from_record,
    prefix_state_from_record,
)
from methods.variable_time_latentloop.defect import normalized_latent_defect
from methods.variable_time_latentloop.metrics import action_error_components
from pi05_stage_gate_v2 import verify_stage
from validate_pi05_cache_v2 import validate_cache_v2


def _error(predicted: torch.Tensor, teacher: torch.Tensor) -> dict[str, float]:
    values = action_error_components(predicted[..., :7], teacher[..., :7], 5)
    return {name: float(value.item()) for name, value in values.items()}


def _noise_hash(value: Any) -> str:
    array = np.ascontiguousarray(torch.as_tensor(value).cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.no_grad()
def evaluate_example(adapter, hook, example: dict[str, Any], device: torch.device) -> dict[str, Any]:
    records = example["records"]
    delta_q = int(example["delta_q"])
    anchor = prefix_state_from_record(records[0], device)
    target = prefix_state_from_record(records[-1], device)
    current = prefix_embedding_from_record(records[-1], device)
    robot = torch.as_tensor(
        records[-1]["robot_state_normalized"], device=device, dtype=torch.float32
    ).view(1, -1)
    noise = torch.as_tensor(records[-1]["action_noise"], device=device, dtype=torch.float32).unsqueeze(0)
    teacher = torch.as_tensor(
        records[-1]["teacher_action_chunk_normalized"], device=device, dtype=torch.float32
    ).unsqueeze(0)
    actions = torch.cat(
        [
            torch.as_tensor(
                record["executed_actions_postprocessed"], device=device, dtype=torch.float32
            ).unsqueeze(0)
            for record in records[:-1]
        ],
        dim=1,
    )
    prefix_history = torch.stack(
        [prefix_embedding_from_record(record, device).embeddings for record in records[1:]], dim=1
    )
    robot_history = torch.stack(
        [
            torch.as_tensor(record["robot_state_normalized"], device=device, dtype=torch.float32).view(1, -1)
            for record in records[1:]
        ],
        dim=1,
    )
    direct = adapter(
        anchor,
        current,
        anchor.embeddings,
        actions,
        robot,
        delta_q=delta_q,
        delta_a=delta_q * 5,
        full_refresh_age=delta_q,
        executed_action_lengths=torch.full(
            (actions.shape[0],), actions.shape[1], device=device, dtype=torch.long
        ),
        intermediate_prefix_embeddings=prefix_history,
        robot_state_history=robot_history,
    )
    sequential_state = anchor
    previous_embeddings = anchor.embeddings
    sequential_encoded = None
    for offset, record in enumerate(records[1:], start=1):
        prefix = prefix_embedding_from_record(record, device)
        step_robot = torch.as_tensor(
            record["robot_state_normalized"], device=device, dtype=torch.float32
        ).view(1, -1)
        step_actions = torch.as_tensor(
            records[offset - 1]["executed_actions_postprocessed"],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        update = adapter(
            sequential_state,
            prefix,
            previous_embeddings,
            step_actions,
            step_robot,
            delta_q=1,
            delta_a=5,
            full_refresh_age=offset,
            executed_action_lengths=torch.full(
                (step_actions.shape[0],), step_actions.shape[1], device=device, dtype=torch.long
            ),
        )
        sequential_state = update.state
        sequential_encoded = update.encoded_state
        previous_embeddings = prefix.embeddings
    assert sequential_encoded is not None
    direct_actions, _ = hook.sample_actions_from_state(direct.state, robot, noise, num_steps=10)
    sequential_actions, _ = hook.sample_actions_from_state(sequential_state, robot, noise, num_steps=10)
    hold_actions, _ = hook.sample_actions_from_state(anchor, robot, noise, num_steps=10)
    direct_error = _error(direct_actions, teacher)
    sequential_error = _error(sequential_actions, teacher)
    hold_error = _error(hold_actions, teacher)
    last = records[-1]
    embedding_change = (current.embeddings.float() - anchor.embeddings.float()).square().mean().sqrt()
    action_magnitude = actions[..., :6].square().mean().sqrt()
    return {
        "suite": last["suite"],
        "task_id": int(last["task_id"]),
        "benchmark_task_index": int(last["benchmark_task_index"]),
        "episode_namespace": str(last["episode_namespace"]),
        "episode_id": int(last["episode_id"]),
        "query_index": int(last["query_index"]),
        "delta_q": delta_q,
        "delta_a": delta_q * 5,
        "noise_seed": int(last["action_noise_seed"]),
        "noise_hash": _noise_hash(last["action_noise"]),
        "direct_state_mse": float(normalized_kv_loss(adapter.codec, direct.state, target).item()),
        "sequential_state_mse": float(normalized_kv_loss(adapter.codec, sequential_state, target).item()),
        "latent_defect": float(normalized_latent_defect(sequential_encoded, direct.encoded_state).item()),
        "observation_change_norm": float(embedding_change.item()),
        "executed_action_magnitude": float(action_magnitude.item()),
        **{f"direct_{name}": value for name, value in direct_error.items()},
        **{f"sequential_{name}": value for name, value in sequential_error.items()},
        **{f"hold_{name}": value for name, value in hold_error.items()},
        "sequential_gripper_std": float(sequential_actions[0, :5, 6].float().std().item()),
        "teacher_gripper_std": float(teacher[0, :5, 6].float().std().item()),
    }


def _mean(rows: list[dict[str, Any]], key: str, ages: set[int] | None = None) -> float:
    values = [float(row[key]) for row in rows if ages is None or int(row["delta_q"]) in ages]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--cache-gate", required=True)
    parser.add_argument("--previous-stage-gate", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--full-cache-inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("v0", "v1", "latent_bridge"), required=True)
    parser.add_argument(
        "--split",
        choices=("checkpoint_validation", "defect_fit", "defect_validity", "scheduler_calibration"),
        default="checkpoint_validation",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--reference-v0-summary")
    parser.add_argument("--reference-no-composition-summary")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_OFFLINE_RUN")
    if args.variant == "latent_bridge":
        raise RuntimeError("Latent Bridge-style KV baseline remains disabled pending official fidelity")
    output = Path(args.output).resolve()
    if args.split == "checkpoint_validation":
        stage = "stage4_v0_offline" if args.variant == "v0" else "stage7_v1_offline"
    else:
        if args.variant != "v1":
            raise ValueError("defect/scheduler metric roles require the V1 model")
        stage = "stage9_defect_fit"
    stage_gate = verify_stage(
        stage,
        args.source_lock,
        [args.previous_stage_gate],
        output_candidate=output,
    )
    cache_status = validate_cache_v2(
        args.cache,
        source_lock_path=args.source_lock,
        final_manifest_path=args.final_evaluation_manifest,
        split_contract_path=args.split_contract,
        full_cache_inventory_path=args.full_cache_inventory,
        verify_hashes=True,
        require_full=True,
    )
    if not cache_status["FULL_CACHE_SCHEMA_V2_PASS"]:
        raise RuntimeError(f"cache validation failed: {cache_status['errors']}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    policy = load_local_policy(args.checkpoint, args.device, flow_steps=10)
    adapter, checkpoint_payload = load_adapter_checkpoint(args.adapter_checkpoint, device)
    checkpoint_variant = checkpoint_payload["config"]["trainer"]["variant"]
    if checkpoint_variant != args.variant:
        raise ValueError(f"adapter is {checkpoint_variant}, requested {args.variant}")
    hook = PrefixKVHook(policy._model)  # noqa: SLF001
    dataset = LatentLoopSequenceDataset(
        args.cache,
        split=args.split,
        max_delta_q=3,
        one_step_only=False,
    )
    rows: list[dict[str, Any]] = []
    noise_by_target: dict[tuple[str, int, int, int], str] = {}
    same_noise_integrity = True
    for index in range(len(dataset)):
        if args.max_examples is not None and index >= args.max_examples:
            break
        row = evaluate_example(adapter, hook, dataset[index], device)
        target_key = (
            str(row["suite"]), int(row["task_id"]), int(row["episode_id"]), int(row["query_index"])
        )
        previous = noise_by_target.setdefault(target_key, str(row["noise_hash"]))
        same_noise_integrity &= previous == row["noise_hash"]
        rows.append(row)
        print(f"offline {index + 1}/{min(len(dataset), args.max_examples or len(dataset))}", flush=True)
    if not rows:
        raise RuntimeError("offline dataset produced no rows")
    csv_path = output / "offline_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cache_manifest_path = Path(args.cache).resolve() / "pi05_latentloop_cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    split_contract = json.loads(Path(args.split_contract).read_text(encoding="utf-8"))

    finite = all(
        np.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in {"suite", "episode_namespace", "noise_hash"}
    )
    sequential = np.asarray([row["sequential_executed_mse"] for row in rows], dtype=np.float64)
    hold = np.asarray([row["hold_executed_mse"] for row in rows], dtype=np.float64)
    gripper_std = np.asarray([row["sequential_gripper_std"] for row in rows], dtype=np.float64)
    teacher_gripper_std = np.asarray([row["teacher_gripper_std"] for row in rows], dtype=np.float64)
    noncollapsed = float(np.mean(gripper_std)) >= max(1e-5, 0.05 * float(np.mean(teacher_gripper_std)))
    p99_noncatastrophic = float(np.quantile(sequential, 0.99)) <= max(
        0.25, 1.25 * float(np.quantile(hold, 0.99))
    )
    summary = {
        "complete": True,
        "variant": args.variant,
        "split": args.split,
        "rows": len(rows),
        "adapter_checkpoint": str(Path(args.adapter_checkpoint).resolve()),
        "adapter_checkpoint_sha256": hashlib.sha256(
            Path(args.adapter_checkpoint).read_bytes()
        ).hexdigest(),
        "source_lock_id": stage_gate["source_lock_id"],
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "base_checkpoint_sha256": cache_manifest["metadata"]["source_hashes"]["checkpoint"],
        "cache_manifest": str(cache_manifest_path),
        "cache_manifest_id": cache_manifest["cache_manifest_id"],
        "cache_manifest_sha256": hashlib.sha256(cache_manifest_path.read_bytes()).hexdigest(),
        "split_contract": str(Path(args.split_contract).resolve()),
        "split_contract_id": split_contract["split_contract_id"],
        "split_contract_sha256": hashlib.sha256(
            Path(args.split_contract).read_bytes()
        ).hexdigest(),
        "final_evaluation_manifest": str(Path(args.final_evaluation_manifest).resolve()),
        "final_evaluation_manifest_sha256": hashlib.sha256(
            Path(args.final_evaluation_manifest).read_bytes()
        ).hexdigest(),
        "offline_metrics": str(csv_path),
        "offline_metrics_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "finite": finite,
        "same_noise_integrity": same_noise_integrity,
        "means": {
            key: _mean(rows, key)
            for key in (
                "direct_executed_mse",
                "sequential_executed_mse",
                "hold_executed_mse",
                "direct_chunk_mse",
                "sequential_chunk_mse",
                "latent_defect",
            )
        },
        "by_age": {
            str(age): {
                "direct_executed_mse": _mean(rows, "direct_executed_mse", {age}),
                "sequential_executed_mse": _mean(rows, "sequential_executed_mse", {age}),
                "hold_executed_mse": _mean(rows, "hold_executed_mse", {age}),
                "latent_defect": _mean(rows, "latent_defect", {age}),
            }
            for age in (1, 2, 3)
        },
        "p99": {
            "sequential_executed_mse": float(np.quantile(sequential, 0.99)),
            "hold_executed_mse": float(np.quantile(hold, 0.99)),
        },
        "gripper_noncollapsed": noncollapsed,
    }
    gate_checks = {
        "finite": finite,
        "same_noise_integrity": same_noise_integrity,
        "gripper_noncollapsed": noncollapsed,
        "bridge_beats_hold_first_r": float(np.mean(sequential)) < float(np.mean(hold)),
        "p99_noncatastrophic": p99_noncatastrophic,
        "source_lock_current": cache_status["source_lock_id"] == stage_gate["source_lock_id"],
    }
    if args.variant == "v1":
        reference_v0 = load_json(args.reference_v0_summary) if args.reference_v0_summary else None
        no_composition = (
            load_json(args.reference_no_composition_summary)
            if args.reference_no_composition_summary
            else None
        )
        age23_direct = _mean(rows, "direct_executed_mse", {2, 3})
        reference_age23 = (
            float(np.mean([
                reference_v0["by_age"][str(age)]["sequential_executed_mse"] for age in (2, 3)
            ]))
            if reference_v0
            else float("nan")
        )
        no_comp_defect = no_composition["means"]["latent_defect"] if no_composition else float("nan")
        gate_checks.update(
            {
                "direct_age23_beats_v0_repeated": bool(reference_v0 and age23_direct < reference_age23),
                "composition_beats_no_composition": bool(
                    no_composition and summary["means"]["latent_defect"] < no_comp_defect
                ),
                "composed_error_does_not_explode": summary["p99"]["sequential_executed_mse"]
                <= max(0.25, 1.25 * summary["p99"]["hold_executed_mse"]),
            }
        )
    gate_marker = "V0_OFFLINE_GATE_PASS" if args.variant == "v0" else "V1_OFFLINE_GATE_PASS"
    checkpoint_split = args.split == "checkpoint_validation"
    gate_pass = bool(checkpoint_split and all(gate_checks.values()))
    gate = {
        "schema_version": 2,
        "variant": args.variant,
        "split": args.split,
        "source_lock_id": stage_gate["source_lock_id"],
        "checks": gate_checks,
        "passed": gate_pass,
        gate_marker: gate_pass,
        "markers": [gate_marker] if gate_pass else [],
        "artifact_provenance": {
            key: summary[key]
            for key in (
                "adapter_checkpoint",
                "adapter_checkpoint_sha256",
                "base_checkpoint",
                "base_checkpoint_sha256",
                "cache_manifest",
                "cache_manifest_id",
                "cache_manifest_sha256",
                "split_contract",
                "split_contract_id",
                "split_contract_sha256",
                "final_evaluation_manifest",
                "final_evaluation_manifest_sha256",
                "offline_metrics",
                "offline_metrics_sha256",
            )
        },
    }
    (output / "offline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "offline_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
