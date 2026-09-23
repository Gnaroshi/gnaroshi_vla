"""Episode-disjoint final-checkpoint offline K=4 gate for native SimVLA V0."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
from architectures.simvla.adapters.latentloop.native_v0_dataset import (
    NativeV0SequenceDataset,
    collate_native_v0_sequences,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    cached_batch_token_layout,
    configure_strict_torch_determinism,
    load_frozen_simvla,
    move_batch,
    native_v0_source_manifest,
    require_gate,
    write_json,
)
from architectures.simvla.wrappers.simvla_two_gpu_guard import parse_selected_gpu_ids
from methods.latentloop.training.native_simvla_v0 import decode_age_conditions


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _action_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = (prediction - target).abs()
    first5 = difference[:, :5]
    return {
        "first5_action_l1": float(first5.mean().item()),
        "full_chunk_action_l1": float(difference.mean().item()),
        "translation_l1": float(first5[..., :3].mean().item()),
        "rotation_l1": float(first5[..., 3:6].mean().item()),
        "continuous_gripper_l1": float(first5[..., 6:].mean().item()),
        "predicted_gripper_std": float(prediction[:, :5, 6].float().std(unbiased=False).item()),
        "reference_gripper_std": float(target[:, :5, 6].float().std(unbiased=False).item()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected = parse_selected_gpu_ids(os.environ.get("SIMVLA_GPU_IDS"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, selected)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must exactly match SIMVLA_GPU_IDS")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("offline K4 gate requires exactly two DDP processes")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    configure_strict_torch_determinism(args.seed)
    source = native_v0_source_manifest(checkpoint=args.checkpoint, norm_stats=args.norm_stats, cache=args.cache)
    source_hash = source["combined_sha256"]
    require_gate(args.parity_gate, verdicts=("K1_HOOK_PARITY_PASS",), source_combined_sha256=source_hash)
    require_gate(args.parameter_gate, verdicts=("PARAMETER_AUDIT_PASS",), source_combined_sha256=source_hash)
    training_gate = require_gate(
        args.training_gate,
        verdicts=("FINAL_150K_TRAINING_COMPLETE",),
        source_combined_sha256=source_hash,
    )
    if int(training_gate.get("global_optimizer_step", -1)) != 150_000:
        raise RuntimeError("offline gate requires the completed 150K training summary")
    model, checkpoint = load_native_v0_checkpoint(args.v0_checkpoint, device=device, require_final_150k=True)
    if checkpoint["source_lock"]["combined_sha256"] != source_hash:
        raise RuntimeError("V0 checkpoint and offline evaluator source locks differ")
    model.eval()
    frozen_model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    dataset = NativeV0SequenceDataset(
        args.cache,
        split="heldout",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    heldout_contract = dataset.contract()
    expected_splits = checkpoint["training_config"].get("dataset_splits")
    if training_gate.get("dataset_splits") != expected_splits:
        raise RuntimeError("training summary and final checkpoint split contracts differ")
    if checkpoint["training_config"].get("dataset_contract_heldout") != heldout_contract:
        raise RuntimeError("offline evaluator held-out split differs from final checkpoint")
    output = Path(args.output).expanduser().resolve()
    exists = torch.tensor([int(output.exists())], device=device)
    dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if int(exists.item()):
        raise FileExistsError(f"refusing existing output: {output}")
    if rank == 0:
        output.mkdir(parents=True)
    dist.barrier()
    local_rows: list[dict[str, Any]] = []
    decode_mode = str(checkpoint["training_config"]["mode"])
    if decode_mode not in {"A", "B"}:
        raise RuntimeError(f"unsupported checkpoint action decode mode: {decode_mode}")

    def decode(conditions: tuple[torch.Tensor, ...], batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        return decode_age_conditions(
            lambda condition, proprio, noise: action_adapter.decode_action_from_condition(
                condition,
                proprio,
                steps=10,
                initial_noise=noise,
            ),
            conditions,
            tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3)),
            tuple(batch["explicit_noises"][:, age - 1] for age in (1, 2, 3)),
            mode=decode_mode,
        )

    with torch.no_grad():
        for dataset_index in range(rank, len(dataset), 2):
            batch = move_batch(collate_native_v0_sequences([dataset[dataset_index]]), device)
            layout = cached_batch_token_layout(
                condition=batch["anchor_condition"],
                language_instructions=batch["language_instruction"],
                processor=processor,
            )
            unroll = model(
                batch["anchor_condition"],
                batch["image_sequence"],
                batch["proprio_sequence"],
                valid_mask=layout.valid_mask,
                group_ids=layout.group_ids,
            )
            predicted_actions = decode(unroll.conditions, batch)
            full_conditions = tuple(batch["teacher_conditions"][:, age - 1] for age in (1, 2, 3))
            full_actions = decode(full_conditions, batch)
            hold_conditions = (batch["anchor_condition"],) * 3
            hold_actions = decode(hold_conditions, batch)
            for offset, age in enumerate((1, 2, 3)):
                mask = layout.valid_mask.unsqueeze(-1)
                condition_difference = (unroll.conditions[offset] - full_conditions[offset]).float()
                condition_mse = float(
                    condition_difference.square().masked_select(mask).mean().item()
                )
                normalized_prediction = F.layer_norm(
                    unroll.conditions[offset].float(),
                    (unroll.conditions[offset].shape[-1],),
                )
                normalized_target = F.layer_norm(
                    full_conditions[offset].float(),
                    (full_conditions[offset].shape[-1],),
                )
                normalized_difference = normalized_prediction - normalized_target
                normalized_condition_mse = float(
                    normalized_difference.square().masked_select(mask).mean().item()
                )
                predicted = _action_metrics(predicted_actions[offset], full_actions[offset])
                hold = _action_metrics(hold_actions[offset], full_actions[offset])
                row: dict[str, Any] = {
                    "dataset_index": dataset_index,
                    "task_id": int(batch["task_id"][0].item()),
                    "episode_id": batch["episode_id"][0],
                    "anchor_query_index": int(batch["anchor_query_index"][0].item()),
                    "age": age,
                    "masked_condition_mse": condition_mse,
                    "masked_normalized_condition_mse": normalized_condition_mse,
                    **{f"predicted/{key}": value for key, value in predicted.items()},
                    **{f"hold/{key}": value for key, value in hold.items()},
                    "full_current/first5_action_l1": 0.0,
                    "finite": bool(
                        math.isfinite(condition_mse)
                        and math.isfinite(normalized_condition_mse)
                        and all(math.isfinite(value) for value in predicted.values())
                        and all(math.isfinite(value) for value in hold.values())
                    ),
                }
                local_rows.append(row)

    gathered: list[list[dict[str, Any]] | None] | None = [None, None] if rank == 0 else None
    dist.gather_object(local_rows, gathered, dst=0)
    result: dict[str, Any] = {}
    if rank == 0:
        rows = [row for shard in gathered or [] for row in shard or []]
        rows.sort(key=lambda row: (row["task_id"], row["episode_id"], row["anchor_query_index"], row["age"]))
        with (output / "offline_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        age_summary: dict[str, Any] = {}
        for age in (1, 2, 3):
            selected_rows = [row for row in rows if row["age"] == age]
            metrics = (
                "masked_condition_mse",
                "masked_normalized_condition_mse",
                "predicted/first5_action_l1",
                "predicted/full_chunk_action_l1",
                "predicted/translation_l1",
                "predicted/rotation_l1",
                "predicted/continuous_gripper_l1",
                "hold/first5_action_l1",
                "hold/full_chunk_action_l1",
                "hold/continuous_gripper_l1",
            )
            age_summary[str(age)] = {
                metric: _summary([float(row[metric]) for row in selected_rows])
                for metric in metrics
            }
        finite = all(bool(row["finite"]) for row in rows)
        mean_better = all(
            age_summary[str(age)]["predicted/first5_action_l1"]["mean"]
            < age_summary[str(age)]["hold/first5_action_l1"]["mean"]
            for age in (1, 2, 3)
        )
        age3_p95 = (
            age_summary["3"]["predicted/first5_action_l1"]["p95"]
            <= age_summary["3"]["hold/first5_action_l1"]["p95"]
        )
        p99_no_catastrophe = all(
            age_summary[str(age)]["predicted/first5_action_l1"]["p99"]
            <= 2.0 * max(age_summary[str(age)]["hold/first5_action_l1"]["p99"], 1e-8)
            for age in (1, 2, 3)
        )
        gripper_no_collapse = all(
            float(row["predicted/predicted_gripper_std"])
            >= 0.1 * float(row["predicted/reference_gripper_std"])
            or float(row["predicted/reference_gripper_std"]) < 1e-6
            for row in rows
        )
        frozen_grads = all(parameter.grad is None for parameter in frozen_model.parameters())
        parameter_cap = bool(model.parameter_audit()["under_hard_cap_1000000"])
        passed = finite and mean_better and age3_p95 and p99_no_catastrophe and gripper_no_collapse and frozen_grads and parameter_cap
        result = {
            "verdict": "OFFLINE_K4_GATE_PASS" if passed else "OFFLINE_K4_GATE_FAIL",
            "source_combined_sha256": source_hash,
            "checkpoint": str(Path(args.v0_checkpoint).resolve()),
            "checkpoint_step": checkpoint["global_optimizer_step"],
            "action_decode_mode": decode_mode,
            "heldout_sequences": len(dataset),
            "dataset_splits": expected_splits,
            "heldout_split_sha256": dataset.split_sha256,
            "rows": len(rows),
            "age_summary": age_summary,
            "checks": {
                "all_finite": finite,
                "first5_mean_lower_than_hold_all_ages": mean_better,
                "age3_first5_p95_no_worse_than_hold": age3_p95,
                "p99_no_catastrophic_tail": p99_no_catastrophe,
                "no_gripper_collapse": gripper_no_collapse,
                "k1_hook_parity_gate_passed": True,
                "parameter_cap_passed": parameter_cap,
                "base_action_gradients_zero": frozen_grads,
            },
            "references": {
                "hold_previous_condition": "q0 anchor held at ages 1/2/3",
                "full_current_condition": "same-noise zero-error oracle",
                "historical_old_v0": "not source-compatible: it consumes executed actions and uses a different updater",
            },
            "k2_online_run": False,
        }
        write_json(output / "simvla_v0_offline_gate.json", result)
        report = [
            "# Correct native SimVLA V0 offline K=4 gate",
            "",
            f"- Verdict: `{result['verdict']}`",
            f"- Final checkpoint step: `{result['checkpoint_step']}`",
            f"- Episode-disjoint held-out sequences: `{len(dataset)}`",
            "- Predicted/full/hold actions use identical proprioception and explicit flow noise.",
            "- The historical V0 is not a valid direct reference because it consumed executed action subchunks.",
            "- No K=2 online evaluation was run.",
            "",
            "```json",
            json.dumps(result["checks"], indent=2, sort_keys=True),
            "```",
        ]
        (output / "simvla_v0_offline_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--v0-checkpoint", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--parity-gate", required=True)
    parser.add_argument("--parameter-gate", required=True)
    parser.add_argument("--training-gate", required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    result = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
