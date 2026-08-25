"""Held-out integrity and fidelity screen for real c_j Generation coupling."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    GENERATION_SCHEDULES,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    COUPLED_CHECKPOINT_SCHEMA,
    audit_projection_only_state,
    build_coupled_query,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_source_lock import (
    verify_coupled_source_lock,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import (
    install_exact_uint8_delta_path,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherSequenceDataset,
    _drop_unused_vlm,
    collate_exact_teacher_sequences,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    full_generation_step_with_hidden,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
    move_batch,
    write_json,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


@torch.no_grad()
def _decode(
    *,
    loop: SimVLAGenerationLoop,
    transformer: Any,
    action_space: Any,
    condition: torch.Tensor,
    valid_mask: torch.Tensor,
    proprio: torch.Tensor,
    noise: torch.Tensor,
    code: torch.Tensor,
) -> torch.Tensor:
    def full_step(
        noisy_action: torch.Tensor, tau: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = full_generation_step_with_hidden(
            transformer,
            condition=condition,
            noisy_action=noisy_action,
            proprio=proprio,
            tau=tau,
            dt=-0.1,
        )
        return output.action_hidden, output.velocity

    trace = loop(
        noise,
        full_step=full_step,
        full_step_indices=GENERATION_SCHEDULES[3],
        proprio=proprio,
        condition=condition,
        condition_valid_mask=valid_mask,
        condition_change_code=code,
    )
    return action_space.postprocess(trace.final_noisy_action)


def _action_l1(action: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = (action.float() - target.float()).abs()
    return {
        "first5_l1": float(difference[:, :5].mean().item()),
        "full_chunk_l1": float(difference.mean().item()),
        "arm_l1": float(difference[:, :5, :6].mean().item()),
        "gripper_l1": float(difference[:, :5, 6:].mean().item()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise RuntimeError("coupled offline screen requires exactly one visible GPU")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    configure_strict_torch_determinism(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parent, parent_payload = load_generation_checkpoint(
        args.parent_generation_checkpoint, device=device
    )
    coupled, coupled_payload = load_generation_checkpoint(
        args.coupled_generation_checkpoint, device=device
    )
    if coupled_payload.get("training_config", {}).get("schema_version") != COUPLED_CHECKPOINT_SCHEMA:
        raise RuntimeError("checkpoint is not a real condition-code coupling checkpoint")
    checkpoint_k_c = int(coupled_payload["training_config"].get("k_c", -1))
    if checkpoint_k_c != args.k_c:
        raise RuntimeError(
            f"coupled checkpoint K_C mismatch: {checkpoint_k_c} != {args.k_c}"
        )
    if int(coupled_payload["training_config"].get("n_g", -1)) != 3:
        raise RuntimeError("coupled checkpoint N_G is not 3")
    source_report = verify_coupled_source_lock(
        coupled_payload["source_lock"],
        parent_generation_checkpoint=args.parent_generation_checkpoint,
        condition_checkpoint=args.condition_checkpoint,
        norm_stats=args.norm_stats,
        exact_cache=args.cache,
    )
    if source_report["verdict"] != "COUPLED_SOURCE_LOCK_PASS":
        raise RuntimeError(json.dumps(source_report, indent=2, sort_keys=True))
    freeze_module(parent)
    freeze_module(coupled)
    condition_adapter, _ = load_native_v0_checkpoint(
        args.condition_checkpoint, device=device, require_final_150k=True
    )
    freeze_module(condition_adapter)
    install_exact_uint8_delta_path(condition_adapter)
    model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    del processor
    _drop_unused_vlm(model)
    freeze_module(model)
    parent_loop = SimVLAGenerationLoop(parent, model.transformer.action_decoder).to(device).eval()
    coupled_loop = SimVLAGenerationLoop(coupled, model.transformer.action_decoder).to(device).eval()
    dataset = ExactTeacherSequenceDataset(
        args.cache,
        split="heldout",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    updated_ages = tuple(age for age in (1, 2, 3) if age % args.k_c != 0)
    total = min(int(args.queries), len(dataset) * len(updated_ages))
    rows: list[dict[str, Any]] = []
    zero_code_equal = True
    for flat_index in range(total):
        sequence_index, age_index = divmod(flat_index, len(updated_ages))
        age = updated_ages[age_index]
        sequence = move_batch(
            collate_exact_teacher_sequences([dataset[sequence_index]]), device
        )
        query = build_coupled_query(
            condition_adapter,
            sequence,
            query_ages=torch.tensor([age], device=device),
            k_c=args.k_c,
        )
        normalized_proprio = action_adapter.normalize_proprio(query["proprio"])
        zero = torch.zeros_like(query["condition_change_code"])
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        uncoupled_action = _decode(
            loop=parent_loop,
            transformer=model.transformer,
            action_space=action_adapter.action_space,
            condition=query["condition"],
            valid_mask=query["valid_mask"],
            proprio=normalized_proprio,
            noise=query["initial_noise"],
            code=zero,
        )
        coupled_zero_action = _decode(
            loop=coupled_loop,
            transformer=model.transformer,
            action_space=action_adapter.action_space,
            condition=query["condition"],
            valid_mask=query["valid_mask"],
            proprio=normalized_proprio,
            noise=query["initial_noise"],
            code=zero,
        )
        zero_code_equal = zero_code_equal and torch.equal(
            uncoupled_action, coupled_zero_action
        )
        coupled_action = _decode(
            loop=coupled_loop,
            transformer=model.transformer,
            action_space=action_adapter.action_space,
            condition=query["condition"],
            valid_mask=query["valid_mask"],
            proprio=normalized_proprio,
            noise=query["initial_noise"],
            code=query["condition_change_code"],
        )
        oracle_action = action_adapter.decode_action_from_condition(
            query["condition"],
            query["proprio"],
            steps=10,
            initial_noise=query["initial_noise"],
            requires_grad=False,
        )
        torch.cuda.synchronize(device)
        target = query["teacher_action"]
        teacher_condition = sequence["teacher_conditions"][:, age - 1]
        valid = query["valid_mask"].bool()
        token_cosine = F.cosine_similarity(
            query["condition"].float(), teacher_condition.float(), dim=-1
        )[valid]
        normalized_prediction = F.layer_norm(
            query["condition"].float(), (query["condition"].shape[-1],)
        )
        normalized_teacher = F.layer_norm(
            teacher_condition.float(), (teacher_condition.shape[-1],)
        )
        condition_mse = (
            (normalized_prediction - normalized_teacher).square()[valid].mean()
        )
        rows.append(
            {
                "flat_query_index": flat_index,
                "sequence_index": sequence_index,
                "query_age_in_window": age,
                "condition_code_norm": float(
                    query["condition_change_code"].float().norm(dim=-1).mean().item()
                ),
                "condition_teacher_cosine": float(token_cosine.mean().item()),
                "condition_teacher_normalized_mse": float(condition_mse.item()),
                "uncoupled_vs_local_oracle": _action_l1(uncoupled_action, oracle_action),
                "coupled_vs_local_oracle": _action_l1(coupled_action, oracle_action),
                "uncoupled_vs_full_condition_teacher": _action_l1(uncoupled_action, target),
                "coupled_vs_full_condition_teacher": _action_l1(coupled_action, target),
                "coupled_vs_uncoupled": _action_l1(coupled_action, uncoupled_action),
                "latency_ms_all_decodes": (time.perf_counter() - started) * 1000.0,
            }
        )

    metric_names = (
        "uncoupled_vs_local_oracle",
        "coupled_vs_local_oracle",
        "uncoupled_vs_full_condition_teacher",
        "coupled_vs_full_condition_teacher",
        "coupled_vs_uncoupled",
    )
    summaries = {
        group: {
            metric: _summary([float(row[group][metric]) for row in rows])
            for metric in ("first5_l1", "full_chunk_l1", "arm_l1", "gripper_l1")
        }
        for group in metric_names
    }
    finite = all(
        math.isfinite(value)
        for row in rows
        for group in metric_names
        for value in row[group].values()
    )
    projection_weight = coupled.condition_code_projection.weight.detach()
    state_audit = audit_projection_only_state(parent, coupled)
    checks = {
        "source_lock": source_report["verdict"] == "COUPLED_SOURCE_LOCK_PASS",
        "parent_optimizer_step_30000": int(parent_payload["optimizer_step"]) == 30_000,
        "coupled_optimizer_step_10000": int(coupled_payload["optimizer_step"]) == 10_000,
        "zero_code_action_path_bitwise_equal": zero_code_equal,
        "all_real_codes_nonzero": all(float(row["condition_code_norm"]) > 0 for row in rows),
        "trained_projection_weight_nonzero": bool(torch.count_nonzero(projection_weight).item()),
        "projection_only_state": state_audit["verdict"]
        == "PROJECTION_ONLY_STATE_PASS",
        "all_metrics_finite": finite,
    }
    result = {
        "verdict": "COUPLED_OFFLINE_INTEGRITY_PASS" if all(checks.values()) else "COUPLED_OFFLINE_INTEGRITY_FAIL",
        "paper_result": False,
        "requires_online_validation": True,
        "k_c": args.k_c,
        "queries": len(rows),
        "updated_query_ages": list(updated_ages),
        "checks": checks,
        "condition_code_norm": _summary([float(row["condition_code_norm"]) for row in rows]),
        "condition_teacher_cosine": _summary([float(row["condition_teacher_cosine"]) for row in rows]),
        "condition_teacher_normalized_mse": _summary(
            [float(row["condition_teacher_normalized_mse"]) for row in rows]
        ),
        "summaries": summaries,
        "projection_only_state_audit": state_audit,
        "source_report": source_report,
    }
    with (output / "offline_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(output / "offline_screen.json", result)
    if result["verdict"] != "COUPLED_OFFLINE_INTEGRITY_PASS":
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--parent-generation-checkpoint", required=True)
    parser.add_argument("--coupled-generation-checkpoint", required=True)
    parser.add_argument("--condition-checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--k-c", type=int, choices=(2, 3), default=2)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
