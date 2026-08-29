"""Paired held-out screening for trained SimVLA Generation Loop checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    GENERATION_SCHEDULES,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    _drop_unused_vlm,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    full_generation_step_with_hidden,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_source_lock import (
    generation_source_lock,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_train import (
    ExactTeacherGenerationDataset,
    collate_generation_sequences,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    load_frozen_simvla,
    move_batch,
    write_json,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop


MEASURED_SCHEDULES = (10, 5, 3, 2)


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
    normalized_proprio: torch.Tensor,
    initial_noise: torch.Tensor,
    n_g: int,
) -> torch.Tensor:
    def full_step(
        noisy_action: torch.Tensor, tau: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = full_generation_step_with_hidden(
            transformer,
            condition=condition,
            noisy_action=noisy_action,
            proprio=normalized_proprio,
            tau=tau,
            dt=-0.1,
        )
        return output.action_hidden, output.velocity

    trace = loop(
        initial_noise,
        full_step=full_step,
        full_step_indices=GENERATION_SCHEDULES[n_g],
        proprio=normalized_proprio,
        condition=condition,
        condition_valid_mask=valid_mask,
        condition_change_code=condition.new_zeros(
            (condition.shape[0], loop.updater.condition_code_dim)
        ),
    )
    return action_space.postprocess(trace.final_noisy_action)


def run(args: argparse.Namespace) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size not in (1, 2):
        raise RuntimeError("offline Generation screening supports WORLD_SIZE in {1,2}")
    selected = tuple(
        int(value.strip())
        for value in os.environ["SIMVLA_GPU_IDS"].split(",")
        if value.strip()
    )
    if len(selected) != world_size or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(
        map(str, selected)
    ):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must exactly match SIMVLA_GPU_IDS and WORLD_SIZE")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    configure_strict_torch_determinism(args.seed)

    output = Path(args.output).expanduser().resolve()
    exists = torch.tensor(int(output.exists()), device=device)
    dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if int(exists.item()):
        raise FileExistsError(f"refusing existing output: {output}")
    if rank == 0:
        output.mkdir(parents=True)
    dist.barrier()

    source = generation_source_lock(
        checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        norm_stats=args.norm_stats,
        exact_cache=args.cache,
    )
    updater, checkpoint = load_generation_checkpoint(args.generation_checkpoint, device=device)
    if checkpoint["source_lock"]["combined_sha256"] != source["combined_sha256"]:
        raise RuntimeError("checkpoint and offline source locks differ")
    model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    del processor
    _drop_unused_vlm(model)
    loop = SimVLAGenerationLoop(updater, model.transformer.action_decoder).to(device).eval()
    dataset = ExactTeacherGenerationDataset(
        args.cache,
        split="heldout",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    total = min(int(args.queries), len(dataset) * 3)
    local_rows: list[dict[str, Any]] = []
    for flat_index in range(rank, total, world_size):
        sequence_index, age_index = divmod(flat_index, 3)
        sequence = move_batch(
            collate_generation_sequences([dataset[sequence_index]]), device
        )
        condition = sequence["conditions"][:, age_index]
        valid_mask = sequence["valid_masks"][:, age_index]
        proprio = sequence["proprio"][:, age_index]
        normalized_proprio = action_adapter.normalize_proprio(proprio)
        initial_noise = sequence["explicit_noises"][:, age_index]
        teacher = sequence["teacher_actions"][:, age_index]
        for n_g in MEASURED_SCHEDULES:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            action = _decode(
                loop=loop,
                transformer=model.transformer,
                action_space=action_adapter.action_space,
                condition=condition,
                valid_mask=valid_mask,
                normalized_proprio=normalized_proprio,
                initial_noise=initial_noise,
                n_g=n_g,
            )
            torch.cuda.synchronize(device)
            difference = (action.float() - teacher.float()).abs()
            local_rows.append(
                {
                    "flat_query_index": flat_index,
                    "sequence_index": sequence_index,
                    "query_age_in_window": age_index + 1,
                    "n_g": n_g,
                    "full_chunk_l1": float(difference.mean().item()),
                    "first5_l1": float(difference[:, :5].mean().item()),
                    "arm_l1": float(difference[..., :6].mean().item()),
                    "gripper_l1": float(difference[..., 6].mean().item()),
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            )

    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_rows)
    result: dict[str, Any] = {}
    if rank == 0:
        rows = sorted(
            [row for shard in gathered for row in (shard or [])],
            key=lambda row: (row["flat_query_index"], row["n_g"]),
        )
        summaries = {}
        for n_g in MEASURED_SCHEDULES:
            selected_rows = [row for row in rows if row["n_g"] == n_g]
            summaries[str(n_g)] = {
                name: _summary([float(row[name]) for row in selected_rows])
                for name in ("full_chunk_l1", "first5_l1", "arm_l1", "gripper_l1", "latency_ms")
            }
        ng3 = summaries["3"]["first5_l1"]
        ng2 = summaries["2"]["first5_l1"]
        candidate = (
            2
            if float(ng2["mean"]) <= 1.10 * float(ng3["mean"])
            and float(ng2["p95"]) <= 1.20 * float(ng3["p95"])
            else 3
        )
        result = {
            "verdict": "GENERATION_OFFLINE_SCREEN_COMPLETE",
            "paper_result": False,
            "requires_online_validation": True,
            "checkpoint": str(Path(args.generation_checkpoint).resolve()),
            "optimizer_step": int(checkpoint["optimizer_step"]),
            "queries_per_schedule": total,
            "paired_schedules": list(MEASURED_SCHEDULES),
            "candidate_n_g": candidate,
            "candidate_rule": "N_G=2 only when mean<=1.10x and p95<=1.20x N_G=3 first5 L1",
            "summaries": summaries,
            "source_combined_sha256": source["combined_sha256"],
        }
        write_json(output / "offline_screen.json", result)
        with (output / "offline_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--generation-checkpoint", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
