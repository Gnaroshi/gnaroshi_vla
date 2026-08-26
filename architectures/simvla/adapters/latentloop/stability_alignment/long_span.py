"""Conditional q0-q7 stability alignment after K_C=4 offline/online gates."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    _drop_unused_vlm,
    collate_exact_teacher_sequences,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
    move_batch,
)
from architectures.simvla.adapters.latentloop.stability_alignment.age_encoding import (
    enable_conditional_kc8_age_support,
)
from architectures.simvla.adapters.latentloop.stability_alignment.checkpoint import (
    GroupWarmupCosine,
    load_checkpoint,
    load_modules_from_checkpoint,
    save_checkpoint,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    GRAD_CLIP_NORM,
    LOSS_SCHEMA,
    atomic_write_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment.data import (
    ReplicatedEventAwareSampler,
    StabilityExactTeacherDataset,
    build_event_index,
)
from architectures.simvla.adapters.latentloop.stability_alignment.model import (
    configure_condition_only_stage,
    optimizer_parameter_groups,
)
from architectures.simvla.adapters.latentloop.stability_alignment.objectives import (
    LOSS_NAMES,
    weighted_total,
)
from architectures.simvla.adapters.latentloop.stability_alignment.trainer import (
    _allreduce_gradients,
    _condition_and_action_rows,
    _forward,
    _summary,
)


LONG_SCHEMA = "simvla_stability_long_span_v1"
UNROLL_PATTERN = (1, 4, 2, 5, 3, 6, 1, 7)


def _distributed() -> tuple[int, int, torch.device]:
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("long-span stability requires exactly two torchrun ranks")
    selected = tuple(
        int(value)
        for value in os.environ.get("SIMVLA_GPU_IDS", "").split(",")
        if value
    )
    if len(selected) != 2:
        raise RuntimeError("SIMVLA_GPU_IDS must contain exactly two physical GPUs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, selected)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES changed from SIMVLA_GPU_IDS")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, torch.device(f"cuda:{local_rank}")


def _cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _slice_batch(batch: Mapping[str, Any], max_age: int) -> dict[str, Any]:
    value = int(max_age)
    if value < 1 or value > 7:
        raise ValueError("long-span unroll length must be in [1,7]")
    result = dict(batch)
    result["image_sequence"] = batch["image_sequence"][:, : value + 1]
    result["proprio_sequence"] = batch["proprio_sequence"][:, : value + 1]
    for name in ("teacher_conditions", "teacher_actions", "explicit_noises"):
        result[name] = batch[name][:, :value]
    return result


def variable_unroll_length(optimizer_step: int) -> int:
    return UNROLL_PATTERN[int(optimizer_step) % len(UNROLL_PATTERN)]


def _source_lock(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema_version": LONG_SCHEMA,
        "short_parent": str(Path(args.short_parent).expanduser().resolve()),
        "short_parent_sha256": sha256_file(args.short_parent),
        "cache": str(Path(args.cache).expanduser().resolve()),
        "cache_manifest_sha256": sha256_file(Path(args.cache) / "manifest.json"),
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "loss_weights": str(Path(args.loss_weights).expanduser().resolve()),
        "loss_weights_sha256": sha256_file(args.loss_weights),
        "checkpoint": str(args.checkpoint),
        "smolvlm_model": str(args.smolvlm_model),
        "split_seed": int(args.split_seed),
        "training_seed": int(args.seed),
        "condition_ages": list(range(1, 8)),
        "unroll_pattern": list(UNROLL_PATTERN),
        "short_unroll_fraction": 0.5,
        "n_g": 3,
        "full_generation_indices": [0, 4, 8],
        "training_mode": "condition_only_frozen_generation",
        "generation_condition_change_code": "zero",
        "optional_joint_branch_automatic": False,
        "scheduler_horizon": 30000,
    }
    payload["combined_sha256"] = canonical_sha256(payload)
    return payload


def _weights(path: str | Path) -> tuple[dict[str, float], float]:
    payload = load_json(path)
    if payload.get("schema_version") != LOSS_SCHEMA:
        raise ValueError("short-span frozen loss-weight schema changed")
    weights = {name: float(payload["weights"][name]) for name in LOSS_NAMES}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("long-span weights must be finite and non-negative")
    return weights, float(payload["base_corrective_lr"])


def _dataset(
    args: argparse.Namespace,
    *,
    split: str,
    output: Path,
    rank: int,
) -> tuple[StabilityExactTeacherDataset, dict[str, Any] | None]:
    dataset = StabilityExactTeacherDataset(
        args.cache,
        split=split,
        split_seed=args.split_seed,
        max_age=7,
    )
    if split != "train":
        return dataset, None
    event_path = output / "event_index_q0_q7.json"
    if rank == 0 and not event_path.exists():
        atomic_write_json(event_path, build_event_index(dataset))
    dist.barrier()
    event = load_json(event_path)
    if event.get("split_sha256") != dataset.split_sha256:
        raise RuntimeError("q0-q7 event index split changed")
    return dataset, event


def _loader(
    dataset: StabilityExactTeacherDataset,
    sampler: ReplicatedEventAwareSampler,
    workers: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 1,
        "sampler": sampler,
        "collate_fn": collate_exact_teacher_sequences,
        "num_workers": int(workers),
        "pin_memory": True,
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def command_catalog(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        observed = load_json(output)
        if observed.get("cache_manifest_sha256") == sha256_file(
            Path(args.cache) / "manifest.json"
        ):
            return observed
        raise FileExistsError(f"refusing incompatible q0-q7 catalog: {output}")
    dataset = StabilityExactTeacherDataset(
        args.cache, split="all", split_seed=args.split_seed, max_age=7
    )
    result = {
        "schema_version": LONG_SCHEMA,
        "verdict": "Q0_Q7_INDEX_READY",
        "cache": str(Path(args.cache).expanduser().resolve()),
        "cache_manifest_sha256": sha256_file(Path(args.cache) / "manifest.json"),
        "sequences": len(dataset),
        "split_sha256": dataset.split_sha256,
        "condition_ages": list(range(1, 8)),
        "query_tensors_duplicated": False,
    }
    atomic_write_json(output, result)
    return result


def command_train(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, device = _distributed()
    try:
        _seed(args.seed + rank)
        determinism = configure_strict_torch_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        resume = bool(args.resume)
        if rank == 0:
            if output.exists() and not resume:
                raise FileExistsError(f"refusing existing long-span output: {output}")
            output.mkdir(parents=True, exist_ok=resume)
        dist.barrier()
        source = _source_lock(args)
        modules, short_payload = load_modules_from_checkpoint(args.short_parent, device=device)
        age_contract = enable_conditional_kc8_age_support(
            modules.condition.condition_updater
        )
        parent_modules = copy.deepcopy(modules).to(device).eval()
        freeze_module(parent_modules)
        stage_audit = configure_condition_only_stage(modules)
        weights, base_lr = _weights(args.loss_weights)
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(
                modules, base_lr=base_lr, weight_decay=args.weight_decay
            )
        )
        scheduler = GroupWarmupCosine(optimizer)
        start_step = 0
        if resume:
            checkpoint = load_checkpoint(
                args.resume, modules=modules, optimizer=optimizer, scheduler=scheduler
            )
            start_step = int(checkpoint["optimizer_step"])
            if checkpoint["source_lock"]["combined_sha256"] != source["combined_sha256"]:
                raise RuntimeError("long-span resume source lock changed")
        if start_step not in {0, 10000} or args.stop_step not in {10000, 30000}:
            raise RuntimeError("long-span boundaries must be 0->10K or 10K->30K")
        scheduler.set_step(start_step)
        frozen_model, processor, action_adapter = load_frozen_simvla(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            device=device,
        )
        del processor
        _drop_unused_vlm(frozen_model)
        dataset, event = _dataset(args, split="train", output=output, rank=rank)
        sampler = ReplicatedEventAwareSampler(
            event,
            seed=args.seed,
            start_step=start_step,
            stop_step=args.stop_step,
        )
        loader = _loader(dataset, sampler, args.num_workers)
        contract = {
            "schema_version": LONG_SCHEMA,
            "source_lock": source,
            "short_parent_optimizer_step": int(short_payload["optimizer_step"]),
            "age_encoding": age_contract,
            "dataset": dataset.contract(),
            "unroll_pattern": list(UNROLL_PATTERN),
            "short_age_fraction": 0.5,
            "loss_weights": weights,
            "original_simvla_frozen": True,
            "generation_ng3_retained": True,
            "generation_updater_frozen": True,
            "condition_code_projection_frozen": True,
            "generation_condition_change_code": "zero; exact validated Generation parent lane",
            "optional_joint_branch_automatic": False,
            "training_mode": "condition_only_frozen_generation",
            "stage_audit": stage_audit,
        }
        if rank == 0:
            atomic_write_json(output / "source_lock.json", source)
            atomic_write_json(output / "training_contract.json", contract)
            atomic_write_json(output / "determinism.json", determinism)
        iterator = iter(loader)
        progress = tqdm(
            total=args.stop_step,
            initial=start_step,
            disable=rank != 0,
            dynamic_ncols=True,
            desc="Stability K_C=8",
        )
        wall = time.perf_counter()
        latest: dict[str, Any] = {}
        for zero_step in range(start_step, args.stop_step):
            batch = move_batch(next(iterator), device)
            unroll = variable_unroll_length(zero_step)
            batch = _slice_batch(batch, unroll)
            optimizer.zero_grad(set_to_none=True)
            lrs = scheduler.set_step(zero_step + 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                raw, diagnostics, _ = _forward(
                    modules=modules,
                    parent_condition=parent_modules.condition,
                    parent_generation=parent_modules.generation,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    optimizer_step=zero_step,
                    requires_grad=True,
                )
                total, weighted = weighted_total(raw, weights)
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(f"non-finite long-span loss at {zero_step + 1}")
            total.backward()
            _allreduce_gradients(modules)
            if any(parameter.grad is not None for parameter in frozen_model.parameters()):
                raise RuntimeError("frozen original SimVLA received gradients")
            if any(parameter.grad is not None for parameter in modules.generation.parameters()):
                raise RuntimeError("frozen validated Generation updater received gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in modules.parameters() if parameter.requires_grad],
                GRAD_CLIP_NORM,
            )
            optimizer.step()
            step = zero_step + 1
            latest = {
                "step": step,
                "unroll_age": unroll,
                "total": float(total.detach().item()),
                "grad_norm_before_clip": float(grad_norm.item()),
                **{f"raw/{name}": float(value.detach().item()) for name, value in raw.items()},
                **{f"weighted/{name}": float(value.detach().item()) for name, value in weighted.items()},
                **{f"diagnostic/{name}": float(value.detach().item()) for name, value in diagnostics.items()},
                **{f"lr/{name}": value for name, value in lrs.items()},
            }
            if rank == 0:
                progress.update(1)
                progress.set_postfix(loss=f"{latest['total']:.4g}", age=unroll)
                if step == 1 or step % args.log_interval == 0 or step == args.stop_step:
                    with (output / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(latest, sort_keys=True) + "\n")
            if step % args.save_interval == 0 or step == args.stop_step:
                if rank == 0:
                    path = output / "checkpoints" / f"stability_long_step_{step:06d}.pt"
                    save_checkpoint(
                        path,
                        modules=modules,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        optimizer_step=step,
                        sampler_state=sampler.state_dict(step),
                        source_lock=source,
                        training_contract=contract,
                        parent_identity={
                            "short_parent": str(Path(args.short_parent).resolve()),
                            "short_parent_sha256": sha256_file(args.short_parent),
                        },
                    )
                    (output / "latest_checkpoint.txt").write_text(
                        str(path) + "\n", encoding="utf-8"
                    )
                dist.barrier()
        progress.close()
        result = {
            "verdict": "STABILITY_LONG_TRAINING_SEGMENT_COMPLETE",
            "optimizer_step": args.stop_step,
            "elapsed_seconds": time.perf_counter() - wall,
            "latest_metrics": latest,
            "source_combined_sha256": source["combined_sha256"],
            "short_age_fraction": 0.5,
        }
        if rank == 0:
            atomic_write_json(output / "run_summary.json", result)
        return result
    finally:
        _cleanup()


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = tuple(
        key
        for key in rows[0]
        if key not in {"dataset_index", "task_id", "episode_id", "age"}
    )
    by_age = {
        str(age): {
            field: _summary(
                [float(row[field]) for row in rows if int(row["age"]) == age]
            )
            for field in fields
        }
        for age in range(1, 8)
    }

    def mean(age: int, name: str) -> float:
        return float(by_age[str(age)][name]["mean"])

    def p99(age: int, name: str) -> float:
        return float(by_age[str(age)][name]["p99"])

    epsilon = 1e-12
    checks = {
        "ages1_3_teacher_forced_preserved": all(
            mean(age, "candidate_teacher_first_r")
            <= 1.05 * max(mean(age, "parent_teacher_first_r"), epsilon)
            for age in (1, 2, 3)
        ),
        "ages1_3_final_ng3_preserved": all(
            mean(age, "candidate_joint_first_r")
            <= 1.05 * max(mean(age, "parent_joint_first_r"), epsilon)
            for age in (1, 2, 3)
        ),
        "age7_recurrence_excess_improved": mean(7, "candidate_recurrence_excess")
        < mean(7, "parent_recurrence_excess"),
        "age7_final_ng3_improved": mean(7, "candidate_joint_first_r")
        < mean(7, "parent_joint_first_r"),
        "age7_p99_no_collapse": p99(7, "candidate_recursive_first_r")
        <= 1.25 * max(p99(7, "parent_recursive_first_r"), epsilon),
        "age7_gripper_no_collapse": mean(7, "candidate_gripper_sign_mismatch")
        <= 1.25 * max(mean(7, "parent_gripper_sign_mismatch"), 1.0),
        "exact_ng3_preserved": sum(
            mean(age, "candidate_exact_ng3_first_r") for age in (1, 2, 3)
        )
        <= 1.05
        * max(
            sum(mean(age, "parent_exact_ng3_first_r") for age in (1, 2, 3)),
            epsilon,
        ),
    }
    return {"by_age": by_age, "checks": checks, "passed": all(checks.values())}


def _final_20pct_slope(candidate: str | Path) -> float:
    path = Path(candidate).expanduser().resolve().parents[1] / "train_metrics.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = rows[max(0, int(0.8 * len(rows))) :]
    if len(selected) < 4:
        raise RuntimeError("long-span slope requires four final-window log points")
    return float(
        np.polyfit(
            [float(row["step"]) for row in selected],
            [float(row["raw/recursive_stability"]) for row in selected],
            1,
        )[0]
    )


def command_offline(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, device = _distributed()
    try:
        _seed(args.seed + rank)
        configure_strict_torch_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"refusing existing long offline output: {output}")
            output.mkdir(parents=True)
        dist.barrier()
        modules, candidate_payload = load_modules_from_checkpoint(args.candidate, device=device)
        parent, _ = load_modules_from_checkpoint(args.short_parent, device=device)
        enable_conditional_kc8_age_support(parent.condition.condition_updater)
        freeze_module(modules)
        freeze_module(parent)
        frozen_model, processor, action_adapter = load_frozen_simvla(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            device=device,
        )
        del processor
        _drop_unused_vlm(frozen_model)
        dataset = StabilityExactTeacherDataset(
            args.cache,
            split="final_offline",
            split_seed=args.split_seed,
            max_age=7,
        )
        rows: list[dict[str, Any]] = []
        for index in tqdm(
            range(rank, len(dataset), 2),
            disable=False,
            dynamic_ncols=True,
            desc=f"long offline rank{rank}",
        ):
            batch = move_batch(
                collate_exact_teacher_sequences([dataset[index]]), device
            )
            rows.extend(
                _condition_and_action_rows(
                    modules=modules,
                    parent_condition=parent.condition,
                    parent_generation=parent.generation,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    dataset_index=index,
                )
            )
        atomic_write_json(output / f"rank_{rank}_rows.json", rows)
        dist.barrier()
        result: dict[str, Any] = {}
        if rank == 0:
            merged = [
                row
                for shard in range(2)
                for row in load_json(output / f"rank_{shard}_rows.json")
            ]
            merged.sort(key=lambda row: (int(row["dataset_index"]), int(row["age"])))
            with (output / "recursive_stability_metrics_age1_7.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
                writer.writeheader()
                writer.writerows(merged)
            aggregate = _aggregate(merged)
            slope = _final_20pct_slope(args.candidate)
            result = {
                "schema_version": LONG_SCHEMA,
                "verdict": "KC8_OFFLINE_READY" if aggregate["passed"] else "KC8_OFFLINE_BLOCKED",
                "passed": aggregate["passed"],
                "optimizer_step": int(candidate_payload["optimizer_step"]),
                "candidate_sha256": sha256_file(args.candidate),
                "short_parent_sha256": sha256_file(args.short_parent),
                "dataset": dataset.contract(),
                "aggregate": aggregate,
                "final_20pct_stability_slope": slope,
                "continuation_to_30k_allowed": bool(aggregate["passed"] and slope < 0.0),
            }
            atomic_write_json(output / "offline_gate.json", result)
        dist.barrier()
        return result
    finally:
        _cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog")
    catalog.add_argument("--output", required=True)
    catalog.add_argument("--cache", required=True)
    catalog.add_argument("--split-seed", type=int, default=20260822)

    def common(value: argparse.ArgumentParser) -> None:
        value.add_argument("--output", required=True)
        value.add_argument("--cache", required=True)
        value.add_argument("--short-parent", required=True)
        value.add_argument("--norm-stats", required=True)
        value.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
        value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
        value.add_argument("--split-seed", type=int, default=20260822)
        value.add_argument("--seed", type=int, default=20260825)
        value.add_argument("--num-workers", type=int, default=2)

    train = sub.add_parser("train")
    common(train)
    train.add_argument("--loss-weights", required=True)
    train.add_argument("--resume", default="")
    train.add_argument("--stop-step", type=int, choices=(10000, 30000), required=True)
    train.add_argument("--save-interval", type=int, default=2000)
    train.add_argument("--log-interval", type=int, default=20)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)

    offline = sub.add_parser("offline")
    common(offline)
    offline.add_argument("--candidate", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "catalog":
        result = command_catalog(args)
    elif args.command == "train":
        result = command_train(args)
    elif args.command == "offline":
        result = command_offline(args)
    else:
        raise AssertionError(args.command)
    if not dist.is_available() or not dist.is_initialized():
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
