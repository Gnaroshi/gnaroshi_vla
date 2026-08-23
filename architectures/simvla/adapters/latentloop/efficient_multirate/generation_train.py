"""Short-budget one- or two-GPU trainer for the SimVLA Generation Loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    GENERATION_SCHEDULES,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherSequenceDataset,
    _drop_unused_vlm,
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    GenerationLoopConfig,
    load_generation_checkpoint,
    save_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_objective import (
    generation_local_oracle_loss,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_source_lock import (
    generation_source_lock,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    append_jsonl,
    configure_strict_torch_determinism,
    load_frozen_simvla,
    move_batch,
    write_json,
)
from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop


class RankDisjointStepSampler(Sampler[int]):
    """Deterministic unique logical samples across ranks and resume points."""

    def __init__(
        self,
        dataset_size: int,
        *,
        seed: int,
        rank: int,
        world_size: int,
        local_batch_size: int = 1,
        start_step: int,
        stop_step: int,
    ) -> None:
        if (
            dataset_size < 1
            or world_size < 1
            or local_batch_size < 1
            or not 0 <= rank < world_size
        ):
            raise ValueError("invalid distributed sampler contract")
        if start_step < 0 or stop_step <= start_step:
            raise ValueError("invalid optimizer-step interval")
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_batch_size = int(local_batch_size)
        self.global_unique_batch = self.world_size * self.local_batch_size
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)

    def index(self, optimizer_step: int, local_offset: int = 0) -> int:
        if not 0 <= int(local_offset) < self.local_batch_size:
            raise ValueError("local_offset is outside the local batch")
        logical = (
            int(optimizer_step) * self.global_unique_batch
            + self.rank * self.local_batch_size
            + int(local_offset)
        )
        epoch, offset = divmod(logical, self.dataset_size)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        return int(torch.randperm(self.dataset_size, generator=generator)[offset])

    def __iter__(self) -> Iterator[int]:
        for step in range(self.start_step, self.stop_step):
            for local_offset in range(self.local_batch_size):
                yield self.index(step, local_offset)

    def __len__(self) -> int:
        return (self.stop_step - self.start_step) * self.local_batch_size


class ExactTeacherGenerationDataset(ExactTeacherSequenceDataset):
    """Read q1-q3 teacher tensors without loading unused RGB observations."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        queries = [self.store.query(value) for value in self.windows[index]][1:]
        noises: list[torch.Tensor] = []
        for item in queries:
            metadata = item["metadata"]
            key_fields = metadata["noise_key"]
            key = ActionNoiseKey(
                checkpoint=str(key_fields["checkpoint"]),
                task_id=int(key_fields["task_id"]),
                episode_id=str(key_fields["episode_id"]),
                policy_query_index=int(key_fields["policy_query_index"]),
                seed_base=int(key_fields["seed_base"]),
            )
            if key.seed() != int(item["noise_seed"]):
                raise RuntimeError("cached query noise key changed")
            noises.append(
                explicit_action_noise(
                    key,
                    batch_size=1,
                    action_horizon=10,
                    action_dim=7,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )[0]
            )
        return {
            "conditions": torch.stack([item["condition"] for item in queries]),
            "valid_masks": torch.stack([item["valid_mask"] for item in queries]),
            "proprio": torch.stack([item["proprio"] for item in queries]),
            "teacher_actions": torch.stack([item["teacher_action"] for item in queries]),
            "explicit_noises": torch.stack(noises),
        }


def collate_generation_sequences(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    keys = ("conditions", "valid_masks", "proprio", "teacher_actions", "explicit_noises")
    return {key: torch.stack([item[key] for item in items]) for key in keys}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _query_batch(sequence: dict[str, Any], *, optimizer_step: int, rank: int) -> dict[str, torch.Tensor]:
    local_batch_size = int(sequence["conditions"].shape[0])
    global_unique_batch = dist.get_world_size() * local_batch_size
    age_indices = torch.tensor(
        [
            (
                int(optimizer_step) * global_unique_batch
                + int(rank) * local_batch_size
                + local_offset
            )
            % 3
            for local_offset in range(local_batch_size)
        ],
        device=sequence["conditions"].device,
        dtype=torch.long,
    )
    batch_indices = torch.arange(
        local_batch_size, device=sequence["conditions"].device
    )
    return {
        "condition": sequence["conditions"][batch_indices, age_indices],
        "valid_mask": sequence["valid_masks"][batch_indices, age_indices],
        "proprio": sequence["proprio"][batch_indices, age_indices],
        "initial_noise": sequence["explicit_noises"][batch_indices, age_indices],
        "teacher_action": sequence["teacher_actions"][batch_indices, age_indices],
        "query_age_in_window": age_indices + 1,
    }


def _lr_lambda(step: int, *, warmup_steps: int, total_steps: int, final_ratio: float) -> float:
    if step < warmup_steps:
        return max(1, step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(final_ratio + (1.0 - final_ratio) * cosine)


def _wandb(args: argparse.Namespace, rank: int, config: dict[str, Any]) -> Any | None:
    if rank != 0 or not args.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or Path(args.output).name,
        dir=args.output,
        config=config,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size not in (1, 2):
        raise RuntimeError("Generation Loop screening supports torchrun WORLD_SIZE in {1,2}")
    selected = tuple(
        int(value.strip())
        for value in os.environ["SIMVLA_GPU_IDS"].split(",")
        if value.strip()
    )
    if len(selected) != world_size or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(
        map(str, selected)
    ):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must exactly match SIMVLA_GPU_IDS and WORLD_SIZE")
    if int(args.local_batch_size) * world_size != 2:
        raise RuntimeError(
            "Generation Loop screening fixes unique global batch to 2: "
            "local_batch_size * WORLD_SIZE must equal 2"
        )
    if args.n_g not in (2, 3):
        raise ValueError("short-budget screening supports only N_G=2 or N_G=3")
    if not 1 <= args.stop_step <= args.schedule_total_steps <= 30_000:
        raise ValueError("require 1 <= stop_step <= schedule_total_steps <= 30000")
    if args.stop_step > 10_000 and not args.resume:
        raise RuntimeError("run the 10K checkpoint first, then resume to a longer budget")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    _seed_everything(args.seed + rank)
    determinism = configure_strict_torch_determinism(args.seed)

    source = generation_source_lock(
        checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        norm_stats=args.norm_stats,
        exact_cache=args.cache,
    )
    cache_validation = validate_exact_cache(args.cache, verify_checksums=False)
    if cache_validation["verdict"] != "EXACT_TEACHER_CACHE_VALID":
        raise RuntimeError(f"exact cache validation failed: {cache_validation}")
    cache_manifest = json.loads(
        (Path(args.cache).expanduser().resolve() / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if cache_manifest.get("checkpoint") != args.checkpoint:
        raise RuntimeError("exact cache checkpoint differs")
    if cache_manifest.get("norm_stats_sha256") != sha256_file(args.norm_stats):
        raise RuntimeError("exact cache normalization differs")

    output = Path(args.output).expanduser().resolve()
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(f"resume output does not exist: {output}")
    else:
        exists = torch.tensor(int(output.exists()), device=device)
        dist.all_reduce(exists, op=dist.ReduceOp.MAX)
        if int(exists.item()):
            raise FileExistsError(f"refusing existing output: {output}")
        if rank == 0:
            output.mkdir(parents=True)
    dist.barrier()

    dataset = ExactTeacherGenerationDataset(
        args.cache,
        split="train",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    config = GenerationLoopConfig(max_generator_age=4)
    updater = config.build().to(device)
    start_step = 0
    resume_payload: dict[str, Any] | None = None
    if args.resume:
        updater, resume_payload = load_generation_checkpoint(args.resume, device=device)
        start_step = int(resume_payload["optimizer_step"])
        if resume_payload["source_lock"]["combined_sha256"] != source["combined_sha256"]:
            raise RuntimeError("resume checkpoint source differs")
        if int(resume_payload["training_config"]["schedule_total_steps"]) != args.schedule_total_steps:
            raise RuntimeError("resume changed the predeclared scheduler horizon")
        if int(resume_payload["training_config"]["n_g"]) != args.n_g:
            raise RuntimeError("resume changed N_G")
    if args.stop_step <= start_step:
        raise ValueError("stop_step must be greater than resumed optimizer step")

    frozen_model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    del processor
    dropped = _drop_unused_vlm(frozen_model)
    loop = SimVLAGenerationLoop(updater, frozen_model.transformer.action_decoder).to(device)
    ddp = DistributedDataParallel(loop, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(
        ddp.module.updater.parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(
            step,
            warmup_steps=args.warmup_steps,
            total_steps=args.schedule_total_steps,
            final_ratio=args.final_lr_ratio,
        ),
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])

    sampler = RankDisjointStepSampler(
        len(dataset),
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        local_batch_size=args.local_batch_size,
        start_step=start_step,
        stop_step=args.stop_step,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.local_batch_size,
        "sampler": sampler,
        "collate_fn": collate_generation_sequences,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    loader = DataLoader(**loader_kwargs)

    trainable = sum(
        parameter.numel()
        for parameter in ddp.module.updater.parameters()
        if parameter.requires_grad
    )
    training_config = {
        "schema_version": "simvla_generation_training_v1",
        "n_g": int(args.n_g),
        "full_step_indices": list(GENERATION_SCHEDULES[args.n_g]),
        "stop_step": int(args.stop_step),
        "schedule_total_steps": int(args.schedule_total_steps),
        "warmup_steps": int(args.warmup_steps),
        "peak_lr": float(args.peak_lr),
        "final_lr_ratio": float(args.final_lr_ratio),
        "weight_decay": float(args.weight_decay),
        "global_unique_batch": int(args.local_batch_size) * world_size,
        "local_batch": int(args.local_batch_size),
        "world_size": world_size,
        "loss": {
            "primary": "layer_normalized_local_oracle_hidden_mse",
            "hidden_weight": 1.0,
            "velocity_weight": 0.0,
            "final_action_weight": 0.0,
            "velocity_and_action": "monitoring_only",
        },
        "condition_change_code": "zero for Generation-only full-condition lane",
        "trainable_parameters": trainable,
        "dataset_contract": dataset.contract(),
        "determinism": determinism,
        "frozen_release": dropped,
    }
    if rank == 0:
        write_json(output / "source_lock.json", source)
        write_json(output / "training_config.json", training_config)
        write_json(output / "parameter_audit.json", ddp.module.updater.parameter_audit())
    run = _wandb(args, rank, training_config)

    progress = tqdm(
        total=args.stop_step,
        initial=start_step,
        desc=f"SimVLA Generation N_G={args.n_g}",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    started = time.perf_counter()
    for zero_step, sequence in enumerate(loader, start=start_step):
        batch = move_batch(sequence, device)
        query = _query_batch(batch, optimizer_step=zero_step, rank=rank)
        normalized_proprio = action_adapter.normalize_proprio(query["proprio"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
            loss = generation_local_oracle_loss(
                loop=ddp,
                transformer=frozen_model.transformer,
                action_space=action_adapter.action_space,
                condition=query["condition"],
                initial_noise=query["initial_noise"],
                normalized_proprio=normalized_proprio,
                condition_valid_mask=query["valid_mask"],
                condition_change_code=query["condition"].new_zeros(
                    (query["condition"].shape[0], config.condition_code_dim)
                ),
                full_step_indices=GENERATION_SCHEDULES[args.n_g],
                teacher_final_action=query["teacher_action"],
                hidden_weight=1.0,
                velocity_weight=0.0,
                final_action_weight=0.0,
            )
        loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ddp.module.updater.parameters(), args.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        step = zero_step + 1

        if rank == 0 and (step == 1 or step % args.log_interval == 0):
            elapsed = time.perf_counter() - started
            query_ages = query["query_age_in_window"].detach().cpu().tolist()
            metrics = {
                "step": step,
                "loss/hidden_normalized_mse": float(loss.hidden_normalized_mse.item()),
                "monitor/velocity_l1": float(loss.velocity_l1.item()),
                "monitor/final_action_l1": float(loss.final_action_l1.item()),
                "optimizer/lr": float(optimizer.param_groups[0]["lr"]),
                "optimizer/grad_norm": float(grad_norm),
                "throughput/mean_step_seconds": elapsed / max(1, step - start_step),
                "memory/peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "query_age_in_window_mean": float(np.mean(query_ages)),
                "query_age_in_window_min": int(min(query_ages)),
                "query_age_in_window_max": int(max(query_ages)),
                "n_g": int(args.n_g),
            }
            append_jsonl(output / "train_metrics.jsonl", metrics)
            progress.set_postfix(
                hidden=f"{metrics['loss/hidden_normalized_mse']:.4g}",
                action=f"{metrics['monitor/final_action_l1']:.4g}",
                sec=f"{metrics['throughput/mean_step_seconds']:.3f}",
            )
            if run is not None:
                run.log(metrics, step=step)
        progress.update(1 if rank == 0 else 0)

        if step % args.save_interval == 0 or step == args.stop_step:
            dist.barrier()
            if rank == 0:
                checkpoint = output / "checkpoints" / f"generation_step_{step:06d}.pt"
                save_generation_checkpoint(
                    checkpoint,
                    updater=ddp.module.updater,
                    config=config,
                    optimizer_step=step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    source_lock=source,
                    training_config=training_config,
                )
                (output / "latest_checkpoint.txt").write_text(
                    str(checkpoint) + "\n", encoding="utf-8"
                )
            dist.barrier()

    progress.close()
    elapsed = time.perf_counter() - started
    summary = {
        "verdict": "GENERATION_LOOP_BUDGET_COMPLETE",
        "optimizer_step": int(args.stop_step),
        "schedule_total_steps": int(args.schedule_total_steps),
        "n_g": int(args.n_g),
        "trainable_parameters": trainable,
        "elapsed_seconds": elapsed,
        "mean_step_seconds": elapsed / max(1, args.stop_step - start_step),
        "source_combined_sha256": source["combined_sha256"],
    }
    if rank == 0:
        write_json(output / f"run_summary_step_{args.stop_step:06d}.json", summary)
        if run is not None:
            run.finish()
    dist.barrier()
    dist.destroy_process_group()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--n-g", type=int, default=2)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--stop-step", type=int, default=10_000)
    parser.add_argument("--schedule-total-steps", type=int, default=30_000)
    parser.add_argument("--resume", default="")
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--final-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
