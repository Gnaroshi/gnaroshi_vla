"""Cache-efficient real adaptation of the released SimVLA action transformer."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from .artifact_validation import (
    validate_real_training_sources,
)
from .condition_cache import RealConditionCacheDataset
from .distributed import initialize_distributed, seed_process
from .io_utils import atomic_write_json
from .model_io import (
    load_exact_official_model,
    official_base_identity,
    save_real_action_checkpoint,
)


def _flow_loss(
    transformer: torch.nn.Module,
    action_space: Any,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    condition = batch["condition"].to(device, non_blocking=True)
    proprio = batch["proprio"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True)
    action_norm = action_space.normalize_action(action)
    proprio_norm = action_space.normalize_state(proprio)
    beta = torch.distributions.Beta(
        torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)
    )
    tau = beta.sample((action.shape[0],)) * 0.999 + 0.001
    noise = torch.randn_like(action_norm)
    noisy_action = tau[:, None, None] * noise + (1.0 - tau[:, None, None]) * action_norm
    target_velocity = noise - action_norm
    predicted = transformer(
        vlm_features=condition,
        action_with_noise=noisy_action,
        proprio=proprio_norm,
        t=tau,
    )
    return torch.square(predicted.float() - target_velocity.float()).mean()


@torch.no_grad()
def _validation_loss(
    transformer: torch.nn.Module,
    action_space: Any,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    transformer.eval()
    losses = []
    generator = torch.Generator(device=device)
    generator.manual_seed(20260904)
    for number, batch in enumerate(loader):
        if number >= max_batches:
            break
        condition = batch["condition"].to(device, non_blocking=True)
        proprio = batch["proprio"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        action_norm = action_space.normalize_action(action)
        proprio_norm = action_space.normalize_state(proprio)
        tau = torch.full((action.shape[0],), 0.5, device=device)
        noise = torch.randn(
            action_norm.shape,
            generator=generator,
            device=device,
            dtype=action_norm.dtype,
        )
        noisy_action = 0.5 * noise + 0.5 * action_norm
        target = noise - action_norm
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predicted = transformer(
                vlm_features=condition,
                action_with_noise=noisy_action,
                proprio=proprio_norm,
                t=tau,
            )
        losses.append(torch.square(predicted.float() - target.float()).mean())
    transformer.train()
    return float(torch.stack(losses).mean().item())


def _learning_rate(step: int, *, total: int, warmup: int, peak: float, floor_ratio: float) -> float:
    if step < warmup:
        return peak * float(step + 1) / float(max(warmup, 1))
    progress = float(step - warmup) / float(max(total - warmup, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return peak * (floor_ratio + (1.0 - floor_ratio) * cosine)


def train(args: argparse.Namespace) -> dict[str, Any]:
    context = initialize_distributed(args.device)
    seed_process(args.seed, context.rank)
    try:
        output = Path(args.output).expanduser().resolve()
        if context.primary:
            output.mkdir(parents=True, exist_ok=True)
            (output / "checkpoints").mkdir(exist_ok=True)
        context.barrier()

        source_contract = validate_real_training_sources(
            condition_cache=args.condition_cache,
            checkpoint=args.checkpoint,
            processor=args.processor,
            norm_stats=args.norm_stats,
            verify_cache_array_checksums=False,
            condition_cache_attestation=args.condition_cache_attestation,
        )
        cache_manifest = source_contract["condition_cache"]
        model, processor, loading = load_exact_official_model(
            model_directory=args.checkpoint,
            processor_directory=args.processor,
            norm_stats=args.norm_stats,
            device="cpu",
            freeze_vlm=True,
            freeze_action_transformer=False,
        )
        del processor
        transformer = model.transformer.to(context.device)
        action_space = model.action_space.to(context.device)
        del model.vlm
        gc.collect()
        if context.device.type == "cuda":
            torch.cuda.empty_cache()

        train_dataset = RealConditionCacheDataset(args.condition_cache, split="train")
        validation_dataset = RealConditionCacheDataset(args.condition_cache, split="validation")
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.local_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=context.device.type == "cuda",
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.local_batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 2),
            pin_memory=context.device.type == "cuda",
        )
        trainable = sum(value.numel() for value in transformer.parameters())
        ddp: torch.nn.Module = transformer
        if context.world_size > 1:
            ddp = DistributedDataParallel(
                transformer,
                device_ids=[context.local_rank],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        optimizer = torch.optim.AdamW(
            ddp.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=context.device.type == "cuda",
        )
        effective_batch = (
            args.local_batch_size * context.world_size * args.gradient_accumulation_steps
        )
        training_config = {
            "protocol": "official_full_checkpoint_then_frozen_vlm_action_transformer_finetune",
            "max_steps": args.max_steps,
            "local_batch_size": args.local_batch_size,
            "world_size": context.world_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch_size": effective_batch,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "minimum_lr_ratio": args.minimum_lr_ratio,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "condition_cache": str(Path(args.condition_cache).resolve()),
            "condition_cache_identity_sha256": cache_manifest[
                "condition_cache_identity_sha256"
            ],
            "condition_cache_attestation_identity_sha256": source_contract[
                "condition_cache_attestation"
            ]["attestation_identity_sha256"],
            "action_transformer_trainable_parameters": trainable,
            "vlm_trainable_parameters": 0,
            "action_transformer_reinitialized": False,
        }
        if context.primary:
            atomic_write_json(output / "training_config.json", training_config)
            atomic_write_json(output / "exact_initialization.json", loading)

        optimizer.zero_grad(set_to_none=True)
        epoch = 0
        iterator = iter(train_loader)
        progress = tqdm(
            total=args.max_steps,
            initial=0,
            disable=not context.primary,
            desc="real SimVLA action-head fine-tune",
        )
        history_path = output / "train_metrics.jsonl"
        last_validation: dict[str, Any] = {}
        started = time.perf_counter()
        step = 0
        while step < args.max_steps:
            accumulated = 0.0
            for micro_step in range(args.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    iterator = iter(train_loader)
                    batch = next(iterator)
                sync = micro_step + 1 == args.gradient_accumulation_steps
                synchronization = (
                    ddp.no_sync() if context.world_size > 1 and not sync else torch.enable_grad()
                )
                with synchronization:
                    with torch.autocast(
                        device_type=context.device.type,
                        dtype=torch.bfloat16,
                        enabled=context.device.type == "cuda",
                    ):
                        loss = _flow_loss(ddp, action_space, batch, context.device)
                    (loss / args.gradient_accumulation_steps).backward()
                accumulated += float(loss.detach().item())
            next_step = step + 1
            lr = _learning_rate(
                next_step - 1,
                total=args.max_steps,
                warmup=args.warmup_steps,
                peak=args.learning_rate,
                floor_ratio=args.minimum_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = next_step
            progress.update(1)
            progress.set_postfix(loss=f"{accumulated / args.gradient_accumulation_steps:.5f}", lr=f"{lr:.2e}")

            checkpoint_due = step % args.save_interval == 0 or step == args.max_steps
            if checkpoint_due:
                context.barrier()
                if context.primary:
                    validation_loss = _validation_loss(
                        transformer,
                        action_space,
                        validation_loader,
                        context.device,
                        args.validation_batches,
                    )
                    last_validation = {
                        "flow_mse": validation_loss,
                        "batches": min(args.validation_batches, len(validation_loader)),
                    }
                    checkpoint = output / "checkpoints" / f"action_transformer_step_{step:06d}.pt"
                    base = official_base_identity(args.checkpoint, args.processor)
                    save_real_action_checkpoint(
                        checkpoint,
                        transformer=transformer,
                        official_base=base,
                        norm_stats_path=args.norm_stats,
                        dataset_identity_sha256=cache_manifest["dataset_identity_sha256"],
                        optimizer_step=step,
                        training_config=training_config,
                        validation=last_validation,
                    )
                    (output / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n", encoding="utf-8")
                    old = sorted((output / "checkpoints").glob("action_transformer_step_*.pt"))
                    for stale in old[: max(0, len(old) - args.keep_checkpoints)]:
                        stale.unlink()
                    with history_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "train_flow_mse": accumulated / args.gradient_accumulation_steps,
                                    "validation_flow_mse": validation_loss,
                                    "learning_rate": lr,
                                    "elapsed_s": time.perf_counter() - started,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                context.barrier()
        progress.close()
        result = {
            "verdict": "REAL_BASELINE_FINETUNE_COMPLETE",
            "optimizer_steps": step,
            "effective_global_batch_size": effective_batch,
            "last_validation": last_validation,
            "elapsed_s": time.perf_counter() - started,
            "output": str(output),
        }
        if context.primary:
            atomic_write_json(output / "run_summary.json", result)
        return result
    finally:
        context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-cache", required=True)
    parser.add_argument("--condition-cache-attestation", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    result = train(build_parser().parse_args())
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
