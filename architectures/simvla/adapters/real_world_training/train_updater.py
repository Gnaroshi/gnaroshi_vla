"""Train real Condition/Generation updaters from one frozen real baseline."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from architectures.simvla.adapters.dcld.simvla_action_adapter import SimVLAActionAdapter
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_objective import (
    generation_local_oracle_loss,
)
from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop

from .artifact_validation import (
    validate_real_baseline_checkpoint,
    validate_real_training_sources,
)
from .condition_cache import RealConditionCacheDataset
from .dataset import align_current_rotvec_proprio
from .distributed import initialize_distributed, seed_process
from .io_utils import atomic_write_json, sha256_file, stable_int_seed
from .model_io import load_exact_official_model
from .updater_data import RealConditionPairDataset
from .updater_io import (
    RealConditionConfig,
    RealGenerationConfig,
    save_real_updater,
)


def _lr(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * float(step + 1) / float(max(warmup, 1))
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _layout(cache_manifest: dict[str, Any], batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    layout = cache_manifest["token_layout"]
    valid = torch.as_tensor(layout["valid_mask"][0], dtype=torch.bool, device=device)
    groups = torch.as_tensor(layout["group_ids"][0], dtype=torch.long, device=device)
    return valid.unsqueeze(0).expand(batch, -1), groups.unsqueeze(0).expand(batch, -1)


def _condition_raw_losses(
    updater: torch.nn.Module,
    projection: torch.nn.Module,
    batch: dict[str, Any],
    cache_manifest: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    previous = batch["previous_condition"].to(device, non_blocking=True)
    target = batch["current_condition"].to(device, non_blocking=True)
    previous_q = batch["previous_proprio"].to(device, non_blocking=True)
    current_q = batch["current_proprio"].to(device, non_blocking=True)
    updater_current_q = align_current_rotvec_proprio(previous_q, current_q)
    previous_images = batch["previous_images"].to(device, non_blocking=True)
    current_images = batch["current_images"].to(device, non_blocking=True)
    valid, groups = _layout(cache_manifest, previous.shape[0], device)
    update = updater.update_once(
        previous,
        NativeV0ObservationPair(
            previous_images=previous_images,
            current_images=current_images,
            previous_proprio=previous_q,
            current_proprio=updater_current_q,
        ),
        valid_mask=valid,
        group_ids=groups,
        age=1,
    )
    mask = valid.unsqueeze(-1).to(dtype=torch.float32)
    predicted_norm = F.layer_norm(update.condition.float(), (update.condition.shape[-1],))
    target_norm = F.layer_norm(target.float(), (target.shape[-1],))
    condition_mse = (((predicted_norm - target_norm) ** 2) * mask).sum() / (
        mask.sum().clamp_min(1) * predicted_norm.shape[-1]
    )
    projected = projection(update.condition)
    target_projected = projection(target).detach()
    projected_norm = F.layer_norm(projected.float(), (projected.shape[-1],))
    target_projected_norm = F.layer_norm(target_projected.float(), (target_projected.shape[-1],))
    projected_mse = (((projected_norm - target_projected_norm) ** 2) * mask).sum() / (
        mask.sum().clamp_min(1) * projected_norm.shape[-1]
    )
    cosine = F.cosine_similarity(update.condition.float(), target.float(), dim=-1)
    cosine_distance = ((1.0 - cosine) * valid.float()).sum() / valid.float().sum().clamp_min(1)
    return {
        "condition_normalized_mse": condition_mse,
        "action_projection_normalized_mse": projected_mse,
        "condition_cosine_distance": cosine_distance,
    }, update.condition


@torch.no_grad()
def _calibrate_condition(
    updater: torch.nn.Module,
    projection: torch.nn.Module,
    loader: DataLoader,
    cache_manifest: dict[str, Any],
    device: torch.device,
    batches: int,
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for number, batch in enumerate(loader):
        if number >= batches:
            break
        losses, _ = _condition_raw_losses(updater, projection, batch, cache_manifest, device)
        for name, value in losses.items():
            values.setdefault(name, []).append(float(value.item()))
    return {name: max(sum(items) / len(items), 1e-8) for name, items in values.items()}


@torch.no_grad()
def _validate_condition(
    updater: torch.nn.Module,
    projection: torch.nn.Module,
    loader: DataLoader,
    cache_manifest: dict[str, Any],
    device: torch.device,
    batches: int,
) -> dict[str, float]:
    updater.eval()
    collected: dict[str, list[float]] = {}
    for number, batch in enumerate(loader):
        if number >= batches:
            break
        losses, _ = _condition_raw_losses(updater, projection, batch, cache_manifest, device)
        for name, value in losses.items():
            collected.setdefault(name, []).append(float(value.item()))
    updater.train()
    return {name: sum(items) / len(items) for name, items in collected.items()}


def _noise(indices: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = []
    for index in indices:
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_int_seed("real-generation", int(index), 20260904))
        values.append(torch.randn((10, 7), generator=generator, device=device, dtype=dtype))
    return torch.stack(values)


def _generation_loss(
    *,
    updater: torch.nn.Module,
    transformer: torch.nn.Module,
    action_space: Any,
    batch: dict[str, Any],
    cache_manifest: dict[str, Any],
    device: torch.device,
    weights: dict[str, float],
) -> Any:
    condition = batch["condition"].to(device, non_blocking=True)
    proprio = batch["proprio"].to(device, non_blocking=True)
    initial_noise = _noise(batch["cache_index"].tolist(), device, proprio.dtype)
    normalized_proprio = action_space.normalize_state(proprio)
    class AdapterHolder:
        num_actions = 10

        def __init__(self) -> None:
            self.transformer = transformer
            self.action_space = action_space

        def eval(self) -> "AdapterHolder":
            self.transformer.eval()
            return self

    adapter_holder = AdapterHolder()
    adapter = SimVLAActionAdapter(adapter_holder)
    with torch.no_grad():
        teacher_action = adapter.decode_action_from_condition(
            condition,
            proprio,
            steps=10,
            initial_noise=initial_noise,
        )
    valid, _ = _layout(cache_manifest, condition.shape[0], device)
    loop = SimVLAGenerationLoop(updater, transformer.action_decoder)
    return generation_local_oracle_loss(
        loop=loop,
        transformer=transformer,
        action_space=action_space,
        condition=condition,
        initial_noise=initial_noise,
        normalized_proprio=normalized_proprio,
        condition_valid_mask=valid,
        condition_change_code=condition.new_zeros((condition.shape[0], 128)),
        full_step_indices=(0, 4, 8),
        teacher_final_action=teacher_action,
        hidden_weight=weights["hidden_normalized_mse"],
        velocity_weight=weights["velocity_l1"],
        final_action_weight=weights["final_action_l1"],
    )


@torch.no_grad()
def _calibrate_generation(
    updater: torch.nn.Module,
    transformer: torch.nn.Module,
    action_space: Any,
    loader: DataLoader,
    cache_manifest: dict[str, Any],
    device: torch.device,
    batches: int,
) -> dict[str, float]:
    raw = {"hidden_normalized_mse": [], "velocity_l1": [], "final_action_l1": []}
    unit = {name: 1.0 for name in raw}
    for number, batch in enumerate(loader):
        if number >= batches:
            break
        loss = _generation_loss(
            updater=updater,
            transformer=transformer,
            action_space=action_space,
            batch=batch,
            cache_manifest=cache_manifest,
            device=device,
            weights=unit,
        )
        for name in raw:
            raw[name].append(float(getattr(loss, name).item()))
    return {name: max(sum(values) / len(values), 1e-8) for name, values in raw.items()}


@torch.no_grad()
def _validate_generation(
    updater: torch.nn.Module,
    transformer: torch.nn.Module,
    action_space: Any,
    loader: DataLoader,
    cache_manifest: dict[str, Any],
    device: torch.device,
    weights: dict[str, float],
    batches: int,
) -> dict[str, float]:
    updater.eval()
    collected = {"hidden_normalized_mse": [], "velocity_l1": [], "final_action_l1": []}
    for number, batch in enumerate(loader):
        if number >= batches:
            break
        loss = _generation_loss(
            updater=updater,
            transformer=transformer,
            action_space=action_space,
            batch=batch,
            cache_manifest=cache_manifest,
            device=device,
            weights=weights,
        )
        for name in collected:
            collected[name].append(float(getattr(loss, name).item()))
    updater.train()
    return {name: sum(values) / len(values) for name, values in collected.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    context = initialize_distributed(args.device)
    if context.world_size != 1:
        context.close()
        raise RuntimeError(
            "Each updater is intentionally single-GPU; launch condition and generation "
            "in parallel on separate GPUs instead of duplicating the small module with DDP"
        )
    seed_process(args.seed, 0)
    try:
        output = Path(args.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)
        source_contract = validate_real_training_sources(
            condition_cache=args.condition_cache,
            checkpoint=args.checkpoint,
            processor=args.processor,
            norm_stats=args.norm_stats,
            verify_cache_array_checksums=False,
            condition_cache_attestation=args.condition_cache_attestation,
        )
        cache_manifest = source_contract["condition_cache"]
        validate_real_baseline_checkpoint(
            args.baseline_action_checkpoint,
            source=source_contract,
            expected_optimizer_step=3000,
        )
        model, _, loading = load_exact_official_model(
            model_directory=args.checkpoint,
            processor_directory=args.processor,
            norm_stats=args.norm_stats,
            device="cpu",
            real_action_checkpoint=args.baseline_action_checkpoint,
            freeze_vlm=True,
            freeze_action_transformer=True,
            expected_dataset_identity_sha256=cache_manifest[
                "dataset_identity_sha256"
            ],
            expected_cache_identity_sha256=cache_manifest[
                "condition_cache_identity_sha256"
            ],
            expected_cache_attestation_identity_sha256=source_contract[
                "condition_cache_attestation"
            ]["attestation_identity_sha256"],
            expected_real_action_optimizer_step=3000,
        )
        transformer = model.transformer.to(context.device).eval()
        action_space = model.action_space.to(context.device)
        del model.vlm
        gc.collect()
        torch.cuda.empty_cache() if context.device.type == "cuda" else None
        for parameter in transformer.parameters():
            parameter.requires_grad_(False)

        if args.kind == "condition":
            config: RealConditionConfig | RealGenerationConfig = RealConditionConfig()
            updater = config.build().to(context.device)
            train_data = RealConditionPairDataset(args.condition_cache, split="train")
            validation_data = RealConditionPairDataset(args.condition_cache, split="validation")
            batch_size = args.condition_batch_size
        else:
            config = RealGenerationConfig()
            updater = config.build().to(context.device)
            train_data = RealConditionCacheDataset(args.condition_cache, split="train")
            validation_data = RealConditionCacheDataset(args.condition_cache, split="validation")
            batch_size = args.generation_batch_size
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=args.num_workers,
            pin_memory=context.device.type == "cuda",
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 2),
            pin_memory=context.device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            updater.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=context.device.type == "cuda",
        )
        atomic_write_json(output / "exact_teacher_initialization.json", loading)

        if args.kind == "condition":
            calibration = _calibrate_condition(
                updater,
                transformer.vlm_proj,
                train_loader,
                cache_manifest,
                context.device,
                args.calibration_batches,
            )
            # The two training terms receive equal contribution after
            # deterministic calibration. Cosine distance is diagnostic only.
            weights = {
                "condition_normalized_mse": 0.5
                / calibration["condition_normalized_mse"],
                "action_projection_normalized_mse": 0.5
                / calibration["action_projection_normalized_mse"],
            }
        else:
            calibration = _calibrate_generation(
                updater,
                transformer,
                action_space,
                train_loader,
                cache_manifest,
                context.device,
                args.calibration_batches,
            )
            weights = {
                name: (1.0 / 3.0) / value
                for name, value in calibration.items()
            }
        objective = {
            "calibration_raw_mean": calibration,
            "loss_weights": weights,
            "weight_rule": "equal contribution after deterministic raw-loss calibration",
            "manual_weight_approval_required": False,
        }
        atomic_write_json(output / "objective.json", objective)

        iterator = iter(train_loader)
        progress = tqdm(total=args.max_steps, initial=0, desc=f"real {args.kind} updater")
        metrics_path = output / "train_metrics.jsonl"
        last_validation: dict[str, float] = {}
        started = time.perf_counter()
        step = 0
        while step < args.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            if args.kind == "condition":
                losses, _ = _condition_raw_losses(
                    updater, transformer.vlm_proj, batch, cache_manifest, context.device
                )
                total = (
                    weights["condition_normalized_mse"] * losses["condition_normalized_mse"]
                    + weights["action_projection_normalized_mse"]
                    * losses["action_projection_normalized_mse"]
                )
                displayed = {name: float(value.detach().item()) for name, value in losses.items()}
            else:
                result = _generation_loss(
                    updater=updater,
                    transformer=transformer,
                    action_space=action_space,
                    batch=batch,
                    cache_manifest=cache_manifest,
                    device=context.device,
                    weights=weights,
                )
                total = result.total
                displayed = {
                    name: float(getattr(result, name).detach().item())
                    for name in ("hidden_normalized_mse", "velocity_l1", "final_action_l1")
                }
            total.backward()
            torch.nn.utils.clip_grad_norm_(updater.parameters(), args.max_grad_norm)
            learning_rate = _lr(step, args.max_steps, args.warmup_steps, args.learning_rate)
            _set_lr(optimizer, learning_rate)
            optimizer.step()
            step += 1
            progress.update(1)
            progress.set_postfix(total=f"{float(total.detach().item()):.4f}", lr=f"{learning_rate:.1e}")

            if step % args.save_interval == 0 or step == args.max_steps:
                if args.kind == "condition":
                    last_validation = _validate_condition(
                        updater,
                        transformer.vlm_proj,
                        validation_loader,
                        cache_manifest,
                        context.device,
                        args.validation_batches,
                    )
                else:
                    last_validation = _validate_generation(
                        updater,
                        transformer,
                        action_space,
                        validation_loader,
                        cache_manifest,
                        context.device,
                        weights,
                        args.validation_batches,
                    )
                checkpoint = output / "checkpoints" / f"{args.kind}_step_{step:06d}.pt"
                shared = {
                    "kind": args.kind,
                    "updater": updater,
                    "config": config,
                    "baseline_action_checkpoint": args.baseline_action_checkpoint,
                    "norm_stats_sha256": sha256_file(args.norm_stats),
                    "dataset_identity_sha256": cache_manifest["dataset_identity_sha256"],
                    "condition_cache_identity_sha256": cache_manifest[
                        "condition_cache_identity_sha256"
                    ],
                    "condition_cache_attestation_identity_sha256": source_contract[
                        "condition_cache_attestation"
                    ]["attestation_identity_sha256"],
                    "optimizer_step": step,
                    "objective": objective,
                    "validation": last_validation,
                }
                save_real_updater(checkpoint, **shared)
                (output / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n", encoding="utf-8")
                old = sorted((output / "checkpoints").glob(f"{args.kind}_step_*.pt"))
                for stale in old[: max(0, len(old) - args.keep_checkpoints)]:
                    stale.unlink()
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "total": float(total.detach().item()),
                                "raw": displayed,
                                "validation": last_validation,
                                "learning_rate": learning_rate,
                                "elapsed_s": time.perf_counter() - started,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        progress.close()
        summary = {
            "verdict": f"REAL_{args.kind.upper()}_UPDATER_COMPLETE",
            "optimizer_steps": step,
            "trainable_parameters": sum(value.numel() for value in updater.parameters()),
            "baseline_action_checkpoint_sha256": sha256_file(args.baseline_action_checkpoint),
            "validation": last_validation,
            "elapsed_s": time.perf_counter() - started,
        }
        atomic_write_json(output / "run_summary.json", summary)
        return summary
    finally:
        context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("condition", "generation"))
    parser.add_argument("--condition-cache", required=True)
    parser.add_argument("--condition-cache-attestation", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--baseline-action-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--condition-batch-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    result = train(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
