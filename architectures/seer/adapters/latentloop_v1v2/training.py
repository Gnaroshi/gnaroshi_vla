"""Reviewed four-GPU offline calibration and training runtime for V1."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from methods.latentloop_v1v2.losses import compute_v1_losses

from .data_runtime import (
    build_frozen_teacher,
    build_seer_args,
    build_split_loader,
    causal_teacher_tuple,
    seed_everything,
    seer_upstream_context,
)
from .runtime import (
    adapter_checkpoint_state,
    sha256_file,
    state_dict_sha256,
    trainable_parameter_report,
)


LOSS_NAMES = (
    "direct_latent",
    "composed_latent",
    "direct_action",
    "composed_action",
    "composition",
    "smooth",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _distributed_start() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 4 or rank < 0 or local_rank not in range(4):
        raise RuntimeError("LatentLoop V1 runtime requires torchrun with exactly four ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("each V1 row must see exactly four CUDA devices")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, torch.device("cuda", local_rank)


def _distributed_stop() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _all_reduce_sum(values: Tensor) -> Tensor:
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def _distributed_refuse_existing(path: Path, device: torch.device) -> None:
    exists = torch.tensor([int(path.exists())], device=device, dtype=torch.int32)
    dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if int(exists.item()):
        raise FileExistsError(path)


def _continuous_action(model: nn.Module, latent: Tensor) -> Tensor:
    diagnostics = model.decode_action_diagnostics_from_latent(latent)
    return torch.cat((diagnostics["arm"], diagnostics["gripper_probability"]), dim=-1)


def _frozen_action_generator(model: nn.Module):
    def decode(latent: Tensor):
        return model.decode_action_diagnostics_from_latent(latent)

    return decode


def _max_gradient_abs(parameters) -> float:
    maximum = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            maximum = max(maximum, float(parameter.grad.detach().abs().max().item()))
    return maximum


def _initialization_consensus(transition: nn.Module, device: torch.device) -> str:
    local = state_dict_sha256(transition.state_dict())
    values: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(values, local)
    if len(set(values)) != 1:
        raise RuntimeError(f"V1 initialization differs across ranks: {values}")
    return local


def _build_runtime(args, *, training: bool):
    rank, world_size, device = _distributed_start()
    _distributed_refuse_existing(args.output_root, device)
    repo_root = Path(__file__).resolve().parents[4]
    seed_everything(args.seed, rank)
    with seer_upstream_context(repo_root):
        seer_args = build_seer_args(
            output_root=args.output_root,
            dataset_root=args.dataset_root,
            vit_checkpoint=args.vit_checkpoint,
            libero_path=getattr(args, "libero_path", repo_root / ".canonical/libero_source"),
            batch_size=args.per_gpu_batch,
            workers=args.workers,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )
        model, transition, load_report = build_frozen_teacher(
            seer_args, device, args.teacher, args.adapter_init
        )
        loader = build_split_loader(
            args=seer_args,
            model=model,
            split_manifest=args.split_manifest,
            role=getattr(args, "split_role", "transition_train"),
            training=training,
        )
    initialization_sha = _initialization_consensus(transition, device)
    return rank, world_size, device, repo_root, model, transition, loader, load_report, initialization_sha


def collect_raw_losses(args) -> None:
    """Collect rank-aggregated initial raw losses from transition-train only."""

    args.split_role = "transition_train"
    args.output_root = args.output.parent
    (
        rank,
        _,
        device,
        _,
        model,
        transition,
        loader,
        load_report,
        initialization_sha,
    ) = _build_runtime(args, training=True)
    transition.eval()
    values = {name: [] for name in LOSS_NAMES}
    try:
        with torch.no_grad():
            for microbatch, batch in enumerate(loader):
                if microbatch >= args.target_microbatches:
                    break
                interval = microbatch % 3 + 1
                teacher = causal_teacher_tuple(model, batch, interval, device)
                output = transition(
                    anchor_latent=teacher["anchor_latent"],
                    primary_sequence=teacher["primary_sequence"],
                    wrist_sequence=teacher["wrist_sequence"],
                    state_sequence=teacher["state_sequence"],
                    executed_actions=teacher["executed_actions"],
                    interval=interval,
                )
                losses = compute_v1_losses(
                    output,
                    teacher["teacher_latent"],
                    teacher["anchor_latent"],
                    _frozen_action_generator(model),
                    {name: 1.0 for name in LOSS_NAMES},
                )
                for name in LOSS_NAMES:
                    values[name].append(float(losses.raw[name].item()))
        if len(next(iter(values.values()))) != args.target_microbatches:
            raise RuntimeError(
                f"raw calibration requested {args.target_microbatches} microbatches per rank "
                f"but loader provided {len(next(iter(values.values())))}"
            )
        gathered: list[dict[str, list[float]] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, values)
        if rank == 0:
            merged = {
                name: [value for row in gathered if row is not None for value in row[name]]
                for name in LOSS_NAMES
            }
            medians = {
                name: float(torch.tensor(metric, dtype=torch.float64).median().item())
                for name, metric in merged.items()
            }
            if any(not math.isfinite(value) or value <= 0 for value in medians.values()):
                raise RuntimeError(f"invalid raw-loss medians: {medians}")
            payload = {
                "schema_version": 1,
                "status": "V1_RAW_LOSS_SAMPLES_COLLECTED",
                "split_role": "train_raw_loss_calibration",
                "dataset_split_role": "transition_train",
                "uses_online_sr": False,
                "world_size": dist.get_world_size(),
                "microbatches_per_rank": args.target_microbatches,
                "global_microbatch_samples": args.target_microbatches * dist.get_world_size(),
                "interval_schedule": "balanced_1_2_3",
                "split_manifest_sha256": sha256_file(args.split_manifest),
                "initialization_sha256": initialization_sha,
                "teacher_sha256": load_report["teacher_sha256"],
                "v0_adapter_sha256": load_report["v0_adapter_sha256"],
                "raw_loss_medians": medians,
            }
            args.output.parent.mkdir(parents=True, exist_ok=False)
            _atomic_json(args.output, payload)
    finally:
        _distributed_stop()


def _lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _save_adapter_checkpoint(path: Path, transition: nn.Module, epoch: int, metadata: Mapping[str, Any]) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": adapter_checkpoint_state(transition),
        "metadata": dict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _validation_metrics(
    *,
    model: nn.Module,
    transition: nn.Module,
    no_composition_control: nn.Module,
    loader,
    weights: Mapping[str, float],
    device: torch.device,
) -> dict[str, Any]:
    transition.eval()
    no_composition_control.eval()
    names = (
        "validation_total_loss",
        "direct_latent_mse",
        "composed_latent_mse",
        "direct_action_l1",
        "composed_action_l1",
        "composition_defect",
        "no_composition_defect",
        "hold_latent_mse",
        "hold_action_l1",
    )
    totals = torch.zeros(len(names) + 8, device=device, dtype=torch.float64)
    # Tail fields: count, age1 sum/count, age2 sum/count, age3 sum/count,
    # predicted gripper sum/square-sum/open-count and teacher open-count are
    # compacted below through explicit indices.
    gripper = torch.zeros(5, device=device, dtype=torch.float64)
    with torch.no_grad():
        for microbatch, batch in enumerate(loader):
            interval = microbatch % 3 + 1
            teacher = causal_teacher_tuple(model, batch, interval, device)
            output = transition(
                anchor_latent=teacher["anchor_latent"],
                primary_sequence=teacher["primary_sequence"],
                wrist_sequence=teacher["wrist_sequence"],
                state_sequence=teacher["state_sequence"],
                executed_actions=teacher["executed_actions"],
                interval=interval,
            )
            control = no_composition_control(
                anchor_latent=teacher["anchor_latent"],
                primary_sequence=teacher["primary_sequence"],
                wrist_sequence=teacher["wrist_sequence"],
                state_sequence=teacher["state_sequence"],
                executed_actions=teacher["executed_actions"],
                interval=interval,
            )
            losses = compute_v1_losses(
                output,
                teacher["teacher_latent"],
                teacher["anchor_latent"],
                _frozen_action_generator(model),
                weights,
            )
            target_action = _continuous_action(model, teacher["teacher_latent"])
            direct_action = _continuous_action(model, output.direct)
            composed_action = _continuous_action(model, output.composed)
            hold_action = _continuous_action(model, teacher["anchor_latent"])
            count = float(teacher["anchor_latent"].shape[0])
            row = (
                losses.total,
                losses.raw["direct_latent"],
                losses.raw["composed_latent"],
                losses.raw["direct_action"],
                losses.raw["composed_action"],
                F.mse_loss(output.composed, output.direct),
                F.mse_loss(control.composed, control.direct),
                F.mse_loss(teacher["anchor_latent"], teacher["teacher_latent"]),
                F.l1_loss(hold_action, target_action),
            )
            totals[: len(names)] += torch.stack([value.double() for value in row]) * count
            totals[len(names)] += count
            age_base = len(names) + 1 + (interval - 1) * 2
            totals[age_base] += losses.raw["composed_action"].double() * count
            totals[age_base + 1] += count
            probability = composed_action[..., 6].double().flatten()
            target_probability = target_action[..., 6].double().flatten()
            gripper += torch.tensor(
                [
                    probability.sum().item(),
                    probability.square().sum().item(),
                    float(probability.numel()),
                    float((probability >= 0.5).sum().item()),
                    float((target_probability >= 0.5).sum().item()),
                ],
                device=device,
                dtype=torch.float64,
            )
    _all_reduce_sum(totals)
    _all_reduce_sum(gripper)
    count = float(totals[len(names)].item())
    if count <= 0:
        raise RuntimeError("checkpoint-validation split produced no samples")
    result = {name: float((totals[index] / count).item()) for index, name in enumerate(names)}
    ages = []
    for interval in (1, 2, 3):
        base = len(names) + 1 + (interval - 1) * 2
        age_count = float(totals[base + 1].item())
        if age_count <= 0:
            raise RuntimeError(f"validation has no interval-{interval} samples")
        ages.append(float((totals[base] / age_count).item()))
    prediction_count = max(1.0, float(gripper[2].item()))
    mean = float(gripper[0].item() / prediction_count)
    variance = max(0.0, float(gripper[1].item() / prediction_count - mean * mean))
    predicted_open = float(gripper[3].item() / prediction_count)
    teacher_open = float(gripper[4].item() / prediction_count)
    teacher_has_both = 0.01 < teacher_open < 0.99
    result.update(
        composed_action_l1_by_age=ages,
        finite_gripper_output=math.isfinite(mean) and math.isfinite(variance),
        predicted_gripper_probability_mean=mean,
        predicted_gripper_probability_std=math.sqrt(variance),
        predicted_gripper_open_fraction=predicted_open,
        teacher_gripper_open_fraction=teacher_open,
        gripper_collapse=bool(
            not math.isfinite(mean)
            or (teacher_has_both and (predicted_open <= 0.01 or predicted_open >= 0.99))
        ),
        validation_samples=int(count),
    )
    transition.train()
    return result


def run_training(args) -> None:
    """Train one independent e20/e40 run and select by validation loss only."""

    args.split_role = "transition_train"
    start_time = time.perf_counter()
    (
        rank,
        world_size,
        device,
        repo_root,
        model,
        transition,
        train_loader,
        load_report,
        initialization_sha,
    ) = _build_runtime(args, training=True)
    # Matched ablation: same initialization, data, optimizer, and schedule;
    # only the explicit direct/composed consistency weight is zero.
    no_composition_control = copy.deepcopy(transition).requires_grad_(True)
    transition_ddp = DDP(transition, device_ids=[device.index], find_unused_parameters=False)
    no_composition_ddp = DDP(
        no_composition_control,
        device_ids=[device.index],
        find_unused_parameters=False,
    )
    try:
        with seer_upstream_context(repo_root):
            validation_args = build_seer_args(
                output_root=args.output_root,
                dataset_root=args.dataset_root,
                vit_checkpoint=args.vit_checkpoint,
                libero_path=getattr(args, "libero_path", repo_root / ".canonical/libero_source"),
                batch_size=args.per_gpu_batch,
                workers=args.workers,
                rank=rank,
                world_size=world_size,
                seed=args.seed,
            )
            validation_loader = build_split_loader(
                args=validation_args,
                model=model,
                split_manifest=args.split_manifest,
                role="checkpoint_validation",
                training=False,
            )
        weight_payload = json.loads(args.loss_weights.read_text(encoding="utf-8"))
        if weight_payload.get("status") != "V1_RAW_LOSS_WEIGHTS_LOCKED":
            raise RuntimeError("V1 loss weights are not frozen")
        if weight_payload.get("split_manifest_sha256") != sha256_file(args.split_manifest):
            raise RuntimeError("V1 loss-weight split provenance mismatch")
        weights = {name: float(weight_payload["weights"][name]) for name in LOSS_NAMES}
        no_composition_weights = dict(weights)
        no_composition_weights["composition"] = 0.0
        report = trainable_parameter_report(model)
        optimizer = torch.optim.AdamW(
            transition.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        no_composition_optimizer = torch.optim.AdamW(
            no_composition_control.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        usable_microbatches = (len(train_loader) // args.gradient_accumulation) * args.gradient_accumulation
        if usable_microbatches == 0:
            raise RuntimeError("transition-train split is too small for one effective batch")
        updates_per_epoch = usable_microbatches // args.gradient_accumulation
        total_updates = args.epochs * updates_per_epoch
        warmup_updates = max(1, int(round(total_updates * args.warmup_fraction)))
        source_lock = repo_root / ".canonical/source_lock/source_lock_manifest.json"
        common_metadata = {
            "source_lock_sha256": sha256_file(source_lock),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "initialization_sha256": initialization_sha,
            "teacher_sha256": load_report["teacher_sha256"],
            "v0_adapter_sha256": load_report["v0_adapter_sha256"],
            "budget_epochs": args.epochs,
            "training_seed": args.seed,
            "world_size": world_size,
            "per_gpu_batch": args.per_gpu_batch,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "effective_batch": args.per_gpu_batch * world_size * args.gradient_accumulation,
            "cosine_horizon_epochs": args.epochs,
            "warmup_fraction": args.warmup_fraction,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "independent_run_id": f"v1_e{args.epochs}_seed{args.seed}",
            "uses_online_sr": False,
            "split_role": "checkpoint_validation",
            "no_composition_control": {
                "kind": "matched_training_ablation",
                "removed_loss": "composition",
                "composition_weight": 0.0,
                "same_initialization_data_optimizer_and_schedule": True,
            },
            **report,
        }
        if rank == 0:
            args.output_root.mkdir(parents=True, exist_ok=False)
            (args.output_root / "checkpoints").mkdir()
            _atomic_json(args.output_root / "training_contract.json", common_metadata)
        dist.barrier()

        best_loss = math.inf
        best_metrics: dict[str, Any] | None = None
        best_epoch = -1
        optimizer_step = 0
        optimizer.zero_grad(set_to_none=True)
        no_composition_optimizer.zero_grad(set_to_none=True)
        for epoch in range(args.epochs):
            train_loader.sampler.set_epoch(epoch)
            transition_ddp.train()
            no_composition_ddp.train()
            epoch_raw = {name: 0.0 for name in LOSS_NAMES}
            control_epoch_raw = {name: 0.0 for name in LOSS_NAMES}
            epoch_count = 0
            for microbatch, batch in enumerate(train_loader):
                if microbatch >= usable_microbatches:
                    break
                interval = (epoch * usable_microbatches + microbatch) % 3 + 1
                teacher = causal_teacher_tuple(model, batch, interval, device)
                sync_now = (microbatch + 1) % args.gradient_accumulation == 0
                candidate_context = nullcontext() if sync_now else transition_ddp.no_sync()
                with candidate_context:
                    output = transition_ddp(
                        anchor_latent=teacher["anchor_latent"],
                        primary_sequence=teacher["primary_sequence"],
                        wrist_sequence=teacher["wrist_sequence"],
                        state_sequence=teacher["state_sequence"],
                        executed_actions=teacher["executed_actions"],
                        interval=interval,
                    )
                    losses = compute_v1_losses(
                        output,
                        teacher["teacher_latent"],
                        teacher["anchor_latent"],
                        _frozen_action_generator(model),
                        weights,
                    )
                    if not torch.isfinite(losses.total):
                        raise RuntimeError(f"non-finite V1 loss at epoch={epoch} microbatch={microbatch}")
                    (losses.total / args.gradient_accumulation).backward()
                control_context = nullcontext() if sync_now else no_composition_ddp.no_sync()
                with control_context:
                    control_output = no_composition_ddp(
                        anchor_latent=teacher["anchor_latent"],
                        primary_sequence=teacher["primary_sequence"],
                        wrist_sequence=teacher["wrist_sequence"],
                        state_sequence=teacher["state_sequence"],
                        executed_actions=teacher["executed_actions"],
                        interval=interval,
                    )
                    control_losses = compute_v1_losses(
                        control_output,
                        teacher["teacher_latent"],
                        teacher["anchor_latent"],
                        _frozen_action_generator(model),
                        no_composition_weights,
                    )
                    if not torch.isfinite(control_losses.total):
                        raise RuntimeError(
                            "non-finite no-composition control loss at "
                            f"epoch={epoch} microbatch={microbatch}"
                        )
                    (control_losses.total / args.gradient_accumulation).backward()
                for name in LOSS_NAMES:
                    epoch_raw[name] += float(losses.raw[name].detach().item())
                    control_epoch_raw[name] += float(
                        control_losses.raw[name].detach().item()
                    )
                epoch_count += 1
                if sync_now:
                    scale = _lr_scale(optimizer_step, total_updates, warmup_updates)
                    for group in optimizer.param_groups:
                        group["lr"] = args.learning_rate * scale
                    for group in no_composition_optimizer.param_groups:
                        group["lr"] = args.learning_rate * scale
                    optimizer.step()
                    no_composition_optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    no_composition_optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
            if epoch_count != usable_microbatches:
                raise RuntimeError("training epoch did not consume the locked microbatch count")
            validation = _validation_metrics(
                model=model,
                transition=transition,
                no_composition_control=no_composition_control,
                loader=validation_loader,
                weights=weights,
                device=device,
            )
            validation.update(common_metadata)
            validation.update(
                epoch=epoch + 1,
                optimizer_steps=optimizer_step,
                total_optimizer_steps=total_updates,
                warmup_optimizer_steps=warmup_updates,
            )
            if rank == 0:
                train_row = {
                    "epoch": epoch + 1,
                    "optimizer_steps": optimizer_step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **{f"train_raw_{name}": epoch_raw[name] / epoch_count for name in LOSS_NAMES},
                    **{
                        f"no_composition_train_raw_{name}":
                        control_epoch_raw[name] / epoch_count
                        for name in LOSS_NAMES
                    },
                    **validation,
                }
                with (args.output_root / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(train_row) + "\n")
                if validation["validation_total_loss"] < best_loss:
                    best_loss = float(validation["validation_total_loss"])
                    best_metrics = dict(validation)
                    best_epoch = epoch + 1
                    checkpoint = args.output_root / "selected_checkpoint.pth"
                    _save_adapter_checkpoint(checkpoint, transition, best_epoch, common_metadata)
                    _save_adapter_checkpoint(
                        args.output_root / "checkpoints" / f"best_epoch_{best_epoch:03d}.pth",
                        transition,
                        best_epoch,
                        common_metadata,
                    )
                    _save_adapter_checkpoint(
                        args.output_root / "selected_no_composition_control.pth",
                        no_composition_control,
                        best_epoch,
                        {**common_metadata, "control": "no_composition"},
                    )
            dist.barrier()

        if optimizer_step != total_updates:
            raise RuntimeError(f"optimizer-step mismatch: {optimizer_step} != {total_updates}")
        if rank == 0:
            if best_metrics is None or best_epoch < 1:
                raise RuntimeError("V1 training did not select a validation checkpoint")
            selected = args.output_root / "selected_checkpoint.pth"
            best_metrics.update(
                selected_epoch=best_epoch,
                selected_checkpoint_sha256=sha256_file(selected),
                measured_wallclock_sec=float(time.perf_counter() - start_time),
            )
            _atomic_json(args.output_root / "validation_metrics.json", best_metrics)
            offline = dict(best_metrics)
            teacher_gradient = _max_gradient_abs(
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("latentloop_plan_adapter.")
            )
            action_generator_gradient = _max_gradient_abs(
                parameter
                for module in model.get_action_head_modules()
                for parameter in module.parameters()
            )
            offline.update(
                status="V1_OFFLINE_METRICS_READY",
                v1_checkpoint_sha256=sha256_file(selected),
                trainable_parameters=int(report["v1_trainable_parameters"]),
                max_teacher_gradient_abs=teacher_gradient,
                max_action_generator_gradient_abs=action_generator_gradient,
            )
            _atomic_json(args.output_root / "offline_gate_metrics.json", offline)
            _save_adapter_checkpoint(
                args.output_root / "final_checkpoint.pth", transition, args.epochs, common_metadata
            )
            _save_adapter_checkpoint(
                args.output_root / "final_no_composition_control.pth",
                no_composition_control,
                args.epochs,
                {**common_metadata, "control": "no_composition"},
            )
            _atomic_json(
                args.output_root / "completion.json",
                {
                    "status": "V1_TRAINING_COMPLETE",
                    "selected_epoch": best_epoch,
                    "optimizer_steps": optimizer_step,
                    "selected_checkpoint_sha256": sha256_file(selected),
                    "selected_no_composition_control_sha256": sha256_file(
                        args.output_root / "selected_no_composition_control.pth"
                    ),
                },
            )
    finally:
        _distributed_stop()
