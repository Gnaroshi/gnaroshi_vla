"""Two-rank scientific trainer for corrected native SimVLA V0."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    NativeV0Config,
    load_native_v0_checkpoint,
    save_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_dataset import (
    NativeV0SequenceDataset,
    collate_native_v0_sequences,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    append_jsonl,
    cached_batch_token_layout,
    configure_strict_torch_determinism,
    load_frozen_simvla,
    move_batch,
    native_v0_source_manifest,
    require_gate,
    write_json,
)
from architectures.simvla.wrappers.simvla_two_gpu_guard import parse_selected_gpu_ids
from methods.latentloop.training.native_simvla_v0 import (
    NativeV0LossWeights,
    WarmupCosineController,
    decode_age_conditions,
    native_v0_raw_losses,
    weighted_native_v0_loss,
)


class ReplicatedLogicalSampler:
    """Give both DDP ranks the same logical sample to preserve global batch one."""

    def __init__(self, dataset_size: int, seed: int) -> None:
        if dataset_size < 1:
            raise ValueError("dataset must not be empty")
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)

    def index(self, optimizer_step: int) -> int:
        logical = int(optimizer_step)
        epoch, offset = divmod(logical, self.dataset_size)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        return int(torch.randperm(self.dataset_size, generator=generator)[offset].item())

    def state_dict(self, optimizer_step: int) -> dict[str, int | str]:
        return {
            "sampler": "replicated_logical_global_batch_one",
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "next_optimizer_step": int(optimizer_step),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _approved_weights(
    path: str | Path,
    source_hash: str,
    train_split_sha256: str,
) -> NativeV0LossWeights:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("approved_by_user") is not True:
        raise RuntimeError("loss weights are not explicitly approved_by_user=true")
    if payload.get("source_combined_sha256") != source_hash:
        raise RuntimeError("approved loss weights use a different source lock")
    if payload.get("train_split_sha256") != train_split_sha256:
        raise RuntimeError("approved loss weights use a different training split")
    names = (
        "condition",
        "first5_action",
        "full_chunk_action",
        "continuous_gripper",
        "update_regularization",
    )
    values = {name: float(payload[name]) for name in names}
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("all approved loss weights must be finite and non-negative")
    if values["first5_action"] <= 0:
        raise ValueError("the primary first-5 action weight must be positive")
    return NativeV0LossWeights(**values)


def _assert_identical_logical_index(index: int, device: torch.device) -> None:
    local = torch.tensor([index], device=device, dtype=torch.long)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    values = [int(item.item()) for item in gathered]
    if len(set(values)) != 1:
        raise RuntimeError(f"DDP ranks received different logical samples: {values}")


def _decode(
    action_adapter: Any,
    condition: torch.Tensor,
    proprio: torch.Tensor,
    noise: torch.Tensor,
    *,
    requires_grad: bool,
) -> torch.Tensor:
    return action_adapter.decode_action_from_condition(
        condition,
        proprio,
        steps=10,
        initial_noise=noise,
        requires_grad=requires_grad,
    )


def _forward_losses(
    *,
    adapter: Any,
    batch: dict[str, Any],
    processor: Any,
    action_adapter: Any,
    mode: str,
    weights: NativeV0LossWeights,
    requires_grad: bool,
    reference_actions: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    layout = cached_batch_token_layout(
        condition=batch["anchor_condition"],
        language_instructions=batch["language_instruction"],
        processor=processor,
    )
    unroll = adapter(
        batch["anchor_condition"],
        batch["image_sequence"],
        batch["proprio_sequence"],
        valid_mask=layout.valid_mask,
        group_ids=layout.group_ids,
    )
    actions = decode_age_conditions(
        lambda condition, proprio, noise: _decode(
            action_adapter,
            condition,
            proprio,
            noise,
            requires_grad=requires_grad,
        ),
        unroll.conditions,
        tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3)),
        tuple(batch["explicit_noises"][:, age - 1] for age in (1, 2, 3)),
        mode=mode,
    )
    full_conditions = tuple(
        batch["teacher_conditions"][:, age - 1].detach() for age in (1, 2, 3)
    )
    teacher_actions = reference_actions
    if teacher_actions is None:
        teacher_actions = decode_age_conditions(
            lambda condition, proprio, noise: _decode(
                action_adapter,
                condition,
                proprio,
                noise,
                requires_grad=False,
            ),
            full_conditions,
            tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3)),
            tuple(batch["explicit_noises"][:, age - 1] for age in (1, 2, 3)),
            mode=mode,
        )
    raw = native_v0_raw_losses(
        unroll=unroll,
        teacher_conditions=full_conditions,
        predicted_actions=actions,
        teacher_actions=teacher_actions,
        valid_mask=layout.valid_mask,
    )
    total, weighted = weighted_native_v0_loss(raw, weights)
    return total, raw, weighted, teacher_actions


@torch.no_grad()
def _validate(
    *,
    model: torch.nn.Module,
    dataset: NativeV0SequenceDataset,
    processor: Any,
    action_adapter: Any,
    mode: str,
    weights: NativeV0LossWeights,
    device: torch.device,
    batches: int,
) -> dict[str, Any]:
    model.eval()
    values: dict[str, list[float]] = {}
    for index in range(min(int(batches), len(dataset))):
        batch = move_batch(collate_native_v0_sequences([dataset[index]]), device)
        total, raw, _, _ = _forward_losses(
            adapter=model,
            batch=batch,
            processor=processor,
            action_adapter=action_adapter,
            mode=mode,
            weights=weights,
            requires_grad=False,
        )
        values.setdefault("total", []).append(float(total.item()))
        for name, value in raw.items():
            values.setdefault(name, []).append(float(value.item()))
    model.train()
    return {
        name: {
            "mean": float(np.mean(items)),
            "p95": float(np.quantile(items, 0.95)),
            "p99": float(np.quantile(items, 0.99)),
        }
        for name, items in values.items()
    }


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
    selected = parse_selected_gpu_ids(os.environ.get("SIMVLA_GPU_IDS"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != ",".join(str(value) for value in selected):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must exactly match SIMVLA_GPU_IDS; got {visible!r} vs {selected}"
        )
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("correct native V0 training requires torchrun WORLD_SIZE=2")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank not in {0, 1}:
        raise RuntimeError("local rank must be 0 or 1")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    _seed_everything(args.seed)
    determinism = configure_strict_torch_determinism(args.seed)

    source = native_v0_source_manifest(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        cache=args.cache,
    )
    source_hash = source["combined_sha256"]
    require_gate(args.parity_gate, verdicts=("K1_HOOK_PARITY_PASS",), source_combined_sha256=source_hash)
    analysis_gate = require_gate(args.analysis_gate, verdicts=("TOKEN_ANALYSIS_PASS",), source_combined_sha256=source_hash)
    require_gate(args.parameter_gate, verdicts=("PARAMETER_AUDIT_PASS",), source_combined_sha256=source_hash)
    mode_gate = require_gate(args.mode_gate, verdicts=("MODE_B_APPROVED", "MODE_A_REQUIRED"), source_combined_sha256=source_hash)
    calibration = require_gate(args.calibration_gate, verdicts=("LOSS_SCALE_CALIBRATION_COMPLETE",), source_combined_sha256=source_hash)
    mode = str(mode_gate["scientific_training_mode"])

    if args.smoke:
        if not 1 <= args.max_steps <= 20:
            raise ValueError("smoke training must use 1-20 optimizer steps")
    elif args.max_steps != 150_000:
        raise ValueError("scientific training is fixed to 150,000 optimizer steps")
    smoke_gate: dict[str, Any] | None = None
    if not args.smoke:
        if not args.smoke_gate:
            raise ValueError("scientific training requires --smoke-gate")
        smoke_gate = require_gate(
            args.smoke_gate,
            verdicts=("TWO_GPU_SMOKE_PASS",),
            source_combined_sha256=source_hash,
        )
    if args.logical_batch_size != 1 or args.gradient_accumulation_steps != 1:
        raise ValueError("historical effective global batch is fixed at one logical sample/update")

    output = Path(args.output).expanduser().resolve()
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
    else:
        exists = torch.tensor([int(output.exists())], device=device)
        dist.all_reduce(exists, op=dist.ReduceOp.MAX)
        if int(exists.item()):
            raise FileExistsError(f"refusing existing output: {output}")
        if rank == 0:
            output.mkdir(parents=True)
    dist.barrier()

    train_dataset = NativeV0SequenceDataset(
        args.cache,
        split="train",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    heldout_dataset = NativeV0SequenceDataset(
        args.cache,
        split="heldout",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    train_contract = train_dataset.contract()
    heldout_contract = heldout_dataset.contract()
    expected_splits = {
        "split_seed": int(args.split_seed),
        "heldout_fraction": float(args.heldout_fraction),
        "train_split_sha256": train_dataset.split_sha256,
        "heldout_split_sha256": heldout_dataset.split_sha256,
        "train_sequences": len(train_dataset),
        "heldout_sequences": len(heldout_dataset),
    }
    for gate_name, gate in (
        ("analysis", analysis_gate),
        ("mode", mode_gate),
        ("calibration", calibration),
    ):
        if gate.get("dataset_splits") != expected_splits:
            raise RuntimeError(f"{gate_name} gate uses a different train/held-out split")
    if smoke_gate is not None and smoke_gate.get("dataset_splits") != expected_splits:
        raise RuntimeError("smoke gate uses a different train/held-out split")
    weights = _approved_weights(
        args.approved_weights,
        source_hash,
        train_dataset.split_sha256,
    )
    sampler = ReplicatedLogicalSampler(len(train_dataset), args.seed)
    config = NativeV0Config()
    model = config.build().to(device)
    start_step = 0
    resume_payload: dict[str, Any] | None = None
    if args.resume:
        model, resume_payload = load_native_v0_checkpoint(args.resume, device=device)
        start_step = int(resume_payload["global_optimizer_step"])
        if resume_payload["source_lock"]["combined_sha256"] != source_hash:
            raise RuntimeError("resume checkpoint source lock differs from current source")
        previous_config = resume_payload["training_config"]
        resume_contract = {
            "mode": mode,
            "weights": weights.to_dict(),
            "peak_lr": float(args.peak_lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "dataset_contract_train": train_contract,
            "dataset_contract_heldout": heldout_contract,
        }
        mismatches = {
            key: (previous_config.get(key), value)
            for key, value in resume_contract.items()
            if previous_config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"resume scientific contract differs: {mismatches}")
    parameter_audit = model.parameter_audit()
    if not parameter_audit["under_hard_cap_1000000"]:
        raise RuntimeError("V0 parameter cap failed")
    ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineController(
        optimizer,
        peak_lr=args.peak_lr,
        total_steps=150_000,
        warmup_steps=7_500,
        final_ratio=0.1,
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        expected_sampler = sampler.state_dict(start_step)
        if resume_payload["sampler_state_dict"] != expected_sampler:
            raise RuntimeError("resume sampler state differs from deterministic contract")

    frozen_model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    training_config = {
        **vars(args),
        "selected_physical_gpu_ids": list(selected),
        "world_size": 2,
        "logical_global_batch_size": 1,
        "physical_replicas_per_logical_sample": 2,
        "unique_samples_per_optimizer_step": 1,
        "age_stacking_scope": "inside_each_rank_local_batch",
        "mode": mode,
        "weights": weights.to_dict(),
        "parameter_audit": parameter_audit,
        "source_combined_sha256": source_hash,
        "determinism": determinism,
        "cached_teacher_actions_used_in_objective": False,
        "teacher_action_target": "frozen_action_generator(full_current_condition,current_proprio,same_explicit_noise)",
        "dataset_splits": expected_splits,
        "dataset_contract_train": train_contract,
        "dataset_contract_heldout": heldout_contract,
    }
    if rank == 0 and not args.resume:
        write_json(output / "source_lock.json", source)
        write_json(output / "training_config.json", training_config)
        write_json(output / "dataset_contract_train.json", train_contract)
        write_json(output / "dataset_contract_heldout.json", heldout_contract)
    wandb_run = _wandb(args, rank, training_config)

    progress = tqdm(
        range(start_step, args.max_steps),
        initial=start_step,
        total=args.max_steps,
        disable=rank != 0,
        dynamic_ncols=True,
        desc="Native SimVLA V0 K4",
    )
    latest_metrics: dict[str, Any] = {}
    reference_action_cache: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    for zero_based_step in progress:
        logical_index = sampler.index(zero_based_step)
        _assert_identical_logical_index(logical_index, device)
        batch = move_batch(
            collate_native_v0_sequences([train_dataset[logical_index]]),
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        lr = scheduler.set_step(zero_based_step + 1)
        cached_reference = reference_action_cache.get(logical_index)
        if cached_reference is not None:
            cached_reference = tuple(value.to(device) for value in cached_reference)
        total, raw, weighted, teacher_actions = _forward_losses(
            adapter=ddp,
            batch=batch,
            processor=processor,
            action_adapter=action_adapter,
            mode=mode,
            weights=weights,
            requires_grad=True,
            reference_actions=cached_reference,
        )
        if logical_index not in reference_action_cache:
            reference_action_cache[logical_index] = tuple(
                value.detach().cpu() for value in teacher_actions
            )
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError(f"non-finite loss at optimizer step {zero_based_step + 1}")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), max_norm=1.0)
        if any(parameter.grad is not None for parameter in frozen_model.parameters()):
            raise RuntimeError("frozen backbone/action generator received gradients")
        optimizer.step()
        step = zero_based_step + 1
        latest_metrics = {
            "step": step,
            "lr": lr,
            "total": float(total.detach().item()),
            "gradient_norm_before_clip": float(gradient_norm.item()),
            **{f"raw/{name}": float(value.detach().item()) for name, value in raw.items()},
            **{f"weighted/{name}": float(value.detach().item()) for name, value in weighted.items()},
        }
        if rank == 0:
            progress.set_postfix(
                loss=f"{latest_metrics['total']:.4g}",
                first5=f"{latest_metrics['raw/first5_action_l1']:.4g}",
                age3=f"{latest_metrics['raw/age3/first5_action_l1']:.4g}",
                lr=f"{lr:.2e}",
            )
            if step % args.log_interval == 0 or step == 1 or step == args.max_steps:
                append_jsonl(output / "train_metrics.jsonl", latest_metrics)
                if wandb_run is not None:
                    wandb_run.log(latest_metrics, step=step)
        if step % args.validation_interval == 0 or step == args.max_steps:
            validation = _validate(
                model=ddp,
                dataset=heldout_dataset,
                processor=processor,
                action_adapter=action_adapter,
                mode=mode,
                weights=weights,
                device=device,
                batches=args.validation_batches,
            )
            if rank == 0:
                payload = {"step": step, "metrics": validation}
                append_jsonl(output / "validation_metrics.jsonl", payload)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"validation/{name}/{statistic}": value for name, stats in validation.items() for statistic, value in stats.items()},
                        step=step,
                    )
        should_save = step % args.save_interval == 0 or step == args.max_steps
        if should_save and rank == 0:
            final = bool(not args.smoke and step == 150_000)
            checkpoint = output / "checkpoints" / f"native_v0_step_{step:06d}.pt"
            save_native_v0_checkpoint(
                checkpoint,
                model=ddp.module,
                config=config,
                global_step=step,
                optimizer=optimizer,
                scheduler_state=scheduler.state_dict(),
                sampler_state=sampler.state_dict(step),
                source_lock=source,
                training_config=training_config,
                final=final,
            )
            (output / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n", encoding="utf-8")
        dist.barrier()

    frozen_grads_zero = all(parameter.grad is None for parameter in frozen_model.parameters())
    result = {
        "verdict": "TWO_GPU_SMOKE_PASS" if args.smoke else "FINAL_150K_TRAINING_COMPLETE",
        "global_optimizer_step": args.max_steps,
        "scientific_primary_checkpoint": bool(not args.smoke and args.max_steps == 150_000),
        "mode": mode,
        "scheduler_state": scheduler.state_dict(),
        "logical_global_batch_size": 1,
        "physical_ddp_replicas": 2,
        "frozen_base_action_gradients_zero": frozen_grads_zero,
        "latest_metrics": latest_metrics,
        "runtime_official_norm_reference_action_cache_entries": len(reference_action_cache),
        "source_combined_sha256": source_hash,
        "dataset_splits": expected_splits,
    }
    if rank == 0:
        write_json(output / "run_summary.json", result)
        if wandb_run is not None:
            wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--parity-gate", required=True)
    parser.add_argument("--analysis-gate", required=True)
    parser.add_argument("--parameter-gate", required=True)
    parser.add_argument("--mode-gate", required=True)
    parser.add_argument("--calibration-gate", required=True)
    parser.add_argument("--approved-weights", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-gate", default="")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logical-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=5000)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    args = parser.parse_args()
    result = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
