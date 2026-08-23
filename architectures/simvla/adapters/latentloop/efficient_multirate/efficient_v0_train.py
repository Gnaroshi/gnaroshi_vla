"""Two-rank cache-backed trainer for the exact native SimVLA Condition Loop V0."""

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
    effective_batch_contract,
    require_gate_payload,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherSequenceDataset,
    _drop_unused_vlm,
    collate_exact_teacher_sequences,
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import (
    install_exact_uint8_delta_path,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.v0_objectives import (
    cache_backed_v0_loss,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    NativeV0Config,
    load_native_v0_checkpoint,
    save_native_v0_checkpoint,
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
from methods.latentloop.training.native_simvla_v0 import (
    NativeV0LossWeights,
    WarmupCosineController,
)


class ReplicatedStepSampler(Sampler[int]):
    """Yield the parent run's deterministic logical sample order on both ranks."""

    def __init__(self, dataset_size: int, seed: int, start_step: int, end_step: int) -> None:
        if dataset_size < 1 or start_step < 0 or end_step <= start_step:
            raise ValueError("invalid deterministic sampler contract")
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.end_step = int(end_step)

    def index(self, optimizer_step: int) -> int:
        epoch, offset = divmod(int(optimizer_step), self.dataset_size)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        return int(torch.randperm(self.dataset_size, generator=generator)[offset].item())

    def __iter__(self) -> Iterator[int]:
        for step in range(self.start_step, self.end_step):
            yield self.index(step)

    def __len__(self) -> int:
        return self.end_step - self.start_step

    def state_dict(self, optimizer_step: int) -> dict[str, int | str]:
        return {
            "sampler": "replicated_logical_global_batch_one_prefetched",
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "next_optimizer_step": int(optimizer_step),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _approved_weights(
    path: str | Path,
    *,
    source: dict[str, Any],
    train_split_sha256: str,
) -> NativeV0LossWeights:
    payload = _load_json(path)
    if payload.get("approved_by_user") is not True:
        raise RuntimeError("loss weights require approved_by_user=true")
    allowed_sources = {
        str(source["combined_sha256"]),
        str(source["parent_source_combined_sha256"]),
    }
    if str(payload.get("source_combined_sha256")) not in allowed_sources:
        raise RuntimeError("approved weights do not belong to this source lineage")
    if payload.get("train_split_sha256") != train_split_sha256:
        raise RuntimeError("approved weights use a different training split")
    names = (
        "condition",
        "first5_action",
        "full_chunk_action",
        "continuous_gripper",
        "update_regularization",
    )
    values = {name: float(payload[name]) for name in names}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("loss weights must be finite and non-negative")
    if values["first5_action"] <= 0:
        raise ValueError("first5_action weight must be positive")
    return NativeV0LossWeights(**values)


def _assert_identical_index(index: int, device: torch.device) -> None:
    local = torch.tensor([int(index)], device=device, dtype=torch.long)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    observed = [int(value.item()) for value in gathered]
    if len(set(observed)) != 1:
        raise RuntimeError(f"DDP ranks selected different logical samples: {observed}")


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


@torch.no_grad()
def _validate(
    *,
    model: torch.nn.Module,
    dataset: ExactTeacherSequenceDataset,
    action_adapter: Any,
    weights: NativeV0LossWeights,
    device: torch.device,
    batches: int,
) -> dict[str, Any]:
    model.eval()
    values: dict[str, list[float]] = {}
    for index in range(min(int(batches), len(dataset))):
        batch = move_batch(collate_exact_teacher_sequences([dataset[index]]), device)
        output = cache_backed_v0_loss(
            adapter=model,
            batch=batch,
            action_adapter=action_adapter,
            weights=weights,
            objective_mode="B",
            zero_based_optimizer_step=0,
            requires_grad=False,
        )
        values.setdefault("total", []).append(float(output.total.item()))
        for name, value in output.raw.items():
            values.setdefault(name, []).append(float(value.item()))
    model.train()
    return {
        name: {
            "count": len(items),
            "mean": float(np.mean(items)),
            "p50": float(np.quantile(items, 0.50)),
            "p95": float(np.quantile(items, 0.95)),
            "p99": float(np.quantile(items, 0.99)),
        }
        for name, items in values.items()
    }


def _monitoring_gate(
    validation: dict[str, Any],
    reference: dict[str, Any],
    *,
    source_hash: str,
    step: int,
) -> dict[str, Any]:
    raw_reference = reference["raw_loss_summary"]
    finite = all(
        math.isfinite(float(stats[statistic]))
        for stats in validation.values()
        for statistic in ("mean", "p95", "p99")
    )
    age_checks = {
        f"age{age}_first5_below_untrained_hold": (
            float(validation[f"age{age}/first5_action_l1"]["mean"])
            < float(raw_reference[f"age{age}/first5_action_l1"]["mean"])
        )
        for age in (1, 2, 3)
    }
    gripper_p99 = float(validation["continuous_gripper_l1"]["p99"])
    reference_gripper_p99 = float(raw_reference["continuous_gripper_l1"]["p99"])
    no_gripper_collapse = gripper_p99 <= reference_gripper_p99
    no_p99_explosion = all(
        float(validation[f"age{age}/first5_action_l1"]["p99"])
        <= 1.25 * float(raw_reference[f"age{age}/first5_action_l1"]["p99"])
        for age in (1, 2, 3)
    )
    checks = {
        "all_metrics_finite": finite,
        **age_checks,
        "no_gripper_collapse": no_gripper_collapse,
        "no_first5_p99_explosion": no_p99_explosion,
    }
    return {
        "verdict": (
            "CONTINUE_TO_150K" if all(checks.values()) else "STOP_EARLY_INVALID_TRAINING"
        ),
        "source_combined_sha256": source_hash,
        "optimizer_step": int(step),
        "offline_heldout_only": True,
        "libero_success_used": False,
        "scheduler_horizon_changed": False,
        "checks": checks,
        "validation": validation,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("cache-backed V0 requires torchrun WORLD_SIZE=2")
    selected = tuple(int(value) for value in os.environ["SIMVLA_GPU_IDS"].split(","))
    if len(selected) != 2 or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, selected)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must exactly match two SIMVLA_GPU_IDS")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    _seed_everything(args.seed)
    determinism = configure_strict_torch_determinism(args.seed)

    source = _load_json(args.source_lock)
    source_hash = str(source["combined_sha256"])
    exact_manifest = _load_json(Path(args.cache) / "manifest.json")
    if exact_manifest.get("source_combined_sha256") != source_hash:
        raise RuntimeError("exact cache and efficient source lock differ")
    cache_validation = validate_exact_cache(args.cache, verify_checksums=False)
    if cache_validation["verdict"] != "EXACT_TEACHER_CACHE_VALID":
        raise RuntimeError(f"exact cache validation failed: {cache_validation}")
    require_gate_payload(
        args.cache_gate,
        verdicts=("EXACT_TEACHER_CACHE_COMPLETE",),
        source_combined_sha256=source_hash,
    )
    objective_verdict = (
        "MODE_B_APPROVED"
        if args.objective_mode == "B" or args.benchmark
        else "MODE_D_APPROVED"
    )
    require_gate_payload(
        args.objective_gate,
        verdicts=(objective_verdict,),
        source_combined_sha256=source_hash,
    )
    batch_gate = require_gate_payload(
        args.batch_gate,
        verdicts=("BATCH_CONFIGURATION_SELECTED",),
        source_combined_sha256=source_hash,
    )
    if (
        int(batch_gate["local_unique_batch"]) != 1
        or int(batch_gate["gradient_accumulation_steps"]) != 1
        or batch_gate.get("replicated_logical_sample") is not True
    ):
        raise RuntimeError("selected batch gate changed effective global batch one")
    if args.smoke and args.benchmark:
        raise ValueError("--smoke and --benchmark are mutually exclusive")
    if args.smoke:
        if not 1 <= args.max_steps <= 20:
            raise ValueError("smoke requires 1-20 optimizer steps")
    elif args.benchmark:
        if not 1_000 <= args.max_steps <= 5_000:
            raise ValueError("bounded throughput benchmark requires 1,000-5,000 steps")
    else:
        if args.max_steps != 150_000:
            raise ValueError("primary efficient V0 training is fixed to 150,000 steps")
        require_gate_payload(
            args.wallclock_gate,
            verdicts=("TRAIN_150K_APPROVED",),
            source_combined_sha256=source_hash,
        )
        require_gate_payload(
            args.smoke_gate,
            verdicts=("TWO_GPU_EFFICIENT_SMOKE_PASS",),
            source_combined_sha256=source_hash,
        )

    output = Path(args.output).expanduser().resolve()
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError("resume output root does not exist")
    else:
        exists = torch.tensor([int(output.exists())], device=device)
        dist.all_reduce(exists, op=dist.ReduceOp.MAX)
        if int(exists.item()):
            raise FileExistsError(f"refusing existing output: {output}")
        if rank == 0:
            output.mkdir(parents=True)
    dist.barrier()

    train_dataset = ExactTeacherSequenceDataset(
        args.cache,
        split="train",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    heldout_dataset = ExactTeacherSequenceDataset(
        args.cache,
        split="heldout",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    parent_training_config = _load_json(args.parent_training_config)
    parent_splits = parent_training_config["dataset_splits"]
    observed_splits = {
        "split_seed": int(args.split_seed),
        "heldout_fraction": float(args.heldout_fraction),
        "train_split_sha256": train_dataset.split_sha256,
        "heldout_split_sha256": heldout_dataset.split_sha256,
        "train_sequences": len(train_dataset),
        "heldout_sequences": len(heldout_dataset),
    }
    if observed_splits != parent_splits:
        raise RuntimeError(
            f"cache-backed split does not equal streaming parent: {observed_splits} != {parent_splits}"
        )
    weights = _approved_weights(
        args.approved_weights,
        source=source,
        train_split_sha256=train_dataset.split_sha256,
    )
    config = NativeV0Config()
    model = config.build().to(device)
    start_step = 0
    resume_payload: dict[str, Any] | None = None
    if args.resume:
        model, resume_payload = load_native_v0_checkpoint(args.resume, device=device)
        start_step = int(resume_payload["global_optimizer_step"])
        if resume_payload["source_lock"]["combined_sha256"] != source_hash:
            raise RuntimeError("resume source lock differs")
    exact_delta_path = install_exact_uint8_delta_path(model)
    ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineController(
        optimizer,
        peak_lr=args.peak_lr,
        total_steps=150_000,
        warmup_steps=7_500,
        final_ratio=0.1,
    )
    sampler = ReplicatedStepSampler(
        len(train_dataset), args.seed, start_step, args.max_steps
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload["sampler_state_dict"] != sampler.state_dict(start_step):
            raise RuntimeError("resume sampler state differs")

    loader_kwargs: dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": 1,
        "sampler": sampler,
        "collate_fn": collate_exact_teacher_sequences,
        "num_workers": int(args.num_workers),
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(args.prefetch_factor),
            }
        )
    loader = DataLoader(**loader_kwargs)

    frozen_model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    unused_release = _drop_unused_vlm(frozen_model)
    del processor
    batch_contract = effective_batch_contract(
        local_unique_batch=1,
        gradient_accumulation_steps=1,
        world_size=2,
        replicated_logical_sample=True,
    )
    gpu_contract = _load_json(args.gpu_contract)
    training_config = {
        **vars(args),
        "experiment_identifier": "simvla_efficient_coupled_multirate_latentloop",
        "source_combined_sha256": source_hash,
        "parent_source_combined_sha256": source["parent_source_combined_sha256"],
        "selected_physical_gpu_ids": list(selected),
        "mode": args.objective_mode,
        "objective_mode": args.objective_mode,
        "cache_backed_teacher_targets": True,
        "teacher_vlm_calls_per_optimizer_step": 0,
        "teacher_action_calls_per_optimizer_step": 0,
        "predicted_age_action_decodes_per_step": 3 if args.objective_mode == "B" else 1,
        "weights": weights.to_dict(),
        "dataset_splits": observed_splits,
        "dataset_contract_train": parent_training_config["dataset_contract_train"],
        "dataset_contract_heldout": parent_training_config["dataset_contract_heldout"],
        "exact_teacher_dataset_contract_train": train_dataset.contract(),
        "exact_teacher_dataset_contract_heldout": heldout_dataset.contract(),
        "batch_contract": batch_contract,
        "dataloader": {
            "pin_memory": True,
            "persistent_workers": args.num_workers > 0,
            "prefetch_factor": int(args.prefetch_factor) if args.num_workers > 0 else None,
            "nonblocking_transfers": True,
            "rank_local_mmap_shards": True,
            "per_step_checksums": False,
        },
        "exact_uint8_delta_path": exact_delta_path,
        "unused_frozen_modules_released": unused_release,
        "determinism": determinism,
        "gpu_contract": gpu_contract,
    }
    if rank == 0 and not args.resume:
        write_json(output / "source_lock.json", source)
        write_json(output / "training_config.json", training_config)
        write_json(output / "dataset_contract_train.json", train_dataset.contract())
        write_json(output / "dataset_contract_heldout.json", heldout_dataset.contract())
        write_json(output / "gpu_contract.json", gpu_contract)
    wandb_run = _wandb(args, rank, training_config)

    progress = tqdm(
        enumerate(loader, start=start_step),
        initial=start_step,
        total=args.max_steps,
        disable=rank != 0,
        dynamic_ncols=True,
        desc=f"Efficient SimVLA V0 {args.objective_mode}",
    )
    latest: dict[str, Any] = {}
    latest_validation: dict[str, Any] = {}
    measured_step_seconds: list[float] = []
    wall_start = time.perf_counter()
    profile_wall_start: float | None = None
    for zero_based_step, host_batch in progress:
        if zero_based_step >= args.max_steps:
            break
        step_start = time.perf_counter()
        if zero_based_step == start_step + args.profile_warmup_steps:
            dist.barrier()
            profile_wall_start = time.perf_counter()
            step_start = profile_wall_start
        if zero_based_step == start_step or (zero_based_step + 1) % 10_000 == 0:
            _assert_identical_index(sampler.index(zero_based_step), device)
        batch = move_batch(host_batch, device)
        data_ready = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        lr = scheduler.set_step(zero_based_step + 1)
        loss = cache_backed_v0_loss(
            adapter=ddp,
            batch=batch,
            action_adapter=action_adapter,
            weights=weights,
            objective_mode=args.objective_mode,
            zero_based_optimizer_step=zero_based_step,
            requires_grad=True,
        )
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), max_norm=1.0)
        if any(parameter.grad is not None for parameter in frozen_model.parameters()):
            raise RuntimeError("frozen SimVLA modules received gradients")
        optimizer.step()
        step = zero_based_step + 1
        should_log = step == 1 or step % args.log_interval == 0 or step == args.max_steps
        if should_log:
            if not bool(torch.isfinite(loss.total).item()):
                raise FloatingPointError(f"non-finite loss at step {step}")
            latest = {
                "step": step,
                "lr": lr,
                "objective_mode": args.objective_mode,
                "selected_action_age": loss.selected_action_age,
                "total": float(loss.total.detach().item()),
                "gradient_norm_before_clip": float(gradient_norm.item()),
                **{f"raw/{name}": float(value.detach().item()) for name, value in loss.raw.items()},
                **{f"weighted/{name}": float(value.detach().item()) for name, value in loss.weighted.items()},
            }
            if rank == 0:
                append_jsonl(output / "train_metrics.jsonl", latest)
                progress.set_postfix(
                    loss=f"{latest['total']:.4g}",
                    first5=f"{latest['raw/first5_action_l1']:.4g}",
                    lr=f"{lr:.2e}",
                )
                if wandb_run is not None:
                    wandb_run.log(latest, step=step)
        validation: dict[str, Any] | None = None
        if step % args.validation_interval == 0 or step == args.max_steps:
            validation = _validate(
                model=ddp,
                dataset=heldout_dataset,
                action_adapter=action_adapter,
                weights=weights,
                device=device,
                batches=args.validation_batches,
            )
            latest_validation = validation
            if rank == 0:
                append_jsonl(output / "validation_metrics.jsonl", {"step": step, "metrics": validation})
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"validation/{name}/{statistic}": value
                            for name, stats in validation.items()
                            for statistic, value in stats.items()
                        },
                        step=step,
                    )
        if step == 10_000:
            if validation is None:
                validation = _validate(
                    model=ddp,
                    dataset=heldout_dataset,
                    action_adapter=action_adapter,
                    weights=weights,
                    device=device,
                    batches=args.validation_batches,
                )
            monitoring = _monitoring_gate(
                validation,
                _load_json(args.monitor_reference),
                source_hash=source_hash,
                step=step,
            )
            if rank == 0:
                write_json(output / "monitoring_step_010000.json", monitoring)
            if monitoring["verdict"] == "STOP_EARLY_INVALID_TRAINING":
                raise RuntimeError("predeclared 10K offline monitoring gate failed")
        if not args.benchmark and (step % args.save_interval == 0 or step == args.max_steps):
            if rank == 0:
                final = bool(not args.smoke and step == 150_000)
                checkpoint = output / "checkpoints" / f"efficient_native_v0_step_{step:06d}.pt"
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
        step_seconds = time.perf_counter() - step_start
        if step > start_step + args.profile_warmup_steps:
            measured_step_seconds.append(step_seconds)
        if rank == 0 and step % args.profile_log_interval == 0:
            append_jsonl(
                output / "throughput_metrics.jsonl",
                {
                    "step": step,
                    "data_seconds": data_ready - step_start,
                    "step_seconds": step_seconds,
                    "mean_measured_step_seconds": (
                        float(np.mean(measured_step_seconds)) if measured_step_seconds else None
                    ),
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                },
            )

    dist.barrier()
    wall_end = time.perf_counter()
    elapsed = wall_end - wall_start
    measured_mean = (
        (wall_end - profile_wall_start) / len(measured_step_seconds)
        if profile_wall_start is not None and measured_step_seconds
        else None
    )
    result = {
        "verdict": (
            "TWO_GPU_EFFICIENT_SMOKE_PASS"
            if args.smoke
            else f"BOUNDED_MODE_{args.objective_mode}_BENCHMARK_COMPLETE"
            if args.benchmark
            else "FINAL_150K_TRAINING_COMPLETE"
        ),
        "source_combined_sha256": source_hash,
        "global_optimizer_step": args.max_steps,
        "scientific_primary_checkpoint": bool(not args.smoke and args.max_steps == 150_000),
        "objective_mode": args.objective_mode,
        "batch_contract": batch_contract,
        "scheduler_state": scheduler.state_dict(),
        "frozen_base_action_gradients_zero": all(
            parameter.grad is None for parameter in frozen_model.parameters()
        ),
        "elapsed_seconds": elapsed,
        "measured_steps": len(measured_step_seconds),
        "mean_measured_step_seconds": measured_mean,
        "cpu_observed_step_seconds_p50": (
            float(np.quantile(measured_step_seconds, 0.50))
            if measured_step_seconds
            else None
        ),
        "cpu_observed_step_seconds_p95": (
            float(np.quantile(measured_step_seconds, 0.95))
            if measured_step_seconds
            else None
        ),
        "measurement_boundary_barriers": 2 if profile_wall_start is not None else 1,
        "per_step_barrier_or_cuda_synchronize": False,
        "projected_150k_hours_from_run": (
            measured_mean * 150_000 / 3600.0 if measured_mean is not None else None
        ),
        "latest_metrics": latest,
        "final_validation": latest_validation,
        "mode_d_age_counts": (
            {
                str(age): sum(
                    1 for step in range(start_step, args.max_steps) if step % 3 + 1 == age
                )
                for age in (1, 2, 3)
            }
            if args.objective_mode == "D"
            else None
        ),
        "dataset_splits": observed_splits,
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
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--cache-gate", required=True)
    parser.add_argument("--objective-gate", required=True)
    parser.add_argument("--batch-gate", required=True)
    parser.add_argument("--wallclock-gate", default="")
    parser.add_argument("--smoke-gate", default="")
    parser.add_argument("--parent-training-config", required=True)
    parser.add_argument("--approved-weights", required=True)
    parser.add_argument("--monitor-reference", required=True)
    parser.add_argument("--gpu-contract", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--objective-mode", choices=("B", "D"), required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=5000)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--profile-warmup-steps", type=int, default=100)
    parser.add_argument("--profile-log-interval", type=int, default=1000)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    args = parser.parse_args()
    result = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
