"""Compute-matched, fixed-step training for the Seer feature bridge.

This entry point intentionally remains separate from ``train.py``.  The latter
implements the literal 200/100-epoch reference contract; this trainer preserves
the global batch and losses while bounding optimization by explicit update
counts and removing avoidable input and synchronization overhead.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from methods.latent_bridge import ComputeMatchedTrainingContract, bridge_distillation_loss

from .bridge import SeerFeatureBridge, SeerFeatureBridgeConfig
from .checkpoint import (
    load_bridge_checkpoint,
    save_bridge_checkpoint,
    validate_bridge_runtime_provenance,
)
from .dataset import BridgeTransitionDataset
from .provenance import PUBLIC_SEER_33_SHA256, sha256_file, verify_official_source
from .train import ExactDistributedEvalSampler, _audit_transition_files, _dataset_files


_BATCH_KEYS = (
    "previous_condition",
    "stable_context",
    "current_state",
    "previous_executed_action",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("R0", "R1"), required=True)
    parser.add_argument("--sync-root", required=True)
    parser.add_argument("--dagger-root")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--preset", choices=("full", "small"), default="full")
    parser.add_argument("--stable-layer")
    parser.add_argument("--stable-token-group")
    parser.add_argument("--stable-seq-len", type=int)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--per-rank-batch", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--validation-every-data-epochs", type=int, default=4)
    parser.add_argument("--checkpoint-every-data-epochs", type=int, default=4)
    parser.add_argument("--log-every-updates", type=int, default=100)
    parser.add_argument("--preload-dataset", action="store_true")
    return parser


def _distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    if torch.cuda.is_available():
        return 0, 1, torch.device("cuda", 0)
    return 0, 1, torch.device("cpu")


def _seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _broadcast(value, *, rank: int):
    if not dist.is_initialized():
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _verified_files(root: str | Path, *, rank: int) -> tuple[list[Path], list[dict]]:
    payload = None
    if rank == 0:
        paths = _dataset_files(root)
        payload = []
        for path in paths:
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload.append({"path": str(path), "sha256": manifest["dataset_sha256"]})
    payload = _broadcast(payload, rank=rank)
    return [Path(record["path"]) for record in payload], payload


def _rank_zero_hash(path: str | Path | None, *, rank: int) -> str | None:
    value = sha256_file(path) if rank == 0 and path else None
    return _broadcast(value, rank=rank)


def _rank_zero_audit(paths: list[Path], config, *, rank: int) -> dict | None:
    value = _audit_transition_files(paths, config) if rank == 0 and paths else None
    return _broadcast(value, rank=rank)


def _make_split(
    paths: list[Path],
    records: list[dict],
    split: str,
    args,
) -> ConcatDataset:
    expected = {record["path"]: record["sha256"] for record in records}
    datasets = [
        BridgeTransitionDataset(
            path,
            split=split,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            expected_sha256=None,
            preload=args.preload_dataset,
        )
        for path in paths
    ]
    # Rank zero already verified every file before the paths were broadcast.
    if set(expected) != {str(path) for path in paths}:
        raise RuntimeError("verified transition records do not match dataset paths")
    nonempty = [dataset for dataset in datasets if len(dataset)]
    if not nonempty:
        raise RuntimeError(f"empty {split} split across {len(paths)} transition files")
    return ConcatDataset(nonempty)


def _preloaded_bytes(dataset) -> int:
    if isinstance(dataset, ConcatDataset):
        return sum(_preloaded_bytes(child) for child in dataset.datasets)
    return int(getattr(dataset, "preloaded_bytes", 0))


def _metric_vector(loss, parts, batch_count: int) -> torch.Tensor:
    return torch.stack(
        (
            loss.detach().double() * batch_count,
            parts["mse"].detach().double() * batch_count,
            parts["cosine_loss"].detach().double() * batch_count,
            loss.detach().new_tensor(float(batch_count), dtype=torch.float64),
        )
    )


def _reduce_metrics(values: torch.Tensor) -> torch.Tensor:
    reduced = values.clone()
    if dist.is_initialized():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def _metric_dict(values: torch.Tensor) -> dict[str, float | int]:
    count = max(float(values[3].item()), 1.0)
    cosine_loss = float(values[2].item() / count)
    return {
        "total": float(values[0].item() / count),
        "mse": float(values[1].item() / count),
        "cosine_loss": cosine_loss,
        "feature_cosine": 1.0 - cosine_loss,
        "samples": int(values[3].item()),
    }


@torch.no_grad()
def _validate(model, loader, device, use_bf16: bool) -> dict[str, float | int]:
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for batch in loader:
        values = [batch[key].to(device=device, non_blocking=True) for key in _BATCH_KEYS]
        target = batch["target_condition"].to(device=device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            raw = model.module if hasattr(model, "module") else model
            predicted = raw.predict_next(*values)
            loss, parts = bridge_distillation_loss(predicted.float(), target.float())
        totals += _metric_vector(loss, parts, target.shape[0])
    return _metric_dict(_reduce_metrics(totals))


def _load_or_initialize_bridge(args, *, rank: int):
    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    resume_payload = None
    if resume_path is not None and resume_path.exists():
        bridge, resume_payload = load_bridge_checkpoint(resume_path, map_location="cpu")
        validate_bridge_runtime_provenance(resume_payload)
        if resume_payload["stage"] != args.stage:
            raise RuntimeError(
                f"resume stage mismatch: expected={args.stage}, actual={resume_payload['stage']}"
            )

    if args.stage == "R0":
        if args.dagger_root or args.initial_checkpoint:
            raise ValueError("R0 starts from zero initialization and cannot consume DAgger data")
        required = (args.stable_layer, args.stable_token_group, args.stable_seq_len)
        if any(value is None for value in required):
            raise ValueError("R0 requires measured stable layer/group/sequence length")
        expected_config = SeerFeatureBridgeConfig.from_preset(
            args.preset,
            stable_seq_len=args.stable_seq_len,
            stable_layer=args.stable_layer,
            stable_token_group=args.stable_token_group,
        )
        if resume_payload is None:
            bridge = SeerFeatureBridge(expected_config)
        elif bridge.config != expected_config:
            raise RuntimeError("R0 resume bridge config differs from measured context")
        return bridge, resume_payload, None

    if not args.dagger_root or not args.initial_checkpoint:
        raise ValueError("R1 requires --dagger-root and --initial-checkpoint")
    initial_bridge, initial_payload = load_bridge_checkpoint(
        args.initial_checkpoint, map_location="cpu"
    )
    validate_bridge_runtime_provenance(initial_payload)
    if initial_bridge.config.preset != args.preset or initial_payload["stage"] != "R0":
        raise RuntimeError("R1 must initialize from a matching R0 checkpoint")
    if resume_payload is None:
        bridge = initial_bridge
    elif bridge.config != initial_bridge.config:
        raise RuntimeError("R1 resume config differs from its R0 initialization checkpoint")
    initial_hash = _rank_zero_hash(args.initial_checkpoint, rank=rank)
    return bridge, resume_payload, initial_hash


def _save_resume(
    output: Path,
    model,
    optimizer,
    scheduler,
    args,
    base_metadata: dict,
    *,
    data_epoch: int,
    global_step: int,
    best_loss: float,
    best_data_epoch: int,
    best_global_step: int,
    history: list[dict],
) -> None:
    raw = model.module if hasattr(model, "module") else model
    save_bridge_checkpoint(
        output / "resume.pt",
        raw,
        stage=args.stage,
        epoch=data_epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata={
            **base_metadata,
            "resume_state": {
                "global_step": global_step,
                "completed_data_epochs": data_epoch,
                "best_validation_total_loss": best_loss,
                "best_data_epoch": best_data_epoch,
                "best_global_step": best_global_step,
                "history": history,
            },
        },
    )


def main() -> None:
    args = _parser().parse_args()
    rank, world_size, device = _distributed()
    _seed_everything(args.seed, rank)
    torch.set_float32_matmul_precision("high")

    for name in (
        "validation_batch_size",
        "validation_every_data_epochs",
        "checkpoint_every_data_epochs",
        "log_every_updates",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    contract = ComputeMatchedTrainingContract(
        stage=args.stage,
        optimizer_steps=args.optimizer_steps,
        learning_rate=args.learning_rate,
        per_rank_batch=args.per_rank_batch,
        world_size=world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    contract.validate()

    official = _broadcast(
        verify_official_source(args.official_source) if rank == 0 else None,
        rank=rank,
    )
    bridge, resume_payload, initial_checkpoint_sha256 = _load_or_initialize_bridge(
        args, rank=rank
    )
    config = bridge.config

    setup_started = time.time()
    sync_files, sync_records = _verified_files(args.sync_root, rank=rank)
    dagger_files: list[Path] = []
    dagger_records: list[dict] = []
    if args.stage == "R1":
        dagger_files, dagger_records = _verified_files(args.dagger_root, rank=rank)

    train_parts = [_make_split(sync_files, sync_records, "train", args)]
    validation_parts = [_make_split(sync_files, sync_records, "validation", args)]
    if dagger_files:
        train_parts.append(_make_split(dagger_files, dagger_records, "train", args))
        validation_parts.append(
            _make_split(dagger_files, dagger_records, "validation", args)
        )
    training = ConcatDataset(train_parts)
    validation = ConcatDataset(validation_parts)
    sync_audit = _rank_zero_audit(sync_files, config, rank=rank)
    dagger_audit = _rank_zero_audit(dagger_files, config, rank=rank)

    train_sampler = (
        DistributedSampler(training, shuffle=True, seed=args.seed)
        if world_size > 1
        else None
    )
    validation_sampler = (
        ExactDistributedEvalSampler(validation, rank=rank, world_size=world_size)
        if world_size > 1
        else None
    )
    loader_workers = 0 if args.preload_dataset else 4
    train_loader = DataLoader(
        training,
        batch_size=args.per_rank_batch,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.validation_batch_size,
        sampler=validation_sampler,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
    )
    usable_batches = (
        len(train_loader) // args.gradient_accumulation_steps
    ) * args.gradient_accumulation_steps
    if usable_batches == 0:
        raise RuntimeError("training split is too small for one effective-batch update")
    updates_per_data_epoch = usable_batches // args.gradient_accumulation_steps

    local_preloaded_bytes = _preloaded_bytes(training) + _preloaded_bytes(validation)
    preload_total = torch.tensor(float(local_preloaded_bytes), device=device)
    if dist.is_initialized():
        dist.all_reduce(preload_total, op=dist.ReduceOp.SUM)

    bridge = bridge.to(device)
    model = (
        DistributedDataParallel(bridge, device_ids=[device.index], find_unused_parameters=False)
        if world_size > 1
        else bridge
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=contract.learning_rate, weight_decay=contract.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=contract.optimizer_steps,
        eta_min=contract.learning_rate * 0.01,
    )
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    base_metadata = {
        "training_protocol": contract.protocol,
        "contract": contract.__dict__,
        "official_source": official,
        "public_seer_checkpoint_sha256": PUBLIC_SEER_33_SHA256,
        "parameter_audit": bridge.parameter_audit(),
        "sync_files": sync_records,
        "dagger_files": dagger_records,
        "precision": args.precision,
        "world_size": world_size,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "sync_dataset_audit": sync_audit,
        "dagger_dataset_audit": dagger_audit,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "preload_dataset": args.preload_dataset,
        "preloaded_bytes_all_ranks": int(preload_total.item()),
        "validation_batch_size": args.validation_batch_size,
        "updates_per_data_epoch": updates_per_data_epoch,
        "validation_every_data_epochs": args.validation_every_data_epochs,
        "checkpoint_every_data_epochs": args.checkpoint_every_data_epochs,
    }
    if resume_payload is not None:
        resume_metadata = resume_payload.get("metadata", {})
        keys = (
            "training_protocol",
            "contract",
            "public_seer_checkpoint_sha256",
            "sync_files",
            "dagger_files",
            "precision",
            "world_size",
            "initial_checkpoint_sha256",
            "preload_dataset",
            "validation_batch_size",
        )
        mismatches = {
            key: {"expected": base_metadata[key], "actual": resume_metadata.get(key)}
            for key in keys
            if resume_metadata.get(key) != base_metadata[key]
        }
        if mismatches:
            raise RuntimeError(f"resume checkpoint contract mismatch: {mismatches}")

    global_step = 0
    data_epoch = 0
    best_loss = float("inf")
    best_data_epoch = 0
    best_global_step = 0
    history: list[dict] = []
    if resume_payload is not None:
        if "optimizer_state_dict" not in resume_payload or "scheduler_state_dict" not in resume_payload:
            raise RuntimeError("resume checkpoint lacks optimizer or scheduler state")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        state = resume_payload.get("metadata", {}).get("resume_state", {})
        global_step = int(state.get("global_step", 0))
        data_epoch = int(state.get("completed_data_epochs", resume_payload["epoch"]))
        best_loss = float(state.get("best_validation_total_loss", float("inf")))
        best_data_epoch = int(state.get("best_data_epoch", 0))
        best_global_step = int(state.get("best_global_step", 0))
        history = list(state.get("history", []))
        if global_step > contract.optimizer_steps:
            raise RuntimeError(
                f"resume step {global_step} exceeds target {contract.optimizer_steps}"
            )

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "SEER_LATENT_BRIDGE_EFFICIENT_START",
                    "stage": args.stage,
                    "target_optimizer_steps": contract.optimizer_steps,
                    "resumed_optimizer_step": global_step,
                    "effective_batch": contract.effective_batch,
                    "examples_seen_at_completion": contract.examples_seen,
                    "updates_per_data_epoch": updates_per_data_epoch,
                    "estimated_data_epochs": math.ceil(
                        contract.optimizer_steps / updates_per_data_epoch
                    ),
                    "dataset_setup_seconds": time.time() - setup_started,
                    "preloaded_bytes_all_ranks": int(preload_total.item()),
                }
            ),
            flush=True,
        )

    use_bf16 = args.precision == "bf16" and device.type == "cuda"
    training_started = time.time()
    starting_step = global_step
    window_totals = torch.zeros(4, device=device, dtype=torch.float64)
    window_start_step = global_step

    while global_step < contract.optimizer_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(data_epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals = torch.zeros(4, device=device, dtype=torch.float64)
        remaining_updates = contract.optimizer_steps - global_step
        batches_this_epoch = min(
            usable_batches,
            remaining_updates * args.gradient_accumulation_steps,
        )

        for batch_index, batch in enumerate(train_loader):
            if batch_index >= batches_this_epoch:
                break
            sync_now = (batch_index + 1) % args.gradient_accumulation_steps == 0
            sync_context = (
                contextlib.nullcontext()
                if sync_now or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync_context:
                values = [
                    batch[key].to(device=device, non_blocking=True) for key in _BATCH_KEYS
                ]
                target = batch["target_condition"].to(device=device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    predicted_delta = model(*values)
                    predicted_next = values[0] + predicted_delta
                    loss, parts = bridge_distillation_loss(
                        predicted_next.float(), target.float()
                    )
                    scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()

            metrics = _metric_vector(loss, parts, target.shape[0])
            epoch_totals += metrics
            window_totals += metrics
            if not sync_now:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), contract.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if (
                global_step % args.log_every_updates == 0
                or global_step == contract.optimizer_steps
            ):
                reduced = _reduce_metrics(window_totals)
                elapsed = max(time.time() - training_started, 1e-9)
                completed_since_start = max(global_step - starting_step, 1)
                updates_per_second = completed_since_start / elapsed
                eta_seconds = (contract.optimizer_steps - global_step) / updates_per_second
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "status": "SEER_LATENT_BRIDGE_EFFICIENT_PROGRESS",
                                "stage": args.stage,
                                "optimizer_step": global_step,
                                "target_optimizer_steps": contract.optimizer_steps,
                                "progress_percent": 100.0 * global_step / contract.optimizer_steps,
                                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "window": _metric_dict(reduced),
                                "updates_per_second": updates_per_second,
                                "eta_seconds": eta_seconds,
                            }
                        ),
                        flush=True,
                    )
                window_totals.zero_()
                window_start_step = global_step

        data_epoch += 1
        reduced_epoch = _reduce_metrics(epoch_totals)
        validate_now = (
            data_epoch % args.validation_every_data_epochs == 0
            or global_step == contract.optimizer_steps
        )
        validation_metrics = None
        if validate_now:
            validation_metrics = _validate(model, validation_loader, device, use_bf16)
            row = {
                "data_epoch": data_epoch,
                "optimizer_step": global_step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": _metric_dict(reduced_epoch),
                "validation": validation_metrics,
            }
            history.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
                raw = model.module if hasattr(model, "module") else model
                if float(validation_metrics["total"]) < best_loss:
                    best_loss = float(validation_metrics["total"])
                    best_data_epoch = data_epoch
                    best_global_step = global_step
                    save_bridge_checkpoint(
                        output / "best.pt",
                        raw,
                        stage=args.stage,
                        epoch=data_epoch,
                        metadata={
                            **base_metadata,
                            "selection_metric": "validation_total_loss",
                            "selection_metric_value": best_loss,
                            "selection_global_step": best_global_step,
                        },
                    )

        checkpoint_now = (
            data_epoch % args.checkpoint_every_data_epochs == 0
            or global_step == contract.optimizer_steps
        )
        if rank == 0 and checkpoint_now:
            _save_resume(
                output,
                model,
                optimizer,
                scheduler,
                args,
                base_metadata,
                data_epoch=data_epoch,
                global_step=global_step,
                best_loss=best_loss,
                best_data_epoch=best_data_epoch,
                best_global_step=best_global_step,
                history=history,
            )
        if dist.is_initialized():
            dist.barrier()

    if window_start_step != global_step:
        # This only occurs when the final update is not a logging boundary.
        reduced = _reduce_metrics(window_totals)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "SEER_LATENT_BRIDGE_EFFICIENT_FINAL_WINDOW",
                        "stage": args.stage,
                        "optimizer_step": global_step,
                        "window": _metric_dict(reduced),
                    }
                ),
                flush=True,
            )

    if rank == 0:
        raw = model.module if hasattr(model, "module") else model
        save_bridge_checkpoint(
            output / "last.pt",
            raw,
            stage=args.stage,
            epoch=data_epoch,
            metadata={
                **base_metadata,
                "completed_optimizer_steps": global_step,
                "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
        )
        if not (output / "best.pt").is_file() or best_global_step <= 0:
            raise RuntimeError("training completed without a selected validation checkpoint")
        summary = {
            "status": "SEER_LATENT_BRIDGE_EFFICIENT_TRAINING_COMPLETE",
            "stage": args.stage,
            "training_protocol": contract.protocol,
            "target_optimizer_steps": contract.optimizer_steps,
            "completed_optimizer_steps": global_step,
            "effective_batch": contract.effective_batch,
            "examples_seen": contract.examples_seen,
            "completed_data_epochs": data_epoch,
            "updates_per_data_epoch": updates_per_data_epoch,
            "best_validation_total_loss": best_loss,
            "best_data_epoch": best_data_epoch,
            "best_global_step": best_global_step,
            "training_elapsed_seconds": time.time() - training_started,
            "history": history,
        }
        (output / "training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (output / "COMPLETE").write_text(summary["status"] + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
