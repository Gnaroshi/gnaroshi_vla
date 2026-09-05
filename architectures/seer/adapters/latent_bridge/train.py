"""Distributed R0/R1 training for the Seer feature bridge."""

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
import h5py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Sampler

from methods.latent_bridge import TrainingContract, bridge_distillation_loss

from .bridge import SeerFeatureBridge, SeerFeatureBridgeConfig
from .checkpoint import (
    load_bridge_checkpoint,
    save_bridge_checkpoint,
    validate_bridge_runtime_provenance,
)
from .dataset import BridgeTransitionDataset
from .provenance import expected_base_checkpoint_sha256, sha256_file, verify_official_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("R0", "R1"), required=True)
    parser.add_argument("--sync-root", required=True)
    parser.add_argument("--dagger-root")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--preset", choices=("full", "small"), default="small")
    parser.add_argument("--stable-layer")
    parser.add_argument("--stable-token-group")
    parser.add_argument("--stable-seq-len", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
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


def _seed(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _dataset_files(root: str | Path) -> list[Path]:
    paths = sorted(Path(root).glob("*_transitions_rank*.h5"))
    if not paths:
        paths = sorted(Path(root).glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"no transition HDF5 files under {root}")
    for path in paths:
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing transition manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = sha256_file(path)
        if actual != manifest.get("dataset_sha256"):
            raise RuntimeError(f"transition hash mismatch: {path}")
    return paths


def _audit_transition_files(paths: list[Path], config: SeerFeatureBridgeConfig) -> dict:
    episode_ids = set()
    transitions = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            expected_shapes = {
                "previous_condition": (config.target_seq_len, config.feature_dim),
                "target_condition": (config.target_seq_len, config.feature_dim),
                "stable_context": (config.stable_seq_len, config.feature_dim),
                "current_state": (config.state_dim,),
                "previous_executed_action": (config.action_dim,),
            }
            for key, shape in expected_shapes.items():
                if tuple(handle[key].shape[1:]) != shape:
                    raise RuntimeError(
                        f"{path}:{key} shape {tuple(handle[key].shape[1:])} != {shape}"
                    )
            local_ids = {
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["episode_id"]
            }
            overlap = episode_ids & local_ids
            if overlap:
                raise RuntimeError(f"duplicate episodes across transition shards: {sorted(overlap)[:5]}")
            episode_ids.update(local_ids)
            transitions += int(handle["previous_condition"].shape[0])
    return {"episodes": len(episode_ids), "transitions": transitions}


def _make_split(paths: list[Path], split: str, args) -> ConcatDataset:
    datasets = [
        BridgeTransitionDataset(
            path,
            split=split,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            expected_sha256=json.loads(
                path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8")
            )["dataset_sha256"],
        )
        for path in paths
    ]
    nonempty = [dataset for dataset in datasets if len(dataset)]
    if not nonempty:
        raise RuntimeError(f"empty {split} split across {len(paths)} transition files")
    return ConcatDataset(nonempty)


def _reduce_metrics(sums: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    return sums


class ExactDistributedEvalSampler(Sampler[int]):
    """Partition validation indices across ranks without padding or duplication."""

    def __init__(self, dataset, *, rank: int, world_size: int):
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} is outside world_size {world_size}")
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size


@torch.no_grad()
def _validate(model, loader, device, use_bf16: bool) -> dict:
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for batch in loader:
        values = [
            batch[key].to(device=device, non_blocking=True)
            for key in (
                "previous_condition",
                "stable_context",
                "current_state",
                "previous_executed_action",
            )
        ]
        target = batch["target_condition"].to(device=device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            predicted = model.module.predict_next(*values) if hasattr(model, "module") else model.predict_next(*values)
            loss, parts = bridge_distillation_loss(predicted.float(), target.float())
        batch_count = target.shape[0]
        totals += torch.tensor(
            [loss.item() * batch_count, parts["mse"].item() * batch_count,
             parts["cosine_loss"].item() * batch_count, batch_count],
            device=device,
            dtype=torch.float64,
        )
    totals = _reduce_metrics(totals)
    count = max(float(totals[3].item()), 1.0)
    return {
        "total": float(totals[0].item() / count),
        "mse": float(totals[1].item() / count),
        "cosine_loss": float(totals[2].item() / count),
        "feature_cosine": float(1.0 - totals[2].item() / count),
        "samples": int(totals[3].item()),
    }


def main() -> None:
    args = _parser().parse_args()
    rank, world_size, device = _distributed()
    _seed(args.seed, rank)
    default_epochs = 200 if args.stage == "R0" else 100
    default_lr = 3e-4 if args.stage == "R0" else 3e-5
    epochs = args.epochs if args.epochs is not None else default_epochs
    learning_rate = args.learning_rate if args.learning_rate is not None else default_lr
    if not args.smoke and (epochs != default_epochs or not math.isclose(learning_rate, default_lr)):
        raise ValueError(
            f"official {args.stage} contract requires epochs={default_epochs}, lr={default_lr}"
        )
    contract = TrainingContract(
        stage=args.stage,
        epochs=epochs,
        learning_rate=learning_rate,
        per_rank_batch=args.per_rank_batch,
        world_size=world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    contract.validate()
    if args.checkpoint_every_epochs < 1:
        raise ValueError("--checkpoint-every-epochs must be positive")
    official = verify_official_source(args.official_source)
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
            config = expected_config
            bridge = SeerFeatureBridge(config)
        else:
            config = bridge.config
            if config != expected_config:
                raise RuntimeError(
                    "R0 resume bridge config differs from the requested measured context"
                )
    else:
        if not args.dagger_root or not args.initial_checkpoint:
            raise ValueError("R1 requires --dagger-root and --initial-checkpoint")
        initial_bridge, initial_payload = load_bridge_checkpoint(
            args.initial_checkpoint, map_location="cpu"
        )
        validate_bridge_runtime_provenance(initial_payload)
        if initial_bridge.config.preset != args.preset:
            raise RuntimeError("R1 preset differs from the R0 checkpoint")
        if initial_payload["stage"] != "R0":
            raise RuntimeError("R1 must initialize from an R0 checkpoint")
        if resume_payload is None:
            bridge = initial_bridge
        elif bridge.config != initial_bridge.config:
            raise RuntimeError("R1 resume config differs from its R0 initialization checkpoint")
        config = bridge.config

    sync_files = _dataset_files(args.sync_root)
    train_parts = [_make_split(sync_files, "train", args)]
    validation_parts = [_make_split(sync_files, "validation", args)]
    dagger_files = []
    if args.stage == "R1":
        dagger_files = _dataset_files(args.dagger_root)
        train_parts.append(_make_split(dagger_files, "train", args))
        validation_parts.append(_make_split(dagger_files, "validation", args))
    training = ConcatDataset(train_parts)
    validation = ConcatDataset(validation_parts)
    sync_audit = _audit_transition_files(sync_files, config)
    dagger_audit = _audit_transition_files(dagger_files, config) if dagger_files else None
    sync_records = [{"path": str(path), "sha256": sha256_file(path)} for path in sync_files]
    dagger_records = [{"path": str(path), "sha256": sha256_file(path)} for path in dagger_files]
    base_checkpoint_sha256 = expected_base_checkpoint_sha256()
    base_metadata = {
        "contract": contract.__dict__,
        "official_source": official,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "parameter_audit": bridge.parameter_audit(),
        "sync_files": sync_records,
        "dagger_files": dagger_records,
        "precision": args.precision,
        "world_size": world_size,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "sync_dataset_audit": sync_audit,
        "dagger_dataset_audit": dagger_audit,
        "initial_checkpoint_sha256": (
            sha256_file(args.initial_checkpoint) if args.initial_checkpoint else None
        ),
    }
    if resume_payload is not None:
        resume_metadata = resume_payload.get("metadata", {})
        expected_resume_fields = {
            "contract": base_metadata["contract"],
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "sync_files": sync_records,
            "dagger_files": dagger_records,
            "precision": args.precision,
            "world_size": world_size,
            "initial_checkpoint_sha256": base_metadata["initial_checkpoint_sha256"],
        }
        mismatches = {
            key: {"expected": value, "actual": resume_metadata.get(key)}
            for key, value in expected_resume_fields.items()
            if resume_metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"resume checkpoint contract mismatch: {mismatches}")

    train_sampler = DistributedSampler(training, shuffle=True, seed=args.seed) if world_size > 1 else None
    val_sampler = (
        ExactDistributedEvalSampler(validation, rank=rank, world_size=world_size)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        training,
        batch_size=args.per_rank_batch,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        validation,
        batch_size=args.per_rank_batch,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    usable_batches = (len(train_loader) // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
    if usable_batches == 0:
        raise RuntimeError("training split is too small for one effective-batch update")

    bridge = bridge.to(device)
    model = (
        DistributedDataParallel(bridge, device_ids=[device.index], find_unused_parameters=False)
        if world_size > 1
        else bridge
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=contract.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    started = time.time()
    start_epoch = 0
    best_loss = float("inf")
    best_epoch = 0
    history = []
    if resume_payload is not None:
        if "optimizer_state_dict" not in resume_payload or "scheduler_state_dict" not in resume_payload:
            raise RuntimeError("resume checkpoint lacks optimizer or scheduler state")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        start_epoch = int(resume_payload["epoch"])
        resume_state = resume_payload.get("metadata", {}).get("resume_state", {})
        best_loss = float(resume_state.get("best_validation_total_loss", float("inf")))
        best_epoch = int(resume_state.get("best_epoch", 0))
        history = list(resume_state.get("history", []))
        if len(history) != start_epoch:
            raise RuntimeError(
                f"resume history length {len(history)} does not match epoch {start_epoch}"
            )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "SEER_LATENT_BRIDGE_TRAINING_RESUME",
                        "checkpoint": str(resume_path),
                        "completed_epochs": start_epoch,
                        "remaining_epochs": epochs - start_epoch,
                    }
                ),
                flush=True,
            )
    if start_epoch > epochs:
        raise RuntimeError(f"resume epoch {start_epoch} exceeds requested epochs {epochs}")
    use_bf16 = args.precision == "bf16" and device.type == "cuda"
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_totals = torch.zeros(4, device=device, dtype=torch.float64)
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= usable_batches:
                break
            sync_now = (batch_index + 1) % args.gradient_accumulation_steps == 0
            sync_context = (
                contextlib.nullcontext()
                if sync_now or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync_context:
                values = [
                    batch[key].to(device=device, non_blocking=True)
                    for key in (
                        "previous_condition",
                        "stable_context",
                        "current_state",
                        "previous_executed_action",
                    )
                ]
                target = batch["target_condition"].to(device=device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    predicted = model(*values)
                    predicted_next = values[0] + predicted
                    loss, parts = bridge_distillation_loss(predicted_next.float(), target.float())
                    scaled = loss / args.gradient_accumulation_steps
                scaled.backward()
            batch_count = target.shape[0]
            train_totals += torch.tensor(
                [loss.item() * batch_count, parts["mse"].item() * batch_count,
                 parts["cosine_loss"].item() * batch_count, batch_count],
                device=device,
                dtype=torch.float64,
            )
            if sync_now:
                torch.nn.utils.clip_grad_norm_(model.parameters(), contract.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        train_totals = _reduce_metrics(train_totals)
        train_count = max(float(train_totals[3].item()), 1.0)
        validation_metrics = _validate(model, val_loader, device, use_bf16)
        row = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_total": float(train_totals[0].item() / train_count),
            "train_mse": float(train_totals[1].item() / train_count),
            "train_cosine_loss": float(train_totals[2].item() / train_count),
            "validation": validation_metrics,
        }
        history.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
            raw = model.module if hasattr(model, "module") else model
            if validation_metrics["total"] < best_loss:
                best_loss = validation_metrics["total"]
                best_epoch = epoch + 1
                save_bridge_checkpoint(
                    output / "best.pt",
                    raw,
                    stage=args.stage,
                    epoch=best_epoch,
                    metadata={
                        **base_metadata,
                        "selection_metric": "validation_total_loss",
                        "selection_metric_value": best_loss,
                    },
                )
            completed_epoch = epoch + 1
            if (
                completed_epoch % args.checkpoint_every_epochs == 0
                or completed_epoch == epochs
            ):
                save_bridge_checkpoint(
                    output / "resume.pt",
                    raw,
                    stage=args.stage,
                    epoch=completed_epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata={
                        **base_metadata,
                        "resume_state": {
                            "best_validation_total_loss": best_loss,
                            "best_epoch": best_epoch,
                            "history": history,
                        },
                    },
                )
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        raw = model.module if hasattr(model, "module") else model
        metadata = {
            **base_metadata,
            "resume_state": {
                "best_validation_total_loss": best_loss,
                "best_epoch": best_epoch,
                "history": history,
            },
        }
        save_bridge_checkpoint(
            output / "last.pt", raw, stage=args.stage, epoch=epochs,
            optimizer=optimizer, scheduler=scheduler, metadata=metadata,
        )
        if not (output / "best.pt").is_file() or best_epoch <= 0:
            raise RuntimeError("training completed without a selected validation checkpoint")
        summary = {
            "status": "SEER_LATENT_BRIDGE_TRAINING_COMPLETE",
            "stage": args.stage,
            "elapsed_seconds": time.time() - started,
            "resumed_from_epoch": start_epoch,
            "epochs": epochs,
            "best_validation_total_loss": best_loss,
            "best_epoch": best_epoch,
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
