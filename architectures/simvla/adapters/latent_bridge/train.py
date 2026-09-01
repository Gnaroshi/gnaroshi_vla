"""R0/R1 training for the official-architecture SimVLA feature bridge."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from .checkpoint import load_bridge_checkpoint, save_bridge_checkpoint
from .dataset import (
    SimVLALatentBridgeDaggerDataset,
    SimVLALatentBridgeDataset,
    SimVLALatentBridgeSyncDataset,
    load_sidecar_manifest,
    sha256_file,
    validate_sidecar,
)
from .model import SimVLALatentBridge, SimVLALatentBridgeConfig
from .provenance import (
    latent_bridge_source_manifest,
    simvla_latent_bridge_integration_manifest,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_mean(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1).mean()


def _bridge_tokens(
    model: SimVLALatentBridge,
    condition: torch.Tensor,
) -> torch.Tensor:
    if model.config.token_mode == "image_only":
        return condition[:, : model.config.image_token_count]
    return condition


def _restore_tokens(
    model: SimVLALatentBridge,
    original: torch.Tensor,
    predicted: torch.Tensor,
) -> torch.Tensor:
    if model.config.token_mode != "image_only":
        return predicted
    restored = original.clone()
    restored[:, : model.config.image_token_count] = predicted
    return restored


def _autocast(device: torch.device, precision: str) -> Any:
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("bf16 training is supported only on CUDA in this entry point")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate(
    model: SimVLALatentBridge,
    loader: DataLoader[Any],
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    totals = {"mse": 0.0, "feature_cosine": 0.0, "copy_cosine": 0.0, "delta_cosine": 0.0}
    samples = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            z0_full = batch["condition_t"]
            z1_full = batch["condition_t1"]
            z0 = _bridge_tokens(model, z0_full)
            z1 = _bridge_tokens(model, z1_full)
            stable = _bridge_tokens(model, batch["stable_t"])
            with _autocast(device, precision):
                delta = model(z0, stable, batch["state_t"], batch["previous_action_t"])
            predicted = z0 + delta
            predicted_full = _restore_tokens(model, z0_full, predicted)
            count = z0.shape[0]
            totals["mse"] += F.mse_loss(delta.float(), (z1 - z0).float()).item() * count
            totals["feature_cosine"] += cosine_mean(predicted_full, z1_full).item() * count
            totals["copy_cosine"] += cosine_mean(z0_full, z1_full).item() * count
            totals["delta_cosine"] += cosine_mean(delta, z1 - z0).item() * count
            samples += count
    return {name: value / max(samples, 1) for name, value in totals.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("epochs, batch_size, and gradient_accumulation_steps must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive when provided")
    if args.weights_only_resume and not args.resume:
        raise ValueError("--weights-only-resume requires --resume")
    if args.dagger_root and (not args.resume or not args.weights_only_resume):
        raise ValueError(
            "R1 DAgger training requires --resume <R0 checkpoint> and "
            "--weights-only-resume"
        )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing training output: {output}")
    output.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    if bool(args.sidecar) == bool(args.sync_root):
        raise ValueError("provide exactly one of --sync-root (primary) or --sidecar (bootstrap)")
    if args.sync_root:
        train_dataset = SimVLALatentBridgeSyncDataset(
            args.sync_root,
            split="train",
            heldout_fraction=args.heldout_fraction,
            split_seed=args.split_seed,
        )
        heldout_dataset = SimVLALatentBridgeSyncDataset(
            args.sync_root,
            split="heldout",
            heldout_fraction=args.heldout_fraction,
            split_seed=args.split_seed,
        )
        data_manifest = train_dataset.manifest
        validation = {
            "passed": True,
            "kind": "on_policy_sync_rollouts",
            "manifest": str(Path(args.sync_root).expanduser().resolve() / "manifest.json"),
        }
        training_data_identity = {
            "kind": "on_policy_sync_rollouts",
            "checkpoint": data_manifest["checkpoint"],
            "norm_stats_sha256": data_manifest["norm_stats_sha256"],
            "simvla_upstream_commit": data_manifest["simvla_upstream_commit"],
            "stable_layer_index": data_manifest["stable_layer_index"],
            "flow_steps_used_for_action_condition": data_manifest["flow_steps"],
            "suite": data_manifest["suite"],
            "trial_offset": data_manifest["trial_offset"],
            "trials_per_task": data_manifest["trials_per_task"],
        }
    else:
        validation = validate_sidecar(
            args.sidecar, verify_hashes=not args.skip_sidecar_hashes
        )
        if not validation["passed"]:
            raise RuntimeError(f"sidecar validation failed: {validation['errors']}")
        data_manifest = load_sidecar_manifest(args.sidecar)
        if (
            int(data_manifest["flow_steps"]) != 10
            and not args.allow_nonproduction_flow_steps
        ):
            raise RuntimeError(
                "production bridge training requires action conditioning generated "
                "with flow_steps=10"
            )
        train_dataset = SimVLALatentBridgeDataset(
            args.sidecar,
            split="train",
            heldout_fraction=args.heldout_fraction,
            split_seed=args.split_seed,
        )
        heldout_dataset = SimVLALatentBridgeDataset(
            args.sidecar,
            split="heldout",
            heldout_fraction=args.heldout_fraction,
            split_seed=args.split_seed,
        )
        training_data_identity = {
            "kind": "training_demonstration_bootstrap",
            "checkpoint": data_manifest["checkpoint"],
            "norm_stats_sha256": data_manifest["norm_stats_sha256"],
            "simvla_upstream_commit": data_manifest["simvla_upstream_commit"],
            "source_cache_manifest_sha256": data_manifest[
                "source_cache_manifest_sha256"
            ],
            "stable_layer_index": data_manifest["stable_layer_index"],
            "flow_steps_used_for_action_condition": data_manifest["flow_steps"],
        }
    current_official_source = latent_bridge_source_manifest()
    current_integration_source = simvla_latent_bridge_integration_manifest()
    recorded_official_source = data_manifest.get("latent_bridge_upstream", {})
    recorded_integration_source = data_manifest.get(
        "simvla_latent_bridge_integration", {}
    )
    if recorded_official_source.get("combined_sha256") != current_official_source[
        "combined_sha256"
    ]:
        raise RuntimeError("training data and runtime official Latent Bridge source differ")
    if recorded_integration_source.get("combined_sha256") != current_integration_source[
        "combined_sha256"
    ]:
        raise RuntimeError("training data and runtime SimVLA integration source differ")
    generator = torch.Generator().manual_seed(args.seed)
    training_source: Any = train_dataset
    dagger_dataset = None
    if args.dagger_root:
        dagger_dataset = SimVLALatentBridgeDaggerDataset(args.dagger_root)
        training_source = ConcatDataset((train_dataset, dagger_dataset))
    train_loader = DataLoader(
        training_source,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.resume:
        model, resume_payload = load_bridge_checkpoint(args.resume, device=device)
        config = model.config
    else:
        config = SimVLALatentBridgeConfig(
            sequence_length=72 if args.token_mode == "image_only" else 122,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            low_rank=args.low_rank,
            stable_layer_index=args.stable_layer_index,
            token_mode=args.token_mode,
        )
        model = SimVLALatentBridge(config).to(device)
        resume_payload = None
    if int(data_manifest["stable_layer_index"]) != config.stable_layer_index:
        raise RuntimeError("bridge configuration and training-data stable layer differ")
    if dagger_dataset is not None:
        dagger_manifest = dagger_dataset.manifest
        dagger_checks = {
            "base_checkpoint": dagger_manifest["base_checkpoint"]
            == training_data_identity["checkpoint"],
            "norm_stats": dagger_manifest["norm_stats_sha256"]
            == training_data_identity["norm_stats_sha256"],
            "stable_layer": int(dagger_manifest["stable_layer_index"])
            == config.stable_layer_index,
            "token_mode": dagger_manifest["token_mode"] == config.token_mode,
            "bridge_checkpoint": dagger_manifest["bridge_checkpoint_sha256"]
            == sha256_file(args.resume),
            "official_source": dagger_manifest["latent_bridge_upstream"][
                "combined_sha256"
            ]
            == current_official_source["combined_sha256"],
            "integration_source": dagger_manifest[
                "simvla_latent_bridge_integration"
            ]["combined_sha256"]
            == current_integration_source["combined_sha256"],
            "action_horizon": int(dagger_manifest["action_horizon"]) == 10,
            "execution_horizon": int(dagger_manifest["execution_horizon"]) == 5,
            "flow_steps": int(dagger_manifest["flow_steps"]) == 10,
        }
        failed = [name for name, passed in dagger_checks.items() if not passed]
        if failed:
            raise RuntimeError(f"R0/DAgger source contract mismatch: {failed}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if resume_payload is not None and not args.weights_only_resume:
        state = resume_payload.get("optimizer_state_dict")
        if state is not None:
            optimizer.load_state_dict(state)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
    start_epoch = 0
    global_step = 0
    if resume_payload is not None and not args.weights_only_resume:
        resume_training = resume_payload.get("training", {})
        start_epoch = int(resume_training.get("epoch", 0))
        global_step = int(resume_training.get("step", 0))
    if start_epoch >= args.epochs:
        raise ValueError(
            f"resume epoch {start_epoch} is not below requested epochs {args.epochs}; "
            "use --weights-only-resume for R1"
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - start_epoch, 1),
        eta_min=args.learning_rate * 0.01,
    )
    source = {
        "latent_bridge": current_official_source,
        "integration": current_integration_source,
        "training_data_validation": validation,
        "training_data_identity": training_data_identity,
        "train_split": train_dataset.contract(),
        "heldout_split": heldout_dataset.contract(),
        "dagger_transitions": len(dagger_dataset) if dagger_dataset is not None else 0,
        "dagger_manifest": (
            dagger_dataset.manifest if dagger_dataset is not None else None
        ),
    }
    run_config = vars(args).copy()
    run_config.update(
        {
            "bridge_config": config.serializable(),
            "parameter_audit": model.parameter_audit(),
            "micro_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "official_recipe_relation": (
                "SingleStepDiT architecture and R0/R1 optimizer defaults; "
                "epoch-wise cosine schedule matches the pinned trainer; direct delta is selected "
                "because the pinned official CLI marks --no_flow recommended"
            ),
        }
    )
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    history = output / "metrics.jsonl"
    best_cosine = -math.inf
    if resume_payload is not None and not args.weights_only_resume:
        best_cosine = float(
            resume_payload.get("training", {})
            .get("heldout", {})
            .get("feature_cosine", -math.inf)
        )
    started = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"bridge epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress):
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            z0_full = batch["condition_t"]
            z1_full = batch["condition_t1"]
            z0 = _bridge_tokens(model, z0_full)
            z1 = _bridge_tokens(model, z1_full)
            stable = _bridge_tokens(model, batch["stable_t"])
            target = z1 - z0
            if args.flow_matching:
                interpolation = torch.rand(z0.shape[0], 1, 1, device=device)
                bridge_input = (1 - interpolation) * z0 + interpolation * z1
            else:
                bridge_input = z0
            with _autocast(device, args.precision):
                predicted_delta = model(
                    bridge_input,
                    stable,
                    batch["state_t"],
                    batch["previous_action_t"],
                )
                mse = F.mse_loss(predicted_delta.float(), target.float())
                if args.cosine_weight > 0:
                    predicted_condition = z0 + predicted_delta
                    cosine_loss = 1 - cosine_mean(predicted_condition, z1)
                else:
                    cosine_loss = mse.new_zeros(())
                loss = mse + args.cosine_weight * cosine_loss
            (loss / args.gradient_accumulation_steps).backward()
            update = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if not update:
                continue
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4g}", step=global_step)
            if args.log_interval and global_step % args.log_interval == 0:
                with history.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "epoch": epoch + 1,
                                "loss": loss.item(),
                                "mse": mse.item(),
                                "cosine_loss": cosine_loss.item(),
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "elapsed_seconds": time.time() - started,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        validation_metrics = evaluate(model, heldout_loader, device, args.precision)
        epoch_record = {
            "step": global_step,
            "epoch": epoch + 1,
            "heldout": validation_metrics,
            "elapsed_seconds": time.time() - started,
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")
        scheduler.step()
        checkpoint_training = {**run_config, **epoch_record}
        save_bridge_checkpoint(
            output / "last.pt",
            model,
            provenance=source,
            training=checkpoint_training,
            optimizer=optimizer,
        )
        if validation_metrics["feature_cosine"] > best_cosine:
            best_cosine = validation_metrics["feature_cosine"]
            save_bridge_checkpoint(
                output / "best.pt",
                model,
                provenance=source,
                training=checkpoint_training,
            )
        if stop:
            break
    summary = {
        "verdict": "SIMVLA_LATENT_BRIDGE_TRAINING_COMPLETE",
        "steps": global_step,
        "epochs_completed": epoch + 1,
        "best_feature_cosine": best_cosine,
        "copy_baseline_cosine": validation_metrics["copy_cosine"],
        "improvement_over_copy": best_cosine - validation_metrics["copy_cosine"],
        "best_checkpoint": str(output / "best.pt"),
        "last_checkpoint": str(output / "last.pt"),
        "parameter_audit": model.parameter_audit(),
        "flow_matching": bool(args.flow_matching),
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--sync-root")
    value.add_argument("--sidecar")
    value.add_argument("--output", required=True)
    value.add_argument("--resume")
    value.add_argument("--weights-only-resume", action="store_true")
    value.add_argument("--dagger-root")
    value.add_argument("--epochs", type=int, default=100)
    value.add_argument("--max-steps", type=int)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--gradient-accumulation-steps", type=int, default=1)
    value.add_argument("--learning-rate", type=float, default=3e-4)
    value.add_argument("--weight-decay", type=float, default=0.01)
    value.add_argument("--num-workers", type=int, default=4)
    value.add_argument("--hidden-dim", type=int, default=768)
    value.add_argument("--num-heads", type=int, default=12)
    value.add_argument("--num-blocks", type=int, default=12)
    value.add_argument("--low-rank", type=int, default=0)
    value.add_argument("--stable-layer-index", type=int, default=10)
    value.add_argument("--token-mode", choices=("all", "image_only"), default="image_only")
    value.add_argument("--flow-matching", action="store_true")
    value.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    value.add_argument("--cosine-weight", type=float, default=0.0)
    value.add_argument("--max-grad-norm", type=float, default=1.0)
    value.add_argument("--heldout-fraction", type=float, default=0.1)
    value.add_argument("--split-seed", type=int, default=42)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--log-interval", type=int, default=20)
    value.add_argument("--skip-sidecar-hashes", action="store_true")
    value.add_argument("--allow-nonproduction-flow-steps", action="store_true")
    value.add_argument("--device", default="cuda")
    return value


def main() -> None:
    result = train(parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
