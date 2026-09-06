"""Joint real-robot fine-tuning using the original SimVLA flow-matching loss.

No VLM-condition cache is used. Validation spans every held-out episode, including
its final valid window. Offline action errors are not robot success rates.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

from .dataset import RealSimVLADataset
from .distributed import initialize_distributed, seed_process
from .io_utils import atomic_write_json, sha256_file
from .model_io import (
    _atomic_torch_save, apply_real_action_checkpoint, load_exact_official_model,
    official_base_identity, save_real_joint_checkpoint,
)


def schedule(step, *, warmup, total, peak, mode):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if mode == "constant":
        return peak
    progress = (step - warmup) / max(1, total - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def enable_checkpointing(model):
    # Keep parameter names/state_dict unchanged, including the saved decoder.
    model.vlm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.vlm.config.use_cache = False
    model.vlm.model.text_model.config.use_cache = False
    for block in model.transformer.blocks:
        original = block.forward

        def forward(*args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        block.forward = forward


def validation_indices(samples, stride):
    if stride < 1:
        raise ValueError("validation_stride must be positive")
    episodes = defaultdict(list)
    for index, (episode, _) in enumerate(samples):
        episodes[episode].append(index)
    # Full temporal coverage, not the beginning of the first demonstration.
    return sorted({i for group in episodes.values() for i in group[::stride] + group[-1:]})


def inputs_for(batch, processor, device):
    return {
        "input_ids": processor.encode_language(batch["language_instruction"])["input_ids"].to(device),
        **{key: batch[key].to(device, non_blocking=True)
           for key in ("image_input", "image_mask", "proprio", "action")},
    }


def sample_noise(episodes, frames, seed, device, horizon=10):
    noises, times = [], []
    for episode, frame in zip(episodes, frames):
        digest = hashlib.sha256(f"{seed}:{episode}:{int(frame)}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
        noises.append(torch.randn(horizon, 7, generator=generator))
        # Inverse CDF of Beta(1.5, 1), identical distribution to upstream.
        times.append(torch.rand((), generator=generator).pow(1 / 1.5) * 0.999 + 0.001)
    return torch.stack(noises).to(device), torch.stack(times).to(device)


def metric_rows(prediction, target, flow_mse, batch):
    error = (prediction.float() - target.float()).abs()
    values = {
        "flow_mse": flow_mse,
        "first5_action_l1": error[:, :5].mean((1, 2)),
        "full10_action_l1": error.mean((1, 2)),
        "translation_mae_mm": error[:, :5, :3].mean((1, 2)) * 20.0,
        "rotation_euler_mae_deg": error[:, :5, 3:6].mean((1, 2)) * 0.05 * 180 / math.pi,
        "continuous_gripper_mae": error[:, :5, 6].mean(1),
        "gripper_sign_agreement": ((prediction[:, :5, 6] >= 0) == (target[:, :5, 6] >= 0)).float().mean(1),
    }
    cpu = {key: value.detach().cpu().tolist() for key, value in values.items()}
    return [{"episode_id": episode, "frame_index": int(batch["frame_index"][i]),
             **{key: value[i] for key, value in cpu.items()}}
            for i, episode in enumerate(batch["episode_id"])]


def summarize_rows(rows):
    if not rows:
        raise ValueError("empty validation set")
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode_id"]].append(row)
    keys = [key for key in rows[0] if key not in ("episode_id", "frame_index")]
    episodes = {}
    for episode, items in groups.items():
        episodes[episode] = {
            "windows": len(items),
            "first_frame": min(x["frame_index"] for x in items),
            "last_frame": max(x["frame_index"] for x in items),
            **{key: sum(x[key] for x in items) / len(items) for key in keys},
        }
    return {
        "windows": len(rows), "episodes": episodes,
        "macro": {key: sum(x[key] for x in episodes.values()) / len(episodes) for key in keys},
        "selection_metric": "episode_macro_first5_action_l1",
        "selection_score": sum(x["first5_action_l1"] for x in episodes.values()) / len(episodes),
        "robot_success_measured": False,
        "evaluation": "held-out observations; fixed per-window noise; fresh H10 Euler10; first R5 action error",
    }


@torch.no_grad()
def validate(model, processor, loader, device, output, step, seed):
    was_training = model.training
    model.eval()
    rows = []
    try:
        for batch in tqdm(loader, desc=f"held-out action evaluation step={step}", leave=False):
            inputs = inputs_for(batch, processor, device)
            target = inputs.pop("action")
            noise, tau = sample_noise(batch["episode_id"], batch["frame_index"], seed, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                condition = model.forward_vlm_efficient(
                    inputs["image_input"], inputs["image_mask"], inputs["input_ids"]
                )["vlm_features"]
                proprio = model.action_space.normalize_state(inputs["proprio"])
                norm_target = model.action_space.normalize_action(target)
                predicted_velocity = model.transformer(
                    vlm_features=condition, proprio=proprio, t=tau,
                    action_with_noise=tau[:, None, None] * noise + (1 - tau[:, None, None]) * norm_target,
                )
                flow = (predicted_velocity.float() - (noise - norm_target)).square().mean((1, 2))
                action = noise.clone()
                for index in range(10):
                    velocity = model.transformer(
                        vlm_features=condition, proprio=proprio,
                        t=torch.full((len(action),), 1 - index / 10, device=device),
                        action_with_noise=action,
                    )
                    action = action - velocity * 0.1
                prediction = model.action_space.postprocess(action)
            rows.extend(metric_rows(prediction, target, flow, batch))
    finally:
        model.train(was_training)
    summary = summarize_rows(rows)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"actions_step_{step:06d}.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(output / f"summary_step_{step:06d}.json", summary)
    return summary


def rng_state():
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state((state["numpy"][0], np.array(state["numpy"][1], dtype=np.uint32),
                         *state["numpy"][2:]))


def gradient_audit(model):
    result = {}
    for name, module in (("vlm", model.vlm), ("action_transformer", model.transformer)):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        result[name] = {
            "parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "gradient_tensors": len(gradients),
            "finite": bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients),
            "nonzero": any(bool(torch.count_nonzero(g)) for g in gradients),
        }
    if not all(x["finite"] and x["nonzero"] for x in result.values()):
        raise RuntimeError(f"joint gradient path is invalid: {result}")
    return result


def check_args(args):
    for name in ("max_steps", "local_batch_size", "accumulation", "validation_interval", "validation_stride"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.warmup_steps < 0 or args.num_workers < 0:
        raise ValueError("warmup_steps and num_workers must be nonnegative")
    if any(not math.isfinite(x) or x <= 0 for x in (args.learning_rate, args.vlm_lr_scale)):
        raise ValueError("learning rates must be finite and positive")
    if socket.gethostname().split(".")[0] in ("jbrserver1", "sd1") and args.device == "cuda":
        ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if not ids or any(x not in ("4", "5", "6", "7") for x in ids):
            raise ValueError("sd1 permits physical GPU4,5,6,7 only; set CUDA_VISIBLE_DEVICES")


def run(args):
    check_args(args)
    if args.preflight:
        from .artifact_validation import validate_real_dataset_manifest
        manifest = validate_real_dataset_manifest(Path(args.dataset) / "manifest.json", verify_episode_checksums=True)
        base = official_base_identity(args.checkpoint, args.processor)
        dataset = RealSimVLADataset(Path(args.dataset) / "manifest.json", split="validation", training=False)
        selected = validation_indices(dataset.samples, args.validation_stride)
        return {"verdict": "JOINT_TRAIN_PREFLIGHT_PASS", "dataset": manifest["dataset_identity_sha256"],
                "official_base": base.to_dict(), "validation_windows": len(selected),
                "validation_episodes": len({dataset.samples[i][0] for i in selected}),
                "robot_initialized": False, "gpu_initialized": False}
    context = initialize_distributed(args.device)
    run_wandb = None
    try:
        seed_process(args.seed, context.rank)
        output = Path(args.output).expanduser().resolve()
        if context.primary:
            output.mkdir(parents=True, exist_ok=True)
            if (output / "training_config.json").exists() and not args.resume:
                raise FileExistsError(f"existing run: use --resume or another --output: {output}")
        context.barrier()
        from .artifact_validation import validate_real_dataset_manifest
        manifest_path = Path(args.dataset).resolve() / "manifest.json"
        dataset_manifest = validate_real_dataset_manifest(manifest_path, verify_episode_checksums=True)
        norm = manifest_path.parent / dataset_manifest["norm_stats"]["path"]
        model, processor, loading = load_exact_official_model(
            model_directory=args.checkpoint, processor_directory=args.processor,
            norm_stats=norm, device=context.device, freeze_vlm=False,
            freeze_action_transformer=False,
        )
        if args.gradient_checkpointing:
            enable_checkpointing(model)
        train_data = RealSimVLADataset(manifest_path, split="train", training=False)
        val_data = RealSimVLADataset(manifest_path, split="validation", training=False)
        indices = validation_indices(val_data.samples, args.validation_stride)
        sampler = DistributedSampler(train_data, num_replicas=context.world_size, rank=context.rank,
                                     seed=args.seed, drop_last=True)
        loader = DataLoader(train_data, batch_size=args.local_batch_size, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=context.device.type == "cuda", drop_last=True)
        val_loader = DataLoader(Subset(val_data, indices), batch_size=args.local_batch_size,
                                num_workers=args.num_workers, shuffle=False)
        if not len(loader):
            raise ValueError("empty training loader")
        base = official_base_identity(args.checkpoint, args.processor)
        config = {
            **vars(args), "protocol": "official_full_checkpoint_joint_vlm_action_finetune",
            "world_size": context.world_size,
            "effective_batch_size": args.local_batch_size * args.accumulation * context.world_size,
            "train_windows": len(train_data), "heldout_windows": len(val_data),
            "evaluated_heldout_windows": len(indices),
            "dataset_identity_sha256": dataset_manifest["dataset_identity_sha256"],
            "norm_stats_sha256": sha256_file(norm), "official_base": base.to_dict(),
            "augmentation": "none; identical image processing at training and deployment",
            "action_transformer_reinitialized": False,
            "task_success_claim": False,
            "torch_version": str(torch.__version__), "hostname": socket.gethostname(),
            "source_sha256": sha256_file(__file__),
        }
        if context.primary:
            repo = Path(__file__).resolve().parents[4]
            for name, command in (("git_head.txt", ["rev-parse", "HEAD"]),
                                  ("git_status.txt", ["status", "--short"]),
                                  ("source_changes.patch", ["diff"])):
                result = subprocess.run(["git", "-C", str(repo), *command],
                                        capture_output=True, text=True, check=True)
                (output / name).write_text(result.stdout)
        ddp = model
        if context.world_size > 1:
            ddp = DistributedDataParallel(model, device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                                          find_unused_parameters=True, broadcast_buffers=False)
        optimizer = torch.optim.AdamW([
            {"params": list(model.vlm.parameters()), "name": "vlm", "scale": args.vlm_lr_scale},
            {"params": list(model.transformer.parameters()), "name": "action", "scale": 1.0},
        ], lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
        step, epoch, consumed = 0, 0, 0
        best, best_path, resume_state = float("inf"), None, None
        if args.resume:
            resume_state = torch.load(output / "resume.pt", map_location="cpu", weights_only=True)
            previous = resume_state["training_config"]
            for key in ("dataset_identity_sha256", "norm_stats_sha256", "effective_batch_size",
                        "world_size", "local_batch_size", "accumulation", "seed", "schedule",
                        "learning_rate", "vlm_lr_scale", "warmup_steps", "max_steps", "source_sha256"):
                if config[key] != previous[key]:
                    raise ValueError(f"resume configuration changed: {key}")
            apply_real_action_checkpoint(model, resume_state["checkpoint"],
                                         expected_base_sha256=base.model_weights_sha256,
                                         expected_norm_sha256=sha256_file(norm),
                                         expected_dataset_identity_sha256=config["dataset_identity_sha256"])
            optimizer.load_state_dict(resume_state["optimizer"])
            step, epoch, consumed = (resume_state[x] for x in ("step", "epoch", "consumed"))
            best, best_path = resume_state["best"], resume_state["best_path"]
        if context.primary:
            atomic_write_json(output / "training_config.json", config)
            atomic_write_json(output / "exact_initialization.json", loading)
            coverage = [{"episode_id": val_data.samples[i][0], "frame_index": val_data.samples[i][1]} for i in indices]
            atomic_write_json(output / "validation_coverage.json", coverage)
            if args.wandb_project:
                import wandb
                run_wandb = wandb.init(project=args.wandb_project, name=output.name,
                                       dir=str(output), config=config,
                                       id=resume_state.get("wandb_id") if resume_state else None,
                                       resume="allow" if resume_state else None)
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(consumed):
            next(iterator)
        if resume_state:
            restore_rng(resume_state["rank_rng"][context.rank])
        else:
            context.barrier()
            if context.primary:
                validate(model, processor, val_loader, context.device, output / "validation", 0, args.seed)
            context.barrier()
        optimizer.zero_grad(set_to_none=True)
        start_step, started = step, time.perf_counter()
        progress = tqdm(total=args.max_steps, initial=step, disable=not context.primary, desc="Doll joint VLM + action")
        while step < args.max_steps:
            loss_sum = 0.0
            for micro in range(args.accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    consumed = 0
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                consumed += 1
                inputs = inputs_for(batch, processor, context.device)
                sync = contextlib.nullcontext() if context.world_size == 1 or micro == args.accumulation - 1 else ddp.no_sync()
                with sync:
                    with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16, enabled=context.device.type == "cuda"):
                        loss = ddp(**inputs)["velocity_loss"]
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"nonfinite flow loss at optimizer step {step}")
                    (loss / args.accumulation).backward()
                loss_sum += float(loss.detach())
            if step == start_step:
                gradients = gradient_audit(model)
                if context.primary:
                    atomic_write_json(output / "joint_gradient_audit.json", gradients)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            lr = schedule(step, warmup=args.warmup_steps, total=args.max_steps, peak=args.learning_rate, mode=args.schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr * group["scale"]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = {"step": step, "train_flow_mse": loss_sum / args.accumulation,
                       "action_lr": lr, "vlm_lr": lr * args.vlm_lr_scale,
                       "grad_norm": float(grad_norm), "elapsed_s": time.perf_counter() - started}
            progress.update(1)
            progress.set_postfix(loss=f'{metrics["train_flow_mse"]:.5f}', lr=f"{lr:.1e}")
            if context.primary and (step == 1 or step % 20 == 0):
                with (output / "train_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                if run_wandb:
                    run_wandb.log(metrics, step=step)
            if step % args.validation_interval and step != args.max_steps:
                continue
            context.barrier()
            local_rng = rng_state()
            rank_rng = [None] * context.world_size
            if context.world_size > 1:
                torch.distributed.all_gather_object(rank_rng, local_rng)
            else:
                rank_rng = [local_rng]
            if context.primary:
                validation = validate(model, processor, val_loader, context.device, output / "validation", step, args.seed)
                path = output / "checkpoints" / f"joint_step_{step:06d}.pt"
                save_real_joint_checkpoint(path, model=model, official_base=base, norm_stats_path=norm,
                                           dataset_identity_sha256=config["dataset_identity_sha256"],
                                           optimizer_step=step, training_config=config, validation=validation)
                if validation["selection_score"] < best:
                    best, best_path = validation["selection_score"], str(path)
                atomic_write_json(output / "selection.json", {
                    "best_checkpoint": best_path, "latest_checkpoint": str(path),
                    "best_score": best, "metric": validation["selection_metric"],
                    "robot_success_measured": False,
                })
                _atomic_torch_save({
                    "checkpoint": str(path), "optimizer": optimizer.state_dict(),
                    "training_config": config, "step": step, "epoch": epoch, "consumed": consumed,
                    "rank_rng": rank_rng, "best": best, "best_path": best_path,
                    "wandb_id": run_wandb.id if run_wandb else None,
                }, output / "resume.pt")
                for old in (output / "checkpoints").glob("joint_step_*.pt"):
                    if str(old) not in {str(path), best_path}:
                        old.unlink()
                (output / "latest_checkpoint.txt").write_text(str(path) + "\n")
                (output / "best_checkpoint.txt").write_text(best_path + "\n")
                if run_wandb:
                    run_wandb.log({f"validation/{k}": v for k, v in validation["macro"].items()}, step=step)
                progress.write(f"Saved step={step}; held-out first5 L1={validation['selection_score']:.6f}; best={best_path}")
            context.barrier()
            restore_rng(local_rng)
        progress.close()
        result = {"verdict": "REAL_JOINT_FINETUNE_COMPLETE", "optimizer_steps": step,
                  "elapsed_s": time.perf_counter() - started, "robot_success_measured": False}
        if context.primary:
            atomic_write_json(output / "run_summary.json", result)
        return result
    finally:
        if run_wandb:
            run_wandb.finish()
        context.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "processor", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--local-batch-size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--vlm-lr-scale", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--validation-interval", type=int, default=500)
    p.add_argument("--validation-stride", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight", action="store_true")
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
