#!/usr/bin/env python3
"""Cacheless local-oracle generation training, with a frozen pi0.5/Condition model."""

import argparse
import math
from pathlib import Path
import time

from dual_loop_runtime import (ROOT, TrainingPairs, atomic_json, load_components, read_json,
                               training_prefix)
import torch
from tqdm import tqdm

from architectures.openpi.adapters.latentloop.dual_loop import DualLoopPolicy, condition_summary, generate, sync
from architectures.openpi.adapters.latentloop.policy_io import explicit_policy_noise
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook
from methods.latentloop.modules.flow_hidden_update import FlowHiddenConfig, FlowHiddenUpdater


def learning_rate(step, total, peak=1e-4):
    warmup = min(200, max(1, total // 20))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup - 1))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def save(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def run(args):
    config = read_json(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base, model, condition = load_components(config)
    pairs = TrainingPairs(config, base)
    observations, _ = pairs.window(0)
    prefix, state = training_prefix(model, None, observations, None, False)
    module = FlowHiddenUpdater(FlowHiddenConfig(
        hidden_dim=model.action_out_proj.in_features, action_dim=model.config.action_dim,
        condition_dim=condition_summary(prefix).shape[-1], state_dim=state.shape[-1])).cuda()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, weight_decay=0.0)
    start_step, best = 0, float("inf")
    latest = output / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved.get("format") != "pi05_simvla_core_generation_v1" or saved["config_id"] != config["config_id"] or saved["total_steps"] != args.steps:
            raise ValueError("resume configuration differs; choose a new output directory")
        module.load_state_dict(saved["updater"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step, best = saved["step"], saved["best_loss"]
    hook = PrefixKVHook(model)
    noise = explicit_policy_noise((1, 10, model.config.action_dim), seed=7, device="cuda")
    with torch.no_grad():
        exact = getattr(model.sample_actions, "_torchdynamo_orig_callable", model.sample_actions)(
            "cuda", observations[1], noise=noise.clone(), num_steps=10)
        rebuilt = generate(model, hook, prefix, state, noise.clone(), None, n_g=10).actions
    max_diff = float((exact - rebuilt).abs().max())
    atomic_json(output / "parity.json", {"max_action_diff": max_diff, "pass": max_diff == 0.0})
    if max_diff != 0.0:
        raise RuntimeError(f"exact original/hook action parity failed: {max_diff}")
    del prefix, state, observations, exact, rebuilt
    trainable = sum(p.numel() for p in module.parameters())
    atomic_json(output / "freeze.json", {
        "baseline_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "condition_trainable": 0,
        "generation_trainable": trainable, "updater_config": module.descriptor(),
        "baseline_total_parameters": sum(p.numel() for p in model.parameters()),
        "condition_used_in_generation_training": False,
        "optimizer_param_names": [n for n, p in module.named_parameters() if p.requires_grad],
        "training_demonstrations": len(pairs.roles["train"]),
        "heldout_demonstrations": len(pairs.roles["checkpoint_validation"])})
    wandb_run = None
    if not args.smoke and config.get("wandb_mode") != "disabled":
        import wandb
        wandb_run = wandb.init(project="gnaroshi-pi05-dual-loop", name=config["run_name"],
                               id=config["config_id"][:14] + "g", resume="allow", dir=str(output),
                               mode=config["wandb_mode"], config=config)
    timer = time.perf_counter()
    progress = tqdm(range(start_step, args.steps), initial=start_step, total=args.steps,
                    mininterval=2.0, desc="pi05 Generation")
    last_step = start_step
    try:
        for step in progress:
            observations, identity = pairs.window(step)
            prefix, state = training_prefix(model, None, observations, None, False)
            noise = explicit_policy_noise((1, 10, model.config.action_dim),
                                          seed=config["train_seed"] * 1000000 + step, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, args.steps)
            result = generate(model, hook, prefix, state, noise, module, n_g=2, train=True)
            loss = result.loss
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("invalid generation distillation loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("frozen model received gradients")
            last_step = step + 1
            metrics = {"step": last_step, "loss": float(loss.detach()), "gradient_norm": float(norm),
                       "lr": optimizer.param_groups[0]["lr"], **result.metrics,
                       "approximate_condition": False, **identity}
            progress.set_postfix(loss=f"{metrics['loss']:.4g}", lr=f"{metrics['lr']:.2g}")
            if step % 50 == 0 or last_step == args.steps:
                with (output / "train_metrics.jsonl").open("a") as handle:
                    import json
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                if wandb_run:
                    wandb_run.log(metrics, step=last_step)
            del result, loss, prefix, state, observations
            if last_step % 1000 == 0 or last_step == args.steps:
                module.eval()
                validation = []
                for index in range(1 if args.smoke else pairs.validation_count):
                    observations, identity = pairs.window(index, "checkpoint_validation")
                    with torch.no_grad():
                        prefix, state = training_prefix(model, None, observations, None, False)
                        val_noise = explicit_policy_noise((1, 10, model.config.action_dim), seed=10000 + index, device="cuda")
                        val = generate(model, hook, prefix, state, val_noise, module, n_g=2, train=True)
                        validation.append({"loss": float(val.loss), **identity, "approximate_condition": False})
                    del val, prefix, state, observations
                val_loss = sum(r["loss"] for r in validation) / len(validation)
                improved = val_loss < best
                best = min(best, val_loss)
                payload = {"format": "pi05_simvla_core_generation_v1",
                           "updater": module.state_dict(), "updater_config": module.descriptor(),
                           "optimizer": optimizer.state_dict(), "step": last_step, "total_steps": args.steps,
                           "config_id": config["config_id"], "best_loss": best}
                save(latest, payload)
                if improved:
                    save(output / "best_diagnostic.pt", payload)
                if last_step == args.steps:
                    save(output / "final.pt", payload)
                atomic_json(output / "validation_latest.json", {"step": last_step, "mean_loss": val_loss, "queries": validation})
                tqdm.write(f"CHECKPOINT step={last_step} val={val_loss:.6g} best={best:.6g} path={latest}")
                module.train()
    finally:
        if wandb_run:
            wandb_run.finish()
    elapsed = time.perf_counter() - timer
    atomic_json(output / "summary.json", {"complete": last_step == args.steps, "step": last_step,
                "config_id": config["config_id"], "elapsed_seconds": elapsed,
                "seconds_per_new_step_including_validation": elapsed / max(1, last_step - start_step),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(), "best_validation_loss": best,
                "generation_parameters": trainable, "original_parity_max_diff": max_diff})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
