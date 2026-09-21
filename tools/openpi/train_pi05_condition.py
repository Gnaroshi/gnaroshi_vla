#!/usr/bin/env python3
"""Cacheless training of the SimVLA Condition core at the pi0.5 KV interface."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from dual_loop_runtime import TrainingPairs, atomic_json, load_components, read_json
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from architectures.openpi.adapters.latentloop.aligned_condition import (
    FORMAT, AlignedConditionUpdater, ConditionConfig, pack_kv,
)
from architectures.openpi.adapters.latentloop.policy_io import explicit_policy_noise, postprocess_policy_actions
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook
from train_pi05_generation import learning_rate, save


def action_postprocessor(config):
    from openpi.training import config as config_api
    c = config_api.get_config("pi05_libero_lora_pytorch")
    if c.data.extra_delta_transform:
        raise ValueError("expected the reproduced Cartesian-delta action contract")
    quantiles = c.data.create(c.assets_dirs, c.model).use_quantile_norm
    path = Path(config["checkpoint"]) / "assets/physical-intelligence/libero/norm_stats.json"
    stats = read_json(path)["norm_stats"]["actions"]
    def tensor(value, ref):
        return torch.as_tensor(value, device=ref.device, dtype=ref.dtype)[:7]
    def postprocess(actions):
        x = actions[..., :7]
        if quantiles:
            low, high = tensor(stats["q01"], x), tensor(stats["q99"], x)
            return (x + 1) / 2 * (high - low + 1e-6) + low
        return x * (tensor(stats["std"], x) + 1e-6) + tensor(stats["mean"], x)
    return postprocess


def objective(model, module, observations, config, index, postprocess, *, train):
    hook = PrefixKVHook(model)
    previous = hook.extract(observations[0]).state.detach()
    losses, metrics = [], {}
    first_action_loss = None
    first_prefix = None
    for age in (1, 2, 3):
        obs = observations[age]
        target = hook.extract(obs).state.detach()
        predicted, update = module(previous, observations[age - 1], obs, age=age)
        if not torch.equal(predicted.pad_mask, target.pad_mask):
            raise ValueError("teacher/student prefix layouts differ")
        pred_kv, target_kv = pack_kv(predicted).float(), pack_kv(target).float()
        squared = (F.layer_norm(pred_kv, (pred_kv.shape[-1],)) -
                   F.layer_norm(target_kv, (target_kv.shape[-1],))).square()
        mask = target.pad_mask[:, None, None, None, :, None].expand_as(squared)
        condition_loss = squared.masked_select(mask).mean()
        noise = explicit_policy_noise((1, 10, model.config.action_dim),
                                      seed=config["train_seed"] * 1000000 + index * 3 + age,
                                      device=obs.state.device)
        with torch.no_grad():
            teacher, _ = hook.sample_actions_from_state(target, obs.state, noise, num_steps=10)
            teacher = postprocess(teacher)
        # Rematerialize expert activations during backward; keep the full recursive KV graph.
        def sample(*kv, template=predicted, state=obs.state, initial_noise=noise):
            layers = template.num_layers
            prefix = replace(template, pre_rope_keys=tuple(kv[:layers]), values=tuple(kv[layers:]))
            return hook.sample_actions_from_state(prefix, state, initial_noise, num_steps=10)[0]
        kv = (*predicted.pre_rope_keys, *predicted.values)
        student = postprocess(checkpoint(sample, *kv, use_reentrant=False) if train else sample(*kv))
        terms = {
            "condition": condition_loss,
            "first5_action": F.l1_loss(student[:, :5], teacher[:, :5]),
            "full_chunk_action": F.l1_loss(student, teacher),
            "continuous_gripper": F.l1_loss(student[:, :5, 6], teacher[:, :5, 6]),
            "update_regularization": update.residual.square().mean(),
        }
        losses.append(sum(config["condition_weights"][k] * v for k, v in terms.items()))
        metrics.update({f"age{age}/{k}": float(v.detach()) for k, v in terms.items()})
        if age == 1:
            first_action_loss, first_prefix = terms["first5_action"], predicted
        previous = predicted
    total = torch.stack(losses).mean()
    metrics["loss"] = float(total.detach())
    metrics["first5_action"] = sum(metrics[f"age{age}/first5_action"] for age in (1, 2, 3)) / 3
    return total, metrics, first_action_loss, first_prefix


def run(args):
    config = read_json(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base, model, _ = load_components(config)
    pairs = TrainingPairs(config, base)
    observations, _ = pairs.window(0)
    hook = PrefixKVHook(model)
    prefix = hook.extract(observations[0])
    module = AlignedConditionUpdater(ConditionConfig.from_prefix(prefix.state, observations[0])).cuda()
    postprocess = action_postprocessor(config)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, weight_decay=0.0)
    latest = output / "latest.pt"
    start_step, best = 0, float("inf")
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT or payload["config_id"] != config["config_id"] or payload["total_steps"] != args.steps:
            raise ValueError("Condition resume identity mismatch")
        module.load_state_dict(payload["updater"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step, best = payload["step"], payload["best_loss"]
    with torch.no_grad():
        noise = explicit_policy_noise((1, 10, model.config.action_dim), seed=7, device="cuda")
        sampler = getattr(model.sample_actions, "_torchdynamo_orig_callable", model.sample_actions)
        native = sampler("cuda", observations[0], noise=noise.clone(), num_steps=10)
        rebuilt, _ = hook.sample_actions_from_state(prefix.state, observations[0].state, noise.clone())
        parity_diff = float((native - rebuilt).abs().max())
        ref_actions = postprocess_policy_actions(base, observations[0].state, native)["actions"]
        transform_diff = float((postprocess(native)[0].cpu() - torch.as_tensor(ref_actions)).abs().max())
    parity = {"original_vs_rebuilt_maxdiff": parity_diff, "torch_vs_original_postprocess_maxdiff": transform_diff,
              "cache": hook.cache_allclose(prefix.state, prefix.source_cache)}
    atomic_json(output / "parity.json", parity)
    if parity_diff != 0 or transform_diff > 1e-6 or not parity["cache"]["passed"]:
        raise RuntimeError("frozen policy/cache/action postprocessing parity failed")
    del observations, prefix, native, rebuilt, noise
    atomic_json(output / "freeze.json", {
        "baseline_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "condition_parameters": sum(p.numel() for p in module.parameters()),
        "updater_config": module.descriptor(),
        "optimizer_param_names": [n for n, p in module.named_parameters() if p.requires_grad],
        "training_episodes": len(pairs.roles["train"]), "heldout_episodes": len(pairs.roles["checkpoint_validation"]),
        "validation_windows": pairs.validation_count, "validation_stages": ["early", "middle", "late"],
        "selection": "fixed_final_step; best_validation_is_diagnostic_only"})
    wandb_run = None
    if not args.smoke and config.get("wandb_mode") != "disabled":
        import wandb
        wandb_run = wandb.init(project="gnaroshi-pi05-dual-loop", name=config["run_name"] + "_condition",
                              id=config["config_id"][:14] + "c", resume="allow", dir=str(output),
                              mode=config["wandb_mode"], config=config)
    timer = time.perf_counter()
    last_step = start_step
    progress = tqdm(range(start_step, args.steps), initial=start_step, total=args.steps,
                    mininterval=2.0, desc="pi05 Condition")
    try:
        for step in progress:
            observations, identity = pairs.window(step)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, args.steps)
            loss, metrics, action_loss, first_prefix = objective(
                model, module, observations, config, step, postprocess, train=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Condition loss is not finite")
            if args.smoke and step == start_step:
                g = torch.autograd.grad(action_loss, first_prefix.values[0], retain_graph=True)[0]
                if not torch.isfinite(g).all() or not bool(g.abs().sum() > 0):
                    raise RuntimeError("action loss does not reach predicted condition")
                atomic_json(output / "action_gradient.json", {"pass": True, "absolute_sum": float(g.abs().sum())})
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("frozen baseline received gradients")
            last_step = step + 1
            metrics.update(step=last_step, gradient_norm=float(norm), lr=optimizer.param_groups[0]["lr"], **identity)
            progress.set_postfix(loss=f"{metrics['loss']:.4g}", action=f"{metrics['first5_action']:.4g}")
            if step % 50 == 0 or last_step == args.steps:
                with (output / "train_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                if wandb_run:
                    wandb_run.log(metrics, step=last_step)
            del loss, action_loss, first_prefix, observations
            if last_step % 1000 == 0 or last_step == args.steps:
                module.eval()
                rows = []
                with torch.no_grad():
                    for index in range(1 if args.smoke else pairs.validation_count):
                        observations, identity = pairs.window(index, "checkpoint_validation")
                        _, m, _, _ = objective(model, module, observations, config, 100000 + index, postprocess, train=False)
                        rows.append({**identity, **m})
                val = sum(r["first5_action"] for r in rows) / len(rows)
                improved = val < best
                best = min(best, val)
                payload = {"format": FORMAT, "updater": module.state_dict(), "updater_config": module.descriptor(),
                           "optimizer": optimizer.state_dict(), "step": last_step, "total_steps": args.steps,
                           "config_id": config["config_id"], "best_loss": best, "validation": rows}
                save(latest, payload)
                if improved:
                    save(output / "best_diagnostic.pt", payload)
                if last_step == args.steps:
                    save(output / "final.pt", payload)
                atomic_json(output / "validation_latest.json", {"step": last_step, "first5_action": val, "windows": rows})
                tqdm.write(f"CHECKPOINT Condition step={last_step} val_action={val:.6g} path={latest}")
                module.train()
    finally:
        if wandb_run:
            wandb_run.finish()
    elapsed = time.perf_counter() - timer
    if args.smoke:
        from unittest.mock import patch
        from architectures.openpi.adapters.latentloop.dual_loop import DualLoopPolicy
        observations, _ = pairs.window(0)
        policy = DualLoopPolicy(model, module.eval(), None, "condition_k2")
        noise = explicit_policy_noise((1, 10, model.config.action_dim), seed=7, device="cuda")
        policy.query(observations[0], noise)
        with patch.object(model, "embed_prefix", side_effect=AssertionError("vision/embedding called on skipped query")):
            _, counters = policy.query(observations[1], noise)
        atomic_json(output / "skip_path.json", {"pass": True, "embed_prefix_calls_on_skip": 0,
                                              "counters": counters})
    atomic_json(output / "summary.json", {"complete": last_step == args.steps, "step": last_step,
                "config_id": config["config_id"], "elapsed_seconds": elapsed,
                "seconds_per_new_step_including_validation": elapsed / max(1, last_step - start_step),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(), "selection": "final.pt",
                "condition_parameters": sum(p.numel() for p in module.parameters())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
