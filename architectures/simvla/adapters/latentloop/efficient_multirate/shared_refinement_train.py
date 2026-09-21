"""Matched four-arm pilot, sharing frozen forward work but no learned weights."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.latentloop.modules.shared_refinement import VARIANTS, SharedRefiner, refine_from_anchor
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save, load_native_v0_checkpoint
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    append_jsonl, configure_strict_torch_determinism, freeze_module, load_frozen_simvla, move_batch,
)
from .condition_mechanism import make_datasets, action_metrics, _balanced_indices
from .efficient_delta import install_exact_uint8_delta_path
from .exact_teacher_cache import _drop_unused_vlm, collate_exact_teacher_sequences
from .generation_train import RankDisjointStepSampler
from .shared_refinement import condition_query, frozen_anchor


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def models_for_seed(seed: int, device: torch.device) -> nn.ModuleDict:
    models = {}
    for variant in VARIANTS:
        torch.manual_seed(seed)
        models[variant] = SharedRefiner(variant).to(device)
    # All identically named, shape-compatible tensors start identically,
    # even when readout construction consumed a different number of RNG draws.
    common = models[VARIANTS[0]].state_dict()
    for variant, model in models.items():
        if variant != VARIANTS[0]:
            state = model.state_dict()
            for name, value in common.items():
                if name in state and state[name].shape == value.shape:
                    state[name] = value.clone()
            model.load_state_dict(state)
    return nn.ModuleDict(models)


def loss_for(model, decoder, hidden, velocity, noise, context, target):
    predictions = [refine_from_anchor(model, decoder, anchor_hidden=hidden,
        anchor_velocity=velocity, noise=noise, context=context, cheap_steps=n) for n in (1, 2)]
    losses = [F.l1_loss(prediction, target) for prediction in predictions]
    return sum(losses) / 2, losses


def load_runtime(c, provenance):
    device = torch.device("cuda:0")
    configure_strict_torch_determinism(c["seed"])
    torch.set_num_threads(1)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction((total - 2 * 1024**3) / total, device)
    adapter, payload = load_native_v0_checkpoint(c["condition_checkpoint"], device=device, require_final_150k=True)
    freeze_module(adapter)
    install_exact_uint8_delta_path(adapter)
    # Load a pinned local snapshot, never mutable HF main.
    model, processor, action = load_frozen_simvla(checkpoint=provenance["checkpoint_snapshot"],
        norm_stats=c["norm_stats"], smolvlm_model=provenance["backbone_snapshot"], device=device)
    _drop_unused_vlm(model)
    del processor
    freeze_module(model)
    train, heldout = make_datasets(c, payload)
    return device, adapter, model, action, train, heldout


def query_inputs(adapter, action, sequence, age):
    context = condition_query(adapter, sequence, age)
    raw_proprio = context.proprio
    context.proprio = action.normalize_proprio(raw_proprio)
    noise = sequence["explicit_noises"][:, age - 1]
    target = action.action_space.normalize_action(sequence["teacher_actions"][:, age - 1])
    return context, raw_proprio, noise, target


def assert_frozen(*modules):
    if any(p.requires_grad or p.grad is not None for m in modules for p in m.parameters()):
        raise RuntimeError("Frozen Condition or SimVLA changed gradient contract")


def gpu_contract_smoke(c, output, runtime, write_json):
    device, adapter, frozen, action, train, _ = runtime
    sequence = move_batch(collate_exact_teacher_sequences([train[0]]), device)
    models = models_for_seed(c["seed"], device)
    report = {"checks": {}, "max_teacher_action_diff": 0.0}
    for age in (1, 2, 3):
        context, raw, noise, target = query_inputs(adapter, action, sequence, age)
        with torch.no_grad():
            original = action.decode_action_from_condition(sequence["teacher_conditions"][:, age - 1], raw,
                steps=10, initial_noise=noise, return_debug=True)
        diff = float((original.action - sequence["teacher_actions"][:, age - 1]).abs().max())
        report["max_teacher_action_diff"] = max(report["max_teacher_action_diff"], diff)
        # This tolerance concerns same-noise cache/runtime identity, not scientific performance.
        if diff > 2e-4:
            raise RuntimeError(f"Teacher/cache action mismatch at age {age}: {diff}")
        hidden, velocity = frozen_anchor(frozen.transformer, context, noise)
        if hidden.shape[-2:] != (10, 1024):
            raise RuntimeError(f"Unexpected hidden shape: {hidden.shape}")
        for name, model in models.items():
            optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=0.0)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            loss, _ = loss_for(model, frozen.transformer.action_decoder, hidden, velocity, noise, context, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite smoke loss: {name}")
            loss.backward()
            if not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in model.parameters()):
                raise RuntimeError(f"No refiner gradient: {name}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if not any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
                raise RuntimeError(f"Optimizer did not update {name}")
            report["checks"][f"{name}/age{age}"] = True
        assert_frozen(adapter, frozen)
    report["verdict"] = "GPU_CONTRACT_SMOKE_PASS"
    report["cheap_steps"] = [1, 2]
    report["full_transformer_calls_per_query"] = 1
    write_json(output / "gpu_contract_smoke.json", report)
    del models
    torch.cuda.empty_cache()
    return report


def run_training(c, output: Path, identity: str, runtime, write_json):
    device, adapter, frozen, action, train, heldout = runtime
    models = models_for_seed(c["seed"], device)
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=0.0)
                  for name, model in models.items()}
    schedulers = {name: torch.optim.lr_scheduler.LambdaLR(opt,
        lambda step: lr_factor(step, c["steps"], c["warmup_steps"])) for name, opt in optimizers.items()}
    checkpoint = output / "checkpoints" / "latest.pt"
    start, elapsed_before = 0, 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise RuntimeError("Resume source/config/input identity mismatch")
        models.load_state_dict(saved["models"])
        for name in VARIANTS:
            optimizers[name].load_state_dict(saved["optimizers"][name])
            schedulers[name].load_state_dict(saved["schedulers"][name])
        start, elapsed_before = saved["step"], saved["training_seconds"]
    parameter_audit = {name: {"trainable_parameters": sum(p.numel() for p in model.parameters()),
        "parameter_names": list(dict(model.named_parameters()))} for name, model in models.items()}
    write_json(output / "parameter_audit.json", {"variants": parameter_audit,
        "frozen_condition_parameters": sum(p.numel() for p in adapter.parameters()),
        "same_budget_claim": "approximate active count, no unused padding; exact counts reported"})
    write_json(output / "dataset_contract.json", {"train": train.contract(), "heldout": heldout.contract(),
        "k_c": 2, "query_cycle": [1, 2, 3, 2], "fresh_fraction": 0.5,
        "target": "full-condition native-10 same-noise final normalized action; all H=10 and 7 channels",
        "loss": "mean of normalized terminal action L1 for one and two cheap refinements, equal weights",
        "optimizer_steps_per_variant": c["steps"], "not_online_success_evaluation": True})
    def export_final(step, elapsed):
        final = {}
        for name, model in models.items():
            destination = output / "checkpoints" / f"{name}_final.pt"
            atomic_torch_save({"format": "simvla_shared_refinement_v1", "identity": identity,
                "variant": name, "model": model.state_dict(), "config": c, "step": step}, destination)
            final[name] = str(destination)
        write_json(output / "training_summary.json", {"verdict": "TRAINING_COMPLETE", "identity": identity,
            "steps_per_variant": step, "checkpoints": final, "training_seconds": elapsed})

    if start == c["steps"]:
        export_final(start, elapsed_before)
        return models
    if not 0 <= start < c["steps"]:
        raise RuntimeError("Checkpoint step outside configured training horizon")
    sampler = RankDisjointStepSampler(len(train), seed=c["seed"], rank=0, world_size=1,
        local_batch_size=c["batch_size"], start_step=start, stop_step=c["steps"])
    options = dict(batch_size=c["batch_size"], sampler=sampler, num_workers=c["num_workers"],
        collate_fn=collate_exact_teacher_sequences, pin_memory=True)
    if c["num_workers"]:
        options.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    # Dataset construction mmaps every shard to establish the episode split.
    # Never pickle those cached tensors into spawned dataloader workers.
    train.store._loaded.clear()
    heldout.store._loaded.clear()
    loader = DataLoader(train, **options)
    begun = time.monotonic()
    progress = tqdm(loader, total=c["steps"], initial=start, desc="Shared refinement (4 controls)", mininterval=2.0)

    def save(step):
        atomic_torch_save({"format": "simvla_shared_refinement_v1", "identity": identity,
            "config": c, "step": step, "models": models.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
            "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
            "training_seconds": elapsed_before + time.monotonic() - begun}, checkpoint)
        print(f"CHECKPOINT_SAVED step={step} path={checkpoint}", flush=True)

    step = start
    tracker = None
    tracker_report = {"requested": c.get("wandb_project", ""), "mode": "disabled"}
    if c.get("wandb_project"):
        try:
            import wandb
            tracker = wandb.init(project=c["wandb_project"], name="shared_refinement_long_seed7",
                id=identity[:16], resume="allow", config=c, dir=str(output),
                mode=os.environ.get("WANDB_MODE", "online"), settings=wandb.Settings(init_timeout=30))
            tracker_report.update(mode=tracker.settings.mode, url=tracker.url)
        except Exception as exc:
            # Local resumable training is not contingent on W&B availability.
            tracker_report.update(mode="local_jsonl_only", error=str(exc))
            print(f"WANDB_UNAVAILABLE: {exc}; local train_metrics.jsonl remains enabled", flush=True)
    write_json(output / "logging_status.json", tracker_report)
    try:
        for step, host in enumerate(progress, start=start + 1):
            sequence = move_batch(host, device)
            age = (1, 2, 3, 2)[(step - 1) % 4]
            context, _, noise, target = query_inputs(adapter, action, sequence, age)
            hidden, velocity = frozen_anchor(frozen.transformer, context, noise)
            values = {}
            for name, model in models.items():
                optimizers[name].zero_grad(set_to_none=True)
                loss, losses = loss_for(model, frozen.transformer.action_decoder, hidden, velocity, noise, context, target)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Nonfinite loss {name} at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizers[name].step()
                schedulers[name].step()
                values[name] = {"loss": float(loss.detach()), "one_cheap": float(losses[0].detach()),
                    "two_cheap": float(losses[1].detach()), "grad_norm": float(norm)}
            if step == 1 or step % c["log_interval"] == 0 or step == c["steps"]:
                assert_frozen(adapter, frozen)
                elapsed = elapsed_before + time.monotonic() - begun
                record = {"step": step, "age": age, "variants": values,
                    "lr": optimizers[VARIANTS[0]].param_groups[0]["lr"], "training_seconds": elapsed,
                    "remaining_seconds_estimate": (c["steps"] - step) * (time.monotonic() - begun) / (step - start),
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                append_jsonl(output / "train_metrics.jsonl", record)
                if tracker is not None:
                    try:
                        tracker.log({**{f"{name}/{key}": value for name, metrics in values.items()
                            for key, value in metrics.items()}, "lr": record["lr"],
                            "training_seconds": elapsed}, step=step)
                    except Exception as exc:
                        print(f"WANDB_LOG_WARNING: {exc}", flush=True)
                progress.set_postfix(shared=f"{values['token_code_hidden']['loss']:.4g}",
                    independent=f"{values['independent_hidden']['loss']:.4g}")
            if step % c["save_interval"] == 0 or step == c["steps"]:
                save(step)
    finally:
        progress.close()
        if tracker is not None:
            try:
                tracker.finish()
            except Exception as exc:
                print(f"WANDB_FINISH_WARNING: {exc}", flush=True)
        # Only fully completed checkpoints are resumable; partial optimizer steps
        # from exceptions must never be mislabeled as a completed step.
        del loader
    export_final(step, elapsed_before + time.monotonic() - begun)
    return models


@torch.no_grad()
def evaluate(c, output, identity, runtime, models, write_json):
    device, adapter, frozen, action, _, heldout = runtime
    models.eval()
    selected = _balanced_indices(heldout.identities, limit=c["heldout_windows"], seed=c["seed"])
    write_json(output / "heldout_selection.json", {"identity": identity,
        "windows": [heldout.identities[i] for i in selected], "split_sha256": heldout.split_sha256})
    all_rows = []
    for index in tqdm(selected, desc="Heldout terminal actions", mininterval=2.0):
        file = output / "heldout_units" / f"window_{index}.json"
        if file.exists():
            unit = json.loads(file.read_text())
            if unit["identity"] != identity:
                raise RuntimeError("Heldout resume identity mismatch")
            all_rows.extend(unit["rows"])
            continue
        sequence = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
        rows = []
        for age in (1, 2, 3):
            context, raw, noise, _ = query_inputs(adapter, action, sequence, age)
            reference = sequence["teacher_actions"][:, age - 1]
            hidden, velocity = frozen_anchor(frozen.transformer, context, noise)
            outputs = {}
            for steps in (1, 2, 3, 10):
                outputs[f"naive_{steps}"] = action.decode_action_from_condition(context.condition, raw,
                    steps=steps, initial_noise=noise)
            for name, model in models.items():
                for cheap in (1, 2):
                    normalized = refine_from_anchor(model, frozen.transformer.action_decoder,
                        anchor_hidden=hidden, anchor_velocity=velocity, noise=noise, context=context, cheap_steps=cheap)
                    outputs[f"{name}_cheap{cheap}"] = action.action_space.postprocess(normalized)
            for name, prediction in outputs.items():
                rows.append({"window_index": index, "task_id": heldout.identities[index][0],
                    "episode_id": heldout.identities[index][1], "age": age, "condition_updated": age != 2,
                    "variant": name, **action_metrics(prediction, reference)})
        write_json(file, {"identity": identity, "rows": rows})
        all_rows.extend(rows)
    summaries = {}
    for name in sorted({r["variant"] for r in all_rows}):
        summaries[name] = {}
        for updated in (False, True):
            rows = [r for r in all_rows if r["variant"] == name and r["condition_updated"] == updated]
            summaries[name]["predicted_condition" if updated else "full_condition"] = {
                "queries": len(rows), **{k: sum(r[k] for r in rows) / len(rows) for k in action_metrics(
                    torch.zeros(1, 10, 7), torch.zeros(1, 10, 7))}}
    report = {"verdict": "TRAINING_AND_OFFLINE_COMPARISON_COMPLETE", "identity": identity,
        "online_success_rate": None, "latency": "not measured; joint training throughput is not inference latency",
        "results": summaries, "next_required": "LIBERO-Long paired online evaluation and isolated latency"}
    write_json(output / "comparison_summary.json", report)
    return report
