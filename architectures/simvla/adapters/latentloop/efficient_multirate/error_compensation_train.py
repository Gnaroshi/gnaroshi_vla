"""Matched local-oracle targets: explain predicted C, or compensate for its error."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.simvla.error_compensation_common import (
    CONFIG, ARMS, arm_inputs, checkpoint_path, configure, identity, read_json, snapshots, write_json,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
from architectures.simvla.adapters.latentloop.native_v0_runtime import append_jsonl, move_batch
from .condition_mechanism import action_metrics, _balanced_indices
from .exact_teacher_cache import collate_exact_teacher_sequences
from .generation_checkpoint import load_generation_checkpoint
from .generation_objective import generation_local_oracle_loss
from .generation_train import RankDisjointStepSampler
from .shared_refinement_train import load_runtime, lr_factor, query_inputs, assert_frozen
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop


def objective(loop, frozen, action, context, noise, target, teacher, arm):
    oracle, code = arm_inputs(arm, context.condition, teacher, context.global_code)
    return generation_local_oracle_loss(loop=loop, transformer=frozen.transformer,
        action_space=action.action_space, condition=context.condition, initial_noise=noise,
        normalized_proprio=context.proprio, condition_valid_mask=context.valid_mask,
        condition_change_code=code, full_step_indices=(0, 4, 8), teacher_final_action=target,
        hidden_weight=1.0, velocity_weight=0.0, final_action_weight=0.0,
        oracle_condition=oracle)


def load_candidate(c, arm, device):
    path = checkpoint_path(c, arm)
    updater, payload = load_generation_checkpoint(path, device=device)
    if payload["training_config"]["identity"] != identity(c) or payload["optimizer_step"] != c["steps"]:
        raise RuntimeError("Incomplete or incompatible trained candidate")
    updater.requires_grad_(False).eval()
    return updater


@torch.no_grad()
def offline(c, arm, runtime, updater, output):
    device, adapter, frozen, action, _, heldout = runtime
    parent, _ = load_generation_checkpoint(c["generation_checkpoint"], device=device)
    parent.requires_grad_(False).eval()
    loops = {"parent": SimVLAGenerationLoop(parent, frozen.transformer.action_decoder).eval(),
        "candidate": SimVLAGenerationLoop(updater, frozen.transformer.action_decoder).eval()}
    rows = []
    indices = _balanced_indices(heldout.identities, limit=c["heldout_windows"], seed=c["seed"])
    for index in tqdm(indices, desc=f"Heldout {arm}", mininterval=2):
        sequence = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
        for age in (1, 3):
            context, raw, noise, _ = query_inputs(adapter, action, sequence, age)
            target = sequence["teacher_actions"][:, age - 1]
            for name, loop in loops.items():
                row_context = replace(context, valid_mask=None) if name == "parent" else context
                result = objective(loop, frozen, action, row_context, noise, target,
                    sequence["teacher_conditions"][:, age - 1],
                    "true_condition_no_code" if name == "parent" else arm)
                prediction = action.action_space.postprocess(result.trace.final_noisy_action)
                rows.append({"index": index, "age": age, "row": name,
                    **action_metrics(prediction, target), "hidden_loss": float(result.hidden_normalized_mse)})
            naive = action.decode_action_from_condition(context.condition, raw,
                initial_noise=noise, steps=3, return_debug=True).action
            rows.append({"index": index, "age": age, "row": "naive3", **action_metrics(naive, target)})
    metrics = list(action_metrics(prediction, target))
    summary = {name: {key: sum(r[key] for r in rows if r["row"] == name) /
        sum(r["row"] == name for r in rows) for key in metrics} for name in ("parent", "candidate", "naive3")}
    write_json(output / "offline_queries.json", rows)
    write_json(output / "offline_summary.json", {"verdict": "OFFLINE_COMPLETE_NO_SR_GATE",
        "heldout_windows": len(indices), "queries_per_row": 2 * len(indices), "metrics": summary})


def run(c, arm, *, smoke=False):
    configure(c)
    run_id = identity(c)
    output = Path(c["output"]) / ("smoke" if smoke else "train") / arm
    output.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(c, snapshots(c))
    device, adapter, frozen, action, train, heldout = runtime
    updater, parent_payload = load_generation_checkpoint(c["generation_checkpoint"], device=device)
    updater.requires_grad_(True).train()
    # Neutral code projection preserves the parent at initialization in all arms.
    with torch.no_grad():
        updater.condition_code_projection.weight.zero_()
    initial_hash = hashlib.sha256()
    for name, tensor in updater.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    loop = SimVLAGenerationLoop(updater, frozen.transformer.action_decoder).to(device)
    optimizer = torch.optim.AdamW(updater.parameters(), lr=c["learning_rate"], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: lr_factor(step, c["steps"], c["warmup_steps"]))
    start, elapsed = 0, 0.0
    latest = output / "latest.pt"
    if latest.exists() and not smoke:
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["training_config"]["identity"] != run_id:
            raise RuntimeError("Resume identity mismatch")
        updater.load_state_dict(saved["updater_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start, elapsed = saved["optimizer_step"], saved["training_seconds"]
    total = 2 if smoke else c["steps"]
    training = {"identity": run_id, "arm": arm, "loss": "normalized hidden MSE only",
        "initial_updater_sha256": initial_hash.hexdigest(),
        "oracle": "predicted condition" if arm == "same_condition" else "full teacher condition at student x,t",
        "condition_code": "zero" if arm.endswith("no_code") else "existing observation delta encoder output",
        "full_transformer_condition": "predicted condition in EVERY arm",
        "full_indices": [0, 4, 8], "integration_steps": 10, "query_ages": [1, 3],
        "fresh_queries": "unchanged original Generation updater, zero code",
        "trainable": [n for n, p in updater.named_parameters() if p.requires_grad],
        "parameters": sum(p.numel() for p in updater.parameters()), "train": train.contract(),
        "heldout": heldout.contract(), "scheduler_total_steps": c["steps"]}
    write_json(output / "training_contract.json", training)
    if start < total:
        sampler = RankDisjointStepSampler(len(train), seed=c["seed"], rank=0, world_size=1,
            local_batch_size=c["batch_size"], start_step=start, stop_step=total)
        train.store._loaded.clear()
        heldout.store._loaded.clear()
        loader = DataLoader(train, batch_size=c["batch_size"], sampler=sampler,
            collate_fn=collate_exact_teacher_sequences, num_workers=0 if smoke else c["num_workers"],
            pin_memory=True, **({"multiprocessing_context": "spawn", "persistent_workers": True}
                if not smoke and c["num_workers"] else {}))
        tracker = None
        if not smoke and c.get("wandb_project"):
            try:
                import wandb
                tracker = wandb.init(project=c["wandb_project"], name=f"error_compensation_{arm}_seed7",
                    id=run_id[:12] + "_" + arm, resume="allow", config={**c, **training},
                    dir=str(output), settings=wandb.Settings(init_timeout=20))
                write_json(output / "wandb.json", {"mode": tracker.settings.mode, "url": tracker.url})
            except Exception as error:
                write_json(output / "wandb.json", {"local_logging_only": True, "error": str(error)})
        started = time.monotonic()
        progress = tqdm(loader, total=total, initial=start, desc=arm, mininterval=2)
        try:
            for step, host in enumerate(progress, start=start + 1):
                sequence = move_batch(host, device)
                age = (1, 3)[(step - 1) % 2]
                context, raw, noise, _ = query_inputs(adapter, action, sequence, age)
                target = sequence["teacher_actions"][:, age - 1]
                teacher = sequence["teacher_conditions"][:, age - 1]
                if step == 1:
                    write_json(output / "first_batch.json", {"task_id": host["task_id"].tolist(),
                        "episode_id": host["episode_id"], "anchor_query_index": host["anchor_query_index"].tolist(), "age": age})
                    with torch.no_grad():
                        actual = action.decode_action_from_condition(teacher, raw,
                            steps=10, initial_noise=noise, return_debug=True).action
                    diff = float((actual - target).abs().max())
                    write_json(output / "teacher_cache_check.json", {"max_action_diff": diff})
                    if diff > 2e-4:
                        raise RuntimeError(f"Teacher/cache numerical contract mismatch: {diff}")
                optimizer.zero_grad(set_to_none=True)
                result = objective(loop, frozen, action, context, noise, target, teacher, arm)
                if not bool(torch.isfinite(result.total)):
                    raise RuntimeError("Nonfinite objective")
                result.total.backward()
                grad = torch.nn.utils.clip_grad_norm_(updater.parameters(), 1.0)
                if not bool(torch.isfinite(grad)) or float(grad) == 0:
                    raise RuntimeError("Nonfinite or zero updater gradient")
                assert_frozen(adapter, frozen)
                optimizer.step()
                scheduler.step()
                if step == 1 or step % c["log_interval"] == 0 or step == total:
                    metrics = {"step": step, "hidden_mse": float(result.hidden_normalized_mse),
                        "velocity_l1": float(result.velocity_l1), "action_l1": float(result.final_action_l1),
                        "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad),
                        "seconds": elapsed + time.monotonic() - started,
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
                    append_jsonl(output / "metrics.jsonl", metrics)
                    if tracker: tracker.log(metrics, step=step)
                    progress.set_postfix(loss=f"{metrics['hidden_mse']:.4g}", act=f"{metrics['action_l1']:.4g}")
                if step % c["save_interval"] == 0 or step == total:
                    atomic_torch_save({"checkpoint_format": parent_payload["checkpoint_format"],
                        "model_config": parent_payload["model_config"], "updater_state_dict": updater.state_dict(),
                        "optimizer_step": step, "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(), "training_config": training,
                        "source_lock": {"identity": run_id}, "training_seconds": elapsed + time.monotonic() - started}, latest)
                    print(f"SAVED {arm} {step}/{total}", flush=True)
        finally:
            if tracker: tracker.finish()
    if smoke:
        write_json(output / "summary.json", {"verdict": "SMOKE_PASS", "steps": total, "identity": run_id})
        return
    updater.requires_grad_(False).eval()
    if not (output / "offline_summary.json").exists():
        offline(c, arm, runtime, updater, output)
    write_json(output / "summary.json", {"verdict": "TRAIN_AND_OFFLINE_COMPLETE", "identity": run_id,
        "steps": c["steps"], "checkpoint": str(latest), "online_evaluation_required": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(read_json(a.config), a.arm, smoke=a.smoke)
