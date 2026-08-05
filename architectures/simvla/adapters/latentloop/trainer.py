"""Stage-T1/T2 cache-backed trainer for SimVLA Chunk-aware LatentLoop."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter  # noqa: E402
from architectures.simvla.adapters.latentloop.checkpoint import (  # noqa: E402
    freeze_module,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
    trainable_parameter_names,
)
from architectures.simvla.adapters.latentloop.condition_adapter import (  # noqa: E402
    build_latentloop_adapter,
    parameter_budget_audit,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
    sha256_file,
)
from methods.latentloop.training import (  # noqa: E402
    DeterministicStepBatchSampler,
    LatentLoopLossWeights,
    LossScaleAccumulator,
    OnPolicyRecordDataset,
    QueryCacheDataset,
    collate_query_records,
    compute_t1_losses,
    deterministic_episode_split_indices,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if device.type == "cuda":
        state["torch_cuda_all"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda" and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda_all"]])


def _gradient_l2_norm(module: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            norm = float(parameter.grad.detach().float().norm(2).item())
            squared += norm * norm
    return squared**0.5


def _weighted_components(
    raw_losses: dict[str, Tensor],
    weights: LatentLoopLossWeights,
    *,
    action_only: bool,
) -> dict[str, Tensor]:
    components = {
        "same_noise_action_chunk_l1": (
            weights.action_chunk * raw_losses["same_noise_action_chunk_l1"]
        ),
        "executed_prefix_l1": weights.executed_prefix * raw_losses["executed_prefix_l1"],
    }
    if not action_only:
        components.update(
            {
                "condition_normalized_mse": (
                    weights.condition * raw_losses["condition_normalized_mse"]
                ),
                "update_regularization_mse": (
                    weights.update_regularization * raw_losses["update_regularization_mse"]
                ),
            }
        )
    return components


_RESUME_LOCKED_ARGUMENTS = (
    "cache",
    "variant",
    "stage",
    "execution_horizon",
    "checkpoint",
    "norm_stats",
    "flow_steps",
    "batch_size",
    "heldout_fraction",
    "split_seed",
    "max_steps",
    "condition_weight",
    "action_chunk_weight",
    "executed_prefix_weight",
    "update_regularization_weight",
    "learning_rate",
    "weight_decay",
    "seed",
)


def _validate_resume_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if "optimizer_state_dict" not in payload or "training_state" not in payload:
        raise ValueError("resume checkpoint lacks optimizer/training state")
    previous = payload.get("metadata", {}).get("args", {})
    mismatches = {
        name: (previous.get(name), getattr(args, name))
        for name in _RESUME_LOCKED_ARGUMENTS
        if previous.get(name) != getattr(args, name)
    }
    if mismatches:
        raise ValueError(f"resume arguments differ from checkpoint: {mismatches}")


def _source_signature(source_lock: dict[str, Any]) -> dict[str, Any]:
    checkpoint = source_lock.get("checkpoint", {})
    return {
        "root_commit": source_lock.get("root_commit"),
        "simvla_upstream_commit": source_lock.get("simvla_upstream_commit"),
        "norm_stats_sha256": source_lock.get("norm_stats_sha256"),
        "checkpoint_revision": checkpoint.get("revision"),
        "checkpoint_blob": checkpoint.get("hf_blob_key_sha256"),
        "packages": source_lock.get("packages"),
        "python": source_lock.get("python"),
        "torch": source_lock.get("torch"),
        "torch_cuda": source_lock.get("torch_cuda"),
        "cuda_visible_devices": source_lock.get("cuda_visible_devices"),
    }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _require_weights(args: argparse.Namespace) -> LatentLoopLossWeights:
    values = {
        "condition": args.condition_weight,
        "action_chunk": args.action_chunk_weight,
        "executed_prefix": args.executed_prefix_weight,
        "update_regularization": args.update_regularization_weight,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "Training weights are intentionally unset. Run --calibrate-losses first, then pass: "
            + ", ".join(missing)
        )
    return LatentLoopLossWeights(**values)


def _check_t2_gate(path: str, execution_horizon: int) -> dict[str, Any]:
    if not path:
        raise ValueError("Stage T2 requires --gate-decision-json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("T1_K2_OFFLINE_PASS", "T1_K2_ONLINE_PASS")
    failed = [name for name in required if not bool(payload.get(name, False))]
    if failed:
        raise RuntimeError(f"Stage T2 is blocked by gate fields: {failed}")
    if int(payload.get("execution_horizon", execution_horizon)) != execution_horizon:
        raise RuntimeError("T2 gate execution horizon does not match this run")
    return payload


def _action_losses(
    prediction: Tensor,
    target: Tensor,
    execution_horizon: Tensor,
) -> dict[str, Tensor]:
    full = F.l1_loss(prediction, target.detach())
    horizon = prediction.shape[1]
    mask = torch.arange(horizon, device=prediction.device).unsqueeze(0) < execution_horizon.unsqueeze(1)
    prefix = (prediction - target.detach()).abs()[mask.unsqueeze(-1).expand_as(prediction)].mean()
    return {
        "same_noise_action_chunk_l1": full,
        "executed_prefix_l1": prefix,
        "condition_normalized_mse": full.new_zeros(()),
        "update_regularization_mse": full.new_zeros(()),
    }


def _build_prediction(
    *,
    adapter: Any,
    batch: dict[str, Any],
    action_adapter: SimVLAActionAdapter,
    flow_steps: int,
    stage: str,
) -> tuple[Tensor | None, Tensor, dict[str, Tensor]]:
    previous_condition = (
        batch["predicted_condition"] if stage == "t2" else batch["full_condition"]
    )
    target_condition = batch["next_full_condition"]
    observation = adapter.encode_observation(
        batch["raw_rgb"],
        batch["next_raw_rgb"],
        batch["proprio"],
        batch["next_proprio"],
    )
    execution_horizon = batch["execution_horizon"]
    elapsed_time = batch["elapsed_time"]
    action_feature = adapter.encode_executed_actions(
        batch["executed_subchunk"],
        execution_horizon,
        elapsed_time,
        reference_feature=observation,
    )
    age = torch.ones_like(execution_horizon)
    if stage == "t2":
        age = batch["rollout_depth"].to(dtype=torch.long) + 1
    if adapter.variant == "action_chunk_correction":
        predicted_action = adapter.correct_action_chunk(
            batch["teacher_action_chunk"],
            observation,
            action_feature,
            execution_horizon=execution_horizon,
            elapsed_time=elapsed_time,
            query_age=age,
        )
        raw_losses = _action_losses(
            predicted_action,
            batch["next_teacher_action_chunk"],
            execution_horizon,
        )
        return None, predicted_action, raw_losses
    if adapter.variant == "nonrecurrent_condition":
        predicted_condition = adapter.predict_nonrecurrent_condition(
            batch["full_condition"],
            observation,
            action_feature,
            execution_horizon=execution_horizon,
            elapsed_time=elapsed_time,
            query_age=age,
        )
    else:
        predicted_condition = adapter.update_recurrent_condition(
            previous_condition,
            observation,
            action_feature,
            execution_horizon=execution_horizon,
            elapsed_time=elapsed_time,
            query_age=age,
        )
    predicted_action = action_adapter.decode_action_from_condition(
        predicted_condition,
        batch["next_proprio"],
        steps=flow_steps,
        initial_noise=batch["next_initial_noise"],
        requires_grad=True,
    )
    unit_weights = LatentLoopLossWeights(1.0, 1.0, 1.0, 1.0)
    raw_losses = compute_t1_losses(
        previous_condition=previous_condition,
        predicted_condition=predicted_condition,
        teacher_condition=target_condition,
        predicted_action_chunk=predicted_action,
        teacher_action_chunk=batch["next_teacher_action_chunk"],
        execution_lengths=execution_horizon,
        weights=unit_weights,
    )
    return predicted_condition, predicted_action, raw_losses


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Calibrate raw scales or optimize only the selected LatentLoop adapter."""

    from models.modeling_smolvlm_vla import SmolVLMVLA

    if args.calibrate_losses and args.resume_from:
        raise ValueError("raw-loss calibration cannot be resumed")
    resume_path = Path(args.resume_from).resolve() if args.resume_from else None
    if resume_path is None:
        output = require_empty_output(args.output).resolve()
    else:
        output = Path(args.output).resolve()
        if not output.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {output}")
        if not resume_path.is_file() or not resume_path.is_relative_to(output):
            raise ValueError("--resume-from must be a checkpoint inside --output")

    device = torch.device(args.device)
    _seed_training(args.seed)
    current_source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    resume_payload: dict[str, Any] | None = None
    if resume_path is None:
        adapter = build_latentloop_adapter(args.variant).to(device)
        start_step = 0
        source_lock = current_source_lock
        _write_json(output / "source_lock.json", source_lock)
    else:
        adapter, resume_payload = load_adapter_checkpoint(resume_path, device=device)
        _validate_resume_payload(resume_payload, args)
        start_step = int(resume_payload["step"])
        if adapter.variant != args.variant:
            raise ValueError(
                f"resume adapter variant {adapter.variant} does not match {args.variant}"
            )
        source_lock = resume_payload.get("metadata", {}).get("source_lock")
        if not isinstance(source_lock, dict):
            raise ValueError("resume checkpoint lacks its original source lock")
        if _source_signature(source_lock) != _source_signature(current_source_lock):
            raise ValueError("current runtime/source signature differs from resume checkpoint")
        _write_json(
            output / f"resume_source_lock_step_{start_step:06d}.json",
            current_source_lock,
        )

    if args.calibrate_losses:
        weights = None
        optimizer = None
        max_steps = int(args.calibration_batches)
    else:
        weights = _require_weights(args)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        max_steps = int(args.max_steps)
    if max_steps < 1:
        raise ValueError("the selected mode requires a positive step count")
    if start_step > max_steps:
        raise ValueError(
            f"resume step {start_step} exceeds requested max_steps {max_steps}"
        )

    if args.stage == "t2":
        gate = _check_t2_gate(args.gate_decision_json, args.execution_horizon)
        cache_dataset = OnPolicyRecordDataset(
            args.cache,
            maximum_rollout_depth=args.maximum_rollout_depth,
        )
        _write_json(output / "t2_gate_snapshot.json", gate)
    else:
        cache_dataset = QueryCacheDataset(args.cache)
    manifest_r = int(cache_dataset.manifest["execution_horizon"])
    if manifest_r != args.execution_horizon:
        raise ValueError(f"cache R={manifest_r} does not match --execution-horizon={args.execution_horizon}")
    train_indices, heldout_indices = deterministic_episode_split_indices(
        cache_dataset,
        heldout_fraction=args.heldout_fraction,
        seed=args.split_seed,
    )
    dataset = Subset(cache_dataset, train_indices)
    split_summary = {
        "cache_manifest_sha256": sha256_file(Path(args.cache) / "manifest.json"),
        "split_unit": "task_id+episode_id",
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "heldout_fraction": args.heldout_fraction,
        "train_records": len(train_indices),
        "heldout_records_excluded_from_training": len(heldout_indices),
    }
    split_path = output / "dataset_split.json"
    if resume_path is None:
        _write_json(split_path, split_summary)
    elif json.loads(split_path.read_text(encoding="utf-8")) != split_summary:
        raise ValueError("current deterministic dataset split differs from original run")
    batch_sampler = DeterministicStepBatchSampler(
        dataset_size=len(dataset),
        batch_size=args.batch_size,
        seed=args.seed,
        start_step=start_step,
        max_steps=max_steps,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_query_records,
        pin_memory=device.type == "cuda",
    )
    adapter.train()
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    action_adapter = SimVLAActionAdapter(model)
    optimizer_names = trainable_parameter_names(adapter)
    if not optimizer_names:
        raise RuntimeError("adapter has no trainable parameters")
    teacher_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if teacher_trainable:
        raise RuntimeError(f"frozen SimVLA has trainable parameters: {teacher_trainable[:10]}")
    freeze_snapshot = {
        "teacher_trainable_parameters": 0,
        "adapter_trainable_parameters": sum(
            parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
        ),
        "optimizer_parameter_names": optimizer_names,
        "variant": args.variant,
    }
    _write_json(output / "freeze_status_snapshot.json", freeze_snapshot)
    _write_json(output / "parameter_budget_audit.json", parameter_budget_audit())
    (output / "optimizer_param_names.txt").write_text(
        "\n".join(optimizer_names) + "\n",
        encoding="utf-8",
    )
    calibration = LossScaleAccumulator()
    elapsed_before = 0.0
    last_losses: dict[str, float] = {}
    last_metrics: dict[str, float] = {}
    previous_peak_allocated = 0
    previous_peak_reserved = 0
    if resume_payload is not None:
        assert optimizer is not None
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        training_state = resume_payload["training_state"]
        calibration.load_state_dict(training_state["loss_accumulator"])
        elapsed_before = float(training_state.get("elapsed_seconds", 0.0))
        last_losses = dict(training_state.get("last_losses", {}))
        last_metrics = dict(training_state.get("last_metrics", {}))
        previous_peak_allocated = int(training_state.get("peak_cuda_allocated_bytes", 0))
        previous_peak_reserved = int(training_state.get("peak_cuda_reserved_bytes", 0))
        _restore_rng_state(training_state["rng_state"], device)
    elif optimizer is not None:
        _write_json(output / "loss_weights.json", asdict(weights))

    progress = output / "train_progress.jsonl"
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_id_path = output / "wandb_run_id.txt"
        if wandb_id_path.is_file():
            wandb_run_id = wandb_id_path.read_text(encoding="utf-8").strip()
        else:
            wandb_run_id = wandb.util.generate_id()
            _atomic_write_text(wandb_id_path, wandb_run_id + "\n")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or output.name,
            dir=str(output),
            id=wandb_run_id,
            resume="allow" if resume_path is not None else "never",
            config=None if resume_path is not None else vars(args),
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    step = start_step
    stop_request: dict[str, int | None] = {"signal": None}

    def request_stop(signum: int, _frame: Any) -> None:
        stop_request["signal"] = int(signum)
        print(
            f"[interrupt] signal {signum} received; stopping after the current step.",
            flush=True,
        )

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_stop)

    progress_bar = tqdm(
        total=max_steps,
        initial=start_step,
        desc=f"LatentLoop {args.stage}/{args.variant}/R{args.execution_horizon}",
        dynamic_ncols=True,
        mininterval=args.tqdm_mininterval,
        disable=args.disable_tqdm,
    )
    peak_allocated = previous_peak_allocated
    peak_reserved = previous_peak_reserved

    def save_training_checkpoint(path: Path) -> None:
        assert optimizer is not None
        elapsed = elapsed_before + (time.time() - started)
        training_state = {
            "elapsed_seconds": elapsed,
            "last_losses": last_losses,
            "last_metrics": last_metrics,
            "loss_accumulator": calibration.state_dict(),
            "rng_state": _capture_rng_state(device),
            "training_seed": args.seed,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        }
        save_adapter_checkpoint(
            path,
            adapter=adapter,
            step=step,
            metadata={"source_lock": source_lock, "args": vars(args)},
            optimizer_state_dict=optimizer.state_dict(),
            training_state=training_state,
        )
        _atomic_write_text(output / "latest_checkpoint.txt", str(path) + "\n")

    try:
        for batch in loader:
            next_step = step + 1
            json_due = next_step == 1 or next_step == max_steps or next_step % args.log_interval == 0
            wandb_due = wandb_run is not None and (
                next_step == 1
                or next_step == max_steps
                or next_step % args.wandb_log_interval == 0
            )
            metric_due = json_due or wandb_due
            batch = _to_device(batch, device)
            _, _, raw_losses = _build_prediction(
                adapter=adapter,
                batch=batch,
                action_adapter=action_adapter,
                flow_steps=args.flow_steps,
                stage=args.stage,
            )
            calibration.update(raw_losses)
            weighted_tensors: dict[str, Tensor] = {}
            gradient_norm: float | None = None
            if optimizer is not None and weights is not None:
                weighted_tensors = _weighted_components(
                    raw_losses,
                    weights,
                    action_only=adapter.variant == "action_chunk_correction",
                )
                total = sum(weighted_tensors.values())
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                if metric_due:
                    gradient_norm = _gradient_l2_norm(adapter)
                optimizer.step()
            else:
                total = raw_losses["same_noise_action_chunk_l1"]
            step = next_step
            progress_bar.update(1)
            last_losses = {
                name: float(value.detach().item())
                for name, value in raw_losses.items()
                if name != "total"
            }
            weighted_losses = {
                name: float(value.detach().item())
                for name, value in weighted_tensors.items()
            }
            segment_elapsed = time.time() - started
            elapsed = elapsed_before + segment_elapsed
            total_loss = float(total.detach().item())
            if device.type == "cuda":
                peak_allocated = max(
                    peak_allocated,
                    int(torch.cuda.max_memory_allocated(device)),
                )
                peak_reserved = max(
                    peak_reserved,
                    int(torch.cuda.max_memory_reserved(device)),
                )
            last_metrics = {
                "total_loss": total_loss,
                "steps_per_second_segment": (step - start_step) / max(segment_elapsed, 1e-9),
                "steps_per_second_total": step / max(elapsed, 1e-9),
                "learning_rate": (
                    float(optimizer.param_groups[0]["lr"]) if optimizer is not None else 0.0
                ),
                "peak_cuda_allocated_bytes": float(peak_allocated),
                "peak_cuda_reserved_bytes": float(peak_reserved),
            }
            if gradient_norm is not None:
                last_metrics["gradient_l2_norm"] = gradient_norm
            event = {
                "step": step,
                "max_steps": max_steps,
                "elapsed_seconds": elapsed,
                "losses": last_losses,
                "weighted_losses": weighted_losses,
                **last_metrics,
            }
            if json_due:
                _append_jsonl(progress, event)
                print(json.dumps(event, sort_keys=True), flush=True)
            if step == 1 or step == max_steps or step % args.tqdm_postfix_interval == 0:
                progress_bar.set_postfix(
                    loss=f"{total_loss:.5g}",
                    prefix=f"{last_losses.get('executed_prefix_l1', 0.0):.5g}",
                    sps=f"{last_metrics['steps_per_second_segment']:.2f}",
                )
            if wandb_due:
                wandb_run.log(
                    {
                        **{f"loss/raw/{name}": value for name, value in last_losses.items()},
                        **{
                            f"loss/weighted/{name}": value
                            for name, value in weighted_losses.items()
                        },
                        **{f"train/{name}": value for name, value in last_metrics.items()},
                        "train/step": step,
                        "train/elapsed_seconds": elapsed,
                    },
                    step=step,
                )
            if optimizer is not None and args.save_interval > 0 and step % args.save_interval == 0:
                save_training_checkpoint(
                    output / "checkpoints" / f"latentloop_step_{step:06d}.pt"
                )
            if stop_request["signal"] is not None:
                break
    finally:
        progress_bar.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    completed = step == max_steps
    interrupted = stop_request["signal"] is not None and not completed
    if not completed and not interrupted:
        raise RuntimeError(f"deterministic loader stopped at step {step}, expected {max_steps}")
    scale_summary = calibration.summary()
    _write_json(output / "raw_loss_scales.json", scale_summary)
    final_checkpoint = None
    if optimizer is not None:
        final_checkpoint = output / "checkpoints" / f"latentloop_step_{step:06d}.pt"
        save_training_checkpoint(final_checkpoint)
    result = {
        "mode": "calibration" if args.calibrate_losses else "training",
        "stage": args.stage,
        "variant": args.variant,
        "execution_horizon": args.execution_horizon,
        "training_seed": args.seed,
        "start_step": start_step,
        "steps": step,
        "max_steps": max_steps,
        "completed": completed,
        "interrupted": interrupted,
        "stop_signal": stop_request["signal"],
        "elapsed_seconds": elapsed_before + (time.time() - started),
        "last_losses": last_losses,
        "last_metrics": last_metrics,
        "raw_loss_scales": scale_summary,
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
        "teacher_trainable_parameters": 0,
        "adapter_trainable_parameters": freeze_snapshot["adapter_trainable_parameters"],
    }
    _write_json(output / "run_summary.json", result)
    if wandb_run is not None:
        wandb_run.summary.update(result)
        wandb_run.finish(exit_code=130 if interrupted else 0)
    return result


def main() -> int:
    """Parse guarded cache-training options."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "chunk_aware_latentloop",
            "old_observation_only",
            "no_observation",
            "nonrecurrent_condition",
            "action_chunk_correction",
        ),
        default="chunk_aware_latentloop",
    )
    parser.add_argument("--stage", choices=("t1", "t2"), default="t1")
    parser.add_argument("--execution-horizon", type=int, choices=(1, 2, 5), required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--calibrate-losses", action="store_true")
    parser.add_argument("--calibration-batches", type=int, default=100)
    parser.add_argument("--condition-weight", type=float, default=None)
    parser.add_argument("--action-chunk-weight", type=float, default=None)
    parser.add_argument("--executed-prefix-weight", type=float, default=None)
    parser.add_argument("--update-regularization-weight", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--tqdm-postfix-interval", type=int, default=20)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-log-interval", type=int, default=1000)
    parser.add_argument("--gate-decision-json", default="")
    parser.add_argument("--maximum-rollout-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 130 if result.get("interrupted", False) else 0


if __name__ == "__main__":
    raise SystemExit(main())
