"""Projection-only training for real Condition-to-Generation code coupling."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    GENERATION_SCHEDULES,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    COUPLED_CHECKPOINT_SCHEMA,
    audit_projection_only_state,
    build_kc2_coupled_query,
    prepare_projection_only_coupling,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_source_lock import (
    build_coupled_source_lock,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import (
    install_exact_uint8_delta_path,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherSequenceDataset,
    _drop_unused_vlm,
    collate_exact_teacher_sequences,
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
    save_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_objective import (
    generation_local_oracle_loss,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_train import (
    RankDisjointStepSampler,
    _lr_lambda,
    _seed_everything,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    append_jsonl,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
    move_batch,
    write_json,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop


def _query_ages(step: int, local_batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [1 if (step * local_batch_size + offset) % 2 == 0 else 3 for offset in range(local_batch_size)],
        device=device,
        dtype=torch.long,
    )


def _zero_code_parity(
    parent: torch.nn.Module,
    candidate: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(20260825)
    kwargs = {
        "previous_hidden": torch.randn(2, 10, 1024, generator=generator, device=device),
        "noisy_action_before": torch.randn(2, 10, 7, generator=generator, device=device),
        "noisy_action_after": torch.randn(2, 10, 7, generator=generator, device=device),
        "tau_before": torch.full((2,), 0.8, device=device),
        "tau_after": torch.full((2,), 0.7, device=device),
        "proprio": torch.randn(2, 8, generator=generator, device=device),
        "condition_change_code": torch.zeros(2, 128, device=device),
        "condition": torch.randn(2, 122, 960, generator=generator, device=device),
        "condition_valid_mask": torch.ones(2, 122, dtype=torch.bool, device=device),
        "generator_age": 1,
    }
    parent.eval()
    candidate.eval()
    with torch.no_grad():
        expected = parent(**kwargs)
        observed = candidate(**kwargs)
    checks = {
        "hidden_bitwise_equal": torch.equal(expected.hidden, observed.hidden),
        "residual_bitwise_equal": torch.equal(expected.residual, observed.residual),
        "gate_bitwise_equal": torch.equal(expected.gate, observed.gate),
    }
    return {
        "verdict": "ZERO_CODE_PARENT_PARITY_PASS" if all(checks.values()) else "ZERO_CODE_PARENT_PARITY_FAIL",
        "checks": checks,
    }


def _wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any | None:
    if not args.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or Path(args.output).name,
        dir=args.output,
        config=config,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "0")) != 1:
        raise RuntimeError("rb2 coupled screening requires torchrun WORLD_SIZE=1")
    gpu_ids = os.environ.get("SIMVLA_GPU_IDS", "")
    if not gpu_ids or "," in gpu_ids or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_ids:
        raise RuntimeError("exactly one physical GPU must match SIMVLA_GPU_IDS")
    if not 1 <= args.stop_step <= 10_000:
        raise ValueError("coupled screening budget must be in [1,10000]")
    if args.n_g != 3:
        raise ValueError("the fixed coupled screening row is N_G=3")

    dist.init_process_group("nccl")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _seed_everything(args.seed)
    determinism = configure_strict_torch_determinism(args.seed)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)

    validation = validate_exact_cache(args.cache, verify_checksums=False)
    if validation.get("verdict") != "EXACT_TEACHER_CACHE_VALID":
        raise RuntimeError(f"exact cache validation failed: {validation}")
    source = build_coupled_source_lock(
        parent_generation_checkpoint=args.parent_generation_checkpoint,
        condition_checkpoint=args.condition_checkpoint,
        norm_stats=args.norm_stats,
        exact_cache=args.cache,
    )
    write_json(output / "source_lock.json", source)

    condition_adapter, condition_payload = load_native_v0_checkpoint(
        args.condition_checkpoint, device=device, require_final_150k=True
    )
    freeze_module(condition_adapter)
    condition_adapter.eval()
    delta_path = install_exact_uint8_delta_path(condition_adapter)

    parent_updater, parent_payload = load_generation_checkpoint(
        args.parent_generation_checkpoint, device=device
    )
    candidate, _ = load_generation_checkpoint(
        args.parent_generation_checkpoint, device=device
    )
    projection_audit = prepare_projection_only_coupling(candidate)
    parity = _zero_code_parity(parent_updater, candidate, device=device)
    if parity["verdict"] != "ZERO_CODE_PARENT_PARITY_PASS":
        raise RuntimeError(json.dumps(parity, indent=2, sort_keys=True))
    write_json(output / "initialization_zero_code_parity.json", parity)

    frozen_model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    del processor
    dropped = _drop_unused_vlm(frozen_model)
    freeze_module(frozen_model)
    loop = SimVLAGenerationLoop(candidate, frozen_model.transformer.action_decoder).to(device)
    ddp = DistributedDataParallel(loop, device_ids=[0], output_device=0)
    trainable = [
        parameter for parameter in ddp.module.updater.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.peak_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(
            step,
            warmup_steps=args.warmup_steps,
            total_steps=args.stop_step,
            final_ratio=args.final_lr_ratio,
        ),
    )

    dataset = ExactTeacherSequenceDataset(
        args.cache,
        split="train",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    sampler = RankDisjointStepSampler(
        len(dataset),
        seed=args.seed,
        rank=0,
        world_size=1,
        local_batch_size=args.local_batch_size,
        start_step=0,
        stop_step=args.stop_step,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.local_batch_size,
        "sampler": sampler,
        "collate_fn": collate_exact_teacher_sequences,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers:
        loader_kwargs.update(
            persistent_workers=True, prefetch_factor=args.prefetch_factor
        )
    loader = DataLoader(**loader_kwargs)

    training_config = {
        "schema_version": COUPLED_CHECKPOINT_SCHEMA,
        "classification": "projection_only_10k_screening",
        "k_c": 2,
        "n_g": 3,
        "full_step_indices": list(GENERATION_SCHEDULES[3]),
        "condition_change_code": "NativeV0DeltaEncoder output used by the same K_C=2 condition update",
        "training_query_ages": [1, 3],
        "full_refresh_zero_code_age": 2,
        "stop_step": args.stop_step,
        "global_unique_batch": args.local_batch_size,
        "peak_lr": args.peak_lr,
        "warmup_steps": args.warmup_steps,
        "final_lr_ratio": args.final_lr_ratio,
        "weight_decay": args.weight_decay,
        "projection_audit": projection_audit,
        "zero_code_parent_parity": parity,
        "parent_generation_optimizer_step": int(parent_payload["optimizer_step"]),
        "dataset_contract": dataset.contract(),
        "determinism": determinism,
        "exact_uint8_delta_path": delta_path,
        "frozen_release": dropped,
        "source_combined_sha256": source["combined_sha256"],
    }
    write_json(output / "training_config.json", training_config)
    write_json(output / "parameter_audit.json", projection_audit)
    run_handle = _wandb(args, training_config)

    progress = tqdm(total=args.stop_step, desc="SimVLA coupled c_j", dynamic_ncols=True)
    started = time.perf_counter()
    for zero_step, host_sequence in enumerate(loader):
        sequence = move_batch(host_sequence, device)
        ages = _query_ages(zero_step, args.local_batch_size, device)
        with torch.no_grad():
            query = build_kc2_coupled_query(
                condition_adapter, sequence, query_ages=ages
            )
        code_norm = query["condition_change_code"].float().norm(dim=-1)
        if not bool((code_norm > 0).all()):
            raise RuntimeError("updated K_C=2 queries produced a zero condition code")
        normalized_proprio = action_adapter.normalize_proprio(query["proprio"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            loss = generation_local_oracle_loss(
                loop=ddp,
                transformer=frozen_model.transformer,
                action_space=action_adapter.action_space,
                condition=query["condition"],
                initial_noise=query["initial_noise"],
                normalized_proprio=normalized_proprio,
                condition_valid_mask=query["valid_mask"],
                condition_change_code=query["condition_change_code"],
                full_step_indices=GENERATION_SCHEDULES[3],
                teacher_final_action=query["teacher_action"],
                hidden_weight=1.0,
                velocity_weight=0.0,
                final_action_weight=0.0,
            )
        loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if any(parameter.grad is not None for parameter in condition_adapter.parameters()):
            raise RuntimeError("frozen Condition updater received gradients")
        if any(parameter.grad is not None for parameter in frozen_model.parameters()):
            raise RuntimeError("frozen SimVLA received gradients")
        optimizer.step()
        scheduler.step()
        step = zero_step + 1
        if step == 1 or step % args.log_interval == 0 or step == args.stop_step:
            elapsed = time.perf_counter() - started
            metrics = {
                "step": step,
                "loss/hidden_normalized_mse": float(loss.hidden_normalized_mse.item()),
                "monitor/velocity_l1": float(loss.velocity_l1.item()),
                "monitor/final_action_l1_to_full_condition_teacher": float(loss.final_action_l1.item()),
                "condition_code/norm_mean": float(code_norm.mean().item()),
                "condition_code/norm_min": float(code_norm.min().item()),
                "optimizer/lr": float(optimizer.param_groups[0]["lr"]),
                "optimizer/grad_norm": float(grad_norm),
                "throughput/mean_step_seconds": elapsed / step,
                "memory/peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "memory/peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            append_jsonl(output / "train_metrics.jsonl", metrics)
            progress.set_postfix(
                hidden=f"{metrics['loss/hidden_normalized_mse']:.4g}",
                code=f"{metrics['condition_code/norm_mean']:.3g}",
                sec=f"{metrics['throughput/mean_step_seconds']:.3f}",
            )
            if run_handle is not None:
                run_handle.log(metrics, step=step)
        progress.update(1)
        if step % args.save_interval == 0 or step == args.stop_step:
            checkpoint = output / "checkpoints" / f"coupled_generation_step_{step:06d}.pt"
            save_generation_checkpoint(
                checkpoint,
                updater=ddp.module.updater,
                config=parent_payload_config(parent_payload),
                optimizer_step=step,
                optimizer=optimizer,
                scheduler=scheduler,
                source_lock=source,
                training_config=training_config,
            )
            (output / "latest_checkpoint.txt").write_text(
                str(checkpoint) + "\n", encoding="utf-8"
            )

    progress.close()
    elapsed = time.perf_counter() - started
    state_audit = audit_projection_only_state(parent_updater, ddp.module.updater)
    if state_audit["verdict"] != "PROJECTION_ONLY_STATE_PASS":
        raise RuntimeError(json.dumps(state_audit, indent=2, sort_keys=True))
    write_json(output / "final_projection_only_state_audit.json", state_audit)
    summary = {
        "verdict": "COUPLED_PROJECTION_TRAINING_COMPLETE",
        "optimizer_step": args.stop_step,
        "trainable_parameters": projection_audit["trainable_parameters"],
        "elapsed_seconds": elapsed,
        "mean_step_seconds": elapsed / args.stop_step,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "source_combined_sha256": source["combined_sha256"],
        "projection_only_state_audit": state_audit,
    }
    write_json(output / "run_summary.json", summary)
    if run_handle is not None:
        run_handle.finish()
    dist.destroy_process_group()
    return summary


def parent_payload_config(payload: dict[str, Any]) -> Any:
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
        GenerationLoopConfig,
    )

    values = dict(payload["model_config"])
    values["supported_n_g"] = tuple(values["supported_n_g"])
    return GenerationLoopConfig(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--parent-generation-checkpoint", required=True)
    parser.add_argument("--condition-checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--n-g", type=int, default=3)
    parser.add_argument("--stop-step", type=int, default=10_000)
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--final-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
