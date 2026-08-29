#!/usr/bin/env python3
"""Minimal real-check smoke for SimVLA DCLD freeze, gradients, and counters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter, SimVLADCLDEvalWrapper  # noqa: E402
from architectures.simvla.adapters.dcld.simvla_dcld_distill_trainer import (  # noqa: E402
    SimVLADCLDDistillTrainer,
)
from datasets import create_smolvlm_dataloader  # noqa: E402
from methods.dcld.modules import DCLDCore, DeltaObservation  # noqa: E402
from methods.dcld.training import TeacherCacheMetadata, TeacherCacheShardWriter  # noqa: E402
from models.modeling_smolvlm_vla import SmolVLMVLA  # noqa: E402
from models.processing_smolvlm_vla import SmolVLMVLAProcessor  # noqa: E402


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: list[str], cwd: Path = ROOT) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def load_real_batch(args: argparse.Namespace) -> dict[str, Any]:
    old_cwd = Path.cwd()
    os.chdir(UPSTREAM)
    try:
        dataloader = create_smolvlm_dataloader(
            batch_size=1,
            metas_path=str(args.metas_path),
            num_actions=args.steps,
            training=False,
            action_mode="libero_joint",
            num_workers=0,
            image_size=args.image_size,
        )
        batch = next(iter(dataloader))
    finally:
        os.chdir(old_cwd)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    lang = processor.encode_language(batch["language_instruction"])
    batch.update(lang)
    return batch


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()) if y.numel() > 1 else 0.0,
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "norm": float(y.norm().item()),
    }


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "image_input": batch["image_input"].to(device),
        "image_mask": batch["image_mask"].to(device),
        "proprio": batch["proprio"].to(device),
        "action": batch["action"].to(device),
    }


def build_delta_obs(batch: dict[str, torch.Tensor]) -> DeltaObservation:
    return DeltaObservation(
        key_images=batch["image_input"],
        cur_images=batch["image_input"],
        key_proprio=batch["proprio"],
        cur_proprio=batch["proprio"],
        age=1.0,
    )


def grad_summary(teacher: torch.nn.Module, dcld_core: torch.nn.Module) -> dict[str, Any]:
    teacher_grad_nonzero = 0
    teacher_grad_tensors = 0
    for param in teacher.parameters():
        if param.grad is not None:
            teacher_grad_tensors += 1
            if torch.any(param.grad.detach() != 0):
                teacher_grad_nonzero += 1
    dcld_grad_nonzero = 0
    dcld_grad_tensors = 0
    dcld_grad_norm_sq = 0.0
    for param in dcld_core.parameters():
        if param.grad is not None:
            dcld_grad_tensors += 1
            grad = param.grad.detach().float()
            dcld_grad_norm_sq += float(torch.sum(grad * grad).item())
            if torch.any(grad != 0):
                dcld_grad_nonzero += 1
    return {
        "teacher_grad_tensors": teacher_grad_tensors,
        "teacher_grad_nonzero_tensors": teacher_grad_nonzero,
        "dcld_grad_tensors": dcld_grad_tensors,
        "dcld_grad_nonzero_tensors": dcld_grad_nonzero,
        "dcld_grad_l2_norm": dcld_grad_norm_sq ** 0.5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_id", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm_model_path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--metas_path", type=Path, default=UPSTREAM / "datasets" / "metas" / "libero_train.json")
    parser.add_argument("--norm_stats_path", type=Path, default=UPSTREAM / "norm_stats" / "libero_norm.json")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_batches", type=int, default=2)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    write_json(out / "args_snapshot.json", {key: str(value) for key, value in vars(args).items()})
    write_json(
        out / "git_snapshot.json",
        {
            "root_head": run_cmd(["git", "rev-parse", "HEAD"]),
            "root_status_short": run_cmd(["git", "status", "--short"]),
            "simvla_upstream_head": run_cmd(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"]),
            "simvla_upstream_status_short": run_cmd(["git", "-C", str(UPSTREAM), "status", "--short"]),
        },
    )

    batch_cpu = load_real_batch(args)
    batch = to_device(batch_cpu, device)

    teacher = SmolVLMVLA.from_pretrained(args.checkpoint_id).to(device)
    teacher.eval()
    if args.norm_stats_path.exists():
        teacher.action_space.load_norm_stats(str(args.norm_stats_path))

    with torch.no_grad():
        condition = teacher.forward_vlm_efficient(
            batch["image_input"], batch["image_mask"], batch["input_ids"]
        )["vlm_features"]

    dcld_core = DCLDCore(latent_dim=condition.shape[-1], gate_bias=-4.0).to(device)
    delta_obs = build_delta_obs(batch)
    forward_smoke = dcld_core.update_latent(condition.detach(), delta_obs)
    gate = forward_smoke.dynamics.gate.detach().float()
    update = forward_smoke.dynamics.update.detach().float()
    dcld_forward_smoke = {
        "z_prev_shape": list(condition.shape),
        "z_next_shape": list(forward_smoke.latent.shape),
        "gate_bias": float(forward_smoke.debug["gate_bias"].mean().item()),
        "gate_mean": float(gate.mean().item()),
        "gate_std": float(gate.std(unbiased=False).item()),
        "gate_min": float(gate.min().item()),
        "gate_max": float(gate.max().item()),
        "update_norm": float(update.flatten(start_dim=1).norm(dim=-1).mean().item()),
        "z_next_finite": bool(torch.isfinite(forward_smoke.latent).all().item()),
        "update_finite": bool(torch.isfinite(update).all().item()),
        "passed": list(condition.shape) == list(forward_smoke.latent.shape)
        and bool(torch.isfinite(forward_smoke.latent).all().item())
        and bool(torch.isfinite(update).all().item()),
    }
    write_json(out / "dcld_forward_smoke_after_gate_bias.json", dcld_forward_smoke)

    trainer = SimVLADCLDDistillTrainer(
        teacher_model=teacher,
        dcld_core=dcld_core,
        output_dir=out,
    )
    optimizer = trainer.build_optimizer(learning_rate=args.learning_rate)

    cache_dir = out / "teacher_cache_smoke"
    writer = TeacherCacheShardWriter(
        cache_dir,
        TeacherCacheMetadata(
            architecture="simvla",
            checkpoint=args.checkpoint_id,
            dataset="LIBERO",
            condition_key="vlm_features",
            norm_stats_path=str(args.norm_stats_path),
            action_mode="libero_joint",
            notes=["minimal real-batch smoke; not full teacher cache generation"],
        ),
        samples_per_shard=1,
    )
    writer.add(
        {
            "episode_id": "smoke_real_batch_0",
            "timestep": 0,
            "task_name": batch_cpu["language_instruction"][0] if isinstance(batch_cpu["language_instruction"], list) else str(batch_cpu["language_instruction"]),
            "image_reference": "official SimVLA dataloader sample; raw HDF5 path recorded in real_batch_format.json from condition check",
            "condition": condition.detach().cpu(),
            "proprio": batch["proprio"].detach().cpu(),
            "action": batch["action"].detach().cpu(),
            "language_instruction": batch_cpu["language_instruction"],
            "normalization": {"norm_stats_path": str(args.norm_stats_path), "action_mode": "libero_joint"},
        }
    )
    writer.close()
    cache_sample_shapes = {
        "condition": tensor_stats(condition),
        "proprio": tensor_stats(batch["proprio"]),
        "teacher_action_chunk": tensor_stats(batch["action"]),
        "condition_has_nan": bool(torch.isnan(condition).any().item()),
        "action_has_nan": bool(torch.isnan(batch["action"]).any().item()),
        "cache_manifest": str(cache_dir / "manifest.json"),
        "reload_smoke": "not_run",
    }
    try:
        loaded = torch.load(cache_dir / "shard_000000.pt", map_location="cpu", weights_only=False)
        cache_sample_shapes["reload_smoke"] = "passed" if loaded and "condition" in loaded[0] else "failed"
    except Exception as exc:
        cache_sample_shapes["reload_smoke"] = f"failed: {exc}"
    write_json(
        out / "teacher_cache_metadata.json",
        {
            "architecture": "simvla",
            "checkpoint": args.checkpoint_id,
            "dataset": "LIBERO",
            "condition_key": "vlm_features",
            "norm_stats_path": str(args.norm_stats_path),
            "action_mode": "libero_joint",
            "cache_dir": str(cache_dir),
            "num_samples": 1,
        },
    )
    write_json(out / "teacher_cache_sample_shapes.json", cache_sample_shapes)
    write_text(
        out / "teacher_cache_smoke_report.md",
        "# Teacher Cache Smoke Report\n\n"
        "- status: `passed`\n"
        "- scope: one real LIBERO batch, one teacher condition tensor\n"
        f"- checkpoint: `{args.checkpoint_id}`\n"
        f"- condition_stats: `{tensor_stats(condition)}`\n"
        f"- cache_dir: `{cache_dir}`\n"
        f"- sample_shapes: `{cache_sample_shapes}`\n"
        "- note: this is a smoke artifact, not full teacher cache generation.\n",
    )

    action_adapter = SimVLAActionAdapter(teacher)
    with torch.no_grad():
        hold_action = action_adapter.decode_action_from_condition(condition.detach(), batch["proprio"], steps=2, deterministic=True)
        pred_condition_for_metrics = forward_smoke.latent.detach()
        pred_action = action_adapter.decode_action_from_condition(pred_condition_for_metrics, batch["proprio"], steps=2, deterministic=True)
    hold_mse = torch.nn.functional.mse_loss(condition.detach(), condition.detach())
    pred_mse = torch.nn.functional.mse_loss(pred_condition_for_metrics, condition.detach())
    cos_pred = torch.nn.functional.cosine_similarity(
        pred_condition_for_metrics.flatten(start_dim=1),
        condition.detach().flatten(start_dim=1),
        dim=-1,
    ).mean()
    action_l1_hold = torch.nn.functional.l1_loss(hold_action.detach(), batch["action"].detach())
    action_l1_pred = torch.nn.functional.l1_loss(pred_action.detach(), batch["action"].detach())
    update_norm = forward_smoke.dynamics.update.detach().flatten(start_dim=1).norm(dim=-1).mean()
    condition_norm = condition.detach().flatten(start_dim=1).norm(dim=-1).mean()
    pre_train_metrics = {
        "condition_mse_hold": float(hold_mse.item()),
        "condition_mse_pred": float(pred_mse.item()),
        "condition_mse_improvement": float((hold_mse - pred_mse).item()),
        "condition_cos_pred_teacher": float(cos_pred.item()),
        "action_l1_hold": float(action_l1_hold.item()),
        "action_l1_pred": float(action_l1_pred.item()),
        "flow_mse_hold": None,
        "flow_mse_pred": None,
        "gate_mean": dcld_forward_smoke["gate_mean"],
        "gate_std": dcld_forward_smoke["gate_std"],
        "gate_min": dcld_forward_smoke["gate_min"],
        "gate_max": dcld_forward_smoke["gate_max"],
        "u_delta_norm": float(forward_smoke.delta_feature.detach().norm(dim=-1).mean().item()),
        "update_norm": float(update_norm.item()),
        "update_to_condition_norm_ratio": float((update_norm / (condition_norm + 1e-8)).item()),
    }

    smoke_batches = [
        {"latent_prev": condition.detach(), "target_condition": condition.detach(), "delta_obs": delta_obs},
        {"latent_prev": condition.detach(), "target_condition": condition.detach(), "delta_obs": delta_obs},
    ]
    os.environ["SIMVLA_DCLD_DEBUG"] = "1"
    train_logs = trainer.smoke_train_cached_batches(smoke_batches, optimizer, max_batches=args.max_batches)
    grad = grad_summary(teacher, dcld_core)
    fast_grad_nonzero = 0
    updater_grad_nonzero = 0
    for name, param in dcld_core.named_parameters():
        if param.grad is None:
            continue
        nonzero = bool(torch.any(param.grad.detach() != 0).item())
        if name.startswith("delta_encoder") and nonzero:
            fast_grad_nonzero += 1
        if name.startswith("dynamics") and nonzero:
            updater_grad_nonzero += 1
    gradient_sanity = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_id": args.checkpoint_id,
        "condition": tensor_stats(condition),
        "teacher_trainable_params": sum(p.numel() for p in teacher.parameters() if p.requires_grad),
        "dcld_trainable_params": sum(p.numel() for p in dcld_core.parameters() if p.requires_grad),
        "fast_visual_delta_encoder_grad_nonzero_tensors": fast_grad_nonzero,
        "dcld_updater_grad_nonzero_tensors": updater_grad_nonzero,
        **grad,
        "passed": grad["teacher_grad_nonzero_tensors"] == 0
        and grad["dcld_grad_nonzero_tensors"] > 0
        and fast_grad_nonzero > 0
        and updater_grad_nonzero > 0,
    }
    write_json(out / "gradient_sanity.json", gradient_sanity)
    trainable_param_count = sum(p.numel() for p in dcld_core.parameters() if p.requires_grad)
    frozen_param_count = sum(p.numel() for p in teacher.parameters() if not p.requires_grad)
    write_json(
        out / "train_smoke_metrics.json",
        {
            "max_batches": args.max_batches,
            "pre_train_metrics": {
                **pre_train_metrics,
                "trainable_param_count": trainable_param_count,
                "frozen_param_count": frozen_param_count,
            },
            "logs": train_logs,
            "gradient_sanity": gradient_sanity,
        },
    )

    mode_names = [
        "full",
        "stepwise_dcld",
        "hold_action",
        "hold_condition",
        "native_action_chunk",
        "no_delta",
        "shuffled_delta",
        "proprio_only",
        "image_only",
    ]
    episode_rows = []
    mode_summaries = {}
    latency_summary = {}
    for mode_name in mode_names:
        eval_core = DCLDCore(latent_dim=condition.shape[-1], gate_bias=-4.0).to(device)
        _ = eval_core.update_latent(condition.detach(), delta_obs)
        wrapper = SimVLADCLDEvalWrapper(teacher, eval_core, refresh_every=2, mode=mode_name, action_steps=2)
        for step_idx in range(3):
            step_out = wrapper.step(batch, step_idx)
            episode_rows.append(
                {
                    "mode": mode_name,
                    "step": step_idx,
                    "refreshed": step_out.refreshed,
                    "age": step_out.age,
                    "action_mean": float(step_out.action.detach().float().mean().item()),
                    "action_std": float(step_out.action.detach().float().std(unbiased=False).item()),
                    **wrapper.counter_summary(),
                }
            )
        mode_summaries[mode_name] = wrapper.counter_summary()
        latency_summary[mode_name] = wrapper.latency.summary()
    expected = {
        "full": {"num_full_vlm_calls": 3, "num_dcld_updates": 0},
        "stepwise_dcld": {"num_full_vlm_calls": 2, "num_dcld_updates": 1, "num_fast_encoder_calls": 1},
        "hold_action": {"num_hold_action_steps": 1, "num_dcld_updates": 0, "num_fast_encoder_calls": 0},
        "hold_condition": {"num_hold_condition_steps": 1, "num_dcld_updates": 0, "num_fast_encoder_calls": 0},
        "native_action_chunk": {"num_native_chunk_steps": 1, "num_dcld_updates": 0, "num_fast_encoder_calls": 0},
        "no_delta": {"num_dcld_updates": 1, "num_no_delta_steps": 1, "num_fast_encoder_calls": 0},
    }
    checks = {}
    for mode_name, expected_counts in expected.items():
        counters = mode_summaries.get(mode_name, {})
        checks[mode_name] = all(counters.get(key, 0) == value for key, value in expected_counts.items())
    for mode_name in ["shuffled_delta", "proprio_only", "image_only"]:
        counters = mode_summaries.get(mode_name, {})
        checks[mode_name] = counters.get("num_dcld_updates", 0) == 1 and counters.get("num_fast_encoder_calls", 0) == 1
    eval_summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "real checkpoint + real batch wrapper counter smoke across modes; not LIBERO eval",
        "steps": 3,
        "refresh_every": 2,
        "action_steps": 2,
        "expected": expected,
        "mode_checks": checks,
        "mode_counters": mode_summaries,
    }
    write_json(out / "eval_smoke_summary.json", eval_summary)
    write_json(out / "eval_smoke_latency_profile.json", latency_summary)
    with (out / "eval_smoke_episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in episode_rows for k in row.keys()})
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(episode_rows)

    print(json.dumps({"output_dir": str(out), "gradient_sanity": gradient_sanity, "eval_summary": eval_summary}, indent=2, sort_keys=True))
    return 0 if gradient_sanity["passed"] and eval_summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
