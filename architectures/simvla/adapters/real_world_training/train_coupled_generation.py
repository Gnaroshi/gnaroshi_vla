"""Projection-only real Condition-to-Generation coupling for K_C=2, N_G=3."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from architectures.simvla.adapters.dcld.simvla_action_adapter import (
    SimVLAActionAdapter,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    audit_projection_only_state,
    condition_update_with_code,
    prepare_projection_only_coupling,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_objective import (
    generation_local_oracle_loss,
)
from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop

from .artifact_validation import (
    validate_real_baseline_checkpoint,
    validate_real_training_sources,
)
from .dataset import align_current_rotvec_proprio
from .distributed import initialize_distributed, seed_process
from .io_utils import atomic_write_json, sha256_file, stable_int_seed
from .model_io import load_exact_official_model
from .updater_data import RealConditionPairDataset
from .updater_io import (
    RealGenerationConfig,
    load_real_updater,
    save_real_coupled_generation,
)


FULL_STEP_INDICES = (0, 4, 8)


def _learning_rate(
    step: int, *, total: int, warmup: int, peak: float, floor_ratio: float
) -> float:
    if step < warmup:
        return peak * float(step + 1) / float(max(warmup, 1))
    progress = float(step - warmup) / float(max(total - warmup, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return peak * (floor_ratio + (1.0 - floor_ratio) * cosine)


def _layout(
    cache_manifest: dict[str, Any], batch: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = cache_manifest["token_layout"]
    valid = torch.as_tensor(payload["valid_mask"][0], dtype=torch.bool, device=device)
    groups = torch.as_tensor(payload["group_ids"][0], dtype=torch.long, device=device)
    return valid.unsqueeze(0).expand(batch, -1), groups.unsqueeze(0).expand(batch, -1)


def _noise(
    indices: Iterable[int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    rows = []
    for index in indices:
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_int_seed("real-generation", int(index), 20260904))
        rows.append(
            torch.randn((10, 7), generator=generator, device=device, dtype=dtype)
        )
    return torch.stack(rows)


class _AdapterHolder:
    num_actions = 10

    def __init__(self, transformer: torch.nn.Module, action_space: Any) -> None:
        self.transformer = transformer
        self.action_space = action_space

    def eval(self) -> "_AdapterHolder":
        self.transformer.eval()
        return self


@torch.no_grad()
def _coupled_query(
    condition_updater: torch.nn.Module,
    batch: dict[str, Any],
    cache_manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    previous = batch["previous_condition"].to(device, non_blocking=True)
    exact_current = batch["current_condition"].to(device, non_blocking=True)
    previous_proprio = batch["previous_proprio"].to(device, non_blocking=True)
    current_proprio = batch["current_proprio"].to(device, non_blocking=True)
    updater_current_proprio = align_current_rotvec_proprio(
        previous_proprio, current_proprio
    )
    valid, groups = _layout(cache_manifest, previous.shape[0], device)
    exposed = condition_update_with_code(
        condition_updater,
        previous,
        NativeV0ObservationPair(
            previous_images=batch["previous_images"].to(device, non_blocking=True),
            current_images=batch["current_images"].to(device, non_blocking=True),
            previous_proprio=previous_proprio,
            current_proprio=updater_current_proprio,
        ),
        valid_mask=valid,
        group_ids=groups,
        age=1,
    )
    code_norm = exposed.condition_change_code.float().norm(dim=-1)
    if not bool((code_norm > 0).all()):
        raise RuntimeError("real Condition Updater produced a zero change code")
    return {
        "predicted_condition": exposed.update.condition,
        "exact_condition": exact_current,
        "condition_change_code": exposed.condition_change_code,
        "condition_valid_mask": valid,
        "proprio": current_proprio,
        "initial_noise": _noise(
            batch["current_cache_index"].tolist(), device, current_proprio.dtype
        ),
        "code_norm": code_norm,
    }


def _generation_result(
    *,
    updater: torch.nn.Module,
    transformer: torch.nn.Module,
    action_adapter: SimVLAActionAdapter,
    query: dict[str, torch.Tensor],
    condition_code: torch.Tensor,
) -> Any:
    with torch.no_grad():
        exact_teacher = action_adapter.decode_action_from_condition(
            query["exact_condition"],
            query["proprio"],
            steps=10,
            initial_noise=query["initial_noise"],
        )
    loop = SimVLAGenerationLoop(updater, transformer.action_decoder)
    return generation_local_oracle_loss(
        loop=loop,
        transformer=transformer,
        action_space=action_adapter.action_space,
        condition=query["predicted_condition"],
        initial_noise=query["initial_noise"],
        normalized_proprio=action_adapter.normalize_proprio(query["proprio"]),
        condition_valid_mask=query["condition_valid_mask"],
        condition_change_code=condition_code,
        full_step_indices=FULL_STEP_INDICES,
        teacher_final_action=exact_teacher,
        hidden_weight=1.0,
        velocity_weight=0.0,
        final_action_weight=0.0,
    )


@torch.no_grad()
def _validate(
    *,
    candidate: torch.nn.Module,
    parent: torch.nn.Module,
    condition_updater: torch.nn.Module,
    transformer: torch.nn.Module,
    action_adapter: SimVLAActionAdapter,
    loader: DataLoader,
    cache_manifest: dict[str, Any],
    device: torch.device,
    batches: int,
) -> dict[str, float]:
    collected: dict[str, list[float]] = {
        "coupled_hidden_normalized_mse": [],
        "uncoupled_hidden_normalized_mse": [],
        "coupled_final_action_l1_to_exact_condition_teacher": [],
        "uncoupled_final_action_l1_to_exact_condition_teacher": [],
        "coupled_vs_uncoupled_action_l1": [],
        "condition_change_code_norm": [],
    }
    for number, batch in enumerate(loader):
        if number >= batches:
            break
        query = _coupled_query(condition_updater, batch, cache_manifest, device)
        coupled = _generation_result(
            updater=candidate,
            transformer=transformer,
            action_adapter=action_adapter,
            query=query,
            condition_code=query["condition_change_code"],
        )
        uncoupled = _generation_result(
            updater=parent,
            transformer=transformer,
            action_adapter=action_adapter,
            query=query,
            condition_code=torch.zeros_like(query["condition_change_code"]),
        )
        coupled_action = action_adapter.action_space.postprocess(
            coupled.trace.final_noisy_action
        )
        uncoupled_action = action_adapter.action_space.postprocess(
            uncoupled.trace.final_noisy_action
        )
        collected["coupled_hidden_normalized_mse"].append(
            float(coupled.hidden_normalized_mse.item())
        )
        collected["uncoupled_hidden_normalized_mse"].append(
            float(uncoupled.hidden_normalized_mse.item())
        )
        collected["coupled_final_action_l1_to_exact_condition_teacher"].append(
            float(coupled.final_action_l1.item())
        )
        collected["uncoupled_final_action_l1_to_exact_condition_teacher"].append(
            float(uncoupled.final_action_l1.item())
        )
        collected["coupled_vs_uncoupled_action_l1"].append(
            float(F.l1_loss(coupled_action.float(), uncoupled_action.float()).item())
        )
        collected["condition_change_code_norm"].append(
            float(query["code_norm"].mean().item())
        )
    if not all(collected.values()):
        raise RuntimeError("coupled validation loader yielded no batches")
    return {name: sum(values) / len(values) for name, values in collected.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    context = initialize_distributed(args.device)
    if context.world_size != 1:
        context.close()
        raise RuntimeError("real coupled projection training requires exactly one GPU")
    seed_process(args.seed, 0)
    try:
        output = Path(args.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)
        source_contract = validate_real_training_sources(
            condition_cache=args.condition_cache,
            checkpoint=args.checkpoint,
            processor=args.processor,
            norm_stats=args.norm_stats,
            verify_cache_array_checksums=False,
            condition_cache_attestation=args.condition_cache_attestation,
        )
        cache_manifest = source_contract["condition_cache"]
        cache_manifest_path = Path(args.condition_cache).expanduser().resolve() / "manifest.json"
        norm_sha = sha256_file(args.norm_stats)
        baseline_sha = sha256_file(args.baseline_action_checkpoint)
        dataset_identity = str(cache_manifest["dataset_identity_sha256"])
        cache_identity = str(cache_manifest["condition_cache_identity_sha256"])
        validate_real_baseline_checkpoint(
            args.baseline_action_checkpoint,
            source=source_contract,
            expected_optimizer_step=3000,
        )

        expected = {
            "expected_baseline_sha256": baseline_sha,
            "expected_norm_sha256": norm_sha,
            "expected_dataset_identity_sha256": dataset_identity,
            "expected_cache_identity_sha256": cache_identity,
            "expected_cache_attestation_identity_sha256": source_contract[
                "condition_cache_attestation"
            ]["attestation_identity_sha256"],
            "expected_optimizer_step": 10_000,
        }
        condition_updater, _ = load_real_updater(
            args.condition_updater_checkpoint,
            kind="condition",
            device=context.device,
            **expected,
        )
        parent, parent_payload = load_real_updater(
            args.parent_generation_checkpoint,
            kind="generation",
            device=context.device,
            **expected,
        )
        for module in (condition_updater, parent):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        model, _, loading = load_exact_official_model(
            model_directory=args.checkpoint,
            processor_directory=args.processor,
            norm_stats=args.norm_stats,
            device="cpu",
            real_action_checkpoint=args.baseline_action_checkpoint,
            freeze_vlm=True,
            freeze_action_transformer=True,
            expected_dataset_identity_sha256=dataset_identity,
            expected_cache_identity_sha256=cache_identity,
            expected_cache_attestation_identity_sha256=source_contract[
                "condition_cache_attestation"
            ]["attestation_identity_sha256"],
            expected_real_action_optimizer_step=3000,
        )
        transformer = model.transformer.to(context.device).eval()
        action_space = model.action_space.to(context.device)
        del model.vlm
        gc.collect()
        if context.device.type == "cuda":
            torch.cuda.empty_cache()
        for parameter in transformer.parameters():
            parameter.requires_grad_(False)
        action_adapter = SimVLAActionAdapter(_AdapterHolder(transformer, action_space))

        config_values = dict(parent_payload["model_config"])
        config_values["full_step_indices"] = tuple(config_values["full_step_indices"])
        config = RealGenerationConfig(**config_values)
        candidate, _ = load_real_updater(
            args.parent_generation_checkpoint,
            kind="generation",
            device=context.device,
            **expected,
        )
        prepare_projection_only_coupling(candidate)

        trainable_names = [
            name for name, parameter in candidate.named_parameters() if parameter.requires_grad
        ]
        trainable = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
        if trainable_names != ["condition_code_projection.weight"] or sum(
            parameter.numel() for parameter in trainable
        ) != 16_384:
            raise RuntimeError("real coupling is not projection-only")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=context.device.type == "cuda",
        )

        train_data = RealConditionPairDataset(args.condition_cache, split="train")
        validation_data = RealConditionPairDataset(args.condition_cache, split="validation")
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=args.num_workers,
            pin_memory=context.device.type == "cuda",
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 2),
            pin_memory=context.device.type == "cuda",
        )
        training_config = {
            "protocol": "real_kc2_ng3_projection_only_coupling",
            "condition_refresh_interval": 2,
            "generation_full_evaluations": 3,
            "full_step_indices": list(FULL_STEP_INDICES),
            "condition_change_code": "condition_updater_delta_encoder",
            "trainable_parameter_names": trainable_names,
            "trainable_parameters": 16_384,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "minimum_lr_ratio": args.minimum_lr_ratio,
            "weight_decay": args.weight_decay,
            "loss": "student_state_local_oracle_hidden_normalized_mse",
            "exact_condition_teacher_action": "monitor_only",
            "condition_cache_array_sha256": {
                name: spec["sha256"] for name, spec in cache_manifest["arrays"].items()
            },
            "condition_cache_attestation_identity_sha256": source_contract[
                "condition_cache_attestation"
            ]["attestation_identity_sha256"],
            "seed": args.seed,
            "exact_initialization": loading,
        }
        atomic_write_json(output / "training_config.json", training_config)

        progress = tqdm(
            total=args.max_steps,
            initial=0,
            desc="real SimVLA coupled projection",
        )
        iterator = iter(train_loader)
        metrics_path = output / "train_metrics.jsonl"
        started = time.perf_counter()
        last_validation: dict[str, float] = {}
        step = 0
        while step < args.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            query = _coupled_query(
                condition_updater, batch, cache_manifest, context.device
            )
            optimizer.zero_grad(set_to_none=True)
            result = _generation_result(
                updater=candidate,
                transformer=transformer,
                action_adapter=action_adapter,
                query=query,
                condition_code=query["condition_change_code"],
            )
            result.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            lr = _learning_rate(
                step,
                total=args.max_steps,
                warmup=args.warmup_steps,
                peak=args.learning_rate,
                floor_ratio=args.minimum_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            step += 1
            progress.update(1)
            progress.set_postfix(
                hidden=f"{float(result.hidden_normalized_mse.item()):.4g}",
                code=f"{float(query['code_norm'].mean().item()):.3g}",
            )

            if step % args.log_interval == 0 or step == args.max_steps:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "hidden_normalized_mse": float(
                                    result.hidden_normalized_mse.item()
                                ),
                                "velocity_l1_monitor": float(result.velocity_l1.item()),
                                "exact_condition_teacher_action_l1_monitor": float(
                                    result.final_action_l1.item()
                                ),
                                "condition_change_code_norm": float(
                                    query["code_norm"].mean().item()
                                ),
                                "gradient_norm": float(grad_norm),
                                "learning_rate": lr,
                                "elapsed_s": time.perf_counter() - started,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

            if step % args.save_interval == 0 or step == args.max_steps:
                last_validation = _validate(
                    candidate=candidate,
                    parent=parent,
                    condition_updater=condition_updater,
                    transformer=transformer,
                    action_adapter=action_adapter,
                    loader=validation_loader,
                    cache_manifest=cache_manifest,
                    device=context.device,
                    batches=args.validation_batches,
                )
                checkpoint = (
                    output / "checkpoints" / f"coupled_generation_step_{step:06d}.pt"
                )
                shared = {
                    "updater": candidate,
                    "config": config,
                    "parent_generation_checkpoint": args.parent_generation_checkpoint,
                    "condition_updater_checkpoint": args.condition_updater_checkpoint,
                    "condition_cache_manifest": cache_manifest_path,
                    "optimizer_step": step,
                    "training_config": training_config,
                    "validation": last_validation,
                }
                save_real_coupled_generation(checkpoint, **shared)
                (output / "latest_checkpoint.txt").write_text(
                    str(checkpoint) + "\n", encoding="utf-8"
                )
                old = sorted((output / "checkpoints").glob("coupled_generation_step_*.pt"))
                for stale in old[: max(0, len(old) - args.keep_checkpoints)]:
                    stale.unlink()
        progress.close()

        state_audit = audit_projection_only_state(parent, candidate)
        if state_audit["verdict"] != "PROJECTION_ONLY_STATE_PASS":
            raise RuntimeError(json.dumps(state_audit, indent=2, sort_keys=True))
        summary = {
            "verdict": "REAL_COUPLED_GENERATION_COMPLETE",
            "optimizer_steps": step,
            "trainable_parameters": 16_384,
            "parent_generation_checkpoint_sha256": sha256_file(
                args.parent_generation_checkpoint
            ),
            "condition_updater_checkpoint_sha256": sha256_file(
                args.condition_updater_checkpoint
            ),
            "projection_only_state_audit": state_audit,
            "validation": last_validation,
            "elapsed_s": time.perf_counter() - started,
        }
        atomic_write_json(output / "projection_only_state_audit.json", state_audit)
        atomic_write_json(output / "run_summary.json", summary)
        return summary
    finally:
        context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-cache", required=True)
    parser.add_argument("--condition-cache-attestation", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--baseline-action-checkpoint", required=True)
    parser.add_argument("--condition-updater-checkpoint", required=True)
    parser.add_argument("--parent-generation-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    result = train(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
