"""Two-rank warm-start trainer and offline gates for stability alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    _drop_unused_vlm,
    collate_exact_teacher_sequences,
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
    move_batch,
)
from architectures.simvla.adapters.latentloop.stability_alignment.checkpoint import (
    GroupWarmupCosine,
    load_checkpoint,
    save_checkpoint,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    CONTRIBUTION_TARGETS,
    GRAD_CLIP_NORM,
    LOSS_SCHEMA,
    SCHEMA,
    atomic_write_json,
    canonical_sha256,
    condition_only_2k_continuation,
    evaluate_2k_gate,
    evaluate_10k_gate,
    k_offline_readiness,
    load_json,
    rotating_condition_age,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment.data import (
    ReplicatedEventAwareSampler,
    StabilityExactTeacherDataset,
    build_event_index,
)
from architectures.simvla.adapters.latentloop.stability_alignment.model import (
    StabilityAlignedModules,
    configure_condition_only_stage,
    generation_rollout,
    load_warm_start,
    optimizer_parameter_groups,
    split_ages,
    zero_code_parity,
)
from architectures.simvla.adapters.latentloop.stability_alignment.objectives import (
    LOSS_NAMES,
    calibrate_loss_weights,
    condition_paths,
    first_r_per_sequence,
    gripper_transition_loss,
    masked_nrms,
    stability_raw_losses,
    weighted_total,
)


DEFAULT_CACHE = (
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/"
    "simvla/latentloop/simvla_efficient_coupled_multirate_latentloop_sigfix_v1/"
    "03_exact_teacher_cache"
)
DEFAULT_CONDITION_50K = (
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/"
    "simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/"
    "checkpoints/native_v0_step_050000.pt"
)
DEFAULT_CONDITION_150K = (
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/"
    "simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/"
    "checkpoints/native_v0_step_150000.pt"
)
DEFAULT_GENERATION_30K = (
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/artifacts/"
    "simvla/generation_eval_bundle_20260824_v1/checkpoint/generation_step_030000.pt"
)
DEFAULT_NORM = (
    "/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream/"
    "norm_stats/libero_norm.json"
)

PARENT_PRESERVATION_PROBE_RELATIVE_NORM = 1e-3
PARENT_PRESERVATION_PROBE_SEED_OFFSET = 104_729


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def deterministic_parameter_probe(
    parameters: Sequence[Tensor],
    *,
    relative_norm: float,
    seed: int,
) -> Iterator[dict[str, Any]]:
    values = tuple(parameters)
    if not values or not 0.0 < float(relative_norm) <= 1e-2:
        raise ValueError("probe needs parameters and relative_norm in (0, 1e-2]")
    originals = tuple(value.detach().clone() for value in values)
    parameter_norm = torch.sqrt(
        sum(value.detach().float().square().sum() for value in values)
    )
    generator = torch.Generator(device=values[0].device)
    generator.manual_seed(int(seed))
    noises = tuple(
        torch.randn(
            value.shape,
            device=value.device,
            dtype=torch.float32,
            generator=generator,
        ).to(dtype=value.dtype)
        for value in values
    )
    noise_norm = torch.sqrt(sum(value.float().square().sum() for value in noises))
    scale = float(relative_norm) * parameter_norm / noise_norm.clamp_min(1e-12)
    with torch.no_grad():
        for value, noise in zip(values, noises):
            value.add_(noise * scale.to(dtype=value.dtype))
    observed = torch.sqrt(
        sum(
            (value.detach().float() - original.float()).square().sum()
            for value, original in zip(values, originals)
        )
    ) / parameter_norm.clamp_min(1e-12)
    audit: dict[str, Any] = {
        "relative_norm_requested": float(relative_norm),
        "relative_norm_observed": float(observed.item()),
        "seed": int(seed),
        "restored_exactly": False,
    }
    try:
        yield audit
    finally:
        with torch.no_grad():
            for value, original in zip(values, originals):
                value.copy_(original)
        audit["restored_exactly"] = all(
            bool(torch.equal(value.detach(), original))
            for value, original in zip(values, originals)
        )


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size != 2:
        raise RuntimeError("stability training requires exactly two torchrun ranks")
    selected = tuple(
        int(value) for value in os.environ.get("SIMVLA_GPU_IDS", "").split(",") if value
    )
    if len(selected) != 2:
        raise RuntimeError("SIMVLA_GPU_IDS must contain exactly two physical GPU IDs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, selected)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must exactly match SIMVLA_GPU_IDS")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _source_lock(args: argparse.Namespace) -> dict[str, Any]:
    cache_manifest = Path(args.cache).expanduser().resolve() / "manifest.json"
    scientific = {
        "schema_version": "simvla_condition_stability_runtime_source_lock_v2",
        "condition_parent": str(Path(args.condition_parent).expanduser().resolve()),
        "condition_parent_sha256": sha256_file(args.condition_parent),
        "generation_parent": str(Path(args.generation_parent).expanduser().resolve()),
        "generation_parent_sha256": sha256_file(args.generation_parent),
        "exact_cache": str(Path(args.cache).expanduser().resolve()),
        "exact_cache_manifest_sha256": sha256_file(cache_manifest),
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "checkpoint": str(args.checkpoint),
        "smolvlm_model": str(args.smolvlm_model),
        "split_seed": int(args.split_seed),
        "training_seed": int(args.seed),
        "action_horizon": 10,
        "execution_horizon": 5,
        "condition_ages": [1, 2, 3],
        "n_g": 3,
        "full_generation_indices": [0, 4, 8],
        "training_mode": "condition_only_frozen_generation",
        "generation_local_oracle_loss": False,
        "full_nfe_training_age": "deterministic rotating 1,2,3",
        "effective_unique_global_batch": 1,
    }
    scientific["combined_sha256"] = canonical_sha256(scientific)
    return scientific


def _assert_parent_contract(payloads: dict[str, Any]) -> dict[str, Any]:
    condition = payloads["condition_payload"]
    generation = payloads["generation_payload"]
    condition_step = int(condition.get("global_optimizer_step", -1))
    if condition_step not in {50_000, 150_000}:
        raise RuntimeError("Condition parent must be the fixed 50K or 150K checkpoint")
    if int(generation.get("optimizer_step", -1)) != 30_000:
        raise RuntimeError("Generation parent must be the validated 30K checkpoint")
    return {
        "condition_optimizer_step": condition_step,
        "condition_source_sha256": condition["source_lock"]["combined_sha256"],
        "generation_optimizer_step": 30_000,
        "generation_source_sha256": generation["source_lock"]["combined_sha256"],
    }


def _dataset_and_index(
    args: argparse.Namespace,
    *,
    split: str,
    output: Path,
    rank: int,
) -> tuple[StabilityExactTeacherDataset, dict[str, Any] | None]:
    dataset = StabilityExactTeacherDataset(
        args.cache, split=split, split_seed=args.split_seed
    )
    if split != "train":
        return dataset, None
    event_path = output / "event_index.json"
    if rank == 0 and not event_path.exists():
        atomic_write_json(event_path, build_event_index(dataset))
    dist.barrier()
    event_index = load_json(event_path)
    if event_index.get("split_sha256") != dataset.split_sha256:
        raise RuntimeError("event index uses another training split")
    return dataset, event_index


def _loader(
    dataset: StabilityExactTeacherDataset,
    sampler: ReplicatedEventAwareSampler,
    *,
    num_workers: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 1,
        "sampler": sampler,
        "collate_fn": collate_exact_teacher_sequences,
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if num_workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def _full_actions(
    action_adapter: Any,
    conditions: Sequence[Tensor],
    batch: Mapping[str, Any],
    *,
    requires_grad: bool,
) -> tuple[Tensor, ...]:
    local = int(conditions[0].shape[0])
    age_count = len(conditions)
    condition = torch.cat(tuple(conditions), dim=0)
    proprio = torch.cat(
        tuple(batch["proprio_sequence"][:, age] for age in range(1, age_count + 1)), dim=0
    )
    noise = torch.cat(
        tuple(batch["explicit_noises"][:, age - 1] for age in range(1, age_count + 1)), dim=0
    )
    action = action_adapter.decode_action_from_condition(
        condition,
        proprio,
        steps=10,
        initial_noise=noise,
        requires_grad=requires_grad,
    )
    return split_ages(action, local)


def _full_actions_at_age(
    action_adapter: Any,
    conditions: Sequence[Tensor],
    batch: Mapping[str, Any],
    *,
    age_index: int,
    requires_grad: bool,
) -> tuple[Tensor, ...]:
    """Decode multiple Condition paths at one deterministic age in one batch."""

    if not conditions:
        raise ValueError("at least one rotating-age Condition is required")
    local = int(conditions[0].shape[0])
    if any(int(value.shape[0]) != local for value in conditions):
        raise ValueError("rotating-age Conditions changed local batch size")
    condition = torch.cat(tuple(conditions), dim=0)
    proprio_value = batch["proprio_sequence"][:, int(age_index) + 1]
    noise_value = batch["explicit_noises"][:, int(age_index)]
    proprio = torch.cat(tuple(proprio_value for _ in conditions), dim=0)
    noise = torch.cat(tuple(noise_value for _ in conditions), dim=0)
    action = action_adapter.decode_action_from_condition(
        condition,
        proprio,
        steps=10,
        initial_noise=noise,
        requires_grad=requires_grad,
    )
    return split_ages(action, local)


def _forward(
    *,
    modules: StabilityAlignedModules,
    parent_condition: torch.nn.Module,
    parent_generation: torch.nn.Module,
    frozen_model: Any,
    action_adapter: Any,
    batch: Mapping[str, Any],
    optimizer_step: int,
    requires_grad: bool,
    instrument: bool = False,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, float]]:
    timings: dict[str, float] = {}

    def tick(name: str, started: float) -> None:
        if instrument:
            torch.cuda.synchronize()
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started

    if instrument:
        torch.cuda.synchronize()
    started = time.perf_counter()
    paths = condition_paths(modules.condition, batch)
    with torch.no_grad():
        parent_paths = condition_paths(parent_condition, batch)
    tick("condition_updater", started)
    age_count = int(batch["teacher_conditions"].shape[1])
    exact_conditions = tuple(
        batch["teacher_conditions"][:, index] for index in range(age_count)
    )
    exact_actions = tuple(batch["teacher_actions"][:, index] for index in range(age_count))

    started = time.perf_counter()
    rotating_age = rotating_condition_age(optimizer_step)
    rotating_index = rotating_age - 1
    rotating_teacher, rotating_recursive = _full_actions_at_age(
        action_adapter,
        (paths.teacher_forced[rotating_index], paths.recursive[rotating_index]),
        batch,
        age_index=rotating_index,
        requires_grad=requires_grad,
    )
    tick("rotating_full_nfe10", started)

    proprio = tuple(batch["proprio_sequence"][:, age] for age in range(1, age_count + 1))
    noises = tuple(batch["explicit_noises"][:, age - 1] for age in range(1, age_count + 1))
    zero_codes = tuple(
        value.new_zeros((value.shape[0], 128)) for value in exact_conditions
    )
    started = time.perf_counter()
    joint = generation_rollout(
        updater=modules.generation,
        transformer=frozen_model.transformer,
        action_space=action_adapter.action_space,
        conditions=paths.recursive,
        change_codes=zero_codes,
        proprio=proprio,
        noises=noises,
        valid_mask=batch["valid_mask"],
        optimizer_step=optimizer_step,
        requires_grad=requires_grad,
        instrument=instrument,
    )
    tick("frozen_ng3_student", started)
    if instrument:
        timings["ng3_student"] = joint.student_seconds

    started = time.perf_counter()
    with torch.no_grad():
        parent = generation_rollout(
            updater=parent_generation,
            transformer=frozen_model.transformer,
            action_space=action_adapter.action_space,
            conditions=parent_paths.recursive[:2],
            change_codes=zero_codes[:2],
            proprio=proprio[:2],
            noises=noises[:2],
            valid_mask=batch["valid_mask"],
            optimizer_step=optimizer_step,
            requires_grad=False,
            instrument=False,
        )
    tick("parent_preservation", started)

    raw, diagnostics = stability_raw_losses(
        paths=paths,
        parent_paths=parent_paths,
        exact_conditions=exact_conditions,
        rotating_recursive_full_action=rotating_recursive,
        rotating_teacher_forced_action=rotating_teacher,
        rotating_age_index=rotating_index,
        joint_actions=joint.actions,
        parent_joint_actions=parent.actions,
        exact_actions=exact_actions,
        valid_mask=batch["valid_mask"],
    )
    return raw, diagnostics, timings


def _allreduce_gradients(module: torch.nn.Module) -> None:
    world = dist.get_world_size()
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def _weights(path: str | Path, source_lock: dict[str, Any]) -> dict[str, float]:
    payload = load_json(path)
    if payload.get("schema_version") != LOSS_SCHEMA:
        raise ValueError("loss-weight schema changed")
    if payload.get("source_contract_sha256") != source_lock["combined_sha256"]:
        raise RuntimeError("loss weights use another branch source contract")
    weights = {name: float(payload["weights"][name]) for name in LOSS_NAMES}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("loss weights must be finite and non-negative")
    return weights


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _wandb(args: argparse.Namespace, rank: int, config: dict[str, Any]) -> Any | None:
    if rank != 0 or not args.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or Path(args.output).name,
        dir=args.output,
        config=config,
    )


def command_calibrate(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, _, device = _distributed()
    try:
        _seed(args.seed + rank)
        configure_strict_torch_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"refusing existing calibration output: {output}")
            output.mkdir(parents=True)
        dist.barrier()
        source = _source_lock(args)
        modules, parent_condition, parent_generation, payloads = load_warm_start(
            condition_checkpoint=args.condition_parent,
            generation_checkpoint=args.generation_parent,
            device=device,
        )
        parent_identity = _assert_parent_contract(payloads)
        stage_audit = configure_condition_only_stage(modules)
        parity = zero_code_parity(parent_generation, modules.generation, device=device)
        if parity["verdict"] != "ZERO_CODE_PARENT_PARITY_PASS":
            raise RuntimeError(json.dumps(parity, indent=2, sort_keys=True))
        frozen_model, processor, action_adapter = load_frozen_simvla(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            device=device,
        )
        del processor
        _drop_unused_vlm(frozen_model)
        dataset, event_index = _dataset_and_index(
            args, split="train", output=output, rank=rank
        )
        sampler = ReplicatedEventAwareSampler(
            event_index,
            seed=args.seed,
            start_step=0,
            stop_step=args.calibration_samples,
        )
        loader = _loader(dataset, sampler, num_workers=args.num_workers)
        raw_values = {name: [] for name in LOSS_NAMES}
        grad_values = {name: [] for name in LOSS_NAMES}
        trainable = [value for value in modules.parameters() if value.requires_grad]
        exact_scale_names = tuple(
            name for name in LOSS_NAMES if name != "parent_preservation"
        )
        probe_audits: list[dict[str, Any]] = []
        for step, host_batch in enumerate(loader):
            batch = move_batch(host_batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                raw, _, _ = _forward(
                    modules=modules,
                    parent_condition=parent_condition,
                    parent_generation=parent_generation,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    optimizer_step=step,
                    requires_grad=True,
                )
            for index, name in enumerate(exact_scale_names):
                gradients = torch.autograd.grad(
                    raw[name],
                    trainable,
                    retain_graph=index < len(exact_scale_names) - 1,
                    allow_unused=True,
                )
                norm = torch.sqrt(
                    sum(
                        gradient.detach().float().square().sum()
                        for gradient in gradients
                        if gradient is not None
                    )
                )
                raw_values[name].append(float(raw[name].detach().item()))
                grad_values[name].append(float(norm.item()))
            with deterministic_parameter_probe(
                trainable,
                relative_norm=PARENT_PRESERVATION_PROBE_RELATIVE_NORM,
                seed=args.seed + PARENT_PRESERVATION_PROBE_SEED_OFFSET,
            ) as probe_audit:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    probe_raw, _, _ = _forward(
                        modules=modules,
                        parent_condition=parent_condition,
                        parent_generation=parent_generation,
                        frozen_model=frozen_model,
                        action_adapter=action_adapter,
                        batch=batch,
                        optimizer_step=step,
                        requires_grad=True,
                    )
                probe_gradients = torch.autograd.grad(
                    probe_raw["parent_preservation"],
                    trainable,
                    allow_unused=True,
                )
                probe_norm = torch.sqrt(
                    sum(
                        gradient.detach().float().square().sum()
                        for gradient in probe_gradients
                        if gradient is not None
                    )
                )
                raw_values["parent_preservation"].append(
                    float(probe_raw["parent_preservation"].detach().item())
                )
                grad_values["parent_preservation"].append(float(probe_norm.item()))
            probe_audits.append(dict(probe_audit))
        if not all(value["restored_exactly"] for value in probe_audits):
            raise RuntimeError("parent-preservation calibration probe was not restored")
        raw_means = {name: float(np.mean(values)) for name, values in raw_values.items()}
        grad_means = {name: float(np.mean(values)) for name, values in grad_values.items()}
        calibrated = calibrate_loss_weights(
            raw_means, grad_means, CONTRIBUTION_TARGETS
        )
        parameter_norm = math.sqrt(
            sum(
                float(value.detach().float().square().sum().item())
                for value in trainable
            )
        )
        weighted_gradient = sum(calibrated["weighted_gradient_norms"].values())
        candidates = (1e-5, 3e-5, 1e-4)
        eligible = [
            value
            for value in candidates
            if value * weighted_gradient / max(parameter_norm, 1e-12) <= 1e-3
        ]
        base_lr = max(eligible) if eligible else min(candidates)
        result = {
            "schema_version": LOSS_SCHEMA,
            "verdict": "STABILITY_LOSS_CALIBRATION_COMPLETE",
            "source_contract_sha256": source["combined_sha256"],
            "branch_condition_parent_step": parent_identity["condition_optimizer_step"],
            "calibration_samples": int(args.calibration_samples),
            "parent_preservation_probe": {
                "relative_parameter_norm": PARENT_PRESERVATION_PROBE_RELATIVE_NORM,
                "seed": args.seed + PARENT_PRESERVATION_PROBE_SEED_OFFSET,
                "observed_relative_norm_min": min(
                    value["relative_norm_observed"] for value in probe_audits
                ),
                "observed_relative_norm_max": max(
                    value["relative_norm_observed"] for value in probe_audits
                ),
                "restored_exactly": True,
                "losses_probed": ["parent_preservation"],
                "other_losses_measured_at_exact_warm_start": True,
            },
            **calibrated,
            "base_corrective_lr": base_lr,
            "base_lr_candidates": list(candidates),
            "predicted_relative_update": base_lr
            * weighted_gradient
            / max(parameter_norm, 1e-12),
            "parameter_group_lr_ratios": {
                "condition_updater": 1.0,
                "observation_change_encoder": 0.40,
            },
            "training_mode": "condition_only_frozen_generation",
            "stage_audit": stage_audit,
            "zero_code_parity": parity,
        }
        if rank == 0:
            atomic_write_json(output / "source_lock.json", source)
            atomic_write_json(output / "stability_alignment_loss_weights.json", result)
        return result
    finally:
        _cleanup_distributed()


def command_merge_calibrations(args: argparse.Namespace) -> dict[str, Any]:
    s50 = load_json(args.s50)
    s150 = load_json(args.s150)
    if s50.get("schema_version") != LOSS_SCHEMA or s150.get("schema_version") != LOSS_SCHEMA:
        raise ValueError("branch calibration schema changed")
    probe_keys = (
        "relative_parameter_norm",
        "seed",
        "restored_exactly",
        "losses_probed",
        "other_losses_measured_at_exact_warm_start",
    )
    if any(
        s50.get("parent_preservation_probe", {}).get(key)
        != s150.get("parent_preservation_probe", {}).get(key)
        for key in probe_keys
    ):
        raise ValueError("S50/S150 parent-preservation probe contracts differ")
    raw = {
        name: math.sqrt(float(s50["raw_means"][name]) * float(s150["raw_means"][name]))
        for name in LOSS_NAMES
    }
    gradients = {
        name: max(float(s50["gradient_norms"][name]), float(s150["gradient_norms"][name]))
        for name in LOSS_NAMES
    }
    calibrated = calibrate_loss_weights(raw, gradients, CONTRIBUTION_TARGETS)
    base_lr = min(float(s50["base_corrective_lr"]), float(s150["base_corrective_lr"]))
    result = {
        "schema_version": LOSS_SCHEMA,
        "verdict": "COMMON_S50_S150_LOSS_WEIGHTS_FROZEN",
        "source_contract_sha256_by_branch": {
            "S50": s50["source_contract_sha256"],
            "S150": s150["source_contract_sha256"],
        },
        "source_contract_sha256": "BRANCH_SPECIFIC_CHECK_REQUIRED",
        **calibrated,
        "base_corrective_lr": base_lr,
        "parameter_group_lr_ratios": s50["parameter_group_lr_ratios"],
        "parent_preservation_probe_by_branch": {
            "S50": s50["parent_preservation_probe"],
            "S150": s150["parent_preservation_probe"],
        },
        "training_mode": "condition_only_frozen_generation",
        "frozen_before_optimizer_step_zero": True,
    }
    atomic_write_json(args.output, result)
    return result


def _branch_weights(path: str | Path, source: dict[str, Any]) -> dict[str, float]:
    payload = load_json(path)
    accepted = {
        payload.get("source_contract_sha256"),
        *payload.get("source_contract_sha256_by_branch", {}).values(),
    }
    if source["combined_sha256"] not in accepted:
        raise RuntimeError("frozen weights do not include this branch source lock")
    return {name: float(payload["weights"][name]) for name in LOSS_NAMES}


def _prepare_training(
    args: argparse.Namespace,
    *,
    rank: int,
    device: torch.device,
    output: Path,
    resume: bool,
) -> tuple[Any, ...]:
    source = _source_lock(args)
    modules, parent_condition, parent_generation, payloads = load_warm_start(
        condition_checkpoint=args.condition_parent,
        generation_checkpoint=args.generation_parent,
        device=device,
    )
    parent_identity = _assert_parent_contract(payloads)
    weights = _branch_weights(args.loss_weights, source)
    weight_payload = load_json(args.loss_weights)
    base_lr = float(weight_payload["base_corrective_lr"])
    groups = optimizer_parameter_groups(
        modules, base_lr=base_lr, weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(groups)
    scheduler = GroupWarmupCosine(optimizer)
    start_step = 0
    if resume:
        payload = load_checkpoint(
            args.resume, modules=modules, optimizer=optimizer, scheduler=scheduler
        )
        start_step = int(payload["optimizer_step"])
        if payload["source_lock"]["combined_sha256"] != source["combined_sha256"]:
            raise RuntimeError("resume source lock changed")
    stage_audit = configure_condition_only_stage(modules)
    scheduler.set_step(start_step)
    parity = zero_code_parity(parent_generation, modules.generation, device=device)
    if start_step == 0 and parity["verdict"] != "ZERO_CODE_PARENT_PARITY_PASS":
        raise RuntimeError("zero-code Generation parent parity failed")
    frozen_model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    del processor
    dropped = _drop_unused_vlm(frozen_model)
    dataset, event_index = _dataset_and_index(
        args, split="train", output=output, rank=rank
    )
    sampler = ReplicatedEventAwareSampler(
        event_index,
        seed=args.seed,
        start_step=start_step,
        stop_step=args.stop_step,
    )
    training_contract = {
        "schema_version": SCHEMA,
        "source_combined_sha256": source["combined_sha256"],
        "condition_parent_step": parent_identity["condition_optimizer_step"],
        "generation_parent_step": 30_000,
        "start_step": start_step,
        "stop_step": int(args.stop_step),
        "schedule_horizon": 30_000,
        "warmup_steps": 1_500,
        "final_lr_ratio": 0.1,
        "effective_unique_global_batch": 1,
        "physical_replicas": 2,
        "event_sampler": "75pct natural + 25pct gripper transition",
        "dataset_contract": dataset.contract(),
        "loss_weights": weights,
        "base_corrective_lr": base_lr,
        "stage_audit": stage_audit,
        "original_simvla_frozen": True,
        "generation_updater_frozen": True,
        "condition_code_projection_frozen": True,
        "generation_condition_change_code": "zero; exact validated Generation parent lane",
        "optional_joint_branch": "not automatic; real c_j projection requires four evidence gates",
        "generation_local_oracle_loss": False,
        "rotating_full_nfe_age_cycle": [1, 2, 3],
        "n_g": 3,
        "full_generation_indices": [0, 4, 8],
        "frozen_release": dropped,
    }
    return (
        source,
        modules,
        parent_condition,
        parent_generation,
        parent_identity,
        weights,
        optimizer,
        scheduler,
        start_step,
        frozen_model,
        action_adapter,
        dataset,
        sampler,
        training_contract,
    )


def benchmark_numerical_stability(
    *,
    total_losses: Sequence[float],
    gradient_norms: Sequence[float],
    weighted_fractions: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    if not total_losses or len(total_losses) != len(gradient_norms):
        raise ValueError("benchmark numerical gate needs aligned loss and gradient rows")
    if set(weighted_fractions) != set(LOSS_NAMES):
        raise ValueError("benchmark weighted-fraction names changed")
    if any(len(values) != len(total_losses) for values in weighted_fractions.values()):
        raise ValueError("benchmark weighted-fraction lengths changed")
    finite = all(
        math.isfinite(value)
        for value in (
            *total_losses,
            *gradient_norms,
            *(item for values in weighted_fractions.values() for item in values),
        )
    )
    total_summary = _summary(total_losses)
    gradient_summary = _summary(gradient_norms)
    fraction_summaries = {
        name: _summary(values) for name, values in weighted_fractions.items()
    }
    clipping_fraction = sum(
        value > GRAD_CLIP_NORM for value in gradient_norms
    ) / len(gradient_norms)
    mean_fraction_sum = sum(
        float(summary["mean"]) for summary in fraction_summaries.values()
    )
    active_losses = {
        name: float(fraction_summaries[name]["mean"])
        >= float(CONTRIBUTION_TARGETS[name]) * 0.01
        for name in LOSS_NAMES
    }
    checks = {
        "all_values_finite": finite,
        "mean_contribution_sum_is_one": math.isclose(
            mean_fraction_sum, 1.0, rel_tol=0.0, abs_tol=1e-5
        ),
        "all_losses_active": all(active_losses.values()),
        "parent_mean_at_most_5_percent": float(
            fraction_summaries["parent_preservation"]["mean"]
        )
        <= 0.05,
        "parent_p95_at_most_25_percent": float(
            fraction_summaries["parent_preservation"]["p95"]
        )
        <= 0.25,
        "total_loss_p95_at_most_10": float(total_summary["p95"]) <= 10.0,
        "gradient_clipping_below_90_percent": clipping_fraction < 0.90,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "active_losses": active_losses,
        "total_loss": total_summary,
        "gradient_norm_before_clip": gradient_summary,
        "gradient_clipping_fraction": clipping_fraction,
        "weighted_contribution_fraction": fraction_summaries,
    }


def command_train_or_benchmark(
    args: argparse.Namespace, *, benchmark: bool
) -> dict[str, Any]:
    rank, _, _, device = _distributed()
    try:
        _seed(args.seed + rank)
        determinism = configure_strict_torch_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        resume = bool(args.resume)
        if rank == 0:
            if output.exists() and not resume:
                raise FileExistsError(f"refusing existing output: {output}")
            output.mkdir(parents=True, exist_ok=resume)
        dist.barrier()
        prepared = _prepare_training(
            args, rank=rank, device=device, output=output, resume=resume
        )
        (
            source,
            modules,
            parent_condition,
            parent_generation,
            parent_identity,
            weights,
            optimizer,
            scheduler,
            start_step,
            frozen_model,
            action_adapter,
            dataset,
            sampler,
            training_contract,
        ) = prepared
        if start_step == 2_000:
            if not args.safety_gate:
                raise ValueError("2K->10K resume requires --safety-gate")
            safety = load_json(args.safety_gate)
            continuation = condition_only_2k_continuation(safety)
            if not continuation["passed"]:
                raise RuntimeError("2K Condition-only safety checks did not pass")
            configure_condition_only_stage(modules)
        if start_step not in {0, 2_000, 10_000}:
            raise RuntimeError("resume boundary must be optimizer step 0, 2K, or 10K")
        if args.stop_step <= start_step or args.stop_step > 30_000:
            raise ValueError("invalid stop step for fixed 30K scheduler")
        loader = _loader(dataset, sampler, num_workers=args.num_workers)
        if rank == 0:
            atomic_write_json(output / "source_lock.json", source)
            atomic_write_json(output / "training_contract.json", training_contract)
            atomic_write_json(output / "determinism.json", determinism)
            atomic_write_json(output / "parameter_audit.json", modules.parameter_audit())
        run = _wandb(args, rank, training_contract)
        progress = tqdm(
            total=args.stop_step,
            initial=start_step,
            disable=rank != 0,
            dynamic_ncols=True,
            desc=f"Stability S{parent_identity['condition_optimizer_step']//1000}",
        )
        timing_values: dict[str, list[float]] = {
            "data_load": [],
            "condition_updater": [],
            "rotating_full_nfe10": [],
            "frozen_ng3_student": [],
            "ng3_student": [],
            "parent_preservation": [],
            "backward": [],
            "optimizer": [],
            "total_step": [],
        }
        total_loss_values: list[float] = []
        gradient_norm_values: list[float] = []
        weighted_fraction_values: dict[str, list[float]] = {
            name: [] for name in LOSS_NAMES
        }
        utilization_values: list[float] = []
        iterator = iter(loader)
        latest: dict[str, Any] = {}
        wall_started = time.perf_counter()
        for zero_step in range(start_step, args.stop_step):
            step_started = time.perf_counter()
            data_started = time.perf_counter()
            host_batch = next(iterator)
            batch = move_batch(host_batch, device)
            torch.cuda.synchronize()
            timing_values["data_load"].append(time.perf_counter() - data_started)
            optimizer.zero_grad(set_to_none=True)
            benchmark_offset = zero_step - start_step
            lrs = scheduler.set_step(zero_step + 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                raw, diagnostics, forward_timings = _forward(
                    modules=modules,
                    parent_condition=parent_condition,
                    parent_generation=parent_generation,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    optimizer_step=zero_step,
                    requires_grad=True,
                    instrument=True,
                )
                total, weighted = weighted_total(raw, weights)
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(f"non-finite loss at step {zero_step + 1}")
            backward_started = time.perf_counter()
            total.backward()
            _allreduce_gradients(modules)
            if any(value.grad is not None for value in frozen_model.parameters()):
                raise RuntimeError("frozen original SimVLA received gradients")
            if any(value.grad is not None for value in modules.generation.parameters()):
                raise RuntimeError("frozen validated Generation updater received gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [value for value in modules.parameters() if value.requires_grad],
                GRAD_CLIP_NORM,
            )
            total_scalar = float(total.detach().item())
            gradient_norm_values.append(float(grad_norm.item()))
            total_loss_values.append(total_scalar)
            for name, value in weighted.items():
                weighted_fraction_values[name].append(
                    float(value.detach().item()) / max(total_scalar, 1e-12)
                )
            torch.cuda.synchronize()
            timing_values["backward"].append(time.perf_counter() - backward_started)
            optimizer_started = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            timing_values["optimizer"].append(time.perf_counter() - optimizer_started)
            for name, value in forward_timings.items():
                timing_values[name].append(value)
            timing_values["total_step"].append(time.perf_counter() - step_started)
            if benchmark and benchmark_offset % 20 == 0:
                physical = os.environ["SIMVLA_GPU_IDS"].split(",")[rank]
                observed = subprocess.check_output(
                    (
                        "nvidia-smi",
                        f"--id={physical}",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ),
                    text=True,
                ).strip()
                utilization_values.append(float(observed))
            step = zero_step + 1
            latest = {
                "step": step,
                "total": float(total.detach().item()),
                "grad_norm_before_clip": float(grad_norm.item()),
                **{f"raw/{name}": float(value.detach().item()) for name, value in raw.items()},
                **{f"weighted/{name}": float(value.detach().item()) for name, value in weighted.items()},
                **{f"diagnostic/{name}": float(value.detach().item()) for name, value in diagnostics.items()},
                **{f"lr/{name}": value for name, value in lrs.items()},
            }
            if rank == 0:
                progress.update(1)
                progress.set_postfix(
                    loss=f"{latest['total']:.4g}",
                    rec=f"{latest['raw/recursive_stability']:.4g}",
                    exec=f"{latest['raw/end_to_end_execution']:.4g}",
                )
                if step == 1 or step % args.log_interval == 0 or step == args.stop_step:
                    _write_jsonl(output / "train_metrics.jsonl", latest)
                    if run is not None:
                        run.log(latest, step=step)
            if benchmark and step - start_step >= args.benchmark_steps:
                break
            if not benchmark and (step % args.save_interval == 0 or step == args.stop_step):
                if rank == 0:
                    checkpoint = output / "checkpoints" / f"stability_step_{step:06d}.pt"
                    save_checkpoint(
                        checkpoint,
                        modules=modules,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        optimizer_step=step,
                        sampler_state=sampler.state_dict(step),
                        source_lock=source,
                        training_contract=training_contract,
                        parent_identity=parent_identity,
                    )
                    (output / "latest_checkpoint.txt").write_text(
                        str(checkpoint) + "\n", encoding="utf-8"
                    )
                dist.barrier()
        progress.close()
        elapsed = time.perf_counter() - wall_started
        timing_summary = {
            name: {
                "mean_seconds": float(np.mean(values)),
                "p95_seconds": float(np.quantile(values, 0.95)),
            }
            for name, values in timing_values.items()
            if values
        }
        mean_step = timing_summary["total_step"]["mean_seconds"]
        numerical_stability = benchmark_numerical_stability(
            total_losses=total_loss_values,
            gradient_norms=gradient_norm_values,
            weighted_fractions=weighted_fraction_values,
        )
        result = {
            "verdict": "STABILITY_500_STEP_BENCHMARK_COMPLETE" if benchmark else "STABILITY_TRAINING_SEGMENT_COMPLETE",
            "optimizer_step": min(
                args.stop_step,
                start_step + args.benchmark_steps if benchmark else args.stop_step,
            ),
            "elapsed_seconds": elapsed,
            "mean_step_seconds": mean_step,
            "projected_10k_seconds": mean_step * 10_000,
            "projected_10k_hours": mean_step * 10_000 / 3600.0,
            "practical_budget_hours": float(args.practical_budget_hours),
            "within_practical_budget": mean_step * 10_000 / 3600.0
            <= float(args.practical_budget_hours),
            "numerical_stability": numerical_stability,
            "timing": timing_summary,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "device": torch.cuda.get_device_name(device),
            "mean_gpu_utilization_percent": (
                float(np.mean(utilization_values)) if utilization_values else None
            ),
            "latest_metrics": latest,
            "source_combined_sha256": source["combined_sha256"],
        }
        if rank == 0:
            filename = "speed_benchmark.json" if benchmark else "run_summary.json"
            atomic_write_json(output / filename, result)
            if run is not None:
                run.finish()
        return result
    finally:
        _cleanup_distributed()


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("cannot summarize empty metrics")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _condition_and_action_rows(
    *,
    modules: StabilityAlignedModules,
    parent_condition: torch.nn.Module,
    parent_generation: torch.nn.Module,
    frozen_model: Any,
    action_adapter: Any,
    batch: Mapping[str, Any],
    dataset_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    age_count = int(batch["teacher_conditions"].shape[1])
    exact = tuple(batch["teacher_conditions"][:, index] for index in range(age_count))
    targets = tuple(batch["teacher_actions"][:, index] for index in range(age_count))
    proprio = tuple(batch["proprio_sequence"][:, age] for age in range(1, age_count + 1))
    noises = tuple(batch["explicit_noises"][:, age - 1] for age in range(1, age_count + 1))
    zeros = tuple(value.new_zeros((value.shape[0], 128)) for value in exact)
    with torch.no_grad():
        candidate_paths = condition_paths(modules.condition, batch)
        parent_paths = condition_paths(parent_condition, batch)
        candidate_tf = _full_actions(action_adapter, candidate_paths.teacher_forced, batch, requires_grad=False)
        candidate_recursive = _full_actions(action_adapter, candidate_paths.recursive, batch, requires_grad=False)
        parent_tf = _full_actions(action_adapter, parent_paths.teacher_forced, batch, requires_grad=False)
        parent_recursive = _full_actions(action_adapter, parent_paths.recursive, batch, requires_grad=False)
        candidate_joint = generation_rollout(
            updater=modules.generation,
            transformer=frozen_model.transformer,
            action_space=action_adapter.action_space,
            conditions=candidate_paths.recursive,
            change_codes=candidate_paths.change_codes,
            proprio=proprio,
            noises=noises,
            valid_mask=batch["valid_mask"],
            optimizer_step=dataset_index,
            requires_grad=False,
            instrument=False,
        ).actions
        parent_joint = generation_rollout(
            updater=parent_generation,
            transformer=frozen_model.transformer,
            action_space=action_adapter.action_space,
            conditions=parent_paths.recursive,
            change_codes=zeros,
            proprio=proprio,
            noises=noises,
            valid_mask=batch["valid_mask"],
            optimizer_step=dataset_index,
            requires_grad=False,
            instrument=False,
        ).actions
        candidate_exact = generation_rollout(
            updater=modules.generation,
            transformer=frozen_model.transformer,
            action_space=action_adapter.action_space,
            conditions=exact,
            change_codes=zeros,
            proprio=proprio,
            noises=noises,
            valid_mask=batch["valid_mask"],
            optimizer_step=dataset_index,
            requires_grad=False,
            instrument=False,
        ).actions
        parent_exact = generation_rollout(
            updater=parent_generation,
            transformer=frozen_model.transformer,
            action_space=action_adapter.action_space,
            conditions=exact,
            change_codes=zeros,
            proprio=proprio,
            noises=noises,
            valid_mask=batch["valid_mask"],
            optimizer_step=dataset_index,
            requires_grad=False,
            instrument=False,
        ).actions
    for index, age in enumerate(range(1, age_count + 1)):
        target_sign = targets[index][:, :5, 6] >= 0
        candidate_sign = candidate_joint[index][:, :5, 6] >= 0
        parent_sign = parent_joint[index][:, :5, 6] >= 0
        rows.append(
            {
                "dataset_index": dataset_index,
                "task_id": int(batch["task_id"].item()),
                "episode_id": str(batch["episode_id"][0]),
                "age": age,
                "candidate_condition_nrms": float(masked_nrms(candidate_paths.recursive[index], exact[index], batch["valid_mask"]).item()),
                "parent_condition_nrms": float(masked_nrms(parent_paths.recursive[index], exact[index], batch["valid_mask"]).item()),
                "candidate_teacher_first_r": float(first_r_per_sequence(candidate_tf[index], targets[index]).item()),
                "parent_teacher_first_r": float(first_r_per_sequence(parent_tf[index], targets[index]).item()),
                "candidate_recursive_first_r": float(first_r_per_sequence(candidate_recursive[index], targets[index]).item()),
                "parent_recursive_first_r": float(first_r_per_sequence(parent_recursive[index], targets[index]).item()),
                "candidate_recurrence_excess": float((first_r_per_sequence(candidate_recursive[index], targets[index]) - first_r_per_sequence(candidate_tf[index], targets[index])).item()),
                "parent_recurrence_excess": float((first_r_per_sequence(parent_recursive[index], targets[index]) - first_r_per_sequence(parent_tf[index], targets[index])).item()),
                "candidate_joint_first_r": float(first_r_per_sequence(candidate_joint[index], targets[index]).item()),
                "parent_joint_first_r": float(first_r_per_sequence(parent_joint[index], targets[index]).item()),
                "candidate_exact_ng3_first_r": float(first_r_per_sequence(candidate_exact[index], targets[index]).item()),
                "parent_exact_ng3_first_r": float(first_r_per_sequence(parent_exact[index], targets[index]).item()),
                "candidate_gripper_sign_mismatch": int((candidate_sign != target_sign).sum().item()),
                "parent_gripper_sign_mismatch": int((parent_sign != target_sign).sum().item()),
            }
        )
    return rows


def _aggregate_offline(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_age: dict[str, Any] = {}
    fields = tuple(
        key for key in rows[0] if key not in {"dataset_index", "task_id", "episode_id", "age"}
    )
    for age in (1, 2, 3):
        selected = [row for row in rows if int(row["age"]) == age]
        by_age[str(age)] = {
            field: _summary([float(row[field]) for row in selected]) for field in fields
        }
    def mean(age: int, field: str) -> float:
        return float(by_age[str(age)][field]["mean"])
    def p95(age: int, field: str) -> float:
        return float(by_age[str(age)][field]["p95"])
    def p99(age: int, field: str) -> float:
        return float(by_age[str(age)][field]["p99"])
    epsilon = 1e-12
    metrics = {
        "frozen_base_gradients_zero": True,
        "age1_first_r_ratio_to_parent": mean(1, "candidate_joint_first_r") / max(mean(1, "parent_joint_first_r"), epsilon),
        "exact_ng3_ratio_to_parent": sum(mean(age, "candidate_exact_ng3_first_r") for age in (1,2,3)) / max(sum(mean(age, "parent_exact_ng3_first_r") for age in (1,2,3)), epsilon),
        "no_gripper_collapse": sum(mean(age, "candidate_gripper_sign_mismatch") for age in (1,2,3)) <= 1.25 * max(sum(mean(age, "parent_gripper_sign_mismatch") for age in (1,2,3)), 1.0),
        "p99_ratio_to_parent": p99(3, "candidate_recursive_first_r") / max(p99(3, "parent_recursive_first_r"), epsilon),
        "stability_slope": float("nan"),
        "age2_recurrence_improvement": (mean(2, "parent_recurrence_excess") - mean(2, "candidate_recurrence_excess")) / max(abs(mean(2, "parent_recurrence_excess")), epsilon),
        "age3_recurrence_improvement": (mean(3, "parent_recurrence_excess") - mean(3, "candidate_recurrence_excess")) / max(abs(mean(3, "parent_recurrence_excess")), epsilon),
        "age3_gripper_sign_improvement": (mean(3, "parent_gripper_sign_mismatch") - mean(3, "candidate_gripper_sign_mismatch")) / max(mean(3, "parent_gripper_sign_mismatch"), 1.0),
        "age3_first_r_p95_ratio": p95(3, "candidate_recursive_first_r") / max(p95(3, "parent_recursive_first_r"), epsilon),
        "teacher_forced_first_r_ratio": sum(mean(age, "candidate_teacher_first_r") for age in (1,2,3)) / max(sum(mean(age, "parent_teacher_first_r") for age in (1,2,3)), epsilon),
        "age1_final_system_ratio": mean(1, "candidate_joint_first_r") / max(mean(1, "parent_joint_first_r"), epsilon),
        "no_p99_or_gripper_collapse": p99(3, "candidate_recursive_first_r") <= 1.25 * p99(3, "parent_recursive_first_r") and sum(mean(age, "candidate_gripper_sign_mismatch") for age in (1,2,3)) <= 1.25 * max(sum(mean(age, "parent_gripper_sign_mismatch") for age in (1,2,3)), 1.0),
        "original_simvla_frozen": True,
    }
    return {"by_age": by_age, "gate_metrics": metrics}


def _training_stability_slope(candidate: str | Path) -> float:
    metrics_path = Path(candidate).expanduser().resolve().parents[1] / "train_metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"training metrics not found: {metrics_path}")
    points: list[tuple[float, float]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "raw/recursive_stability" in payload:
            points.append((float(payload["step"]), float(payload["raw/recursive_stability"])))
    if len(points) < 4:
        raise RuntimeError("at least four logged recursive-stability points are required")
    start = max(0, int(0.8 * len(points)))
    selected = points[start:]
    x = np.asarray([item[0] for item in selected], dtype=np.float64)
    y = np.asarray([item[1] for item in selected], dtype=np.float64)
    return float(np.polyfit(x, y, deg=1)[0])


def command_offline(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, world, device = _distributed()
    try:
        _seed(args.seed + rank)
        configure_strict_torch_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"refusing existing offline output: {output}")
            output.mkdir(parents=True)
        dist.barrier()
        modules, parent_condition, parent_generation, payloads = load_warm_start(
            condition_checkpoint=args.condition_parent,
            generation_checkpoint=args.generation_parent,
            device=device,
        )
        payload = load_checkpoint(args.candidate, modules=modules)
        freeze_module(modules)
        freeze_module(parent_condition)
        frozen_model, processor, action_adapter = load_frozen_simvla(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            device=device,
        )
        del processor
        _drop_unused_vlm(frozen_model)
        dataset = StabilityExactTeacherDataset(
            args.cache, split=args.offline_split, split_seed=args.split_seed
        )
        rows: list[dict[str, Any]] = []
        for dataset_index in tqdm(
            range(rank, len(dataset), world),
            desc=f"offline rank{rank}",
            dynamic_ncols=True,
        ):
            batch = move_batch(
                collate_exact_teacher_sequences([dataset[dataset_index]]), device
            )
            rows.extend(
                _condition_and_action_rows(
                    modules=modules,
                    parent_condition=parent_condition,
                    parent_generation=parent_generation,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    dataset_index=dataset_index,
                )
            )
        shard_path = output / f"rank_{rank}_rows.json"
        atomic_write_json(shard_path, rows)
        dist.barrier()
        result: dict[str, Any] = {}
        if rank == 0:
            merged: list[dict[str, Any]] = []
            for shard in range(world):
                merged.extend(load_json(output / f"rank_{shard}_rows.json"))
            merged.sort(key=lambda row: (int(row["dataset_index"]), int(row["age"])))
            with (output / "recursive_stability_metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
                writer.writeheader()
                writer.writerows(merged)
            aggregate = _aggregate_offline(merged)
            aggregate["gate_metrics"]["stability_slope"] = _training_stability_slope(
                args.candidate
            )
            step = int(payload["optimizer_step"])
            gate = (
                evaluate_2k_gate(aggregate["gate_metrics"])
                if step == 2_000
                else evaluate_10k_gate(aggregate["gate_metrics"])
            )
            readiness = {}
            if step >= 10_000:
                age_pass = {
                    1: aggregate["gate_metrics"]["age1_final_system_ratio"] <= 1.05,
                    2: aggregate["gate_metrics"]["age2_recurrence_improvement"] >= 0.20,
                    3: aggregate["gate_metrics"]["age3_recurrence_improvement"] >= 0.30 and aggregate["gate_metrics"]["age3_first_r_p95_ratio"] < 1.0,
                }
                readiness = {
                    "kc3": k_offline_readiness(3, age_pass),
                    "kc4": k_offline_readiness(4, age_pass),
                }
            result = {
                "verdict": gate.verdict,
                "optimizer_step": step,
                "split": args.offline_split,
                "dataset_contract": dataset.contract(),
                "aggregate": aggregate,
                "gate": gate.to_dict(),
                "readiness": readiness,
                "candidate_sha256": sha256_file(args.candidate),
            }
            atomic_write_json(output / "offline_gate.json", result)
        dist.barrier()
        return result
    finally:
        _cleanup_distributed()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(value: argparse.ArgumentParser) -> None:
        value.add_argument("--output", required=True)
        value.add_argument("--cache", default=DEFAULT_CACHE)
        value.add_argument("--condition-parent", required=True)
        value.add_argument("--generation-parent", default=DEFAULT_GENERATION_30K)
        value.add_argument("--norm-stats", default=DEFAULT_NORM)
        value.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
        value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
        value.add_argument("--split-seed", type=int, default=20260822)
        value.add_argument("--seed", type=int, default=20260825)
        value.add_argument("--num-workers", type=int, default=2)
        value.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)

    calibrate = subparsers.add_parser("calibrate")
    common(calibrate)
    calibrate.add_argument("--calibration-samples", type=int, default=8)

    merge = subparsers.add_parser("merge-calibrations")
    merge.add_argument("--s50", required=True)
    merge.add_argument("--s150", required=True)
    merge.add_argument("--output", required=True)

    for name in ("benchmark", "train"):
        value = subparsers.add_parser(name)
        common(value)
        value.add_argument("--loss-weights", required=True)
        value.add_argument("--resume", default="")
        value.add_argument("--safety-gate", default="")
        value.add_argument("--stop-step", type=int, required=True)
        value.add_argument("--benchmark-steps", type=int, default=500)
        value.add_argument("--practical-budget-hours", type=float, default=12.0)
        value.add_argument("--weight-decay", type=float, default=0.0)
        value.add_argument("--max-grad-norm", type=float, default=1.0)
        value.add_argument("--log-interval", type=int, default=20)
        value.add_argument("--save-interval", type=int, default=2_000)
        value.add_argument("--wandb-project", default="")
        value.add_argument("--wandb-name", default="")

    offline = subparsers.add_parser("offline")
    common(offline)
    offline.add_argument("--candidate", required=True)
    offline.add_argument(
        "--offline-split",
        choices=("checkpoint_validation", "final_offline"),
        required=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "calibrate":
        result = command_calibrate(args)
    elif args.command == "merge-calibrations":
        result = command_merge_calibrations(args)
    elif args.command == "benchmark":
        result = command_train_or_benchmark(args, benchmark=True)
    elif args.command == "train":
        result = command_train_or_benchmark(args, benchmark=False)
    elif args.command == "offline":
        result = command_offline(args)
    else:
        raise AssertionError(args.command)
    if not dist.is_available() or not dist.is_initialized():
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
