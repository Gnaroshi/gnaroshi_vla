"""Shared deterministic runtime helpers for stability-alignment V3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment.model import (
    StabilityAlignedModules,
    generation_rollout,
    split_ages,
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
    "simvla/generation_eval_bundle_20260824_v1/checkpoint/"
    "generation_step_030000.pt"
)
DEFAULT_NORM = (
    "/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream/"
    "norm_stats/libero_norm.json"
)

V3_SOURCE_FILES = (
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_contracts.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_data.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_objectives.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_runtime.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_diagnostics.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_prepare.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_checkpoint.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_trainer.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_pipeline.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/v3_rb2_pipeline.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/online_eval.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/model.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/checkpoint.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/data.py",
    "architectures/simvla/adapters/latentloop/stability_alignment/objectives.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/generation_hidden.py",
    "methods/latentloop/modules/native_simvla_v0.py",
    "methods/latentloop/modules/simvla_generation_loop.py",
    "architectures/simvla/wrappers/run_sd1_simvla_stability_v3.sh",
    "architectures/simvla/wrappers/run_rb2_simvla_stability_v3.sh",
)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def distributed_pair() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "0"))
    if world != 2:
        raise RuntimeError("stability V3 requires exactly two torchrun ranks")
    physical = tuple(
        int(value)
        for value in os.environ.get("SIMVLA_GPU_IDS", "").split(",")
        if value
    )
    if len(physical) != 2:
        raise RuntimeError("SIMVLA_GPU_IDS must contain exactly two physical IDs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, physical)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal SIMVLA_GPU_IDS")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return dist.get_rank(), local_rank, world, torch.device(f"cuda:{local_rank}")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def configure_determinism(seed: int) -> None:
    seed_everything(seed)
    configure_strict_torch_determinism(int(seed))


def v3_source_lock(
    *,
    repository: str | Path,
    cache: str | Path,
    condition_parent: str | Path,
    generation_parent: str | Path,
    norm_stats: str | Path,
    checkpoint: str = DEFAULT_CHECKPOINT,
    smolvlm_model: str = DEFAULT_SMOLVLM,
    split_seed: int,
    training_seed: int,
) -> dict[str, Any]:
    root = Path(repository).expanduser().resolve()
    source_hashes: dict[str, str] = {}
    for relative in V3_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V3 source file missing: {path}")
        source_hashes[relative] = sha256_file(path)
    cache_root = Path(cache).expanduser().resolve()
    payload: dict[str, Any] = {
        "schema_version": "simvla_stability_v3_source_lock_v1",
        "source_files": source_hashes,
        "source_files_combined_sha256": canonical_sha256(source_hashes),
        "cache": str(cache_root),
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "condition_parent": str(Path(condition_parent).expanduser().resolve()),
        "condition_parent_sha256": sha256_file(condition_parent),
        "generation_parent": str(Path(generation_parent).expanduser().resolve()),
        "generation_parent_sha256": sha256_file(generation_parent),
        "norm_stats": str(Path(norm_stats).expanduser().resolve()),
        "norm_stats_sha256": sha256_file(norm_stats),
        "checkpoint": str(checkpoint),
        "smolvlm_model": str(smolvlm_model),
        "split_seed": int(split_seed),
        "training_seed": int(training_seed),
        "condition_ages": [1, 2, 3],
        "execution_horizon": 5,
        "action_horizon": 10,
        "generation_n_g": 3,
        "generation_change_code": "zero_128d",
        "scheduler_horizon": 30_000,
    }
    payload["combined_sha256"] = canonical_sha256(payload)
    return payload


def source_locked_indices(
    identities: Sequence[Any], *, count: int, seed: int
) -> tuple[int, ...]:
    if int(count) < 1 or int(count) > len(identities):
        raise ValueError("fixed audit count is outside the dataset")
    keyed = []
    for index, identity in enumerate(identities):
        encoded = json.dumps(
            [int(seed), identity], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        keyed.append((hashlib.sha256(encoded).hexdigest(), index))
    return tuple(index for _, index in sorted(keyed)[: int(count)])


def condition_updater_parameters(
    modules: StabilityAlignedModules,
) -> tuple[Tensor, ...]:
    values = tuple(
        parameter
        for parameter in modules.condition.condition_updater.parameters()
        if parameter.requires_grad
    )
    if not values:
        raise RuntimeError("Condition updater has no trainable parameters")
    return values


def flatten_gradients(
    loss: Tensor,
    parameters: Sequence[Tensor],
    *,
    retain_graph: bool,
) -> Tensor:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=bool(retain_graph),
        allow_unused=True,
    )
    pieces = [
        (
            gradient.detach().float().reshape(-1)
            if gradient is not None
            else parameter.detach().new_zeros(parameter.numel(), dtype=torch.float32)
        )
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(pieces)


def cosine(left: Tensor, right: Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator.item()) <= 1e-20:
        return float("nan")
    return float(torch.dot(left, right).div(denominator).item())


def pairwise_gradient_rows(
    vectors: Mapping[str, Tensor],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    names = tuple(vectors)
    return [
        {
            "scope": str(scope),
            "loss_a": left,
            "loss_b": right,
            "cosine": cosine(vectors[left], vectors[right]),
            "norm_a": float(vectors[left].norm().item()),
            "norm_b": float(vectors[right].norm().item()),
        }
        for left_index, left in enumerate(names)
        for right in names[left_index:]
    ]


def decode_full_actions(
    action_adapter: Any,
    *,
    conditions: Sequence[Tensor],
    proprio: Sequence[Tensor],
    noises: Sequence[Tensor],
    requires_grad: bool,
) -> tuple[Tensor, ...]:
    if not conditions or not (len(conditions) == len(proprio) == len(noises)):
        raise ValueError("full action decode inputs changed length")
    local = int(conditions[0].shape[0])
    if any(int(value.shape[0]) != local for value in conditions):
        raise ValueError("Condition batch sizes differ")
    action = action_adapter.decode_action_from_condition(
        torch.cat(tuple(conditions), dim=0),
        torch.cat(tuple(proprio), dim=0),
        steps=10,
        initial_noise=torch.cat(tuple(noises), dim=0),
        requires_grad=requires_grad,
    )
    return split_ages(action, local)


def zero_code_ng3_actions(
    *,
    modules: StabilityAlignedModules,
    frozen_model: Any,
    action_adapter: Any,
    conditions: Sequence[Tensor],
    proprio: Sequence[Tensor],
    noises: Sequence[Tensor],
    valid_mask: Tensor,
    optimizer_step: int,
    requires_grad: bool,
) -> tuple[Tensor, ...]:
    zeros = tuple(
        condition.new_zeros((condition.shape[0], 128)) for condition in conditions
    )
    return generation_rollout(
        updater=modules.generation,
        transformer=frozen_model.transformer,
        action_space=action_adapter.action_space,
        conditions=conditions,
        change_codes=zeros,
        proprio=proprio,
        noises=noises,
        valid_mask=valid_mask,
        optimizer_step=int(optimizer_step),
        requires_grad=requires_grad,
        instrument=False,
    ).actions


def summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("metric summary needs finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("weighted quantile inputs differ or are empty")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    pairs = sorted((float(value), float(weight)) for value, weight in zip(values, weights))
    if any(not math.isfinite(value) or not math.isfinite(weight) or weight < 0.0 for value, weight in pairs):
        raise ValueError("weighted quantile inputs must be finite and non-negative")
    total = sum(weight for _, weight in pairs)
    if total <= 0.0:
        raise ValueError("weighted quantile needs positive total weight")
    threshold = float(q) * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def all_reduce_mean(vector: Tensor) -> Tensor:
    result = vector.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result.div_(dist.get_world_size())
    return result


def write_rank_json(output: Path, rank: int, name: str, payload: Any) -> Path:
    path = output / "shards" / f"rank_{int(rank)}_{name}.json"
    atomic_write_json(path, payload)
    return path
