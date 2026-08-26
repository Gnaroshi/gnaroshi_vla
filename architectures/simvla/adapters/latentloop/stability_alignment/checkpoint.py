"""Combined checkpoint and exact 30K-horizon optimizer schedule."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    CHECKPOINT_SCHEMA,
    SCHEDULER_HORIZON,
    scheduler_multiplier,
)
from architectures.simvla.adapters.latentloop.stability_alignment.model import (
    StabilityAlignedModules,
)
from architectures.simvla.adapters.latentloop.stability_alignment.age_encoding import (
    enable_conditional_kc8_age_support,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import NativeV0Config
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    GenerationLoopConfig,
)


class GroupWarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.optimizer_step = 0
        for group in optimizer.param_groups:
            if "name" not in group or "peak_lr" not in group:
                raise ValueError("optimizer group lacks name/peak_lr")
        self.set_step(0)

    def set_step(self, step: int) -> dict[str, float]:
        self.optimizer_step = int(step)
        multiplier = scheduler_multiplier(self.optimizer_step)
        result: dict[str, float] = {}
        for group in self.optimizer.param_groups:
            lr = float(group["peak_lr"]) * multiplier
            group["lr"] = lr
            result[str(group["name"])] = lr
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "simvla_condition_stability_30k_scheduler_v2",
            "horizon": SCHEDULER_HORIZON,
            "optimizer_step": self.optimizer_step,
            "groups": {
                str(group["name"]): float(group["peak_lr"])
                for group in self.optimizer.param_groups
            },
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != "simvla_condition_stability_30k_scheduler_v2":
            raise ValueError("scheduler schema changed")
        if int(payload.get("horizon", -1)) != SCHEDULER_HORIZON:
            raise ValueError("scheduler horizon changed")
        expected = {
            str(group["name"]): float(group["peak_lr"])
            for group in self.optimizer.param_groups
        }
        if payload.get("groups") != expected:
            raise ValueError("scheduler parameter groups changed")
        self.set_step(int(payload["optimizer_step"]))


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_checkpoint(
    path: str | Path,
    *,
    modules: StabilityAlignedModules,
    optimizer: torch.optim.Optimizer,
    scheduler: GroupWarmupCosine,
    optimizer_step: int,
    sampler_state: dict[str, Any],
    source_lock: dict[str, Any],
    training_contract: dict[str, Any],
    parent_identity: dict[str, Any],
) -> Path:
    return atomic_torch_save(
        {
            "checkpoint_format": CHECKPOINT_SCHEMA,
            "condition_state_dict": modules.condition.state_dict(),
            "generation_state_dict": modules.generation.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "optimizer_step": int(optimizer_step),
            "sampler_state": sampler_state,
            "source_lock": source_lock,
            "training_contract": training_contract,
            "parent_identity": parent_identity,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    modules: StabilityAlignedModules,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: GroupWarmupCosine | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != CHECKPOINT_SCHEMA:
        raise ValueError("stability checkpoint format changed")
    modules.condition.load_state_dict(payload["condition_state_dict"], strict=True)
    modules.generation.load_state_dict(payload["generation_state_dict"], strict=True)
    step = int(payload["optimizer_step"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload


def load_modules_from_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[StabilityAlignedModules, dict[str, Any]]:
    """Load a transfer bundle without requiring either parent file on rb2."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != CHECKPOINT_SCHEMA:
        raise ValueError("stability checkpoint format changed")
    condition = NativeV0Config().build().to(device)
    generation = GenerationLoopConfig().build().to(device)
    modules = StabilityAlignedModules(condition, generation).to(device)
    if any(
        key.endswith("condition_updater.age_embedding.source_weight")
        for key in payload["condition_state_dict"]
    ):
        enable_conditional_kc8_age_support(modules.condition.condition_updater)
    modules.condition.load_state_dict(payload["condition_state_dict"], strict=True)
    modules.generation.load_state_dict(payload["generation_state_dict"], strict=True)
    modules.eval()
    for parameter in modules.parameters():
        parameter.requires_grad_(False)
    return modules, payload
