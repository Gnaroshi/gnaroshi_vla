"""Checkpoint format for recurrence-focused stability V3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.stability_alignment.checkpoint import (
    GroupWarmupCosine,
    atomic_torch_save,
)
from architectures.simvla.adapters.latentloop.stability_alignment.model import (
    StabilityAlignedModules,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_contracts import (
    V3_CHECKPOINT_SCHEMA,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import NativeV0Config
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    GenerationLoopConfig,
)
from architectures.simvla.adapters.latentloop.stability_alignment.age_encoding import (
    enable_conditional_kc8_age_support,
)


def save_v3_checkpoint(
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
    loss_weight_contract: dict[str, Any],
    moving_window_audit: dict[str, Any],
) -> Path:
    return atomic_torch_save(
        {
            "checkpoint_format": V3_CHECKPOINT_SCHEMA,
            "condition_state_dict": modules.condition.state_dict(),
            "generation_state_dict": modules.generation.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "optimizer_step": int(optimizer_step),
            "sampler_state": sampler_state,
            "source_lock": source_lock,
            "training_contract": training_contract,
            "parent_identity": parent_identity,
            "loss_weight_contract": loss_weight_contract,
            "moving_window_audit": moving_window_audit,
        },
        path,
    )


def load_v3_checkpoint(
    path: str | Path,
    *,
    modules: StabilityAlignedModules,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: GroupWarmupCosine | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != V3_CHECKPOINT_SCHEMA:
        raise ValueError("stability V3 checkpoint format changed")
    modules.condition.load_state_dict(payload["condition_state_dict"], strict=True)
    modules.generation.load_state_dict(payload["generation_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload


def load_v3_modules(
    path: str | Path, *, device: torch.device | str
) -> tuple[StabilityAlignedModules, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != V3_CHECKPOINT_SCHEMA:
        raise ValueError("stability V3 checkpoint format changed")
    modules = StabilityAlignedModules(
        NativeV0Config().build().to(device),
        GenerationLoopConfig().build().to(device),
    ).to(device)
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
