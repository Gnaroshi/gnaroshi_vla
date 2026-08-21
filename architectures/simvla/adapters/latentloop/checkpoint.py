"""Adapter-only checkpoint serialization for SimVLA LatentLoop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .condition_adapter import LatentLoopAdapterConfig, SimVLAChunkAwareAdapter


CHECKPOINT_TYPE = "simvla_chunkaware_latentloop_adapter_v1"


def save_adapter_checkpoint(
    path: str | Path,
    *,
    adapter: SimVLAChunkAwareAdapter,
    step: int,
    metadata: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically save adapter parameters and optional resumable training state."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "step": int(step),
        "adapter_config": adapter.config.to_dict(),
        "adapter_state_dict": adapter.state_dict(),
        "metadata": metadata,
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if training_state is not None:
        payload["training_state"] = training_state
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_adapter_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SimVLAChunkAwareAdapter, dict[str, Any]]:
    """Load an adapter and reject legacy PM047M checkpoint formats."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(
            f"Expected {CHECKPOINT_TYPE}; legacy DCLD/PM047M checkpoints are not accepted"
        )
    config = LatentLoopAdapterConfig(**payload["adapter_config"])
    adapter = SimVLAChunkAwareAdapter(config).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    return adapter, payload


def freeze_module(module: nn.Module) -> None:
    """Put a module in eval mode and disable gradients for all parameters."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def trainable_parameter_names(module: nn.Module) -> list[str]:
    """Return sorted optimizer-eligible parameter names."""

    return sorted(name for name, parameter in module.named_parameters() if parameter.requires_grad)
