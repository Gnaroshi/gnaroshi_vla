"""Versioned checkpoints for the SimVLA Latent Bridge adaptation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .model import SimVLALatentBridge, SimVLALatentBridgeConfig
from .provenance import (
    latent_bridge_source_manifest,
    simvla_latent_bridge_integration_manifest,
)


CHECKPOINT_SCHEMA = "simvla_latent_bridge_v1"


def save_bridge_checkpoint(
    path: str | Path,
    model: SimVLALatentBridge,
    *,
    provenance: dict[str, Any],
    training: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config": model.config.serializable(),
        "model_state_dict": model.state_dict(),
        "parameter_audit": model.parameter_audit(),
        "provenance": provenance,
        "training": training,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_bridge_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
    official_upstream_root: str | None = None,
    verify_official_source: bool = True,
) -> tuple[SimVLALatentBridge, dict[str, Any]]:
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location=device, weights_only=False
    )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported Latent Bridge checkpoint: {payload.get('schema_version')}"
        )
    if verify_official_source:
        expected = payload.get("provenance", {}).get("latent_bridge")
        if not isinstance(expected, dict) or "combined_sha256" not in expected:
            raise RuntimeError("checkpoint lacks pinned Latent Bridge source provenance")
        current = latent_bridge_source_manifest(official_upstream_root)
        if current["combined_sha256"] != expected["combined_sha256"]:
            raise RuntimeError("checkpoint and runtime Latent Bridge source differ")
        expected_integration = payload.get("provenance", {}).get("integration")
        if (
            not isinstance(expected_integration, dict)
            or "combined_sha256" not in expected_integration
        ):
            raise RuntimeError("checkpoint lacks SimVLA Latent Bridge integration provenance")
        current_integration = simvla_latent_bridge_integration_manifest()
        if (
            current_integration["combined_sha256"]
            != expected_integration["combined_sha256"]
        ):
            raise RuntimeError("checkpoint and runtime SimVLA Latent Bridge integration differ")
    config = SimVLALatentBridgeConfig(**payload["config"])
    model = SimVLALatentBridge(
        config, official_upstream_root=official_upstream_root
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
