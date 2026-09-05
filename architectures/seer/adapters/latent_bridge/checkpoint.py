"""Strict bridge checkpoint serialization."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import torch

from .bridge import SeerFeatureBridge, SeerFeatureBridgeConfig
from .provenance import (
    OFFICIAL_COMMIT,
    OFFICIAL_MODEL_SHA256,
    expected_base_checkpoint_sha256,
    sha256_file,
)


FORMAT_VERSION = 1


def save_bridge_checkpoint(
    path: str | Path,
    model: SeerFeatureBridge,
    *,
    stage: str,
    epoch: int,
    optimizer=None,
    scheduler=None,
    metadata: dict | None = None,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "architecture": "seer_feature_bridge",
        "config": asdict(model.config),
        "stage": stage,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": sha256_file(path), "stage": stage, "epoch": int(epoch)}


def load_bridge_checkpoint(path: str | Path, *, map_location="cpu") -> tuple[SeerFeatureBridge, dict]:
    payload = torch.load(path, map_location=map_location)
    if payload.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(f"unsupported bridge checkpoint format: {payload.get('format_version')}")
    if payload.get("architecture") != "seer_feature_bridge":
        raise RuntimeError(f"not a Seer feature-bridge checkpoint: {path}")
    config = SeerFeatureBridgeConfig(**payload["config"])
    model = SeerFeatureBridge(config)
    status = model.load_state_dict(payload["model_state_dict"], strict=True)
    if status.missing_keys or status.unexpected_keys:
        raise RuntimeError(f"strict bridge load failed: {status}")
    return model, payload


def validate_bridge_runtime_provenance(payload: dict) -> dict:
    """Fail closed when a trained bridge is detached from its locked sources."""

    stage = payload.get("stage")
    if stage not in {"R0", "R1"}:
        raise RuntimeError(f"runtime bridge must be a trained R0/R1 checkpoint, got {stage!r}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("bridge checkpoint has no metadata mapping")
    official = metadata.get("official_source")
    if not isinstance(official, dict):
        raise RuntimeError("bridge checkpoint lacks official-source provenance")
    expected = {
        "commit": OFFICIAL_COMMIT,
        "model_sha256": OFFICIAL_MODEL_SHA256,
    }
    mismatches = {
        key: {"expected": value, "actual": official.get(key)}
        for key, value in expected.items()
        if official.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"bridge official-source provenance mismatch: {mismatches}")
    base_hash = metadata.get(
        "base_checkpoint_sha256", metadata.get("public_seer_checkpoint_sha256")
    )
    expected_base_hash = expected_base_checkpoint_sha256()
    if base_hash != expected_base_hash:
        raise RuntimeError(
            "bridge base-checkpoint provenance mismatch: "
            f"expected={expected_base_hash}, actual={base_hash}"
        )
    if not metadata.get("sync_files"):
        raise RuntimeError("bridge checkpoint does not identify synchronized training data")
    if stage == "R1" and not metadata.get("dagger_files"):
        raise RuntimeError("R1 bridge checkpoint does not identify DAgger training data")
    return {
        "stage": stage,
        "official_commit": official["commit"],
        "official_model_sha256": official["model_sha256"],
        "base_checkpoint_sha256": base_hash,
        "sync_file_count": len(metadata["sync_files"]),
        "dagger_file_count": len(metadata.get("dagger_files", [])),
    }
