"""Load SimVLA and its SmolVLM dependency from pinned local HF snapshots."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch


def _snapshot(source_lock: dict[str, Any], key: str) -> Path:
    snapshot = Path(str(source_lock[key].get("snapshot_path", "")))
    if not snapshot.is_dir():
        raise FileNotFoundError(f"source-locked {key} snapshot is not cached: {snapshot}")
    return snapshot


def load_source_locked_simvla(
    model_class: Any,
    source_lock: dict[str, Any],
    *,
    device: torch.device,
) -> Any:
    """Prevent both outer SimVLA and nested SmolVLM loads from resolving `main`."""

    model_snapshot = _snapshot(source_lock, "checkpoint")
    smolvlm_snapshot = _snapshot(source_lock, "processor_checkpoint")
    config = model_class.config_class.from_pretrained(
        str(model_snapshot), local_files_only=True
    )
    config.smolvlm_model_path = str(smolvlm_snapshot)
    return model_class.from_pretrained(
        str(model_snapshot),
        config=config,
        local_files_only=True,
    ).to(device)


def load_source_locked_processor(
    processor_class: Any,
    source_lock: dict[str, Any],
) -> Any:
    """Load the processor without allowing its upstream remote-fallback branch."""

    snapshot = str(_snapshot(source_lock, "processor_checkpoint"))
    parameters = inspect.signature(processor_class).parameters
    if "smolvlm_model_path" in parameters:
        return processor_class(smolvlm_model_path=snapshot)
    return processor_class.from_pretrained(snapshot, local_files_only=True)
