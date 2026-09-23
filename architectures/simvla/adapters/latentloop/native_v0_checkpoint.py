"""Strict checkpoint I/O for corrected native SimVLA V0."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0


CHECKPOINT_FORMAT = "simvla_correct_native_v0_v1"


@dataclass(frozen=True)
class NativeV0Config:
    num_views: int = 2
    proprio_dim: int = 8
    condition_dim: int = 960
    condition_tokens: int = 122
    delta_dim: int = 128
    rank_dim: int = 64
    max_tokens: int = 256
    num_token_groups: int = 8
    fixed_k: int = 4
    action_horizon: int = 10
    execution_horizon: int = 5

    def validate_primary(self) -> None:
        expected = {
            "num_views": 2,
            "proprio_dim": 8,
            "condition_dim": 960,
            "condition_tokens": 122,
            "rank_dim": 64,
            "fixed_k": 4,
            "action_horizon": 10,
            "execution_horizon": 5,
        }
        actual = asdict(self)
        mismatches = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
        if mismatches:
            raise ValueError(f"not the primary corrected SimVLA V0 contract: {mismatches}")

    def build(self) -> NativeSimVLAV0:
        return NativeSimVLAV0(
            num_views=self.num_views,
            proprio_dim=self.proprio_dim,
            condition_dim=self.condition_dim,
            delta_dim=self.delta_dim,
            rank_dim=self.rank_dim,
            max_tokens=self.max_tokens,
            num_token_groups=self.num_token_groups,
        )


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_native_v0_checkpoint(
    path: str | Path,
    *,
    model: NativeSimVLAV0,
    config: NativeV0Config,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict[str, Any],
    sampler_state: dict[str, Any],
    source_lock: dict[str, Any],
    training_config: dict[str, Any],
    final: bool,
) -> Path:
    config.validate_primary()
    return atomic_torch_save(
        {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "global_optimizer_step": int(global_step),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state,
            "sampler_state_dict": sampler_state,
            "source_lock": source_lock,
            "training_config": training_config,
            "scientific_primary_checkpoint": bool(final and int(global_step) == 150_000),
            "final": bool(final),
        },
        path,
    )


def load_native_v0_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
    require_final_150k: bool = False,
) -> tuple[NativeSimVLAV0, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported native V0 checkpoint: {payload.get('checkpoint_format')}")
    config = NativeV0Config(**payload["model_config"])
    config.validate_primary()
    if require_final_150k and not bool(payload.get("scientific_primary_checkpoint")):
        raise RuntimeError("scientific evaluation requires the final 150K checkpoint")
    model = config.build().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(Path(path).resolve()),
        "checkpoint_format": payload.get("checkpoint_format"),
        "global_optimizer_step": payload.get("global_optimizer_step"),
        "scientific_primary_checkpoint": payload.get("scientific_primary_checkpoint"),
        "model_config": payload.get("model_config"),
        "source_lock": payload.get("source_lock"),
    }


def write_checkpoint_summary(path: str | Path, output: str | Path) -> None:
    Path(output).write_text(
        json.dumps(checkpoint_summary(path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
