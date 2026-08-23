"""Checkpoint contract for the SimVLA Generation Loop updater."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from methods.latentloop.modules.simvla_generation_loop import (
    SimVLAGenerationHiddenUpdater,
)


CHECKPOINT_FORMAT = "simvla_generation_loop_v1"


@dataclass(frozen=True)
class GenerationLoopConfig:
    hidden_dim: int = 1024
    condition_dim: int = 960
    action_dim: int = 7
    proprio_dim: int = 8
    condition_code_dim: int = 128
    rank_dim: int = 128
    max_generator_age: int = 4
    gate_bias: float = -4.0
    action_horizon: int = 10
    supported_n_g: tuple[int, ...] = (3, 2)

    def validate(self) -> None:
        if self.action_horizon != 10:
            raise ValueError("released SimVLA Generation Loop requires H=10")
        if self.max_generator_age < 4:
            raise ValueError("N_G=2 requires generator age 4")
        if tuple(self.supported_n_g) != (3, 2):
            raise ValueError("screening checkpoint must support N_G=(3,2)")

    def build(self) -> SimVLAGenerationHiddenUpdater:
        self.validate()
        return SimVLAGenerationHiddenUpdater(
            hidden_dim=self.hidden_dim,
            condition_dim=self.condition_dim,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            condition_code_dim=self.condition_code_dim,
            rank_dim=self.rank_dim,
            max_generator_age=self.max_generator_age,
            gate_bias=self.gate_bias,
        )


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_generation_checkpoint(
    path: str | Path,
    *,
    updater: SimVLAGenerationHiddenUpdater,
    config: GenerationLoopConfig,
    optimizer_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    source_lock: dict[str, Any],
    training_config: dict[str, Any],
) -> Path:
    config.validate()
    return _atomic_torch_save(
        {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "updater_state_dict": updater.state_dict(),
            "model_config": asdict(config),
            "optimizer_step": int(optimizer_step),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "source_lock": source_lock,
            "training_config": training_config,
        },
        path,
    )


def load_generation_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[SimVLAGenerationHiddenUpdater, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported Generation Loop checkpoint: {payload.get('checkpoint_format')}"
        )
    config_payload = dict(payload["model_config"])
    config_payload["supported_n_g"] = tuple(config_payload["supported_n_g"])
    config = GenerationLoopConfig(**config_payload)
    updater = config.build().to(device)
    updater.load_state_dict(payload["updater_state_dict"], strict=True)
    updater.eval()
    return updater, payload
