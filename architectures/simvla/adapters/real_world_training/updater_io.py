"""Checkpoint contracts for real-world Condition and Generation updaters."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.simvla_generation_loop import (
    SimVLAGenerationHiddenUpdater,
)

from .io_utils import sha256_file


CONDITION_FORMAT = "simvla_real_condition_updater_kc2_v1"
GENERATION_FORMAT = "simvla_real_generation_updater_ng3_v1"


@dataclass(frozen=True)
class RealConditionConfig:
    num_views: int = 2
    proprio_dim: int = 8
    condition_dim: int = 960
    delta_dim: int = 128
    rank_dim: int = 64
    max_tokens: int = 256
    num_token_groups: int = 8
    condition_refresh_interval: int = 2

    def build(self) -> NativeSimVLAV0:
        if self.condition_refresh_interval != 2:
            raise ValueError("real deployment Condition Updater is fixed to K_C=2")
        return NativeSimVLAV0(
            num_views=self.num_views,
            proprio_dim=self.proprio_dim,
            condition_dim=self.condition_dim,
            delta_dim=self.delta_dim,
            rank_dim=self.rank_dim,
            max_tokens=self.max_tokens,
            num_token_groups=self.num_token_groups,
        )


@dataclass(frozen=True)
class RealGenerationConfig:
    hidden_dim: int = 1024
    condition_dim: int = 960
    action_dim: int = 7
    proprio_dim: int = 8
    condition_code_dim: int = 128
    rank_dim: int = 128
    max_generator_age: int = 3
    gate_bias: float = -4.0
    generation_full_evaluations: int = 3
    full_step_indices: tuple[int, ...] = (0, 4, 8)

    def build(self) -> SimVLAGenerationHiddenUpdater:
        if self.generation_full_evaluations != 3 or tuple(self.full_step_indices) != (0, 4, 8):
            raise ValueError("real deployment Generation Updater is fixed to N_G=3 at (0,4,8)")
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


def _atomic_save(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_real_updater(
    path: str | Path,
    *,
    kind: str,
    updater: torch.nn.Module,
    config: RealConditionConfig | RealGenerationConfig,
    baseline_action_checkpoint: str | Path,
    norm_stats_sha256: str,
    dataset_identity_sha256: str,
    condition_cache_identity_sha256: str,
    optimizer_step: int,
    objective: Mapping[str, Any],
    validation: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None = None,
) -> Path:
    expected_type = RealConditionConfig if kind == "condition" else RealGenerationConfig
    if kind not in {"condition", "generation"} or not isinstance(config, expected_type):
        raise ValueError("updater kind and config type do not match")
    return _atomic_save(
        {
            "checkpoint_format": CONDITION_FORMAT if kind == "condition" else GENERATION_FORMAT,
            "updater_state_dict": {
                key: value.detach().cpu() for key, value in updater.state_dict().items()
            },
            "model_config": asdict(config),
            "source_lock": {
                "real_baseline_action_checkpoint": str(Path(baseline_action_checkpoint).resolve()),
                "real_baseline_action_checkpoint_sha256": sha256_file(baseline_action_checkpoint),
                "norm_stats_sha256": str(norm_stats_sha256),
                "dataset_identity_sha256": str(dataset_identity_sha256),
                "condition_cache_identity_sha256": str(condition_cache_identity_sha256),
            },
            "optimizer_step": int(optimizer_step),
            "objective": dict(objective),
            "validation": dict(validation),
            "optimizer_state_dict": optimizer_state,
        },
        path,
    )


def load_real_updater(
    path: str | Path,
    *,
    kind: str,
    device: torch.device | str,
    expected_baseline_sha256: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    expected_format = CONDITION_FORMAT if kind == "condition" else GENERATION_FORMAT
    if payload.get("checkpoint_format") != expected_format:
        raise ValueError(
            f"unsupported {kind} updater checkpoint: {payload.get('checkpoint_format')!r}"
        )
    raw_config = dict(payload["model_config"])
    if kind == "condition":
        config: RealConditionConfig | RealGenerationConfig = RealConditionConfig(**raw_config)
    else:
        raw_config["full_step_indices"] = tuple(raw_config["full_step_indices"])
        config = RealGenerationConfig(**raw_config)
    updater = config.build().to(device)
    updater.load_state_dict(payload["updater_state_dict"], strict=True)
    updater.eval()
    observed = payload["source_lock"]["real_baseline_action_checkpoint_sha256"]
    if expected_baseline_sha256 is not None and observed != expected_baseline_sha256:
        raise ValueError(
            f"{kind} updater was trained from a different real baseline: "
            f"{observed} != {expected_baseline_sha256}"
        )
    return updater, payload

