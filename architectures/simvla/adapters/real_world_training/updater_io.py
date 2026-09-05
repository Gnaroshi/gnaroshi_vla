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
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    audit_projection_only_state,
)

from .io_utils import sha256_file


CONDITION_FORMAT = "simvla_real_condition_updater_kc2_v2"
GENERATION_FORMAT = "simvla_real_generation_updater_ng3_v2"
COUPLED_GENERATION_FORMAT = "simvla_real_coupled_generation_kc2_ng3_v2"


SOURCE_LOCK_FIELDS = {
    "baseline": "real_baseline_action_checkpoint_sha256",
    "norm": "norm_stats_sha256",
    "dataset": "dataset_identity_sha256",
    "cache": "condition_cache_identity_sha256",
    "attestation": "condition_cache_attestation_identity_sha256",
}


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
    condition_cache_attestation_identity_sha256: str,
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
                "condition_cache_attestation_identity_sha256": str(
                    condition_cache_attestation_identity_sha256
                ),
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
    expected_norm_sha256: str | None = None,
    expected_dataset_identity_sha256: str | None = None,
    expected_cache_identity_sha256: str | None = None,
    expected_cache_attestation_identity_sha256: str | None = None,
    expected_optimizer_step: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if kind not in {"condition", "generation"}:
        raise ValueError(f"unsupported real updater kind: {kind!r}")
    payload = torch.load(path, map_location=device, weights_only=True)
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
    expected = {
        "baseline": expected_baseline_sha256,
        "norm": expected_norm_sha256,
        "dataset": expected_dataset_identity_sha256,
        "cache": expected_cache_identity_sha256,
        "attestation": expected_cache_attestation_identity_sha256,
    }
    source_lock = payload.get("source_lock", {})
    for label, value in expected.items():
        if value is None:
            continue
        field = SOURCE_LOCK_FIELDS[label]
        observed = source_lock.get(field)
        if observed != value:
            raise ValueError(
                f"{kind} updater {label} source mismatch: {observed} != {value}"
            )
    if (
        expected_optimizer_step is not None
        and int(payload.get("optimizer_step", -1)) != int(expected_optimizer_step)
    ):
        raise ValueError(
            f"{kind} updater optimizer step mismatch: "
            f"{payload.get('optimizer_step')} != {expected_optimizer_step}"
        )
    return updater, payload


def save_real_coupled_generation(
    path: str | Path,
    *,
    updater: SimVLAGenerationHiddenUpdater,
    config: RealGenerationConfig,
    parent_generation_checkpoint: str | Path,
    condition_updater_checkpoint: str | Path,
    condition_cache_manifest: str | Path,
    optimizer_step: int,
    training_config: Mapping[str, Any],
    validation: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
) -> Path:
    parent_payload = torch.load(
        parent_generation_checkpoint, map_location="cpu", weights_only=True
    )
    condition_payload = torch.load(
        condition_updater_checkpoint, map_location="cpu", weights_only=True
    )
    if parent_payload.get("checkpoint_format") != GENERATION_FORMAT:
        raise ValueError("coupled parent is not a real Generation Updater checkpoint")
    if condition_payload.get("checkpoint_format") != CONDITION_FORMAT:
        raise ValueError("coupled condition source is not a real Condition Updater checkpoint")
    parent_source = dict(parent_payload.get("source_lock", {}))
    condition_source = dict(condition_payload.get("source_lock", {}))
    for field in SOURCE_LOCK_FIELDS.values():
        if parent_source.get(field) != condition_source.get(field):
            raise ValueError(f"coupled parent source mismatch for {field}")
    raw_parent_config = dict(parent_payload["model_config"])
    raw_parent_config["full_step_indices"] = tuple(raw_parent_config["full_step_indices"])
    parent_config = RealGenerationConfig(**raw_parent_config)
    if config != parent_config:
        raise ValueError("coupled Generation config differs from its parent")
    try:
        candidate_device = next(updater.parameters()).device
    except StopIteration as error:
        raise ValueError("coupled Generation updater has no parameters") from error
    parent = parent_config.build().to(candidate_device)
    parent.load_state_dict(parent_payload["updater_state_dict"], strict=True)
    projection_audit = audit_projection_only_state(parent, updater)
    if projection_audit["verdict"] != "PROJECTION_ONLY_STATE_PASS":
        raise ValueError(
            "refusing to save a coupled updater that changed parent tensors outside "
            "condition_code_projection.weight"
        )
    return _atomic_save(
        {
            "checkpoint_format": COUPLED_GENERATION_FORMAT,
            "updater_state_dict": {
                key: value.detach().cpu() for key, value in updater.state_dict().items()
            },
            "model_config": asdict(config),
            "source_lock": {
                **parent_source,
                "parent_generation_checkpoint": str(
                    Path(parent_generation_checkpoint).expanduser().resolve()
                ),
                "parent_generation_checkpoint_sha256": sha256_file(
                    parent_generation_checkpoint
                ),
                "condition_updater_checkpoint": str(
                    Path(condition_updater_checkpoint).expanduser().resolve()
                ),
                "condition_updater_checkpoint_sha256": sha256_file(
                    condition_updater_checkpoint
                ),
                "condition_cache_manifest": str(
                    Path(condition_cache_manifest).expanduser().resolve()
                ),
                "condition_cache_manifest_sha256": sha256_file(
                    condition_cache_manifest
                ),
            },
            "optimizer_step": int(optimizer_step),
            "training_config": dict(training_config),
            "validation": dict(validation),
            "projection_only_state_audit": projection_audit,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
        },
        path,
    )


def load_real_coupled_generation(
    path: str | Path,
    *,
    device: torch.device | str,
    expected_parent_generation_sha256: str | None = None,
    expected_condition_updater_sha256: str | None = None,
    expected_cache_manifest_sha256: str | None = None,
    expected_baseline_sha256: str | None = None,
    expected_norm_sha256: str | None = None,
    expected_dataset_identity_sha256: str | None = None,
    expected_cache_identity_sha256: str | None = None,
    expected_cache_attestation_identity_sha256: str | None = None,
    expected_optimizer_step: int | None = 10_000,
) -> tuple[SimVLAGenerationHiddenUpdater, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("checkpoint_format") != COUPLED_GENERATION_FORMAT:
        raise ValueError(
            "unsupported coupled Generation checkpoint: "
            f"{payload.get('checkpoint_format')!r}"
        )
    raw_config = dict(payload["model_config"])
    raw_config["full_step_indices"] = tuple(raw_config["full_step_indices"])
    config = RealGenerationConfig(**raw_config)
    updater = config.build().to(device)
    updater.load_state_dict(payload["updater_state_dict"], strict=True)
    updater.eval()

    training = payload.get("training_config", {})
    required_training = {
        "condition_refresh_interval": 2,
        "generation_full_evaluations": 3,
        "full_step_indices": [0, 4, 8],
        "trainable_parameter_names": ["condition_code_projection.weight"],
        "trainable_parameters": 16_384,
        "condition_change_code": "condition_updater_delta_encoder",
    }
    mismatches = {
        key: {"observed": training.get(key), "required": value}
        for key, value in required_training.items()
        if training.get(key) != value
    }
    if mismatches:
        raise ValueError(f"coupled Generation training contract mismatch: {mismatches}")
    saved_projection_audit = payload.get("projection_only_state_audit", {})
    if saved_projection_audit.get("verdict") != "PROJECTION_ONLY_STATE_PASS":
        raise ValueError("coupled Generation checkpoint lacks a passing projection-only audit")

    source = payload.get("source_lock", {})
    expected_sources = {
        "parent_generation_checkpoint_sha256": expected_parent_generation_sha256,
        "condition_updater_checkpoint_sha256": expected_condition_updater_sha256,
        "condition_cache_manifest_sha256": expected_cache_manifest_sha256,
        SOURCE_LOCK_FIELDS["baseline"]: expected_baseline_sha256,
        SOURCE_LOCK_FIELDS["norm"]: expected_norm_sha256,
        SOURCE_LOCK_FIELDS["dataset"]: expected_dataset_identity_sha256,
        SOURCE_LOCK_FIELDS["cache"]: expected_cache_identity_sha256,
        SOURCE_LOCK_FIELDS["attestation"]: expected_cache_attestation_identity_sha256,
    }
    for field, expected_value in expected_sources.items():
        if expected_value is not None and source.get(field) != expected_value:
            raise ValueError(
                f"coupled Generation source mismatch for {field}: "
                f"{source.get(field)} != {expected_value}"
            )
    if (
        expected_optimizer_step is not None
        and int(payload.get("optimizer_step", -1)) != int(expected_optimizer_step)
    ):
        raise ValueError(
            "coupled Generation optimizer step mismatch: "
            f"{payload.get('optimizer_step')} != {expected_optimizer_step}"
        )
    return updater, payload


def audit_real_coupled_checkpoint(
    *,
    parent_generation_checkpoint: str | Path,
    coupled_generation_checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    parent, _ = load_real_updater(
        parent_generation_checkpoint, kind="generation", device=device
    )
    coupled, _ = load_real_coupled_generation(
        coupled_generation_checkpoint,
        device=device,
        expected_parent_generation_sha256=sha256_file(parent_generation_checkpoint),
    )
    return audit_projection_only_state(parent, coupled)
