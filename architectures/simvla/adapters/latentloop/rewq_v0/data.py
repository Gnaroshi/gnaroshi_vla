"""Compact, episode-disjoint datasets for rewq v0 branch supervision."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset

from methods.latentloop.modules.rewq_v0 import NextAnchorRecoveryTargets
from .features import SimVLARecoverabilityFeatureConfig


RECOVERY_DATASET_SCHEMA = "simvla_rewq_v0_recovery_dataset_v1"
SAFE_REFERENCE_SCHEMA = "simvla_rewq_v0_kc2_ng3_safe_reference_v1"
ALLOWED_SPLITS = ("train", "checkpoint_validation", "final_offline")


def _atomic_save(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_recovery_dataset(
    path: str | Path,
    *,
    split: str,
    features: Tensor,
    targets: NextAnchorRecoveryTargets,
    candidate_age: Tensor,
    episode_ids: list[str],
    feature_config: SimVLARecoverabilityFeatureConfig,
    source_metadata: Mapping[str, Any],
) -> Path:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}")
    targets.validate()
    rows = int(features.shape[0])
    if features.shape != (rows, feature_config.input_dim):
        raise ValueError("features violate the configured dimension")
    if targets.continuous.shape[0] != rows or candidate_age.shape != (rows,):
        raise ValueError("features, targets, and candidate ages must be row-aligned")
    if len(episode_ids) != rows or not rows:
        raise ValueError("one non-empty episode ID is required per row")
    if not bool(torch.isfinite(features.float()).all()):
        raise ValueError("runtime features contain non-finite values")
    required_contract = {
        "paired_environment_initialization": True,
        "paired_action_noise": True,
        "candidate_queries": 1,
        "exact_recovery_queries": 1,
        "total_environment_actions": 10,
    }
    for key, expected in required_contract.items():
        if source_metadata.get(key) != expected:
            raise ValueError(f"source metadata must declare {key}={expected!r}")
    age = candidate_age.detach().cpu().long()
    if bool((age < 1).any()) or bool((age > feature_config.max_age).any()):
        raise ValueError("candidate age exceeds the declared feature contract")
    return _atomic_save(
        Path(path).expanduser().resolve(),
        {
            "schema_version": RECOVERY_DATASET_SCHEMA,
            "split": split,
            "feature_config": feature_config.to_dict(),
            "features": features.detach().cpu().float(),
            "continuous": targets.continuous.detach().cpu().float(),
            "continuous_valid": targets.continuous_valid.detach().cpu().bool(),
            "events": targets.events.detach().cpu().float(),
            "mode_valid": targets.mode_valid.detach().cpu().bool(),
            "candidate_age": age,
            "episode_ids": [str(value) for value in episode_ids],
            "source_metadata": dict(source_metadata),
            "scientific_contract": {
                **required_contract,
                "simulator_object_state_is_runtime_input": False,
                "exact_condition_is_runtime_input": False,
                "exact_action_is_runtime_input": False,
            },
        },
    )


def save_safe_reference(
    path: str | Path,
    *,
    continuous: Tensor,
    events: Tensor,
    episode_ids: list[str],
    source_metadata: Mapping[str, Any],
) -> Path:
    if continuous.ndim != 2 or events.ndim != 2:
        raise ValueError("safe reference tensors must be [N,C] and [N,E]")
    rows = int(continuous.shape[0])
    if events.shape[0] != rows or len(episode_ids) != rows:
        raise ValueError("safe reference rows must be aligned")
    if rows < 20:
        raise ValueError("safe reference requires at least 20 successful rows")
    if source_metadata.get("row_name") != "condition_kc2_ng3":
        raise ValueError("safe reference row must be condition_kc2_ng3")
    if source_metadata.get("success_only") is not True:
        raise ValueError("safe reference must contain successful episodes only")
    return _atomic_save(
        Path(path).expanduser().resolve(),
        {
            "schema_version": SAFE_REFERENCE_SCHEMA,
            "continuous": continuous.detach().cpu().float(),
            "events": events.detach().cpu().float(),
            "episode_ids": [str(value) for value in episode_ids],
            "source_metadata": dict(source_metadata),
        },
    )


class CompactRecoveryDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, path: str | Path, *, expected_split: str) -> None:
        payload = torch.load(
            Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
        )
        if payload.get("schema_version") != RECOVERY_DATASET_SCHEMA:
            raise ValueError("unsupported rewq v0 recovery dataset")
        if payload.get("split") != expected_split or expected_split not in ALLOWED_SPLITS:
            raise ValueError("recovery dataset split does not match its role")
        raw = payload["feature_config"]
        self.feature_config = SimVLARecoverabilityFeatureConfig(
            **{
                key: int(raw[key])
                for key in (
                    "delta_dim",
                    "proprio_dim",
                    "action_dim",
                    "first_r",
                    "num_token_groups",
                    "max_age",
                    "condition_summary_bins",
                )
            }
        )
        self.payload = payload
        self.features = payload["features"].float()
        self.episode_ids = tuple(str(value) for value in payload["episode_ids"])
        if self.features.shape != (len(self.episode_ids), self.feature_config.input_dim):
            raise ValueError("stored recovery features violate their contract")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "features": self.features[index],
            "continuous": self.payload["continuous"][index],
            "continuous_valid": self.payload["continuous_valid"][index],
            "events": self.payload["events"][index],
            "mode_valid": self.payload["mode_valid"][index],
            "candidate_age": self.payload["candidate_age"][index],
        }


def load_safe_reference(path: str | Path) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if payload.get("schema_version") != SAFE_REFERENCE_SCHEMA:
        raise ValueError("unsupported rewq v0 safe reference")
    if payload.get("source_metadata", {}).get("row_name") != "condition_kc2_ng3":
        raise ValueError("safe reference lineage is not condition_kc2_ng3")
    if payload.get("source_metadata", {}).get("success_only") is not True:
        raise ValueError("safe reference includes unsuccessful rows")
    return payload
