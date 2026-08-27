"""Fail-closed rewq v0 checkpoint serialization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from methods.latentloop.modules.rewq_v0 import (
    ComputeCostTable,
    RecoverabilityHead,
    RecoverySafetyEnvelope,
    SplitConformalCalibration,
)
from .features import SimVLARecoverabilityFeatureConfig


CHECKPOINT_SCHEMA = "simvla_rewq_v0_recoverability_checkpoint_v1"
MAX_PARAMETERS = 100_000


def save_rewq_v0_checkpoint(
    path: str | Path,
    *,
    head: RecoverabilityHead,
    feature_config: SimVLARecoverabilityFeatureConfig,
    envelope: RecoverySafetyEnvelope,
    conformal: SplitConformalCalibration,
    costs: ComputeCostTable,
    metadata: Mapping[str, Any],
) -> Path:
    audit = head.parameter_audit()
    if audit["total_parameters"] > MAX_PARAMETERS:
        raise ValueError("rewq v0 head exceeds the 100K parameter ceiling")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "head_state_dict": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
        },
        "head": audit,
        "feature_config": feature_config.to_dict(),
        "envelope": envelope.to_dict(),
        "conformal": conformal.to_dict(),
        "costs": costs.to_dict(),
        "metadata": dict(metadata),
        "frozen_external_modules": ["SimVLA", "U_C", "U_G"],
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_rewq_v0_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[
    RecoverabilityHead,
    SimVLARecoverabilityFeatureConfig,
    RecoverySafetyEnvelope,
    SplitConformalCalibration,
    ComputeCostTable,
    dict[str, Any],
]:
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported rewq v0 checkpoint")
    raw = payload["feature_config"]
    config = SimVLARecoverabilityFeatureConfig(
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
    raw_head = payload["head"]
    head = RecoverabilityHead(
        config.input_dim,
        hidden_dim=int(raw_head["hidden_dim"]),
        bottleneck_dim=int(raw_head["bottleneck_dim"]),
        quantile=float(raw_head["quantile"]),
    )
    head.load_state_dict(payload["head_state_dict"], strict=True)
    head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    envelope = RecoverySafetyEnvelope.from_dict(payload["envelope"])
    conformal = SplitConformalCalibration.from_dict(payload["conformal"])
    raw_cost = payload["costs"]
    costs = ComputeCostTable(
        exact_condition_ms=float(raw_cost["exact_condition_ms"]),
        approximate_condition_ms=float(raw_cost["approximate_condition_ms"]),
        generation_ng3_ms=float(raw_cost["generation_ng3_ms"]),
        generation_ng2_ms=float(raw_cost["generation_ng2_ms"]),
        router_ms=float(raw_cost["router_ms"]),
        provenance=str(raw_cost["provenance"]),
    )
    return head, config, envelope, conformal, costs, payload
