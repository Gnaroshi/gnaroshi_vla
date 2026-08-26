"""Strict checkpoint contract for the SimVLA action-fidelity head."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.action_equivalent_refresh.features import (
    SimVLAActionFidelityFeatureConfig,
)
from methods.latentloop.modules.action_equivalent_refresh import (
    ActionFidelityHead,
    ExactCallBudgetCalibration,
)


CHECKPOINT_SCHEMA = "simvla_action_equivalent_refresh_v1"
MAX_PRIMARY_RISK_HEAD_PARAMETERS = 50_000


def save_action_fidelity_checkpoint(
    path: str | Path,
    *,
    head: ActionFidelityHead,
    feature_config: SimVLAActionFidelityFeatureConfig,
    calibration: ExactCallBudgetCalibration,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_parameters = sum(parameter.numel() for parameter in head.parameters())
    if total_parameters > MAX_PRIMARY_RISK_HEAD_PARAMETERS:
        raise ValueError("risk head exceeds the primary 50K-parameter contract")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "feature_config": feature_config.to_dict(),
        "head_config": {
            "input_dim": head.input_dim,
            "hidden_dim": head.hidden_dim,
            "bottleneck_dim": head.bottleneck_dim,
            "quantile": head.quantile,
        },
        "head_state_dict": head.state_dict(),
        "calibration": calibration.to_dict(),
        "scientific_contract": {
            "policy_query_cadence_fixed": True,
            "action_execution_horizon_fixed": 5,
            "action_horizon_fixed": 10,
            "generation_n_g_fixed": 3,
            "maximum_approximate_age": 3,
            "runtime_exact_action_input": False,
            "runtime_exact_condition_input": False,
            "risk_head_parameters": int(total_parameters),
        },
        "metadata": dict(metadata or {}),
    }
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_action_fidelity_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[
    ActionFidelityHead,
    SimVLAActionFidelityFeatureConfig,
    ExactCallBudgetCalibration,
    dict[str, Any],
]:
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location=device, weights_only=False
    )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported action-fidelity checkpoint schema")
    raw_feature = payload["feature_config"]
    feature = SimVLAActionFidelityFeatureConfig(
        **{
            key: int(raw_feature[key])
            for key in (
                "delta_dim",
                "proprio_dim",
                "action_dim",
                "first_r",
                "num_token_groups",
                "max_age",
            )
        }
    )
    head_config = payload["head_config"]
    if int(head_config["input_dim"]) != feature.input_dim:
        raise ValueError("checkpoint feature/head dimensions differ")
    head = ActionFidelityHead(
        int(head_config["input_dim"]),
        hidden_dim=int(head_config["hidden_dim"]),
        bottleneck_dim=int(head_config["bottleneck_dim"]),
        quantile=float(head_config["quantile"]),
    ).to(device)
    head.load_state_dict(payload["head_state_dict"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    calibration = ExactCallBudgetCalibration.from_dict(payload["calibration"])
    if calibration.max_approximate_age != feature.max_age:
        raise ValueError("checkpoint calibration and feature age contracts differ")
    return head, feature, calibration, payload
