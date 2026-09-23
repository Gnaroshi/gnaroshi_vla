"""Calibrated q2-primary losses for exact condition regeneration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from methods.latentloop.training.losses import normalized_condition_mse


@dataclass(frozen=True)
class ExactQ2LossWeights:
    """Immutable weights approved only after raw-scale calibration."""

    q2_prefix: float
    q2_chunk: float
    q2_condition: float
    q1_auxiliary: float
    update_regularization: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _prefix_elements(action: Tensor, execution_horizon: int) -> Tensor:
    if action.ndim != 3 or action.shape[-1] != 7:
        raise ValueError("action must be [B,H,7]")
    if not 0 < int(execution_horizon) <= action.shape[1]:
        raise ValueError("invalid execution horizon")
    return action[:, : int(execution_horizon)]


def per_example_prefix_l1(prediction: Tensor, target: Tensor, execution_horizon: int) -> Tensor:
    """Return one exact environment-boundary first-R action error per example."""

    return (
        _prefix_elements(prediction, execution_horizon)
        - _prefix_elements(target.detach(), execution_horizon)
    ).abs().mean(dim=(1, 2))


def compute_exact_q2_losses(
    *,
    c0_full: Tensor,
    c1_pred: Tensor,
    c2_pred: Tensor,
    c1_full: Tensor,
    c2_full: Tensor,
    a1_pred: Tensor,
    a2_pred: Tensor,
    a1_full: Tensor,
    a2_full: Tensor,
    execution_horizon: int,
    weights: ExactQ2LossWeights | None,
) -> dict[str, Tensor]:
    """Compute raw components and, when approved, their frozen weighted total."""

    q2_prefix = per_example_prefix_l1(a2_pred, a2_full, execution_horizon).mean()
    q2_chunk = F.l1_loss(a2_pred, a2_full.detach())
    q2_condition = normalized_condition_mse(c2_pred, c2_full)
    q1_prefix = per_example_prefix_l1(a1_pred, a1_full, execution_horizon).mean()
    q1_chunk = F.l1_loss(a1_pred, a1_full.detach())
    q1_condition = normalized_condition_mse(c1_pred, c1_full)
    q1_auxiliary = q1_prefix + q1_chunk + q1_condition
    update_regularization = 0.5 * (
        F.mse_loss(c1_pred - c0_full, torch.zeros_like(c1_pred))
        + F.mse_loss(c2_pred - c1_pred, torch.zeros_like(c2_pred))
    )
    q2_gripper_command = F.l1_loss(
        _prefix_elements(a2_pred, execution_horizon)[..., 6],
        _prefix_elements(a2_full.detach(), execution_horizon)[..., 6],
    )
    raw = {
        "q2_prefix_l1": q2_prefix,
        "q2_chunk_l1": q2_chunk,
        "q2_condition_normalized_mse": q2_condition,
        "q2_gripper_command_l1": q2_gripper_command,
        "q1_prefix_l1": q1_prefix,
        "q1_chunk_l1": q1_chunk,
        "q1_condition_normalized_mse": q1_condition,
        "q1_auxiliary": q1_auxiliary,
        "update_regularization_mse": update_regularization,
    }
    if weights is None:
        return raw
    total = (
        weights.q2_prefix * q2_prefix
        + weights.q2_chunk * q2_chunk
        + weights.q2_condition * q2_condition
        + weights.q1_auxiliary * q1_auxiliary
        + weights.update_regularization * update_regularization
    )
    return {"total": total, **raw}


class RawLossScaleAccumulator:
    """Collect unweighted scales without proposing result-dependent coefficients."""

    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def update(self, losses: Mapping[str, Tensor | float]) -> None:
        for name, value in losses.items():
            if name == "total":
                continue
            scalar = float(value.detach().item()) if torch.is_tensor(value) else float(value)
            self.values.setdefault(name, []).append(scalar)

    def summary(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(self.values.items()):
            tensor = torch.tensor(values, dtype=torch.float64)
            result[name] = {
                "count": len(values),
                "mean": float(tensor.mean().item()),
                "p50": float(torch.quantile(tensor, 0.50).item()),
                "p90": float(torch.quantile(tensor, 0.90).item()),
                "p95": float(torch.quantile(tensor, 0.95).item()),
                "max": float(tensor.max().item()),
            }
        return result


def load_approved_loss_contract(path: str | Path) -> tuple[ExactQ2LossWeights, dict]:
    """Reject unset, unapproved, or non-q2-primary loss contracts."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("experiment_identifier") != "simvla_r5_exact_q2_regeneration":
        raise ValueError("loss contract belongs to another experiment")
    if payload.get("approval_status") != "APPROVED_AFTER_RAW_SCALE_REVIEW":
        raise RuntimeError("loss contract was not approved after raw-scale calibration")
    weights = ExactQ2LossWeights(**payload["weights"])
    intended = payload.get("intended_weighted_contributions", {})
    required = {
        "q2_prefix",
        "q2_chunk",
        "q2_condition",
        "q1_auxiliary",
        "update_regularization",
    }
    if set(intended) != required:
        raise ValueError("loss contract lacks intended contribution declarations")
    if float(intended["q2_prefix"]) <= max(
        float(value) for name, value in intended.items() if name != "q2_prefix"
    ):
        raise ValueError("q2 prefix is not the largest intended contribution")
    if any(value < 0.0 or not torch.isfinite(torch.tensor(value)) for value in weights.to_dict().values()):
        raise ValueError("loss weights must be finite and nonnegative")
    return weights, payload
