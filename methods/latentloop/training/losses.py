"""Predeclared Stage-T1 losses and raw-scale calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentLoopLossWeights:
    """Explicit loss weights selected only after raw-scale calibration."""

    condition: float
    action_chunk: float
    executed_prefix: float
    update_regularization: float


def normalized_condition_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """MSE after parameter-free LayerNorm over the condition channel."""

    if prediction.shape != target.shape:
        raise ValueError("condition prediction and target shapes must match")
    return F.mse_loss(
        F.layer_norm(prediction, (prediction.shape[-1],)),
        F.layer_norm(target.detach(), (target.shape[-1],)),
    )


def _prefix_mask(lengths: Tensor | int, action_chunk: Tensor) -> Tensor:
    batch_size, horizon, _ = action_chunk.shape
    length_tensor = torch.as_tensor(lengths, device=action_chunk.device, dtype=torch.long)
    if length_tensor.ndim == 0:
        length_tensor = length_tensor.expand(batch_size)
    if length_tensor.shape != (batch_size,):
        raise ValueError("execution lengths must be scalar or [B]")
    if bool((length_tensor < 1).any()) or bool((length_tensor > horizon).any()):
        raise ValueError("execution lengths must be in [1, action_horizon]")
    return torch.arange(horizon, device=action_chunk.device).unsqueeze(0) < length_tensor.unsqueeze(1)


def compute_t1_losses(
    *,
    previous_condition: Tensor,
    predicted_condition: Tensor,
    teacher_condition: Tensor,
    predicted_action_chunk: Tensor,
    teacher_action_chunk: Tensor,
    execution_lengths: Tensor | int,
    weights: LatentLoopLossWeights,
) -> dict[str, Tensor]:
    """Compute condition, same-noise chunk, prefix, and update losses."""

    if predicted_action_chunk.shape != teacher_action_chunk.shape:
        raise ValueError("predicted and teacher action chunk shapes must match")
    if predicted_action_chunk.ndim != 3:
        raise ValueError("action chunks must be [B,H,A]")
    condition = normalized_condition_mse(predicted_condition, teacher_condition)
    action_chunk = F.l1_loss(predicted_action_chunk, teacher_action_chunk.detach())
    mask = _prefix_mask(execution_lengths, predicted_action_chunk)
    element_mask = mask.unsqueeze(-1).expand_as(predicted_action_chunk)
    prefix_abs = (predicted_action_chunk - teacher_action_chunk.detach()).abs()
    executed_prefix = prefix_abs[element_mask].mean()
    update_regularization = F.mse_loss(
        predicted_condition - previous_condition,
        torch.zeros_like(predicted_condition),
    )
    total = (
        float(weights.condition) * condition
        + float(weights.action_chunk) * action_chunk
        + float(weights.executed_prefix) * executed_prefix
        + float(weights.update_regularization) * update_regularization
    )
    return {
        "total": total,
        "condition_normalized_mse": condition,
        "same_noise_action_chunk_l1": action_chunk,
        "executed_prefix_l1": executed_prefix,
        "update_regularization_mse": update_regularization,
    }


class LossScaleAccumulator:
    """Record raw, unweighted loss scales before choosing coefficients."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = {}

    def update(self, losses: Mapping[str, Tensor | float]) -> None:
        """Append scalar losses, excluding the weighted total."""

        for name, value in losses.items():
            if name == "total":
                continue
            scalar = float(value.detach().item()) if torch.is_tensor(value) else float(value)
            self._values.setdefault(name, []).append(scalar)

    def state_dict(self) -> dict[str, list[float]]:
        """Return the exact accumulated values for interruption-safe resume."""

        return {name: list(values) for name, values in self._values.items()}

    def load_state_dict(self, state: Mapping[str, list[float]]) -> None:
        """Restore exact accumulated values from a training checkpoint."""

        restored: dict[str, list[float]] = {}
        for name, values in state.items():
            restored[name] = [float(value) for value in values]
        self._values = restored

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Return count, mean, and robust percentiles for every loss."""

        output: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(self._values.items()):
            tensor = torch.tensor(values, dtype=torch.float64)
            output[name] = {
                "count": len(values),
                "mean": float(tensor.mean().item()),
                "p50": float(torch.quantile(tensor, 0.50).item()),
                "p90": float(torch.quantile(tensor, 0.90).item()),
                "p95": float(torch.quantile(tensor, 0.95).item()),
                "max": float(tensor.max().item()),
            }
        return output
