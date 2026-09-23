"""Runtime-aligned losses for recurrent LatentLoop distillation.

The helpers in this file mirror Seer's legacy temporal ensemble without
thresholding the gripper probability, so gradients remain available during
adapter training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class RuntimeAlignedLosses:
    gripper_distill: Tensor
    overlap: Tensor
    ensemble: Tensor
    gripper_switch: Tensor
    predicted_executed: Tensor
    teacher_executed: Tensor


def _validate_chunks(name: str, chunks: Tensor) -> None:
    if chunks.ndim != 4 or chunks.shape[-1] != 7:
        raise ValueError(f"{name} must be [B,H,P,7], got {tuple(chunks.shape)}")
    if chunks.shape[1] < 1 or chunks.shape[2] < 1:
        raise ValueError(f"{name} must contain at least one query and one action token")


def age_weighted_loss(prediction: Tensor, target: Tensor, ages: Tensor, kind: str) -> Tensor:
    """Average a loss within each cache age, then combine with frozen age weights."""

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("age-weighted tensors must have the same shape and an age axis")
    if ages.ndim != 1 or ages.numel() != prediction.shape[1]:
        raise ValueError("age weights must be one-dimensional and match axis 1")
    if bool((ages <= 0).any()):
        raise ValueError("age weights must be positive")
    if kind == "mse":
        elementwise = (prediction - target.detach()).square()
    elif kind == "l1":
        elementwise = (prediction - target.detach()).abs()
    else:
        raise ValueError(f"unsupported age-weighted loss kind: {kind}")
    per_age = elementwise.flatten(2).mean(dim=(0, 2))
    weights = ages.to(device=prediction.device, dtype=prediction.dtype)
    return (per_age * weights).sum() / weights.sum()


def differentiable_temporal_ensemble(
    anchor_chunk: Tensor,
    recurrent_chunks: Tensor,
    temperature: float,
) -> Tensor:
    """Return executed probabilities for recurrent ages 1..H.

    Query zero is the full-refresh anchor. Query ``q > 0`` is a recurrent
    prediction. At age ``t``, token ``t-q`` from every still-valid query is
    combined in the same oldest-to-newest order as Seer's action buffer.
    """

    _validate_chunks("recurrent_chunks", recurrent_chunks)
    if anchor_chunk.ndim != 3 or anchor_chunk.shape[-1] != 7:
        raise ValueError(f"anchor_chunk must be [B,P,7], got {tuple(anchor_chunk.shape)}")
    batch, horizon, action_tokens, action_dim = recurrent_chunks.shape
    if anchor_chunk.shape != (batch, action_tokens, action_dim):
        raise ValueError("anchor and recurrent action chunks must share [B,P,7]")
    if temperature < 0:
        raise ValueError("temporal-ensemble temperature cannot be negative")

    queries = torch.cat((anchor_chunk.unsqueeze(1), recurrent_chunks), dim=1)
    executed = []
    for timestep in range(1, horizon + 1):
        first_query = max(0, timestep - action_tokens + 1)
        candidates = [
            queries[:, query, timestep - query]
            for query in range(first_query, timestep + 1)
        ]
        stacked = torch.stack(candidates, dim=1)
        indices = torch.arange(
            stacked.shape[1], device=stacked.device, dtype=stacked.dtype
        )
        weights = torch.exp(-float(temperature) * indices)
        weights = weights / weights.sum()
        executed.append((stacked * weights.view(1, -1, 1)).sum(dim=1))
    return torch.stack(executed, dim=1)


def runtime_aligned_action_losses(
    predicted_chunks: Tensor,
    teacher_chunks: Tensor,
    anchor_teacher_chunk: Tensor,
    *,
    temperature: float = 0.01,
    probability_epsilon: float = 1e-5,
) -> RuntimeAlignedLosses:
    """Compare raw chunks and the actions that the runtime would execute."""

    _validate_chunks("predicted_chunks", predicted_chunks)
    _validate_chunks("teacher_chunks", teacher_chunks)
    if predicted_chunks.shape != teacher_chunks.shape:
        raise ValueError("predicted and teacher chunks must share [B,H,P,7]")
    if anchor_teacher_chunk.shape != predicted_chunks.shape[:1] + predicted_chunks.shape[2:]:
        raise ValueError("anchor_teacher_chunk must be [B,P,7]")

    teacher_chunks = teacher_chunks.detach()
    anchor_teacher_chunk = anchor_teacher_chunk.detach()
    pred_gripper = predicted_chunks[..., 6:].clamp(
        probability_epsilon, 1.0 - probability_epsilon
    )
    teacher_gripper = teacher_chunks[..., 6:].clamp(0.0, 1.0)
    gripper_distill = F.binary_cross_entropy(pred_gripper, teacher_gripper)

    overlap_terms = []
    for age in range(1, predicted_chunks.shape[1]):
        predicted_delta = (
            predicted_chunks[:, age - 1, 1:] - predicted_chunks[:, age, :-1]
        )
        teacher_delta = teacher_chunks[:, age - 1, 1:] - teacher_chunks[:, age, :-1]
        overlap_terms.append(F.smooth_l1_loss(predicted_delta, teacher_delta))
    overlap = (
        torch.stack(overlap_terms).mean()
        if overlap_terms
        else predicted_chunks.sum() * 0.0
    )

    predicted_executed = differentiable_temporal_ensemble(
        anchor_teacher_chunk, predicted_chunks, temperature
    )
    teacher_executed = differentiable_temporal_ensemble(
        anchor_teacher_chunk, teacher_chunks, temperature
    ).detach()
    ensemble_arm = F.smooth_l1_loss(
        predicted_executed[..., :6], teacher_executed[..., :6]
    )
    ensemble_gripper = F.binary_cross_entropy(
        predicted_executed[..., 6:].clamp(
            probability_epsilon, 1.0 - probability_epsilon
        ),
        teacher_executed[..., 6:].clamp(0.0, 1.0),
    )
    ensemble = ensemble_arm + ensemble_gripper

    anchor_probability = anchor_teacher_chunk[:, 0, 6:7]
    predicted_probability_sequence = torch.cat(
        (anchor_probability.unsqueeze(1), predicted_executed[..., 6:]), dim=1
    )
    teacher_probability_sequence = torch.cat(
        (anchor_probability.unsqueeze(1), teacher_executed[..., 6:]), dim=1
    )
    predicted_change = (
        predicted_probability_sequence[:, 1:] - predicted_probability_sequence[:, :-1]
    )
    teacher_change = (
        teacher_probability_sequence[:, 1:] - teacher_probability_sequence[:, :-1]
    ).detach()
    transition_alignment = F.smooth_l1_loss(predicted_change, teacher_change)
    sign_target = (teacher_executed[..., 6:] >= 0.5).to(predicted_executed.dtype)
    sign_alignment = F.binary_cross_entropy(
        predicted_executed[..., 6:].clamp(
            probability_epsilon, 1.0 - probability_epsilon
        ),
        sign_target,
    )
    gripper_switch = transition_alignment + sign_alignment

    return RuntimeAlignedLosses(
        gripper_distill=gripper_distill,
        overlap=overlap,
        ensemble=ensemble,
        gripper_switch=gripper_switch,
        predicted_executed=predicted_executed,
        teacher_executed=teacher_executed,
    )
