"""Predeclared V0/V1 state and same-noise action losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from methods.variable_time_latentloop.composition import normalized_composition_distance
from methods.variable_time_latentloop.metrics import action_error_components

from .kv_codec import LayerSharedKVCodec
from .prefix_kv_hook import PrefixKVState


@dataclass(frozen=True)
class LossWeights:
    state: float = 1.0
    chunk: float = 1.0
    executed: float = 1.0
    gripper: float = 1.0
    composition: float = 1.0


def normalized_kv_loss(
    codec: LayerSharedKVCodec,
    predicted: PrefixKVState,
    target: PrefixKVState,
    scales: Tensor | None = None,
) -> Tensor:
    predicted_packed = codec.pack(predicted).float()
    target_packed = codec.pack(target).float()
    difference = predicted_packed - target_packed
    if scales is None:
        scales = target_packed.detach().square().mean(dim=(0, 2, 3), keepdim=True).sqrt()
    scales = torch.as_tensor(scales, device=difference.device, dtype=difference.dtype).clamp_min(1e-5)
    return (difference / scales).square().mean()


def action_losses(predicted: Tensor, target: Tensor, execution_horizon: int = 5) -> dict[str, Tensor]:
    components = action_error_components(predicted[..., :7], target[..., :7], execution_horizon)
    return {
        "chunk": components["chunk_mse"].mean(),
        "executed": components["executed_mse"].mean(),
        "translation": components["translation_mse"].mean(),
        "rotation": components["rotation_mse"].mean(),
        "gripper": components["gripper_mse"].mean(),
    }


def weighted_v0_loss(state: Tensor, action: dict[str, Tensor], weights: LossWeights) -> Tensor:
    return (
        weights.state * state
        + weights.chunk * action["chunk"]
        + weights.executed * action["executed"]
        + weights.gripper * action["gripper"]
    )


def composition_loss(direct_encoded: Tensor, composed_encoded: Tensor) -> Tensor:
    return normalized_composition_distance(direct_encoded, composed_encoded).mean()
