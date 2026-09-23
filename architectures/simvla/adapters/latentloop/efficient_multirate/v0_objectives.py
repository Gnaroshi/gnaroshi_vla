"""Cache-backed exact Mode B and balanced stochastic Mode D objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    balanced_mode_d_age,
)
from methods.latentloop.modules.native_simvla_v0 import NativeV0UnrollOutput
from methods.latentloop.training.native_simvla_v0 import (
    NativeV0LossWeights,
    decode_age_conditions,
    native_v0_raw_losses,
    weighted_native_v0_loss,
)


@dataclass(frozen=True)
class EfficientV0LossOutput:
    total: Tensor
    raw: dict[str, Tensor]
    weighted: dict[str, Tensor]
    selected_action_age: int | None


def _decode(
    action_adapter: Any,
    condition: Tensor,
    proprio: Tensor,
    noise: Tensor,
    *,
    requires_grad: bool,
) -> Tensor:
    return action_adapter.decode_action_from_condition(
        condition,
        proprio,
        steps=10,
        initial_noise=noise,
        requires_grad=requires_grad,
    )


def _normalized_condition_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    target = F.layer_norm(target.detach().float(), (target.shape[-1],))
    weight = mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
    denominator = weight.sum().clamp_min(1.0) * prediction.shape[-1]
    return ((prediction - target).square() * weight).sum() / denominator


def _mode_d_losses(
    *,
    unroll: NativeV0UnrollOutput,
    teacher_conditions: tuple[Tensor, Tensor, Tensor],
    predicted_action: Tensor,
    teacher_action: Tensor,
    valid_mask: Tensor,
    selected_age: int,
) -> dict[str, Tensor]:
    """Unbiased one-age action estimate with all-age condition objectives."""

    condition_values = [
        _normalized_condition_mse(prediction, target, valid_mask)
        for prediction, target in zip(unroll.conditions, teacher_conditions)
    ]
    regularizers = [update.residual.float().square().mean() for update in unroll.updates]
    difference = (predicted_action - teacher_action.detach()).abs()
    selected = selected_age - 1
    raw: dict[str, Tensor] = {
        "condition_normalized_mse": torch.stack(condition_values).mean(),
        "first5_action_l1": difference[:, :5].mean(),
        "full_chunk_action_l1": difference.mean(),
        "translation_l1": difference[:, :5, :3].mean(),
        "rotation_l1": difference[:, :5, 3:6].mean(),
        "continuous_gripper_l1": difference[:, :5, 6:].mean(),
        "update_regularization_mse": torch.stack(regularizers).mean(),
    }
    for age, value in enumerate(condition_values, start=1):
        raw[f"age{age}/condition_normalized_mse"] = value
        raw[f"age{age}/update_regularization_mse"] = regularizers[age - 1]
    raw[f"age{selected_age}/first5_action_l1"] = raw["first5_action_l1"]
    raw[f"age{selected_age}/full_chunk_action_l1"] = raw["full_chunk_action_l1"]
    raw[f"age{selected_age}/translation_l1"] = raw["translation_l1"]
    raw[f"age{selected_age}/rotation_l1"] = raw["rotation_l1"]
    raw[f"age{selected_age}/continuous_gripper_l1"] = raw["continuous_gripper_l1"]
    raw["selected_action_age"] = raw["first5_action_l1"].new_tensor(float(selected + 1))
    return raw


def cache_backed_v0_loss(
    *,
    adapter: Any,
    batch: dict[str, Any],
    action_adapter: Any,
    weights: NativeV0LossWeights,
    objective_mode: str,
    zero_based_optimizer_step: int,
    requires_grad: bool,
) -> EfficientV0LossOutput:
    """Compute V0 loss without any teacher VLM/action forward in the step."""

    unroll = adapter(
        batch["anchor_condition"],
        batch["image_sequence"],
        batch["proprio_sequence"],
        valid_mask=batch["valid_mask"],
        group_ids=batch["group_ids"],
    )
    teacher_conditions = tuple(
        batch["teacher_conditions"][:, age - 1].detach() for age in (1, 2, 3)
    )
    teacher_actions = tuple(
        batch["teacher_actions"][:, age - 1].detach() for age in (1, 2, 3)
    )
    proprio = tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3))
    noises = tuple(batch["explicit_noises"][:, age - 1] for age in (1, 2, 3))

    if objective_mode == "B":
        predicted = decode_age_conditions(
            lambda condition, state, noise: _decode(
                action_adapter,
                condition,
                state,
                noise,
                requires_grad=requires_grad,
            ),
            unroll.conditions,
            proprio,
            noises,
            mode="B",
        )
        raw = native_v0_raw_losses(
            unroll=unroll,
            teacher_conditions=teacher_conditions,
            predicted_actions=predicted,
            teacher_actions=teacher_actions,
            valid_mask=batch["valid_mask"],
        )
        total, weighted = weighted_native_v0_loss(raw, weights)
        return EfficientV0LossOutput(total, raw, weighted, None)

    if objective_mode != "D":
        raise ValueError("cache-backed scientific objective mode must be B or D")
    selected_age = balanced_mode_d_age(zero_based_optimizer_step)
    index = selected_age - 1
    predicted_action = _decode(
        action_adapter,
        unroll.conditions[index],
        proprio[index],
        noises[index],
        requires_grad=requires_grad,
    )
    raw = _mode_d_losses(
        unroll=unroll,
        teacher_conditions=teacher_conditions,
        predicted_action=predicted_action,
        teacher_action=teacher_actions[index],
        valid_mask=batch["valid_mask"],
        selected_age=selected_age,
    )
    total, weighted = weighted_native_v0_loss(raw, weights)
    return EfficientV0LossOutput(total, raw, weighted, selected_age)


def attach_static_group_ids(batch: dict[str, Any], group_ids: Tensor) -> dict[str, Any]:
    """Attach a precomputed token-group layout without allocating it per step."""

    expected = (batch["anchor_condition"].shape[0], batch["anchor_condition"].shape[1])
    if group_ids.ndim == 1:
        group_ids = group_ids.unsqueeze(0).expand(expected[0], -1)
    if group_ids.shape != expected:
        raise ValueError(f"group_ids must be {expected}, got {tuple(group_ids.shape)}")
    batch["group_ids"] = group_ids
    return batch
