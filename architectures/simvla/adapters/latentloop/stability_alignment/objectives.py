"""Differentiable losses for recurrent Condition stability alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0ObservationPair,
)


LOSS_NAMES = (
    "recursive_reference",
    "teacher_forced_preservation",
    "recursive_stability",
    "end_to_end_execution",
    "gripper_transition",
    "tail_cvar",
    "rotating_full_nfe_execution",
    "parent_preservation",
)


@dataclass(frozen=True)
class ConditionPaths:
    recursive: tuple[Tensor, ...]
    teacher_forced: tuple[Tensor, ...]
    change_codes: tuple[Tensor, ...]
    residuals: tuple[Tensor, ...]


def masked_nrms(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("condition tensors must share [B,T,D]")
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError("valid_mask must match [B,T]")
    mask = valid_mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
    difference = prediction.float() - target.detach().float()
    target_scale = target.detach().float().square()
    denominator = (target_scale * mask).sum().clamp_min(1e-8)
    return torch.sqrt(((difference.square() * mask).sum() / denominator).clamp_min(1e-12))


def first_r_per_sequence(prediction: Tensor, target: Tensor, first_r: int = 5) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("actions must share [B,H,7]")
    return (prediction[:, :first_r].float() - target[:, :first_r].detach().float()).abs().mean(
        dim=(1, 2)
    )


def first_r_action_distance(prediction: Tensor, target: Tensor, first_r: int = 5) -> Tensor:
    return first_r_per_sequence(prediction, target, first_r).mean()


def bounded_top_cvar(values: Tensor, fraction: float = 0.10) -> Tensor:
    flattened = values.reshape(-1)
    if not flattened.numel():
        raise ValueError("CVaR needs at least one value")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("CVaR fraction must be in (0,1]")
    count = max(1, int(round(flattened.numel() * float(fraction))))
    count = min(count, flattened.numel())
    return torch.topk(flattened, k=count, largest=True, sorted=False).values.mean()


def gripper_transition_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    first_r: int = 5,
    sign_margin: float = 0.10,
    switch_temperature: float = 8.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    pred = prediction[:, :first_r, 6].float()
    truth = target[:, :first_r, 6].detach().float()
    target_sign = torch.where(truth >= 0.0, torch.ones_like(truth), -torch.ones_like(truth))
    continuous = F.l1_loss(pred, truth)
    sign = F.relu(float(sign_margin) - target_sign * pred).mean()
    target_switch = (target_sign[:, 1:] != target_sign[:, :-1]).float()
    switch_logits = -float(switch_temperature) * pred[:, 1:] * pred[:, :-1]
    switch_probability = torch.sigmoid(switch_logits)
    switch = F.binary_cross_entropy_with_logits(switch_logits, target_switch)
    position = torch.arange(
        target_switch.shape[1], device=prediction.device, dtype=prediction.dtype
    ).unsqueeze(0)
    target_count = target_switch.sum(dim=1).clamp_min(1.0)
    target_timing = (target_switch * position).sum(dim=1) / target_count
    predicted_count = switch_probability.sum(dim=1).clamp_min(1e-6)
    predicted_timing = (switch_probability * position).sum(dim=1) / predicted_count
    event_present = target_switch.sum(dim=1) > 0
    timing = (
        (predicted_timing[event_present] - target_timing[event_present]).abs().mean()
        if bool(event_present.any())
        else prediction.sum() * 0.0
    )
    total = continuous + sign + switch + 0.25 * timing
    return total, {
        "continuous": continuous,
        "sign_margin": sign,
        "switch_event": switch,
        "switch_timing": timing,
        "event_fraction": event_present.float().mean(),
    }


def condition_paths(
    adapter: NativeSimVLAV0,
    batch: Mapping[str, Any],
) -> ConditionPaths:
    anchor = batch["anchor_condition"]
    age_count = int(batch["teacher_conditions"].shape[1])
    if age_count not in {1, 2, 3, 4, 5, 6, 7}:
        raise ValueError("Condition path requires 1--7 update ages")
    exact = tuple(batch["teacher_conditions"][:, index] for index in range(age_count))
    valid_mask = batch["valid_mask"]
    group_ids = batch["group_ids"]
    recursive_previous = anchor
    recursive: list[Tensor] = []
    teacher_forced: list[Tensor] = []
    codes: list[Tensor] = []
    residuals: list[Tensor] = []
    for index, age in enumerate(range(1, age_count + 1)):
        pair = NativeV0ObservationPair(
            previous_images=batch["image_sequence"][:, age - 1],
            current_images=batch["image_sequence"][:, age],
            previous_proprio=batch["proprio_sequence"][:, age - 1],
            current_proprio=batch["proprio_sequence"][:, age],
        )
        code = adapter.delta_encoder(pair)
        recursive_update = adapter.condition_updater(
            recursive_previous,
            code,
            valid_mask=valid_mask,
            group_ids=group_ids,
            age=age,
        )
        teacher_previous = anchor if age == 1 else exact[index - 1]
        teacher_update = adapter.condition_updater(
            teacher_previous,
            code,
            valid_mask=valid_mask,
            group_ids=group_ids,
            age=age,
        )
        recursive.append(recursive_update.condition)
        teacher_forced.append(teacher_update.condition)
        codes.append(code)
        residuals.append(recursive_update.residual)
        recursive_previous = recursive_update.condition
    return ConditionPaths(
        recursive=tuple(recursive),
        teacher_forced=tuple(teacher_forced),
        change_codes=tuple(codes),
        residuals=tuple(residuals),
    )


def stability_pair_loss(recursive: Tensor, teacher_forced: Tensor, valid_mask: Tensor) -> Tensor:
    """The teacher-forced target is intentionally detached."""

    return masked_nrms(recursive, teacher_forced.detach(), valid_mask)


def stability_raw_losses(
    *,
    paths: ConditionPaths,
    parent_paths: ConditionPaths,
    exact_conditions: Sequence[Tensor],
    rotating_recursive_full_action: Tensor,
    rotating_teacher_forced_action: Tensor,
    rotating_age_index: int,
    joint_actions: Sequence[Tensor],
    parent_joint_actions: Sequence[Tensor],
    exact_actions: Sequence[Tensor],
    valid_mask: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    sequences = (
        paths.recursive,
        paths.teacher_forced,
        parent_paths.recursive,
        exact_conditions,
        joint_actions,
        exact_actions,
    )
    age_count = len(paths.recursive)
    if not age_count or any(len(value) != age_count for value in sequences):
        raise ValueError("stability objective age sequences changed length")
    if len(parent_joint_actions) < min(2, age_count):
        raise ValueError("age-1/K_C=2 parent actions are missing")
    if not 0 <= int(rotating_age_index) < age_count:
        raise ValueError("rotating full-NFE age is outside this batch")
    cond_reference: list[Tensor] = []
    tf_condition_preservation: list[Tensor] = []
    recurrence_condition: list[Tensor] = []
    execution: list[Tensor] = []
    per_sequence_tail: list[Tensor] = []
    gripper_values: list[Tensor] = []
    gripper_diag: dict[str, list[Tensor]] = {}
    parent: list[Tensor] = []
    for index in range(age_count):
        cond_reference.append(
            masked_nrms(paths.recursive[index], exact_conditions[index], valid_mask)
        )
        tf_condition_preservation.append(
            masked_nrms(paths.teacher_forced[index], exact_conditions[index], valid_mask)
        )
        recurrence_condition.append(
            stability_pair_loss(
                paths.recursive[index], paths.teacher_forced[index], valid_mask
            )
        )
        execution.append(first_r_action_distance(joint_actions[index], exact_actions[index]))
        per_sequence_tail.append(first_r_per_sequence(joint_actions[index], exact_actions[index]))
        grip, details = gripper_transition_loss(joint_actions[index], exact_actions[index])
        gripper_values.append(grip)
        for name, value in details.items():
            gripper_diag.setdefault(name, []).append(value)
        if index < 2:
            parent.append(
                masked_nrms(
                    paths.recursive[index], parent_paths.recursive[index], valid_mask
                )
                + first_r_action_distance(
                    joint_actions[index], parent_joint_actions[index]
                )
            )
    rotating_target = exact_actions[int(rotating_age_index)]
    tf_action_preservation = first_r_action_distance(
        rotating_teacher_forced_action, rotating_target
    )
    recurrence_action = first_r_action_distance(
        rotating_recursive_full_action,
        rotating_teacher_forced_action.detach(),
    )
    raw = {
        "recursive_reference": torch.stack(cond_reference).mean(),
        "teacher_forced_preservation": torch.stack(tf_condition_preservation).mean()
        + tf_action_preservation,
        "recursive_stability": torch.stack(recurrence_condition).mean()
        + recurrence_action,
        "end_to_end_execution": torch.stack(execution).mean(),
        "gripper_transition": torch.stack(gripper_values).mean(),
        "tail_cvar": bounded_top_cvar(torch.cat(per_sequence_tail), 0.10),
        "rotating_full_nfe_execution": first_r_action_distance(
            rotating_recursive_full_action, rotating_target
        ),
        "parent_preservation": torch.stack(parent).mean(),
    }
    diagnostics = {
        f"gripper/{name}": torch.stack(values).mean()
        for name, values in gripper_diag.items()
    }
    diagnostics.update(
        {
            "rotating_age": raw["recursive_reference"].new_tensor(
                float(int(rotating_age_index) + 1)
            ),
            "rotating_recursive_first_r": raw["rotating_full_nfe_execution"],
            "rotating_teacher_forced_first_r": tf_action_preservation,
            "joint_first_r": torch.stack(execution).mean(),
        }
    )
    return raw, diagnostics


def weighted_total(
    raw: Mapping[str, Tensor], weights: Mapping[str, float]
) -> tuple[Tensor, dict[str, Tensor]]:
    if set(raw) != set(LOSS_NAMES):
        raise ValueError(f"raw loss names changed: {sorted(raw)}")
    if set(weights) != set(LOSS_NAMES):
        raise ValueError(f"loss-weight names changed: {sorted(weights)}")
    weighted = {name: float(weights[name]) * raw[name] for name in LOSS_NAMES}
    total = torch.stack(tuple(weighted.values())).sum()
    return total, weighted


def calibrate_loss_weights(
    raw_means: Mapping[str, float],
    gradient_norms: Mapping[str, float],
    contribution_targets: Mapping[str, float],
    *,
    gradient_budget: float = 1.0,
) -> dict[str, Any]:
    if set(raw_means) != set(LOSS_NAMES) or set(gradient_norms) != set(LOSS_NAMES):
        raise ValueError("calibration loss names changed")
    if set(contribution_targets) != set(LOSS_NAMES):
        raise ValueError("contribution target names changed")
    if not math.isclose(
        sum(float(value) for value in contribution_targets.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("contribution targets must sum to one")
    invalid_raw = {
        name: float(raw_means[name])
        for name in LOSS_NAMES
        if not math.isfinite(float(raw_means[name])) or float(raw_means[name]) < 0.0
    }
    invalid_gradients = {
        name: float(gradient_norms[name])
        for name in LOSS_NAMES
        if not math.isfinite(float(gradient_norms[name]))
        or float(gradient_norms[name]) <= 1e-10
    }
    if invalid_raw:
        raise ValueError(f"non-finite or negative raw calibration losses: {invalid_raw}")
    if invalid_gradients:
        raise ValueError(
            "calibration requires a measured nonzero gradient for every loss: "
            f"{invalid_gradients}"
        )
    if not math.isfinite(float(gradient_budget)) or float(gradient_budget) <= 0.0:
        raise ValueError("gradient_budget must be finite and positive")
    epsilon = 1e-12
    raw_scale_weights = {
        name: float(contribution_targets[name]) / max(float(raw_means[name]), epsilon)
        for name in LOSS_NAMES
    }
    raw_scale_gradient_norms = {
        name: raw_scale_weights[name] * max(float(gradient_norms[name]), epsilon)
        for name in LOSS_NAMES
    }
    raw_values = list(raw_scale_gradient_norms.values())
    raw_gradient_spread = max(raw_values) / max(min(raw_values), epsilon)
    conflict = raw_gradient_spread > 100.0
    if conflict:
        weights = {
            name: float(contribution_targets[name])
            / max(float(gradient_norms[name]), epsilon)
            for name in LOSS_NAMES
        }
        expected_raw_total = sum(
            weights[name] * max(float(raw_means[name]), epsilon)
            for name in LOSS_NAMES
        )
        normalization = 1.0 / max(expected_raw_total, epsilon)
        weights = {name: value * normalization for name, value in weights.items()}
    else:
        weights = raw_scale_weights
    weighted_gradient_norms_before_global_scale = {
        name: weights[name] * max(float(gradient_norms[name]), epsilon)
        for name in LOSS_NAMES
    }
    gradient_l1_before_global_scale = sum(
        weighted_gradient_norms_before_global_scale.values()
    )
    global_weight_scale = min(
        1.0,
        float(gradient_budget) / max(gradient_l1_before_global_scale, epsilon),
    )
    weights = {
        name: value * global_weight_scale for name, value in weights.items()
    }
    weighted_gradient_norms = {
        name: weights[name] * max(float(gradient_norms[name]), epsilon)
        for name in LOSS_NAMES
    }
    weighted_raw_values = {
        name: weights[name] * max(float(raw_means[name]), epsilon)
        for name in LOSS_NAMES
    }
    weighted_raw_total = sum(weighted_raw_values.values())
    weighted_raw_fractions = {
        name: value / max(weighted_raw_total, epsilon)
        for name, value in weighted_raw_values.items()
    }
    values = list(weighted_gradient_norms.values())
    gradient_spread = max(values) / max(min(values), epsilon)
    return {
        "weights": weights,
        "raw_scale_weights_before_conflict_handling": raw_scale_weights,
        "raw_means": {name: float(raw_means[name]) for name in LOSS_NAMES},
        "gradient_norms": {name: float(gradient_norms[name]) for name in LOSS_NAMES},
        "weighted_gradient_norms_before_global_scale": (
            weighted_gradient_norms_before_global_scale
        ),
        "weighted_gradient_norms": weighted_gradient_norms,
        "weighted_gradient_l1_before_global_scale": float(
            gradient_l1_before_global_scale
        ),
        "weighted_gradient_l1_after_global_scale": float(
            sum(weighted_gradient_norms.values())
        ),
        "gradient_budget": float(gradient_budget),
        "global_weight_scale": float(global_weight_scale),
        "weighted_raw_values": weighted_raw_values,
        "weighted_raw_fractions": weighted_raw_fractions,
        "gradient_norm_spread": float(gradient_spread),
        "gradient_norm_spread_before_conflict_handling": float(raw_gradient_spread),
        "severe_gradient_scale_conflict_observed": bool(conflict),
        "gradient_norm_balancing_applied": bool(conflict),
        "calibration_strategy": (
            "fixed_gradient_norm_balancing" if conflict else "raw_semantic_contribution"
        ),
        "contribution_targets": dict(contribution_targets),
    }
