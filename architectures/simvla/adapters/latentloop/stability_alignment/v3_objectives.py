"""Recurrence-focused V3 objectives with fixed teacher targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from architectures.simvla.adapters.latentloop.stability_alignment.objectives import (
    first_r_per_sequence,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_contracts import (
    V3_LOSS_NAMES,
)
from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0ObservationPair,
)


@dataclass(frozen=True)
class V3ConditionPaths:
    student_recursive: tuple[Tensor, ...]
    frozen_parent_recursive: tuple[Tensor, ...]
    frozen_teacher_targets: tuple[Tensor, ...]
    change_codes: tuple[Tensor, ...]


def masked_nrms_per_sequence(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("condition tensors must share [B,T,D]")
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError("valid_mask must match [B,T]")
    mask = valid_mask.to(device=prediction.device, dtype=torch.float32).unsqueeze(-1)
    difference = prediction.float() - target.detach().float()
    numerator = (difference.square() * mask).sum(dim=(1, 2))
    denominator = (target.detach().float().square() * mask).sum(dim=(1, 2))
    return torch.sqrt((numerator / denominator.clamp_min(1e-8)).clamp_min(1e-12))


def v3_condition_paths(
    student: NativeSimVLAV0,
    frozen_parent: NativeSimVLAV0,
    batch: Mapping[str, Any],
) -> V3ConditionPaths:
    anchor = batch["anchor_condition"]
    exact = tuple(
        batch["teacher_conditions"][:, index]
        for index in range(int(batch["teacher_conditions"].shape[1]))
    )
    if len(exact) != 3:
        raise ValueError("V3 requires exactly three update ages")
    valid_mask = batch["valid_mask"]
    group_ids = batch["group_ids"]
    student_previous = anchor
    parent_previous = anchor
    student_recursive: list[Tensor] = []
    parent_recursive: list[Tensor] = []
    teacher_targets: list[Tensor] = []
    codes: list[Tensor] = []
    for index, age in enumerate((1, 2, 3)):
        pair = NativeV0ObservationPair(
            previous_images=batch["image_sequence"][:, age - 1],
            current_images=batch["image_sequence"][:, age],
            previous_proprio=batch["proprio_sequence"][:, age - 1],
            current_proprio=batch["proprio_sequence"][:, age],
        )
        code = student.delta_encoder(pair)
        update = student.condition_updater(
            student_previous,
            code,
            valid_mask=valid_mask,
            group_ids=group_ids,
            age=age,
        )
        with torch.no_grad():
            parent_code = frozen_parent.delta_encoder(pair)
            parent_update = frozen_parent.condition_updater(
                parent_previous,
                parent_code,
                valid_mask=valid_mask,
                group_ids=group_ids,
                age=age,
            )
            teacher_previous = anchor if age == 1 else exact[index - 1]
            teacher_update = frozen_parent.condition_updater(
                teacher_previous,
                parent_code,
                valid_mask=valid_mask,
                group_ids=group_ids,
                age=age,
            )
        student_recursive.append(update.condition)
        parent_recursive.append(parent_update.condition)
        teacher_targets.append(teacher_update.condition.detach())
        codes.append(code)
        student_previous = update.condition
        parent_previous = parent_update.condition
    return V3ConditionPaths(
        student_recursive=tuple(student_recursive),
        frozen_parent_recursive=tuple(parent_recursive),
        frozen_teacher_targets=tuple(teacher_targets),
        change_codes=tuple(codes),
    )


def parent_recurrence_gains(
    paths: V3ConditionPaths,
    exact_conditions: Sequence[Tensor],
    valid_mask: Tensor,
    *,
    epsilon: float = 1e-6,
) -> dict[int, Tensor]:
    if len(exact_conditions) != 3:
        raise ValueError("parent gain requires ages 1,2,3")
    gains: dict[int, Tensor] = {}
    for index, age in ((1, 2), (2, 3)):
        input_error = masked_nrms_per_sequence(
            paths.frozen_parent_recursive[index - 1],
            exact_conditions[index - 1],
            valid_mask,
        )
        output_divergence = masked_nrms_per_sequence(
            paths.frozen_parent_recursive[index],
            paths.frozen_teacher_targets[index],
            valid_mask,
        )
        gains[age] = output_divergence / input_error.detach().clamp_min(float(epsilon))
    return gains


def recurrence_gain_loss(
    paths: V3ConditionPaths,
    exact_conditions: Sequence[Tensor],
    valid_mask: Tensor,
    *,
    gamma: float,
    epsilon: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    values: list[Tensor] = []
    diagnostics: dict[str, Tensor] = {}
    for index, age, weight in ((1, 2, 1.0), (2, 3, 2.0)):
        input_error = masked_nrms_per_sequence(
            paths.student_recursive[index - 1],
            exact_conditions[index - 1],
            valid_mask,
        )
        output_divergence = masked_nrms_per_sequence(
            paths.student_recursive[index],
            paths.frozen_teacher_targets[index],
            valid_mask,
        )
        gain = output_divergence / input_error.detach().clamp_min(float(epsilon))
        values.append(float(weight) * F.relu(gain - float(gamma)).square().mean())
        diagnostics[f"gain_age{age}"] = gain.detach().mean()
        diagnostics[f"input_error_age{age - 1}"] = input_error.detach().mean()
        diagnostics[f"output_divergence_age{age}"] = output_divergence.detach().mean()
    return torch.stack(values).sum() / 3.0, diagnostics


def gripper_transition_loss_with_boundaries(
    predictions: Sequence[Tensor],
    targets: Sequence[Tensor],
    anchor_target: Tensor,
    *,
    first_r: int = 5,
    sign_margin: float = 0.10,
    switch_temperature: float = 8.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if len(predictions) != 3 or len(targets) != 3:
        raise ValueError("V3 gripper supervision requires three ages")
    age_weights = (1.0, 1.0, 2.0)
    losses: list[Tensor] = []
    continuous_values: list[Tensor] = []
    sign_values: list[Tensor] = []
    switch_values: list[Tensor] = []
    mismatch_sequences: list[Tensor] = []
    mismatch_positions: list[Tensor] = []
    switch_mismatch_sequences: list[Tensor] = []
    for index, (prediction, target, age_weight) in enumerate(
        zip(predictions, targets, age_weights)
    ):
        pred = prediction[:, : int(first_r), 6].float()
        truth = target[:, : int(first_r), 6].detach().float()
        truth_sign = torch.where(truth >= 0.0, 1.0, -1.0)
        continuous = F.l1_loss(pred, truth)
        sign = F.relu(float(sign_margin) - truth_sign * pred).mean()

        if index == 0:
            previous_pred = anchor_target[:, int(first_r) - 1, 6].detach().float()
            previous_truth = previous_pred
        else:
            previous_pred = predictions[index - 1][:, int(first_r) - 1, 6].float()
            previous_truth = targets[index - 1][:, int(first_r) - 1, 6].detach().float()
        pred_pairs_before = torch.cat((previous_pred[:, None], pred[:, :-1]), dim=1)
        pred_pairs_after = pred
        truth_pairs_before = torch.cat((previous_truth[:, None], truth[:, :-1]), dim=1)
        truth_pairs_after = truth
        target_switch = (
            (truth_pairs_before >= 0.0) != (truth_pairs_after >= 0.0)
        ).float()
        switch_logits = (
            -float(switch_temperature) * pred_pairs_before * pred_pairs_after
        )
        switch = F.binary_cross_entropy_with_logits(switch_logits, target_switch)
        losses.append(float(age_weight) * (continuous + sign + switch))
        continuous_values.append(continuous.detach())
        sign_values.append(sign.detach())
        switch_values.append(switch.detach())

        sign_mismatch = (pred >= 0.0) != (truth >= 0.0)
        predicted_switch = (pred_pairs_before >= 0.0) != (pred_pairs_after >= 0.0)
        target_switch_bool = target_switch >= 0.5
        mismatch_sequences.append(sign_mismatch.any(dim=1).float().sum())
        mismatch_positions.append(sign_mismatch.float().sum())
        switch_mismatch_sequences.append(
            (predicted_switch != target_switch_bool).any(dim=1).float().sum()
        )
    return torch.stack(losses).sum() / sum(age_weights), {
        "gripper_continuous": torch.stack(continuous_values).mean(),
        "gripper_sign": torch.stack(sign_values).mean(),
        "gripper_switch": torch.stack(switch_values).mean(),
        "mismatch_sequences": torch.stack(mismatch_sequences),
        "mismatch_positions": torch.stack(mismatch_positions),
        "switch_mismatch_sequences": torch.stack(switch_mismatch_sequences),
    }


def v3_raw_losses(
    *,
    paths: V3ConditionPaths,
    exact_conditions: Sequence[Tensor],
    ng3_actions: Sequence[Tensor],
    exact_actions: Sequence[Tensor],
    rotating_full_action: Tensor,
    rotating_age_index: int,
    anchor_teacher_action: Tensor,
    valid_mask: Tensor,
    gamma: float,
    hard_sample: bool,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    if any(len(values) != 3 for values in (exact_conditions, ng3_actions, exact_actions)):
        raise ValueError("V3 losses require exactly three ages")
    age_weights = (1.0, 1.0, 2.0)
    gain, diagnostics = recurrence_gain_loss(
        paths, exact_conditions, valid_mask, gamma=float(gamma)
    )
    exact_reference = torch.stack(
        [
            weight
            * masked_nrms_per_sequence(
                paths.student_recursive[index], exact_conditions[index], valid_mask
            ).mean()
            for index, weight in enumerate(age_weights)
        ]
    ).sum() / sum(age_weights)
    teacher_preservation = torch.stack(
        [
            weight
            * masked_nrms_per_sequence(
                paths.student_recursive[index],
                paths.frozen_teacher_targets[index],
                valid_mask,
            ).mean()
            for index, weight in enumerate(age_weights)
        ]
    ).sum() / sum(age_weights)
    per_age_action = [
        first_r_per_sequence(ng3_actions[index], exact_actions[index])
        for index in range(3)
    ]
    ng3_execution = torch.stack(
        [weight * per_age_action[index].mean() for index, weight in enumerate(age_weights)]
    ).sum() / sum(age_weights)
    gripper, gripper_diagnostics = gripper_transition_loss_with_boundaries(
        ng3_actions, exact_actions, anchor_teacher_action
    )
    rotating = first_r_per_sequence(
        rotating_full_action, exact_actions[int(rotating_age_index)]
    ).mean()
    age3_tail = per_age_action[2].mean()
    hard_execution = age3_tail if bool(hard_sample) else age3_tail * 0.0
    raw = {
        "recurrence_gain": gain,
        "exact_condition_reference": exact_reference,
        "frozen_teacher_preservation": teacher_preservation,
        "frozen_ng3_execution": ng3_execution,
        "gripper_transition": gripper,
        "rotating_full_nfe_execution": rotating,
        "hard_sequence_execution": hard_execution,
    }
    if set(raw) != set(V3_LOSS_NAMES):
        raise AssertionError("V3 objective changed loss names")
    diagnostics.update(gripper_diagnostics)
    diagnostics.update(
        {
            "age1_ng3_first_r": per_age_action[0].detach().mean(),
            "age2_ng3_first_r": per_age_action[1].detach().mean(),
            "age3_ng3_first_r": per_age_action[2].detach().mean(),
            "hard_sample": raw["recurrence_gain"].new_tensor(float(bool(hard_sample))),
        }
    )
    return raw, diagnostics


def weighted_v3_total(
    raw: Mapping[str, Tensor], weights: Mapping[str, float]
) -> tuple[Tensor, dict[str, Tensor]]:
    if set(raw) != set(V3_LOSS_NAMES) or set(weights) != set(V3_LOSS_NAMES):
        raise ValueError("V3 weighted objective changed loss names")
    weighted = {name: float(weights[name]) * raw[name] for name in V3_LOSS_NAMES}
    return torch.stack(tuple(weighted.values())).sum(), weighted
