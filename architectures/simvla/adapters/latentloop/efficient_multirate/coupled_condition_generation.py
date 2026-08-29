"""Shared condition-code coupling primitives for SimVLA LatentLoop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0ObservationPair,
    NativeV0UpdateOutput,
)
from methods.latentloop.modules.simvla_generation_loop import (
    SimVLAGenerationHiddenUpdater,
)


COUPLED_K_C_VALUES = (2, 3)
COUPLED_N_G_VALUES = (2, 3, 5)
COUPLED_CONFIGS = tuple(
    (k_c, n_g) for k_c in COUPLED_K_C_VALUES for n_g in COUPLED_N_G_VALUES
)


def coupled_row_name(k_c: int, n_g: int = 3) -> str:
    if int(k_c) not in {2, 3}:
        raise ValueError("coupled condition refresh interval must be 2 or 3")
    if int(n_g) not in COUPLED_N_G_VALUES:
        raise ValueError("coupled generation requires N_G in {2,3,5}")
    return f"condition_kc{int(k_c)}_ng{int(n_g)}_coupled"


COUPLED_ROW = coupled_row_name(2)
COUPLED_KC3_ROW = coupled_row_name(3)
COUPLED_ROWS = tuple(coupled_row_name(k_c, n_g) for k_c, n_g in COUPLED_CONFIGS)
COUPLED_CHECKPOINT_SCHEMA = "simvla_condition_generation_coupling_v1"


@dataclass(frozen=True)
class ConditionUpdateWithCode:
    """One Condition update and the exact delta code consumed by that update."""

    update: NativeV0UpdateOutput
    condition_change_code: Tensor


def condition_update_with_code(
    adapter: NativeSimVLAV0,
    previous_condition: Tensor,
    pair: NativeV0ObservationPair,
    *,
    valid_mask: Tensor,
    group_ids: Tensor,
    age: Tensor | int,
) -> ConditionUpdateWithCode:
    """Expose the existing delta feature without adding another encoder call."""

    code = adapter.delta_encoder(pair)
    update = adapter.condition_updater(
        previous_condition,
        code,
        valid_mask=valid_mask,
        group_ids=group_ids,
        age=age,
    )
    return ConditionUpdateWithCode(update=update, condition_change_code=code)


def build_coupled_query(
    adapter: NativeSimVLAV0,
    sequence: dict[str, Any],
    *,
    query_ages: Tensor,
    k_c: int,
) -> dict[str, Tensor]:
    """Build coupled queries from exact q0-q1-q2-q3 cache windows."""

    batch = int(sequence["anchor_condition"].shape[0])
    k_c = int(k_c)
    if k_c not in {2, 3}:
        raise ValueError("coupled query construction requires K_C in {2,3}")
    if query_ages.shape != (batch,):
        raise ValueError(f"query_ages must be {(batch,)}, got {tuple(query_ages.shape)}")
    if bool((query_ages < 1).any()) or bool((query_ages > 3).any()):
        raise ValueError("query ages must be in {1,2,3}")

    conditions: list[Tensor] = []
    codes: list[Tensor] = []
    for index, raw_age in enumerate(query_ages.detach().cpu().tolist()):
        age = int(raw_age)
        local_age = age % k_c
        if local_age == 0:
            conditions.append(
                sequence["teacher_conditions"][index : index + 1, age - 1]
            )
            codes.append(
                sequence["anchor_condition"].new_zeros((1, adapter.delta_dim))
            )
            continue

        refresh_age = age - local_age
        condition = (
            sequence["anchor_condition"][index : index + 1]
            if refresh_age == 0
            else sequence["teacher_conditions"][index : index + 1, refresh_age - 1]
        )
        exposed: ConditionUpdateWithCode | None = None
        for current_age in range(refresh_age + 1, age + 1):
            pair = NativeV0ObservationPair(
                previous_images=sequence["image_sequence"][
                    index : index + 1, current_age - 1
                ],
                current_images=sequence["image_sequence"][
                    index : index + 1, current_age
                ],
                previous_proprio=sequence["proprio_sequence"][
                    index : index + 1, current_age - 1
                ],
                current_proprio=sequence["proprio_sequence"][
                    index : index + 1, current_age
                ],
            )
            exposed = condition_update_with_code(
                adapter,
                condition,
                pair,
                valid_mask=sequence["valid_mask"][index : index + 1],
                group_ids=sequence["group_ids"][index : index + 1],
                age=current_age - refresh_age,
            )
            condition = exposed.update.condition
        if exposed is None:
            raise RuntimeError("updated coupled query produced no Condition update")
        conditions.append(condition)
        codes.append(exposed.condition_change_code)

    batch_indices = torch.arange(batch, device=query_ages.device)
    target_indices = query_ages - 1
    return {
        "condition": torch.cat(conditions, dim=0),
        "condition_change_code": torch.cat(codes, dim=0),
        "valid_mask": sequence["valid_mask"],
        "proprio": sequence["proprio_sequence"][batch_indices, query_ages],
        "initial_noise": sequence["explicit_noises"][batch_indices, target_indices],
        "teacher_action": sequence["teacher_actions"][batch_indices, target_indices],
        "query_age_in_window": query_ages,
    }


def build_kc2_coupled_query(
    adapter: NativeSimVLAV0,
    sequence: dict[str, Any],
    *,
    query_ages: Tensor,
) -> dict[str, Tensor]:
    """Compatibility wrapper for the original K_C=2 coupling lane."""

    return build_coupled_query(
        adapter,
        sequence,
        query_ages=query_ages,
        k_c=2,
    )


def prepare_projection_only_coupling(
    updater: SimVLAGenerationHiddenUpdater,
) -> dict[str, Any]:
    """Freeze the parent updater and neutral-initialize only the c_j weight."""

    for parameter in updater.parameters():
        parameter.requires_grad_(False)
    weight = updater.condition_code_projection.weight
    with torch.no_grad():
        weight.zero_()
    weight.requires_grad_(True)
    trainable = [
        name for name, parameter in updater.named_parameters() if parameter.requires_grad
    ]
    if trainable != ["condition_code_projection.weight"]:
        raise RuntimeError(f"unexpected coupled trainable parameters: {trainable}")
    return {
        "trainable_parameter_names": trainable,
        "trainable_parameters": int(weight.numel()),
        "condition_code_dim": int(updater.condition_code_dim),
        "rank_dim": int(updater.rank_dim),
        "zero_code_preserves_parent_path": True,
        "condition_code_projection_bias_frozen": True,
    }


def audit_projection_only_state(
    parent: SimVLAGenerationHiddenUpdater,
    candidate: SimVLAGenerationHiddenUpdater,
) -> dict[str, Any]:
    """Prove that c_j's projection weight is the only changed state tensor."""

    parent_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    same_keys = tuple(parent_state) == tuple(candidate_state)
    changed = (
        [
            name
            for name in parent_state
            if name in candidate_state
            and not torch.equal(parent_state[name], candidate_state[name])
        ]
        if same_keys
        else []
    )
    checks = {
        "same_state_keys": same_keys,
        "only_projection_weight_changed": changed
        == ["condition_code_projection.weight"],
        "projection_bias_unchanged": (
            same_keys
            and torch.equal(
                parent_state["condition_code_projection.bias"],
                candidate_state["condition_code_projection.bias"],
            )
        ),
    }
    return {
        "verdict": (
            "PROJECTION_ONLY_STATE_PASS"
            if all(checks.values())
            else "PROJECTION_ONLY_STATE_FAIL"
        ),
        "checks": checks,
        "changed_state_names": changed,
    }
