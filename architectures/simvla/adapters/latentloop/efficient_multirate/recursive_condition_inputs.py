"""Build teacher-recorded query inputs using the live recursive Condition update."""
from dataclasses import dataclass

import torch
from torch import Tensor

from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair


@dataclass
class RecursiveConditionContext:
    condition: Tensor
    valid_mask: Tensor
    proprio: Tensor
    global_code: Tensor


def query_inputs(adapter, action, sequence, age, *, track_grad=False):
    with torch.set_grad_enabled(track_grad):
        return _query_inputs(adapter, action, sequence, age)


def _query_inputs(adapter, action, sequence, age):
    if age not in (1, 2, 3):
        raise ValueError("Recursive Condition age must be 1, 2 or 3")
    condition = sequence["anchor_condition"]
    # Never refresh from teacher_conditions inside this window. At age>1 the
    # preceding prediction, not a fresh teacher condition, is the next input.
    for offset in range(1, age + 1):
        pair = NativeV0ObservationPair(
            previous_images=sequence["image_sequence"][:, offset - 1],
            current_images=sequence["image_sequence"][:, offset],
            previous_proprio=sequence["proprio_sequence"][:, offset - 1],
            current_proprio=sequence["proprio_sequence"][:, offset],
        )
        code = adapter.delta_encoder(pair)
        condition = adapter.condition_updater(condition, code,
            valid_mask=sequence["valid_mask"], group_ids=sequence["group_ids"], age=offset).condition
    raw = sequence["proprio_sequence"][:, age]
    context = RecursiveConditionContext(condition=condition,
        valid_mask=sequence["valid_mask"].bool(), proprio=action.normalize_proprio(raw), global_code=code)
    return (context, raw, sequence["explicit_noises"][:, age - 1],
        action.action_space.normalize_action(sequence["teacher_actions"][:, age - 1]))
