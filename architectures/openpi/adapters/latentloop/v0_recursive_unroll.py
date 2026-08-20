"""Differentiable fixed-Kq=4 V0 unroll over predicted states at ages 1, 2, 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from .prefix_kv_hook import PrefixEmbeddingState, PrefixKVState
from .transition_core import OpenPITransitionOutput


@dataclass(frozen=True)
class V0AgeInput:
    current_prefix: PrefixEmbeddingState
    executed_actions: Tensor
    robot_state: Tensor
    target_state: PrefixKVState
    action_noise: Tensor
    teacher_action_chunk: Tensor


@dataclass(frozen=True)
class V0AgeOutput:
    age: int
    transition: OpenPITransitionOutput
    target_state: PrefixKVState
    action_noise: Tensor
    teacher_action_chunk: Tensor
    consumed_predicted_previous: bool


def recursive_v0_unroll(
    adapter: Any,
    anchor_state: PrefixKVState,
    steps: Sequence[V0AgeInput],
    *,
    execution_horizon: int = 5,
) -> list[V0AgeOutput]:
    if len(steps) != 3:
        raise ValueError("fixed K_q=4 V0 requires exactly three recurrent ages")
    previous_state = anchor_state
    previous_embeddings = anchor_state.embeddings
    outputs: list[V0AgeOutput] = []
    for age, step in enumerate(steps, start=1):
        if step.executed_actions.shape[1] != execution_horizon:
            raise ValueError("each V0 age must receive exactly the actually executed R-action subchunk")
        transition = adapter(
            previous_state,
            step.current_prefix,
            previous_embeddings,
            step.executed_actions,
            step.robot_state,
            delta_q=1,
            delta_a=execution_horizon,
            full_refresh_age=age,
            executed_action_lengths=torch.full(
                (step.executed_actions.shape[0],),
                execution_horizon,
                device=step.executed_actions.device,
                dtype=torch.long,
            ),
        )
        outputs.append(
            V0AgeOutput(
                age=age,
                transition=transition,
                target_state=step.target_state,
                action_noise=step.action_noise,
                teacher_action_chunk=step.teacher_action_chunk,
                consumed_predicted_previous=age > 1,
            )
        )
        previous_state = transition.state
        previous_embeddings = step.current_prefix.embeddings
    return outputs
