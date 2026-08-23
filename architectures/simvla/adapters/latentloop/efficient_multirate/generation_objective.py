"""Local-oracle objective for the schedule-independent SimVLA Generation Loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    full_generation_step_with_hidden,
)
from methods.latentloop.modules.simvla_generation_loop import (
    GenerationFlowTrace,
    SimVLAGenerationLoop,
)


@dataclass(frozen=True)
class GenerationLossOutput:
    total: Tensor
    hidden_normalized_mse: Tensor
    velocity_l1: Tensor
    final_action_l1: Tensor
    trace: GenerationFlowTrace
    oracle_calls: int
    oracle_batched_positions: int


def generation_local_oracle_loss(
    *,
    loop: SimVLAGenerationLoop,
    transformer: Any,
    action_space: Any,
    condition: Tensor,
    initial_noise: Tensor,
    normalized_proprio: Tensor,
    condition_valid_mask: Tensor | None,
    condition_change_code: Tensor,
    full_step_indices: Sequence[int],
    teacher_final_action: Tensor,
    hidden_weight: float = 1.0,
    velocity_weight: float = 0.1,
    final_action_weight: float = 0.1,
) -> GenerationLossOutput:
    """Roll out the student, then batch all student-state local-oracle calls once."""

    def full_step(noisy_action: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            output = full_generation_step_with_hidden(
                transformer,
                condition=condition,
                noisy_action=noisy_action.detach(),
                proprio=normalized_proprio,
                tau=tau,
                dt=-0.1,
            )
        return output.action_hidden, output.velocity

    trace = loop(
        initial_noise,
        full_step=full_step,
        full_step_indices=full_step_indices,
        proprio=normalized_proprio,
        condition=condition,
        condition_valid_mask=condition_valid_mask,
        condition_change_code=condition_change_code,
    )
    if not trace.predicted_hidden:
        zero = trace.final_noisy_action.sum() * 0.0
        final = F.l1_loss(action_space.postprocess(trace.final_noisy_action), teacher_final_action)
        return GenerationLossOutput(
            total=float(final_action_weight) * final,
            hidden_normalized_mse=zero,
            velocity_l1=zero,
            final_action_l1=final,
            trace=trace,
            oracle_calls=0,
            oracle_batched_positions=0,
        )

    batch = condition.shape[0]
    positions = len(trace.predicted_hidden)
    oracle_condition = condition.repeat(positions, 1, 1)
    oracle_proprio = normalized_proprio.repeat(positions, 1)
    oracle_action = torch.cat([value.detach() for value in trace.skipped_noisy_actions], dim=0)
    oracle_tau = torch.cat([value.detach() for value in trace.skipped_times], dim=0)
    with torch.no_grad():
        oracle = full_generation_step_with_hidden(
            transformer,
            condition=oracle_condition,
            noisy_action=oracle_action,
            proprio=oracle_proprio,
            tau=oracle_tau,
            dt=-0.1,
        )
    oracle_hidden = oracle.action_hidden.split(batch, dim=0)
    oracle_velocity = oracle.velocity.split(batch, dim=0)
    hidden_losses = [
        F.mse_loss(
            F.layer_norm(predicted.float(), (predicted.shape[-1],)),
            F.layer_norm(target.float(), (target.shape[-1],)),
        )
        for predicted, target in zip(trace.predicted_hidden, oracle_hidden)
    ]
    velocity_losses = [
        F.l1_loss(predicted.float(), target.float())
        for predicted, target in zip(trace.predicted_velocity, oracle_velocity)
    ]
    hidden_loss = torch.stack(hidden_losses).mean()
    velocity_loss = torch.stack(velocity_losses).mean()
    final_action = action_space.postprocess(trace.final_noisy_action)
    final_loss = F.l1_loss(final_action.float(), teacher_final_action.detach().float())
    total = (
        float(hidden_weight) * hidden_loss
        + float(velocity_weight) * velocity_loss
        + float(final_action_weight) * final_loss
    )
    return GenerationLossOutput(
        total=total,
        hidden_normalized_mse=hidden_loss,
        velocity_l1=velocity_loss,
        final_action_l1=final_loss,
        trace=trace,
        oracle_calls=1,
        oracle_batched_positions=positions,
    )
