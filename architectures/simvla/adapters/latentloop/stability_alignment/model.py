"""Warm-started Condition and Generation components without new architecture."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    full_generation_step_with_hidden,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    GENERATION_NG3_FULL_INDICES,
)
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.simvla_generation_loop import (
    GenerationFlowTrace,
    SimVLAGenerationHiddenUpdater,
    SimVLAGenerationLoop,
)


@dataclass(frozen=True)
class JointRollout:
    actions: tuple[Tensor, ...]
    trace: GenerationFlowTrace
    student_seconds: float


class StabilityAlignedModules(nn.Module):
    """Existing Condition and Generation components; no new architecture."""

    def __init__(
        self,
        condition: NativeSimVLAV0,
        generation: SimVLAGenerationHiddenUpdater,
    ) -> None:
        super().__init__()
        self.condition = condition
        self.generation = generation

    def parameter_audit(self) -> dict[str, Any]:
        condition = self.condition.parameter_audit()
        generation = self.generation.parameter_audit()
        return {
            "condition": condition,
            "generation": generation,
            "total_parameters": sum(value.numel() for value in self.parameters()),
            "new_observation_encoder": False,
            "new_action_head": False,
            "new_latent_representation": False,
            "condition_code_projection_shape": list(
                self.generation.condition_code_projection.weight.shape
            ),
        }


def load_warm_start(
    *,
    condition_checkpoint: str,
    generation_checkpoint: str,
    device: torch.device | str,
) -> tuple[
    StabilityAlignedModules,
    NativeSimVLAV0,
    SimVLAGenerationHiddenUpdater,
    dict[str, Any],
]:
    condition, condition_payload = load_native_v0_checkpoint(
        condition_checkpoint, device=device, require_final_150k=False
    )
    generation, generation_payload = load_generation_checkpoint(
        generation_checkpoint, device=device
    )
    parent_condition = copy.deepcopy(condition).to(device).eval()
    parent_generation = copy.deepcopy(generation).to(device).eval()
    for parameter in parent_condition.parameters():
        parameter.requires_grad_(False)
    for parameter in parent_generation.parameters():
        parameter.requires_grad_(False)
    modules = StabilityAlignedModules(condition, generation).to(device)
    return modules, parent_condition, parent_generation, {
        "condition_payload": condition_payload,
        "generation_payload": generation_payload,
        "initialization": {
            "condition": "exact parent state",
            "generation": "exact validated parent state; fully frozen",
            "condition_code_projection": "exact validated parent state; fully frozen",
            "zero_code_parent_parity": True,
        },
    }


def configure_condition_only_stage(modules: StabilityAlignedModules) -> dict[str, Any]:
    for parameter in modules.parameters():
        parameter.requires_grad_(False)
    for parameter in modules.condition.condition_updater.parameters():
        parameter.requires_grad_(True)
    for parameter in modules.condition.delta_encoder.parameters():
        parameter.requires_grad_(True)
    trainable = [
        name for name, parameter in modules.named_parameters() if parameter.requires_grad
    ]
    return {
        "training_mode": "condition_only",
        "generation_updater_frozen": True,
        "condition_code_projection_frozen": True,
        "trainable_names": trainable,
        "trainable_parameters": sum(
            parameter.numel() for parameter in modules.parameters() if parameter.requires_grad
        ),
    }


def optimizer_parameter_groups(
    modules: StabilityAlignedModules,
    *,
    base_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "condition_updater",
            "params": list(modules.condition.condition_updater.parameters()),
            "peak_lr": float(base_lr),
            "lr": 0.0,
            "weight_decay": float(weight_decay),
        },
        {
            "name": "observation_change_encoder",
            "params": list(modules.condition.delta_encoder.parameters()),
            "peak_lr": 0.40 * float(base_lr),
            "lr": 0.0,
            "weight_decay": float(weight_decay),
        },
    ]


def _stack(values: Sequence[Tensor]) -> tuple[Tensor, int]:
    if not values:
        raise ValueError("age stack cannot be empty")
    batch_sizes = {int(value.shape[0]) for value in values}
    if len(batch_sizes) != 1:
        raise ValueError("age tensors have different local batch sizes")
    batch = batch_sizes.pop()
    return torch.cat(tuple(values), dim=0), batch


def split_ages(value: Tensor, batch: int) -> tuple[Tensor, ...]:
    pieces = value.split(int(batch), dim=0)
    if not pieces or any(int(piece.shape[0]) != int(batch) for piece in pieces):
        raise RuntimeError("stacked generation output did not preserve age batches")
    return tuple(pieces)


def generation_rollout(
    *,
    updater: SimVLAGenerationHiddenUpdater,
    transformer: Any,
    action_space: Any,
    conditions: Sequence[Tensor],
    change_codes: Sequence[Tensor],
    proprio: Sequence[Tensor],
    noises: Sequence[Tensor],
    valid_mask: Tensor,
    optimizer_step: int,
    requires_grad: bool,
    instrument: bool = False,
) -> JointRollout:
    condition, batch = _stack(conditions)
    code, code_batch = _stack(change_codes)
    qpos, qpos_batch = _stack(proprio)
    noise, noise_batch = _stack(noises)
    if {batch, code_batch, qpos_batch, noise_batch} != {batch}:
        raise RuntimeError("generation age stack changed batch size")
    mask = valid_mask.repeat(len(conditions), 1)
    normalized_qpos = action_space.normalize_state(qpos)
    loop = SimVLAGenerationLoop(updater, transformer.action_decoder).to(condition.device)

    def full_step(noisy_action: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
        output = full_generation_step_with_hidden(
            transformer,
            condition=condition,
            noisy_action=noisy_action,
            proprio=normalized_qpos,
            tau=tau,
            dt=-0.1,
        )
        return output.action_hidden, output.velocity

    if instrument and condition.is_cuda:
        torch.cuda.synchronize(condition.device)
    student_started = __import__("time").perf_counter()
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        trace = loop(
            noise,
            full_step=full_step,
            full_step_indices=GENERATION_NG3_FULL_INDICES,
            proprio=normalized_qpos,
            condition=condition,
            condition_valid_mask=mask,
            condition_change_code=code,
        )
        actions = action_space.postprocess(trace.final_noisy_action)
    if instrument and condition.is_cuda:
        torch.cuda.synchronize(condition.device)
    student_seconds = __import__("time").perf_counter() - student_started

    return JointRollout(
        actions=split_ages(actions, batch),
        trace=trace,
        student_seconds=student_seconds,
    )


def zero_code_parity(
    parent: SimVLAGenerationHiddenUpdater,
    candidate: SimVLAGenerationHiddenUpdater,
    *,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(20260825)
    kwargs = {
        "previous_hidden": torch.randn(1, 10, 1024, generator=generator, device=device),
        "noisy_action_before": torch.randn(1, 10, 7, generator=generator, device=device),
        "noisy_action_after": torch.randn(1, 10, 7, generator=generator, device=device),
        "tau_before": torch.full((1,), 0.8, device=device),
        "tau_after": torch.full((1,), 0.7, device=device),
        "proprio": torch.randn(1, 8, generator=generator, device=device),
        "condition_change_code": torch.zeros(1, 128, device=device),
        "condition": torch.randn(1, 122, 960, generator=generator, device=device),
        "condition_valid_mask": torch.ones(1, 122, dtype=torch.bool, device=device),
        "generator_age": 1,
    }
    with torch.no_grad():
        expected = parent(**kwargs)
        observed = candidate(**kwargs)
    checks = {
        "hidden": torch.equal(expected.hidden, observed.hidden),
        "residual": torch.equal(expected.residual, observed.residual),
        "gate": torch.equal(expected.gate, observed.gate),
    }
    return {
        "verdict": "ZERO_CODE_PARENT_PARITY_PASS" if all(checks.values()) else "ZERO_CODE_PARENT_PARITY_FAIL",
        "checks": checks,
    }
