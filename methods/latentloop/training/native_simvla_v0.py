"""Losses, action-decoder modes, and schedule for corrected SimVLA V0."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from methods.latentloop.modules.native_simvla_v0 import NativeV0UnrollOutput


ActionDecoder = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class NativeV0LossWeights:
    """Explicitly approved weights; no defaults are used by scientific training."""

    condition: float
    first5_action: float
    full_chunk_action: float
    continuous_gripper: float
    update_regularization: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _masked_normalized_condition_mse(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    target = F.layer_norm(target.detach().float(), (target.shape[-1],))
    mask = valid_mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
    denominator = mask.sum().clamp_min(1.0) * prediction.shape[-1]
    return ((prediction - target).square() * mask).sum() / denominator


def decode_age_conditions(
    decoder: ActionDecoder,
    conditions: Sequence[Tensor],
    proprioceptions: Sequence[Tensor],
    explicit_noises: Sequence[Tensor],
    *,
    mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode three ages separately (A) or as one local stacked batch (B)."""

    if not (len(conditions) == len(proprioceptions) == len(explicit_noises) == 3):
        raise ValueError("exactly age-1/2/3 inputs are required")
    if mode == "A":
        values = [
            decoder(condition, proprio, noise)
            for condition, proprio, noise in zip(conditions, proprioceptions, explicit_noises)
        ]
    elif mode == "B":
        batch_sizes = [condition.shape[0] for condition in conditions]
        if len(set(batch_sizes)) != 1:
            raise ValueError("Mode B requires equal local batch sizes for all ages")
        stacked = decoder(
            torch.cat(tuple(conditions), dim=0),
            torch.cat(tuple(proprioceptions), dim=0),
            torch.cat(tuple(explicit_noises), dim=0),
        )
        values = list(stacked.split(batch_sizes[0], dim=0))
    else:
        raise ValueError("action decode mode must be A or B")
    return values[0], values[1], values[2]


def native_v0_raw_losses(
    *,
    unroll: NativeV0UnrollOutput,
    teacher_conditions: Sequence[Tensor],
    predicted_actions: Sequence[Tensor],
    teacher_actions: Sequence[Tensor],
    valid_mask: Tensor,
    executed_prefix: int = 5,
) -> dict[str, Tensor]:
    """Compute equally weighted age-1/2/3 raw objectives and diagnostics."""

    sequences = (
        teacher_conditions,
        predicted_actions,
        teacher_actions,
        unroll.conditions,
        unroll.updates,
    )
    if any(len(sequence) != 3 for sequence in sequences):
        raise ValueError("loss contract requires exactly ages 1, 2, and 3")
    if executed_prefix != 5:
        raise ValueError("native SimVLA V0 primary first-action window is fixed at R=5")

    age_components: dict[str, list[Tensor]] = {
        "condition_normalized_mse": [],
        "first5_action_l1": [],
        "full_chunk_action_l1": [],
        "translation_l1": [],
        "rotation_l1": [],
        "continuous_gripper_l1": [],
        "update_regularization_mse": [],
    }
    result: dict[str, Tensor] = {}
    for index, age in enumerate((1, 2, 3)):
        condition = unroll.conditions[index]
        target_condition = teacher_conditions[index]
        action = predicted_actions[index]
        target_action = teacher_actions[index].detach()
        difference = (action - target_action).abs()
        if action.ndim != 3 or action.shape[1] < executed_prefix or action.shape[-1] != 7:
            raise ValueError("actions must be [B,H>=5,7]")
        values = {
            "condition_normalized_mse": _masked_normalized_condition_mse(
                condition, target_condition, valid_mask
            ),
            "first5_action_l1": difference[:, :executed_prefix].mean(),
            "full_chunk_action_l1": difference.mean(),
            "translation_l1": difference[:, :executed_prefix, :3].mean(),
            "rotation_l1": difference[:, :executed_prefix, 3:6].mean(),
            "continuous_gripper_l1": difference[:, :executed_prefix, 6:].mean(),
            "update_regularization_mse": unroll.updates[index].residual.float().square().mean(),
        }
        for name, value in values.items():
            age_components[name].append(value)
            result[f"age{age}/{name}"] = value
    for name, values in age_components.items():
        result[name] = torch.stack(values).mean()
    return result


def weighted_native_v0_loss(
    raw: dict[str, Tensor],
    weights: NativeV0LossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    weighted = {
        "condition": weights.condition * raw["condition_normalized_mse"],
        "first5_action": weights.first5_action * raw["first5_action_l1"],
        "full_chunk_action": weights.full_chunk_action * raw["full_chunk_action_l1"],
        "continuous_gripper": (
            weights.continuous_gripper * raw["continuous_gripper_l1"]
        ),
        "update_regularization": (
            weights.update_regularization * raw["update_regularization_mse"]
        ),
    }
    return torch.stack(tuple(weighted.values())).sum(), weighted


def lr_multiplier(
    optimizer_step: int,
    *,
    total_steps: int = 150_000,
    warmup_steps: int = 7_500,
    final_ratio: float = 0.1,
) -> float:
    """Linear warmup followed by cosine decay to ``final_ratio``."""

    step = min(max(int(optimizer_step), 0), int(total_steps))
    if not 0 < warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in (0,total_steps)")
    if not 0.0 <= final_ratio <= 1.0:
        raise ValueError("final_ratio must be in [0,1]")
    if step <= warmup_steps:
        return float(step) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_ratio + (1.0 - final_ratio) * cosine


class WarmupCosineController:
    """Explicit optimizer-step scheduler with checkpointable state."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        peak_lr: float = 1e-4,
        total_steps: int = 150_000,
        warmup_steps: int = 7_500,
        final_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.final_ratio = float(final_ratio)
        self.optimizer_step = 0
        self.set_step(0)

    def set_step(self, optimizer_step: int) -> float:
        self.optimizer_step = int(optimizer_step)
        lr = self.peak_lr * lr_multiplier(
            self.optimizer_step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            final_ratio=self.final_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def step(self) -> float:
        return self.set_step(self.optimizer_step + 1)

    def state_dict(self) -> dict[str, float | int]:
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "final_ratio": self.final_ratio,
            "optimizer_step": self.optimizer_step,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        expected = {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "final_ratio": self.final_ratio,
        }
        mismatches = {
            key: (expected[key], state.get(key))
            for key in expected
            if expected[key] != state.get(key)
        }
        if mismatches:
            raise ValueError(f"scheduler contract mismatch: {mismatches}")
        self.set_step(int(state["optimizer_step"]))


def flattened_gradients(module: nn.Module) -> Tensor:
    """Concatenate gradients, inserting zeros for absent gradients."""

    values = [
        parameter.grad.detach().float().reshape(-1)
        if parameter.grad is not None
        else torch.zeros_like(parameter.detach(), dtype=torch.float32).reshape(-1)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return torch.cat(values) if values else torch.empty(0)
