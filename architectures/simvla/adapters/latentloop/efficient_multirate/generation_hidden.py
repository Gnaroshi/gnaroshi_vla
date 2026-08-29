"""Noninvasive hook for SimVLA action-token hidden states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class FullGenerationStep:
    action_hidden: Tensor
    velocity: Tensor
    next_noisy_action: Tensor
    tau: Tensor
    dt: float


def full_generation_step_with_hidden(
    transformer: Any,
    *,
    condition: Tensor,
    noisy_action: Tensor,
    proprio: Tensor,
    tau: Tensor,
    dt: float,
) -> FullGenerationStep:
    """Run the frozen original transformer and capture decoder input exactly."""

    if bool(getattr(transformer, "use_adaln", False)):
        raise RuntimeError("concat-mode hidden hook is required for released SimVLA-LIBERO")
    decoder = getattr(transformer, "action_decoder", None)
    if decoder is None:
        raise RuntimeError("original concat action_decoder was not found")
    captured: list[Tensor] = []

    def capture(_module: Any, inputs: tuple[Tensor, ...]) -> None:
        if len(inputs) != 1:
            raise RuntimeError("unexpected action_decoder input signature")
        captured.append(inputs[0])

    handle = decoder.register_forward_pre_hook(capture)
    try:
        velocity = transformer(
            vlm_features=condition,
            action_with_noise=noisy_action,
            proprio=proprio,
            t=tau,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected exactly one decoder pre-hook call, got {len(captured)}")
    hidden = captured[0]
    replay_velocity = decoder(hidden)
    if not torch.equal(replay_velocity, velocity):
        difference = (replay_velocity.float() - velocity.float()).abs().max().item()
        raise RuntimeError(f"decoder replay parity failed: max_abs={difference}")
    return FullGenerationStep(
        action_hidden=hidden,
        velocity=velocity,
        next_noisy_action=noisy_action + float(dt) * velocity,
        tau=tau,
        dt=float(dt),
    )


def hidden_hook_parity_report(
    transformer: Any,
    *,
    condition: Tensor,
    noisy_action: Tensor,
    proprio: Tensor,
    tau: Tensor,
    dt: float,
) -> dict[str, Any]:
    hooked = full_generation_step_with_hidden(
        transformer,
        condition=condition,
        noisy_action=noisy_action,
        proprio=proprio,
        tau=tau,
        dt=dt,
    )
    direct = transformer(
        vlm_features=condition,
        action_with_noise=noisy_action,
        proprio=proprio,
        t=tau,
    )
    difference = (hooked.velocity.float() - direct.float()).abs()
    passed = bool(torch.equal(hooked.velocity, direct))
    return {
        "verdict": "GENERATOR_HIDDEN_HOOK_PASS" if passed else "GENERATOR_HIDDEN_HOOK_FAIL",
        "velocity_bitwise_equal": passed,
        "velocity_max_abs_difference": float(difference.max().item()),
        "velocity_mean_abs_difference": float(difference.mean().item()),
        "hidden_shape": list(hooked.action_hidden.shape),
        "velocity_shape": list(hooked.velocity.shape),
        "next_noisy_action_shape": list(hooked.next_noisy_action.shape),
        "hook_target": "SmolVLMActionTransformer.action_decoder forward pre-hook",
        "frozen_decoder_reused": True,
    }
