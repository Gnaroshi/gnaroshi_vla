"""Decode SimVLA actions from a precomputed condition latent."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SimVLAActionDecodeOutput:
    action: torch.Tensor
    initial_noise: torch.Tensor
    final_action_latent: torch.Tensor
    debug: dict[str, Any]


class SimVLAActionAdapter:
    """External reproduction of ``SmolVLMVLA.generate_actions`` after encoding."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @property
    def action_space(self) -> Any:
        return self.model.action_space

    @property
    def dim_action(self) -> int:
        return int(self.action_space.dim_action)

    @property
    def num_actions(self) -> int:
        return int(self.model.num_actions)

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        if hasattr(self.action_space, "normalize_state"):
            return self.action_space.normalize_state(proprio)
        if hasattr(self.action_space, "normalize"):
            return self.action_space.normalize(proprio)
        return proprio

    def sample_initial_action_noise(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        shape = (batch_size, self.num_actions, self.dim_action)
        if deterministic:
            return torch.zeros(shape, device=device, dtype=dtype)
        if generator is None:
            return torch.randn(shape, device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    def decode_action_from_condition(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        steps: int = 10,
        initial_noise: torch.Tensor | None = None,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        requires_grad: bool = False,
        return_debug: bool = False,
    ) -> torch.Tensor | SimVLAActionDecodeOutput:
        """Decode action using the same flow loop as upstream ``generate_actions``.

        Exact equivalence tests should pass ``initial_noise`` explicitly, or
        reset the same RNG state before both official and adapter paths.
        """

        context = nullcontext() if requires_grad else torch.no_grad()
        with context:
            if not requires_grad:
                self.model.eval()

            batch_size = condition.shape[0]
            device = proprio.device
            dtype = proprio.dtype
            proprio_norm = self.normalize_proprio(proprio)

            if initial_noise is None:
                x_t = self.sample_initial_action_noise(
                    batch_size,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                    deterministic=deterministic,
                )
            else:
                x_t = initial_noise.to(device=device, dtype=dtype)
            initial_noise_used = x_t.detach().clone()

            steps = max(1, int(steps))
            dt = -1.0 / steps
            t = 1.0
            iterations = 0

            while t > -dt / 2:
                t_tensor = torch.full((batch_size,), t, device=device, dtype=dtype)
                v_t = self.model.transformer(
                    vlm_features=condition,
                    action_with_noise=x_t,
                    proprio=proprio_norm,
                    t=t_tensor,
                )
                x_t = x_t + dt * v_t
                t = t + dt
                iterations += 1

            action = self.action_space.postprocess(x_t)

        if not return_debug:
            return action
        return SimVLAActionDecodeOutput(
            action=action,
            initial_noise=initial_noise_used,
            final_action_latent=x_t,
            debug={
                "steps": steps,
                "iterations": iterations,
                "deterministic": deterministic,
            },
        )
