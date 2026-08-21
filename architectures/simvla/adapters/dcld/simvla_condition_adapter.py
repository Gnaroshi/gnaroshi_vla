"""Condition-latent hook for SimVLA."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from .simvla_action_adapter import SimVLAActionAdapter


@dataclass
class SimVLAConditionOutput:
    condition: torch.Tensor
    action: torch.Tensor | None
    aux: dict[str, Any]


class SimVLAConditionAdapter:
    """Expose SimVLA ``enc["vlm_features"]`` as the DCLD condition latent."""

    def __init__(self, model: Any, action_adapter: SimVLAActionAdapter | None = None) -> None:
        self.model = model
        self.action_adapter = action_adapter or SimVLAActionAdapter(model)

    def encode_condition(
        self,
        *,
        input_ids: torch.Tensor,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        context = nullcontext() if requires_grad else torch.no_grad()
        with context:
            if not requires_grad:
                self.model.eval()
            enc = self.model.forward_vlm_efficient(image_input, image_mask, input_ids)
            if "vlm_features" not in enc:
                raise KeyError("SimVLA forward_vlm_efficient did not return 'vlm_features'")
            return enc["vlm_features"]

    def full_forward_return_latent(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_action: bool = True,
        steps: int = 10,
        initial_noise: torch.Tensor | None = None,
        deterministic: bool = False,
        requires_grad: bool = False,
    ) -> SimVLAConditionOutput:
        condition = self.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
            requires_grad=requires_grad,
        )
        action = None
        if return_action:
            action = self.action_adapter.decode_action_from_condition(
                condition,
                batch["proprio"],
                steps=steps,
                initial_noise=initial_noise,
                deterministic=deterministic,
                requires_grad=requires_grad,
            )
        return SimVLAConditionOutput(
            condition=condition,
            action=action,
            aux={
                "condition_key": "vlm_features",
                "condition_shape": tuple(condition.shape),
                "steps": steps,
            },
        )

    def compare_hook_equivalence(
        self,
        batch: dict[str, torch.Tensor],
        *,
        steps: int = 10,
        seed: int = 0,
    ) -> dict[str, float | bool]:
        """Compare official and hooked paths by resetting RNG state.

        This is a lightweight helper. It still requires a loaded SimVLA model and
        a real batch, so wrappers keep it opt-in.
        """

        self.model.eval()
        device = batch["proprio"].device
        if device.type == "cuda":
            devices = [device.index] if device.index is not None else []
        else:
            devices = []

        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            official = self.model.generate_actions(
                batch["input_ids"],
                batch["image_input"],
                batch["image_mask"],
                batch["proprio"],
                steps=steps,
            )

        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            condition = self.encode_condition(
                input_ids=batch["input_ids"],
                image_input=batch["image_input"],
                image_mask=batch["image_mask"],
            )
            hooked = self.action_adapter.decode_action_from_condition(
                condition,
                batch["proprio"],
                steps=steps,
                deterministic=False,
            )

        diff = (official - hooked).detach().float()
        return {
            "mean_abs": float(diff.abs().mean().item()),
            "max_abs": float(diff.abs().max().item()),
            "l2": float(diff.flatten(start_dim=1).norm(dim=-1).mean().item()),
            "allclose_1e_5": bool(torch.allclose(official, hooked, atol=1e-5, rtol=1e-5)),
            "allclose_1e_4": bool(torch.allclose(official, hooked, atol=1e-4, rtol=1e-4)),
        }
