"""Local decoder geometry primitives used by offline feasibility analyses."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def decoder_jacobian(decoder: Callable[[Tensor], Tensor], latent: Tensor) -> Tensor:
    """Return per-sample Jacobians with shape ``[B, output_dim, latent_dim]``."""
    if latent.ndim < 2:
        raise ValueError(f"latent must include batch and feature axes, got {latent.shape}")
    flat = latent.reshape(latent.shape[0], -1)

    def single(sample: Tensor) -> Tensor:
        output = decoder(sample.unsqueeze(0).reshape(1, *latent.shape[1:]))
        return output.reshape(-1)

    return torch.func.vmap(torch.func.jacrev(single))(flat)


def damped_action_to_latent_lift(
    jacobian: Tensor,
    desired_action_delta: Tensor,
    damping: float,
) -> Tensor:
    """Lift an action delta with ``J^T (J J^T + damping I)^-1 delta_y``."""
    if jacobian.ndim != 3:
        raise ValueError(f"jacobian must be [B,M,N], got {jacobian.shape}")
    if desired_action_delta.shape != jacobian.shape[:2]:
        raise ValueError(
            "desired_action_delta must match Jacobian batch/output axes: "
            f"{desired_action_delta.shape} vs {jacobian.shape[:2]}"
        )
    if damping <= 0:
        raise ValueError(f"damping must be positive, got {damping}")
    gram = jacobian @ jacobian.transpose(-1, -2)
    identity = torch.eye(
        gram.shape[-1], device=gram.device, dtype=gram.dtype
    ).expand_as(gram)
    dual = torch.linalg.solve(
        gram + damping * identity, desired_action_delta.unsqueeze(-1)
    )
    return (jacobian.transpose(-1, -2) @ dual).squeeze(-1)


def damped_latent_decomposition(
    jacobian: Tensor,
    latent_delta: Tensor,
    damping: float,
) -> tuple[Tensor, Tensor]:
    """Split a latent delta into damped action-sensitive and residual components."""
    if latent_delta.shape != (jacobian.shape[0], jacobian.shape[2]):
        raise ValueError(
            f"latent_delta must be [B,N], got {latent_delta.shape} for {jacobian.shape}"
        )
    action_delta = (jacobian @ latent_delta.unsqueeze(-1)).squeeze(-1)
    parallel = damped_action_to_latent_lift(jacobian, action_delta, damping)
    return parallel, latent_delta - parallel
