"""SimVLA action-noise and executed-subchunk helpers for LatentLoop."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ActionNoiseKey:
    """Row-independent key for paired SimVLA flow initial noise."""

    checkpoint: str
    task_id: int
    episode_id: str
    policy_query_index: int
    seed_base: int

    def seed(self) -> int:
        """Return a deterministic 63-bit seed without method/K/R identifiers."""

        payload = "|".join(
            (
                str(self.checkpoint),
                str(self.task_id),
                str(self.episode_id),
                str(self.policy_query_index),
                str(self.seed_base),
            )
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def explicit_action_noise(
    key: ActionNoiseKey,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create paired initial noise without consuming global RNG state."""

    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(key.seed())
    return torch.randn(
        (batch_size, action_horizon, action_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def executed_subchunk(action_chunk: Tensor, execution_horizon: int) -> Tensor:
    """Return only the final postprocessed actions that will be sent to env."""

    if action_chunk.ndim != 3 or action_chunk.shape[-1] != 7:
        raise ValueError("action_chunk must be [B,H,7]")
    if execution_horizon < 1 or execution_horizon > action_chunk.shape[1]:
        raise ValueError("execution_horizon must be in [1,H]")
    return action_chunk[:, :execution_horizon].clone()
