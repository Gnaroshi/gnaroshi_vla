"""Reuse OpenPI's configured transforms around external action samplers."""

from __future__ import annotations

import hashlib
from typing import Any

import jax
import numpy as np
import torch


def prepare_policy_observation(policy: Any, raw_observation: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from openpi.models import model as model_api

    inputs = jax.tree.map(lambda value: value, raw_observation)
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    device = policy._pytorch_device  # noqa: SLF001
    tensors = jax.tree.map(
        lambda value: torch.from_numpy(np.asarray(value)).to(device)[None, ...], inputs
    )
    return model_api.Observation.from_dict(tensors), tensors


def postprocess_policy_actions(policy: Any, normalized_state: Any, actions: torch.Tensor) -> dict[str, Any]:
    outputs = {
        "state": jax.tree.map(lambda value: np.asarray(value[0].detach().cpu()), normalized_state),
        "actions": np.asarray(actions[0].detach().cpu()),
    }
    return policy._output_transform(outputs)  # noqa: SLF001


def policy_noise_seed(
    seed_base: int,
    suite: str,
    task_id: int,
    episode_id: int,
    query_index: int,
) -> int:
    key = f"{seed_base}:{suite}:{task_id}:{episode_id}:{query_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def explicit_policy_noise(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: str | torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)
