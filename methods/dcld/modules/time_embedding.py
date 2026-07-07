"""Small time/age embeddings for fixed-step latent dynamics."""

from __future__ import annotations

import math

import torch


def sinusoidal_time_embedding(
    value: torch.Tensor | float,
    num_frequencies: int = 4,
    include_value: bool = True,
) -> torch.Tensor:
    """Return scalar + sin/cos powers-of-two frequency embedding.

    Args:
        value: Scalar or batch tensor, usually normalized skip age or dt.
        num_frequencies: Number of powers-of-two sinusoidal bands.
        include_value: Whether to include the raw scalar as the first feature.

    Returns:
        Tensor with shape ``value.shape + [1 + 2 * num_frequencies]`` when
        ``include_value`` is true.
    """

    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32)

    value = value.float()
    if value.ndim == 0:
        value = value.unsqueeze(0)

    freqs = torch.pow(
        torch.tensor(2.0, device=value.device, dtype=value.dtype),
        torch.arange(num_frequencies, device=value.device, dtype=value.dtype),
    )
    angles = value.unsqueeze(-1) * freqs * math.pi

    parts = []
    if include_value:
        parts.append(value.unsqueeze(-1))
    parts.extend([torch.sin(angles), torch.cos(angles)])
    return torch.cat(parts, dim=-1)
