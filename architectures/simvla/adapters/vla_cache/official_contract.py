"""VLA-Cache source contract and architecture-scaled token selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch.nn import functional as F


VLA_CACHE_REPOSITORY = "https://github.com/siyuhsu/VLA-Cache"
VLA_CACHE_COMMIT = "a4909880573868dee2769343d52e793c0341678b"
VLA_CACHE_TRANSFORMERS_REPOSITORY = "https://github.com/siyuhsu/transformers"
VLA_CACHE_TRANSFORMERS_BRANCH = "vla-cache-openvla"
VLA_CACHE_TRANSFORMERS_COMMIT = "2302fce58afa3a4f8461625b1394f9e9c8a7f1ea"


@dataclass(frozen=True)
class VLACacheConfig:
    """Published VLA-Cache choices mapped to SmolVLM connector tokens.

    The official OpenVLA-OFT implementation selects 150 stable and 100
    task-relevant tokens from each 16x16 (256-token) image. SmolVLM exposes a
    6x6 (36-token) connector grid to its text decoder. We preserve the two
    published fractions at the actual compute-token granularity.
    """

    pruning_layers: tuple[int, ...] = (2, 6, 9, 11)
    reference_attention_layer: int = 15
    source_visual_tokens_per_view: int = 256
    source_stable_top_k: int = 150
    source_task_relevant_top_k: int = 100
    connector_grid_size: int = 6
    similarity_threshold: float = 0.996
    positive_growth_factor: float = 0.55

    @property
    def visual_tokens_per_view(self) -> int:
        return self.connector_grid_size**2

    @property
    def stable_top_k(self) -> int:
        return round(
            self.source_stable_top_k
            * self.visual_tokens_per_view
            / self.source_visual_tokens_per_view
        )

    @property
    def task_relevant_top_k(self) -> int:
        return round(
            self.source_task_relevant_top_k
            * self.visual_tokens_per_view
            / self.source_visual_tokens_per_view
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "visual_tokens_per_view": self.visual_tokens_per_view,
                "stable_top_k": self.stable_top_k,
                "task_relevant_top_k": self.task_relevant_top_k,
                "official_repository": VLA_CACHE_REPOSITORY,
                "official_commit": VLA_CACHE_COMMIT,
                "official_transformers_repository": VLA_CACHE_TRANSFORMERS_REPOSITORY,
                "official_transformers_branch": VLA_CACHE_TRANSFORMERS_BRANCH,
                "official_transformers_commit": VLA_CACHE_TRANSFORMERS_COMMIT,
            }
        )
        return payload


def connector_patch_cosine(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    grid_size: int,
) -> torch.Tensor:
    """Return cosine similarity for patches aligned with connector tokens."""

    if previous.shape != current.shape or previous.ndim != 4:
        raise ValueError("previous/current images must have equal [V,C,H,W] shape")
    height, width = current.shape[-2:]
    if height != width or height % grid_size:
        raise ValueError("image size must be square and divisible by connector grid")
    patch = height // grid_size

    def flatten(value: torch.Tensor) -> torch.Tensor:
        return (
            value.unfold(-2, patch, patch)
            .unfold(-1, patch, patch)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(value.shape[0], grid_size * grid_size, -1)
            .float()
        )

    return F.cosine_similarity(flatten(previous), flatten(current), dim=-1)


def reusable_visual_positions(
    *,
    previous_images: torch.Tensor,
    current_images: torch.Tensor,
    previous_visual_importance: torch.Tensor,
    config: VLACacheConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Apply official stable-minus-task-relevant selection per camera view."""

    similarities = connector_patch_cosine(
        previous_images,
        current_images,
        grid_size=config.connector_grid_size,
    )
    views, tokens = similarities.shape
    if previous_visual_importance.shape != (views, tokens):
        raise ValueError(
            "visual importance must match [views, connector_tokens_per_view]"
        )
    reusable: list[int] = []
    diagnostics: list[dict[str, object]] = []
    for view in range(views):
        scores = similarities[view]
        eligible = torch.nonzero(
            scores >= config.similarity_threshold, as_tuple=False
        ).flatten()
        if eligible.numel():
            order = torch.argsort(scores[eligible], descending=True, stable=True)
            stable = eligible[order[: config.stable_top_k]]
        else:
            stable = eligible
        important = torch.argsort(
            previous_visual_importance[view], descending=True, stable=True
        )[: config.task_relevant_top_k]
        keep = sorted(set(stable.tolist()) - set(important.tolist()))
        reusable.extend(view * tokens + token for token in keep)
        diagnostics.append(
            {
                "view": view,
                "stable_candidates": int(eligible.numel()),
                "stable_selected": int(stable.numel()),
                "task_relevant_selected": int(important.numel()),
                "reusable_selected": len(keep),
                "similarity_mean": float(scores.mean().item()),
                "similarity_min": float(scores.min().item()),
            }
        )
    return (
        torch.tensor(reusable, dtype=torch.long, device=current_images.device),
        {"per_view": diagnostics, "reusable_positions": reusable},
    )


def layer_reuse_schedule(
    attentions: Sequence[torch.Tensor],
    *,
    growth_factor: float,
) -> torch.Tensor:
    """Reproduce the official attention-entropy reuse schedule."""

    if len(attentions) < 2:
        raise ValueError("at least two decoder attention maps are required")
    entropy = []
    for attention in attentions[:-1]:
        probabilities = attention.float().mean(dim=1)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        token_entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum(dim=-1)
        entropy.append(token_entropy.mean())
    values = torch.stack(entropy)
    normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
    reuse = (1.0 - normalized).tolist()
    for index in range(1, len(reuse)):
        delta = reuse[index] - reuse[index - 1]
        if delta > 0:
            reuse[index] = reuse[index - 1] + delta * growth_factor
    return torch.tensor(reuse, dtype=torch.float32, device=attentions[0].device)
