"""Conditional long-span age support without changing ages 1--3."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ParityPreservingAgeEncoding(nn.Module):
    """Reuse the learned lookup at ages 1--3 and extrapolate without new weights.

    The short-span campaign keeps the source lookup untouched.  This adapter is
    installed only after the K_C=4 offline and online gates pass.  Integer ages
    1--3 are exact table lookups; ages 4--7 linearly continue the learned 2->3
    direction.  It therefore adds no trainable representation or capacity.
    """

    def __init__(self, source_weight: Tensor, *, max_age: int = 7) -> None:
        super().__init__()
        if source_weight.ndim != 2 or source_weight.shape[0] < 4:
            raise ValueError("source age table must contain indices 0--3")
        if int(max_age) < 3:
            raise ValueError("max_age must preserve ages 1--3")
        self.max_age = int(max_age)
        self.register_buffer("source_weight", source_weight.detach().clone())

    @classmethod
    def from_embedding(
        cls, embedding: nn.Embedding, *, max_age: int = 7
    ) -> "ParityPreservingAgeEncoding":
        return cls(embedding.weight, max_age=max_age)

    def forward(self, age: Tensor) -> Tensor:
        values = torch.as_tensor(age, device=self.source_weight.device)
        if bool((values < 0).any()) or bool((values > self.max_age).any()):
            raise ValueError(f"age must be in [0,{self.max_age}]")
        lower = values.floor().long().clamp(max=3)
        exact = F.embedding(lower, self.source_weight)
        older = values.to(self.source_weight.dtype).unsqueeze(-1)
        slope = self.source_weight[3] - self.source_weight[2]
        extrapolated = self.source_weight[3] + (older - 3.0) * slope
        return torch.where((values <= 3).unsqueeze(-1), exact, extrapolated)


def enable_conditional_kc8_age_support(condition_updater: nn.Module) -> dict[str, object]:
    """Install long-span support after external K_C=4 gates have passed."""

    embedding = getattr(condition_updater, "age_embedding", None)
    if not isinstance(embedding, nn.Embedding):
        raise TypeError("Condition updater does not use the audited fixed age lookup")
    replacement = ParityPreservingAgeEncoding.from_embedding(embedding, max_age=7)
    replacement = replacement.to(device=embedding.weight.device, dtype=embedding.weight.dtype)
    ages = torch.tensor([1, 2, 3], device=embedding.weight.device)
    with torch.no_grad():
        parity = torch.equal(embedding(ages), replacement(ages))
    if not parity:
        raise RuntimeError("long-span age adapter changed ages 1--3")
    condition_updater.age_embedding = replacement
    condition_updater.max_age = 7
    return {
        "verdict": "KC8_AGE_SUPPORT_STATIC_READY",
        "ages_1_3_exact_parity": True,
        "supported_ages": list(range(1, 8)),
        "new_trainable_parameters": 0,
        "extrapolation": "linear continuation of learned age-2 to age-3 direction",
    }
