"""Defect-normalized three-level refresh scheduler for LatentLoop V2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor


class RefreshDecision(IntEnum):
    KEEP_SEQUENTIAL = 0
    DIRECT_REANCHOR = 1
    FULL_SEER = 2


class DefectNormalizer:
    """Normalize z_seq-z_dir with train-split-only per-coordinate RMS."""

    def __init__(self, scale: Tensor, epsilon: float = 1e-6) -> None:
        if scale.ndim < 1 or not bool(torch.isfinite(scale).all()):
            raise ValueError("defect scale must be a finite tensor")
        self.scale = scale.detach().clone().clamp_min(float(epsilon))

    @classmethod
    def fit(cls, train_differences: Tensor) -> "DefectNormalizer":
        if train_differences.ndim < 2:
            raise ValueError("training differences must include sample and latent dimensions")
        scale = train_differences.detach().float().square().mean(dim=0).sqrt()
        return cls(scale)

    def __call__(self, z_seq: Tensor, z_dir: Tensor) -> Tensor:
        if z_seq.shape != z_dir.shape:
            raise ValueError("sequential and direct latent shapes differ")
        scale = self.scale.to(device=z_seq.device, dtype=z_seq.dtype)
        return ((z_seq - z_dir) / scale).square().flatten(start_dim=1).mean(dim=1).sqrt()


@dataclass
class SchedulerState:
    full_anchor: Tensor | None = None
    current: Tensor | None = None
    sequential_age: int = 0
    full_age: int = 0
    full_calls: int = 0
    transition_calls: int = 0
    direct_calls: int = 0
    direct_reanchors: int = 0
    policy_steps: int = 0

    def reset_full(self, full_latent: Tensor) -> None:
        self.full_anchor = full_latent.detach()
        self.current = full_latent.detach()
        self.sequential_age = 0
        self.full_age = 0
        self.full_calls += 1

    def apply(self, decision: RefreshDecision, z_seq: Tensor, z_dir: Tensor, full_latent: Tensor | None = None) -> Tensor:
        self.policy_steps += 1
        if decision == RefreshDecision.KEEP_SEQUENTIAL:
            self.current = z_seq.detach()
            self.sequential_age += 1
            self.full_age += 1
            self.transition_calls += 1
            self.direct_calls += 1
        elif decision == RefreshDecision.DIRECT_REANCHOR:
            self.current = z_dir.detach()
            self.sequential_age = 0
            self.full_age += 1
            self.transition_calls += 1
            self.direct_calls += 1
            self.direct_reanchors += 1
        elif decision == RefreshDecision.FULL_SEER:
            if full_latent is None:
                raise ValueError("Level-2 reset requires the actual Full Seer latent")
            self.reset_full(full_latent)
        else:
            raise ValueError(f"unknown refresh decision: {decision}")
        assert self.current is not None
        return self.current

    @property
    def effective_k(self) -> float:
        return self.policy_steps / self.full_calls if self.full_calls else 0.0

    def to_dict(self) -> dict[str, int | float]:
        payload = {
            "sequential_age": self.sequential_age,
            "full_age": self.full_age,
            "full_calls": self.full_calls,
            "transition_calls": self.transition_calls,
            "direct_calls": self.direct_calls,
            "direct_reanchors": self.direct_reanchors,
            "policy_steps": self.policy_steps,
        }
        payload["effective_k"] = self.effective_k
        return payload


@dataclass(frozen=True)
class AdaptiveRefreshScheduler:
    seq_unsafe_threshold: float
    direct_unsafe_threshold: float
    direct_advantage_margin: float
    max_sequential_age: int
    max_full_age: int

    def decide(
        self,
        *,
        predicted_seq_error: float,
        predicted_direct_error: float,
        sequential_age: int,
        full_age: int,
    ) -> RefreshDecision:
        if full_age >= self.max_full_age:
            return RefreshDecision.FULL_SEER
        both_unsafe = (
            predicted_seq_error >= self.seq_unsafe_threshold
            and predicted_direct_error >= self.direct_unsafe_threshold
        )
        if both_unsafe:
            return RefreshDecision.FULL_SEER
        direct_better = (
            predicted_direct_error + self.direct_advantage_margin < predicted_seq_error
        )
        if sequential_age >= self.max_sequential_age or direct_better:
            return RefreshDecision.DIRECT_REANCHOR
        return RefreshDecision.KEEP_SEQUENTIAL
