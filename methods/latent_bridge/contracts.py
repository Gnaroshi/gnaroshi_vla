"""Shared scientific contracts for official-style Latent Bridge adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BridgePreset:
    """The two feature-bridge capacities reported by Latent Bridge."""

    name: str
    hidden_dim: int
    num_blocks: int
    num_heads: int

    @classmethod
    def official_full(cls) -> "BridgePreset":
        return cls("full", hidden_dim=768, num_blocks=12, num_heads=12)

    @classmethod
    def official_small(cls) -> "BridgePreset":
        return cls("small", hidden_dim=384, num_blocks=4, num_heads=6)


@dataclass(frozen=True)
class TrainingContract:
    """Optimizer and schedule contract, independent of the VLA architecture."""

    stage: str
    epochs: int
    learning_rate: float
    per_rank_batch: int
    world_size: int
    gradient_accumulation_steps: int
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine"

    @property
    def effective_batch(self) -> int:
        return self.per_rank_batch * self.world_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        if self.stage not in {"R0", "R1"}:
            raise ValueError(f"stage must be R0 or R1, got {self.stage!r}")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if self.effective_batch != 64:
            raise ValueError(
                "Official effective batch must be 64: "
                f"per_rank={self.per_rank_batch}, world_size={self.world_size}, "
                f"accumulation={self.gradient_accumulation_steps}, "
                f"effective={self.effective_batch}"
            )
        if self.scheduler != "cosine":
            raise ValueError("Official training contract requires cosine scheduling")


@dataclass(frozen=True)
class ComputeMatchedTrainingContract:
    """Step-budgeted adapter training with the official global batch size."""

    stage: str
    optimizer_steps: int
    learning_rate: float
    per_rank_batch: int
    world_size: int
    gradient_accumulation_steps: int
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine_per_update"
    protocol: str = "compute_matched_v1"

    @property
    def effective_batch(self) -> int:
        return self.per_rank_batch * self.world_size * self.gradient_accumulation_steps

    @property
    def examples_seen(self) -> int:
        return self.optimizer_steps * self.effective_batch

    def validate(self) -> None:
        if self.stage not in {"R0", "R1"}:
            raise ValueError(f"stage must be R0 or R1, got {self.stage!r}")
        if self.optimizer_steps <= 0 or self.learning_rate <= 0:
            raise ValueError("optimizer_steps and learning_rate must be positive")
        if self.effective_batch != 64:
            raise ValueError(
                "Compute-matched effective batch must be 64: "
                f"per_rank={self.per_rank_batch}, world_size={self.world_size}, "
                f"accumulation={self.gradient_accumulation_steps}, "
                f"effective={self.effective_batch}"
            )
        if self.scheduler != "cosine_per_update":
            raise ValueError("Compute-matched training requires per-update cosine scheduling")


def bridge_distillation_loss(
    predicted_next: torch.Tensor,
    target_next: torch.Tensor,
    *,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Official-style feature MSE plus per-token cosine loss."""

    if predicted_next.shape != target_next.shape:
        raise ValueError(
            f"feature shape mismatch: {tuple(predicted_next.shape)} != {tuple(target_next.shape)}"
        )
    mse = F.mse_loss(predicted_next, target_next)
    cosine = 1.0 - F.cosine_similarity(predicted_next, target_next, dim=-1).mean()
    total = mse_weight * mse + cosine_weight * cosine
    return total, {"mse": mse, "cosine_loss": cosine, "total": total}


def should_full_refresh(step: int, refresh_period: int) -> bool:
    """Return whether policy step ``step`` runs the frozen full backbone."""

    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if refresh_period < 1:
        raise ValueError(f"refresh_period must be >= 1, got {refresh_period}")
    return step % refresh_period == 0
