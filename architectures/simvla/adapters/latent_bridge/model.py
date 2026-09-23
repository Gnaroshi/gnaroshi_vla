"""Official-architecture feature bridge instantiated at SimVLA's condition boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .provenance import load_official_bridge_core


@dataclass(frozen=True)
class SimVLALatentBridgeConfig:
    feature_dim: int = 960
    sequence_length: int = 122
    hidden_dim: int = 768
    num_heads: int = 12
    num_blocks: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    state_dim: int = 8
    action_dim: int = 7
    low_rank: int = 0
    stable_layer_index: int = 10
    token_mode: str = "all"
    image_token_count: int = 72

    def validate(self) -> None:
        if self.feature_dim <= 0 or self.sequence_length <= 0:
            raise ValueError("feature_dim and sequence_length must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if self.token_mode not in {"all", "image_only"}:
            raise ValueError("token_mode must be all or image_only")
        if not 0 < self.image_token_count <= self.sequence_length:
            raise ValueError("image_token_count is outside the sequence")
        if self.token_mode == "image_only" and self.sequence_length != self.image_token_count:
            raise ValueError("image_only bridge sequence_length must equal image_token_count")

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


class SimVLALatentBridge(nn.Module):
    """Single-step feature-delta predictor with the official Latent Bridge DiT core.

    The wiring matches official ``SingleStepDiT``. The primary SimVLA
    configuration predicts all 122 fused action-condition positions because
    both visual and language positions change materially across policy queries.
    """

    def __init__(
        self,
        config: SimVLALatentBridgeConfig | None = None,
        *,
        official_upstream_root: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SimVLALatentBridgeConfig()
        self.config.validate()
        core = load_official_bridge_core(official_upstream_root)
        cfg = self.config
        self.input_proj = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.stable_proj = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.sequence_length, cfg.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.state_embed = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                core.DiTCrossBlock(
                    cfg.hidden_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout
                )
                for _ in range(cfg.num_blocks)
            ]
        )
        if cfg.low_rank > 0:
            self.final_layer = core.DiTFinalLayer(cfg.hidden_dim, cfg.low_rank)
            self.rank_up = nn.Linear(cfg.low_rank, cfg.feature_dim)
            nn.init.zeros_(self.rank_up.weight)
            nn.init.zeros_(self.rank_up.bias)
        else:
            self.final_layer = core.DiTFinalLayer(cfg.hidden_dim, cfg.feature_dim)
            self.rank_up = None

    def _validate_inputs(
        self,
        condition: Tensor,
        stable: Tensor,
        state: Tensor,
        previous_action: Tensor,
    ) -> None:
        expected = (self.config.sequence_length, self.config.feature_dim)
        if condition.ndim != 3 or tuple(condition.shape[1:]) != expected:
            raise ValueError(f"condition must be [B,{expected[0]},{expected[1]}]")
        if stable.shape != condition.shape:
            raise ValueError("stable feature shape differs from condition")
        if state.shape != (condition.shape[0], self.config.state_dim):
            raise ValueError("state shape differs from bridge configuration")
        if previous_action.shape != (condition.shape[0], self.config.action_dim):
            raise ValueError("previous_action shape differs from bridge configuration")

    def forward(
        self,
        condition: Tensor,
        stable: Tensor,
        state: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        """Predict `condition[t+1] - condition[t]`."""

        self._validate_inputs(condition, stable, state, previous_action)
        x = self.input_proj(condition) + self.pos_embed
        stable_context = self.stable_proj(stable) + self.pos_embed
        state_embedding = self.state_embed(state)
        action_embedding = self.action_embed(previous_action)
        conditioning = self.cond_fuse(
            torch.cat((state_embedding, action_embedding), dim=-1)
        )
        for block in self.blocks:
            x = block(x, stable_context, conditioning)
        delta = self.final_layer(x, conditioning)
        if self.rank_up is not None:
            delta = self.rank_up(delta)
        return delta

    def predict_next(
        self,
        condition: Tensor,
        stable: Tensor,
        state: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        return condition + self(condition, stable, state, previous_action)

    def parameter_audit(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "total_millions": total / 1_000_000,
        }
