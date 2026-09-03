"""Seer-sized residual feature bridge built from the official DiT blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from architectures.latent_bridge.upstream.qcvla.model.rectified_flow_bridge import (
    DiTCrossBlock,
    DiTFinalLayer,
)
from methods.latent_bridge import BridgePreset


@dataclass(frozen=True)
class SeerFeatureBridgeConfig:
    feature_dim: int = 384
    target_seq_len: int = 3
    stable_seq_len: int = 3
    state_dim: int = 8
    action_dim: int = 7
    hidden_dim: int = 768
    num_blocks: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    preset: str = "full"
    stable_layer: str = "block_00"
    stable_token_group: str = "action"

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        stable_seq_len: int,
        stable_layer: str,
        stable_token_group: str,
    ) -> "SeerFeatureBridgeConfig":
        presets = {
            "full": BridgePreset.official_full(),
            "small": BridgePreset.official_small(),
        }
        if preset_name not in presets:
            raise ValueError(f"unknown bridge preset {preset_name!r}")
        preset = presets[preset_name]
        return cls(
            stable_seq_len=stable_seq_len,
            hidden_dim=preset.hidden_dim,
            num_blocks=preset.num_blocks,
            num_heads=preset.num_heads,
            preset=preset.name,
            stable_layer=stable_layer,
            stable_token_group=stable_token_group,
        )

    def to_dict(self) -> dict:
        return asdict(self)


class SeerFeatureBridge(nn.Module):
    """Predict ``z_t - z_{t-1}`` for Seer's three action-query tokens.

    The official GR00T implementation uses one positional tensor because target
    and stable sequences have equal lengths. Seer selects these two sequences
    independently after a measured token-mode audit, so this adapter uses two
    positional tensors. The official attention blocks and zero-output final
    layer are used without modification.
    """

    def __init__(self, config: SeerFeatureBridgeConfig):
        super().__init__()
        self.config = config
        if config.feature_dim != 384:
            raise ValueError("The audited Seer action-condition width is 384")
        if config.target_seq_len != 3:
            raise ValueError("The audited public Seer checkpoint has three action-query tokens")
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.input_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        self.stable_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        self.target_pos_embed = nn.Parameter(torch.zeros(1, config.target_seq_len, config.hidden_dim))
        self.stable_pos_embed = nn.Parameter(torch.zeros(1, config.stable_seq_len, config.hidden_dim))
        nn.init.trunc_normal_(self.target_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.stable_pos_embed, std=0.02)

        self.state_embed = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.blocks = nn.ModuleList([
            DiTCrossBlock(config.hidden_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.num_blocks)
        ])
        self.final_layer = DiTFinalLayer(config.hidden_dim, config.feature_dim)

    def forward(self, previous_condition, stable_context, current_state, previous_executed_action):
        self._validate_inputs(previous_condition, stable_context, current_state, previous_executed_action)
        x = self.input_proj(previous_condition) + self.target_pos_embed
        stable = self.stable_proj(stable_context) + self.stable_pos_embed
        state = self.state_embed(current_state)
        action = self.action_embed(previous_executed_action)
        condition = self.cond_fuse(torch.cat([state, action], dim=-1))
        for block in self.blocks:
            x = block(x, stable, condition)
        return self.final_layer(x, condition)

    def predict_next(self, previous_condition, stable_context, current_state, previous_executed_action):
        return previous_condition + self(
            previous_condition, stable_context, current_state, previous_executed_action
        )

    def _validate_inputs(self, previous_condition, stable_context, current_state, previous_action):
        expected = {
            "previous_condition": (self.config.target_seq_len, self.config.feature_dim),
            "stable_context": (self.config.stable_seq_len, self.config.feature_dim),
            "current_state": (self.config.state_dim,),
            "previous_executed_action": (self.config.action_dim,),
        }
        tensors = {
            "previous_condition": previous_condition,
            "stable_context": stable_context,
            "current_state": current_state,
            "previous_executed_action": previous_action,
        }
        batch = previous_condition.shape[0]
        for name, tensor in tensors.items():
            if tensor.shape[0] != batch or tuple(tensor.shape[1:]) != expected[name]:
                raise ValueError(
                    f"{name} expected [B, {', '.join(map(str, expected[name]))}], got {tuple(tensor.shape)}"
                )

    def parameter_audit(self) -> dict:
        by_module = {
            "input_projection": sum(p.numel() for p in self.input_proj.parameters()),
            "stable_projection": sum(p.numel() for p in self.stable_proj.parameters()),
            "position_embeddings": self.target_pos_embed.numel() + self.stable_pos_embed.numel(),
            "state_embedding": sum(p.numel() for p in self.state_embed.parameters()),
            "action_embedding": sum(p.numel() for p in self.action_embed.parameters()),
            "condition_fusion": sum(p.numel() for p in self.cond_fuse.parameters()),
            "dit_blocks": sum(p.numel() for p in self.blocks.parameters()),
            "final_layer": sum(p.numel() for p in self.final_layer.parameters()),
        }
        return {
            "config": self.config.to_dict(),
            "by_module": by_module,
            "total": sum(by_module.values()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "zero_initialized_output": bool(
                torch.count_nonzero(self.final_layer.linear.weight).item() == 0
                and torch.count_nonzero(self.final_layer.linear.bias).item() == 0
            ),
        }
