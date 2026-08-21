"""Local adaptation of the official pi0.5 Latent Bridge KV architecture.

Source contract: 1999Lyd/Latent-Bridge commit
``ed556014aa96bae8ed85768194f02360389b9365``. This module preserves the
pre-RoPE K/V delta architecture but uses Gnaroshi's cache, split, frozen model,
and evaluation harness. Until the official DAgger stages are reproduced, rows
from this implementation must be called ``Latent Bridge-style KV baseline``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .prefix_kv_hook import PrefixKVState
from .transition_core import OpenPITransitionOutput


class AdaLNBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(width, elementwise_affine=False)
        mlp_width = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_width), nn.GELU(), nn.Linear(mlp_width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, value: Tensor, context: Tensor, condition: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).unsqueeze(1).chunk(6, dim=-1)
        hidden = self.norm1(value) * (1 + scale1) + shift1
        value = value + gate1 * self.self_attention(hidden, hidden, hidden, need_weights=False)[0]
        hidden = self.norm2(value)
        value = value + self.cross_attention(hidden, context, context, need_weights=False)[0]
        hidden = self.norm3(value) * (1 + scale2) + shift2
        return value + gate2 * self.mlp(hidden)


@dataclass(frozen=True)
class LatentBridgeConfig:
    kv_dim: int = 256
    num_layers: int = 18
    max_prefix_tokens: int = 1024
    embedding_dim: int = 2048
    hidden_dim: int = 768
    num_heads: int = 12
    num_blocks: int = 10
    state_dim: int = 32
    action_dim: int = 7


class LocalLatentBridgeKV(nn.Module):
    def __init__(self, config: LatentBridgeConfig = LatentBridgeConfig()) -> None:
        super().__init__()
        self.config = config
        c = config
        self.embedding_delta_projection = nn.Linear(c.embedding_dim, c.hidden_dim)
        self.current_embedding_projection = nn.Linear(c.embedding_dim, c.hidden_dim)
        self.previous_kv_projection = nn.Linear(c.num_layers * c.kv_dim * 2, c.hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, c.max_prefix_tokens, c.hidden_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.condition = nn.Sequential(
            nn.Linear(c.state_dim + c.action_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )
        self.blocks = nn.ModuleList(AdaLNBlock(c.hidden_dim, c.num_heads) for _ in range(c.num_blocks))
        self.layer_heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(c.hidden_dim), nn.Linear(c.hidden_dim, c.kv_dim * 2))
            for _ in range(c.num_layers)
        )
        for head in self.layer_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        embedding_delta: Tensor,
        current_embedding: Tensor,
        previous_kv: Tensor,
        robot_state: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        if previous_kv.ndim != 4:
            raise ValueError("previous_kv must be [B,L,S,512]")
        batch, layers, tokens, width = previous_kv.shape
        c = self.config
        if layers != c.num_layers or width != c.kv_dim * 2 or tokens > c.max_prefix_tokens:
            raise ValueError("previous KV shape is incompatible with the Latent Bridge config")
        flattened = previous_kv.permute(0, 2, 1, 3).reshape(batch, tokens, layers * width)
        position = self.position_embedding[:, :tokens]
        value = self.embedding_delta_projection(embedding_delta) + self.previous_kv_projection(flattened) + position
        context = self.current_embedding_projection(current_embedding) + position
        condition = self.condition(torch.cat((robot_state, previous_action), dim=-1))
        for block in self.blocks:
            value = block(value, context, condition)
        deltas = torch.stack([head(value) for head in self.layer_heads], dim=1)
        return previous_kv + deltas


def official_style_config() -> LatentBridgeConfig:
    return LatentBridgeConfig(hidden_dim=768, num_heads=12, num_blocks=10)


def small_under_19m_config() -> LatentBridgeConfig:
    return LatentBridgeConfig(hidden_dim=320, num_heads=8, num_blocks=4)


class RawKVCodec:
    """Minimal packer used by shared state losses without extra parameters."""

    @staticmethod
    def pack(state: PrefixKVState) -> Tensor:
        return torch.stack(
            [
                torch.cat((key[:, 0], value[:, 0]), dim=-1)
                for key, value in zip(state.pre_rope_keys, state.values, strict=True)
            ],
            dim=1,
        )


def latent_bridge_config_from_dict(payload: dict[str, Any]) -> LatentBridgeConfig:
    return LatentBridgeConfig(**payload)


class LocalLatentBridgeAdapter(nn.Module):
    """Expose the official-style bridge through the common pi0.5 state API."""

    def __init__(self, config: LatentBridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.bridge = LocalLatentBridgeKV(config)
        self.codec = RawKVCodec()

    @property
    def trainable_parameters(self) -> int:
        return self.bridge.trainable_parameters

    def forward(
        self,
        previous_state: PrefixKVState,
        current_prefix: Any,
        previous_prefix_embeddings: Tensor,
        executed_actions: Tensor,
        robot_state: Tensor,
        **_unused,
    ) -> OpenPITransitionOutput:
        packed = self.codec.pack(previous_state).float()
        current = current_prefix.embeddings.float()
        previous = previous_prefix_embeddings.float()
        action = executed_actions[:, -1, :7].float()
        state = robot_state[:, : self.config.state_dim].float()
        predicted = self.bridge(current - previous, current, packed, state, action)
        predicted = predicted.to(dtype=packed.dtype)
        keys, values = predicted[..., : self.config.kv_dim], predicted[..., self.config.kv_dim :]
        output_state = PrefixKVState(
            embeddings=current_prefix.embeddings,
            pad_mask=current_prefix.pad_mask,
            attention_pattern=current_prefix.attention_pattern,
            position_ids=current_prefix.position_ids,
            pre_rope_keys=tuple(
                keys[:, layer, None].to(previous_state.pre_rope_keys[0].dtype)
                for layer in range(keys.shape[1])
            ),
            values=tuple(
                values[:, layer, None].to(previous_state.values[0].dtype)
                for layer in range(values.shape[1])
            ),
        )
        delta = predicted - packed
        return OpenPITransitionOutput(
            state=output_state,
            encoded_state=predicted,
            encoded_delta=delta,
            gate=torch.ones((*predicted.shape[:-1], 1), device=predicted.device),
            action_feature=action,
        )
