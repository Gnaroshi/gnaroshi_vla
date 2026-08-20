"""Token/layer-shared low-rank codec for pre-RoPE K and V deltas."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .prefix_kv_hook import PrefixKVState


class LayerSharedKVCodec(nn.Module):
    def __init__(self, head_dim: int = 256, state_width: int = 128) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.raw_width = 2 * self.head_dim
        self.state_width = int(state_width)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.raw_width),
            nn.Linear(self.raw_width, self.state_width),
            nn.GELU(),
            nn.LayerNorm(self.state_width),
        )
        self.delta_decoder = nn.Linear(self.state_width, self.raw_width)

    def pack(self, state: PrefixKVState) -> Tensor:
        state.validate()
        if any(key.shape[1] != 1 or key.shape[-1] != self.head_dim for key in state.pre_rope_keys):
            raise ValueError("the pi0.5 codec requires one KV head with the configured head dimension")
        layers = [
            torch.cat((key[:, 0], value[:, 0]), dim=-1)
            for key, value in zip(state.pre_rope_keys, state.values, strict=True)
        ]
        return torch.stack(layers, dim=1)

    def encode(self, packed_state: Tensor) -> Tensor:
        if packed_state.ndim != 4 or packed_state.shape[-1] != self.raw_width:
            raise ValueError("packed_state must be [B,L,S,2*head_dim]")
        dtype = self.encoder[1].weight.dtype
        return self.encoder(packed_state.to(dtype=dtype))

    def apply_delta(self, previous_packed: Tensor, encoded_delta: Tensor) -> Tensor:
        if previous_packed.shape[:-1] != encoded_delta.shape[:-1]:
            raise ValueError("delta and previous KV layouts do not match")
        decoded = self.delta_decoder(encoded_delta).to(dtype=previous_packed.dtype)
        return previous_packed + decoded

    def unpack_like(self, packed: Tensor, template: PrefixKVState) -> PrefixKVState:
        if packed.ndim != 4 or packed.shape[-1] != self.raw_width:
            raise ValueError("packed KV must be [B,L,S,2*head_dim]")
        keys, values = packed[..., : self.head_dim], packed[..., self.head_dim :]
        return PrefixKVState(
            embeddings=template.embeddings,
            pad_mask=template.pad_mask,
            attention_pattern=template.attention_pattern,
            position_ids=template.position_ids,
            pre_rope_keys=tuple(keys[:, layer, None].contiguous() for layer in range(keys.shape[1])),
            values=tuple(values[:, layer, None].contiguous() for layer in range(values.shape[1])),
        )
