"""SimVLA's unmodified Condition core at OpenPI's layer-wise pre-RoPE KV interface."""

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

from methods.latentloop.modules.native_simvla_v0 import (
    NativeV0DeltaEncoder, NativeV0ObservationPair, TokenSharedConditionUpdater,
)


FORMAT = "pi05_simvla_core_condition_v1"
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")


@dataclass(frozen=True)
class ConditionConfig:
    layers: int
    kv_heads: int
    head_dim: int
    max_tokens: int
    proprio_dim: int
    rank_dim: int = 64
    delta_dim: int = 128
    max_age: int = 3

    @classmethod
    def from_prefix(cls, prefix, observation):
        prefix.validate()
        return cls(prefix.num_layers, prefix.values[0].shape[1], prefix.values[0].shape[-1],
                   prefix.num_tokens, observation.state.shape[-1])


def pack_kv(prefix):
    # Batch stays outermost; layer, K/V type, and head become independent token groups.
    return torch.stack([torch.stack((k, v), dim=1) for k, v in
                        zip(prefix.pre_rope_keys, prefix.values, strict=True)], dim=1)


def unpack_kv(prefix, packed):
    return replace(prefix, pre_rope_keys=tuple(packed[:, i, 0] for i in range(prefix.num_layers)),
                   values=tuple(packed[:, i, 1] for i in range(prefix.num_layers)))


def observation_pair(previous, current):
    views = []
    for observation in (previous, current):
        ordered = []
        for camera in CAMERAS:
            if not bool(observation.image_masks[camera].all()):
                raise ValueError(f"required camera is masked: {camera}")
            image = observation.images[camera]
            if not image.is_floating_point():
                raise TypeError("expected OpenPI transformed images in [-1,1]")
            ordered.append((image.float() + 1.0) * 0.5)
        views.append(ordered)
    for field in ("tokenized_prompt", "tokenized_prompt_mask"):
        if not torch.equal(getattr(previous, field), getattr(current, field)):
            raise ValueError("prompt/layout changed inside a Condition refresh interval")
    return NativeV0ObservationPair(views[0], views[1], previous.state, current.state)


class AlignedConditionUpdater(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.delta_encoder = NativeV0DeltaEncoder(num_views=2, proprio_dim=config.proprio_dim,
                                                  output_dim=config.delta_dim)
        self.condition_updater = TokenSharedConditionUpdater(
            condition_dim=config.head_dim, delta_dim=config.delta_dim, rank_dim=config.rank_dim,
            max_tokens=config.max_tokens, num_token_groups=config.layers * 2 * config.kv_heads,
            max_age=config.max_age, gate_bias=-4.0)

    def descriptor(self):
        return asdict(self.config)

    def forward(self, previous_prefix, previous_observation, current_observation, *, age):
        previous_prefix.validate()
        packed = pack_kv(previous_prefix)
        b, layers, kinds, heads, tokens, dim = packed.shape
        if (layers, heads, dim) != (self.config.layers, self.config.kv_heads, self.config.head_dim):
            raise ValueError("prefix KV geometry differs from the trained Condition interface")
        delta = self.delta_encoder(observation_pair(previous_observation, current_observation))
        groups = layers * kinds * heads
        valid = previous_prefix.pad_mask[:, None].expand(b, groups, tokens).reshape(b * groups, tokens)
        group_ids = torch.arange(groups, device=packed.device)[None, :, None].expand(b, groups, tokens)
        update = self.condition_updater(
            packed.float().reshape(b * groups, tokens, dim),
            delta[:, None].expand(b, groups, self.config.delta_dim).reshape(b * groups, self.config.delta_dim),
            valid_mask=valid, group_ids=group_ids.reshape(b * groups, tokens), age=age)
        predicted = update.condition.reshape_as(packed).to(packed.dtype)
        # Embeddings remain only as RoPE shape/dtype metadata, never as updater input.
        return unpack_kv(previous_prefix, predicted), update


def load_condition(path, device="cuda", config_id=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("legacy pi0.5 KV checkpoints are not method-aligned Condition checkpoints")
    if config_id is not None and payload.get("config_id") != config_id:
        raise ValueError("Condition checkpoint belongs to another experiment")
    model = AlignedConditionUpdater(ConditionConfig(**payload["updater_config"]))
    model.load_state_dict(payload["updater"], strict=True)
    return model.to(device).eval(), payload
