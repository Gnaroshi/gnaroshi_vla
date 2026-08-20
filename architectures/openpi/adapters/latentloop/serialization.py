"""Tensor-only serialization helpers for prefix states."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from .prefix_kv_hook import PrefixEmbeddingState, PrefixKVState
from .transition_core import OpenPIKVLatentLoop, adapter_config_from_dict


def adapter_checkpoint_config(adapter: torch.nn.Module) -> dict[str, Any]:
    """Return an explicit, extensible adapter descriptor for checkpoints."""

    from .latent_bridge_baseline import LocalLatentBridgeAdapter

    if isinstance(adapter, OpenPIKVLatentLoop):
        from .transition_core import adapter_config_to_dict

        return {
            "adapter_type": "openpi_variable_time_latentloop",
            "adapter": adapter_config_to_dict(adapter.config),
        }
    if isinstance(adapter, LocalLatentBridgeAdapter):
        return {
            "adapter_type": "latent_bridge_style_kv_baseline",
            "scientific_label": "Latent Bridge-style KV baseline",
            "adapter": asdict(adapter.config),
        }
    raise TypeError(f"unsupported adapter class: {type(adapter).__name__}")


def prefix_state_to_dict(state: PrefixKVState, *, squeeze_batch: bool = True) -> dict[str, torch.Tensor]:
    state.validate()
    batch = state.embeddings.shape[0]
    if squeeze_batch and batch != 1:
        raise ValueError("batch squeezing is only valid for B=1")

    def maybe_squeeze(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().cpu()
        return value[0] if squeeze_batch else value

    keys = torch.stack(state.pre_rope_keys, dim=1)
    values = torch.stack(state.values, dim=1)
    return {
        "prefix_embeddings": maybe_squeeze(state.embeddings),
        "prefix_pad_mask": maybe_squeeze(state.pad_mask),
        "prefix_attention_pattern": maybe_squeeze(state.attention_pattern),
        "prefix_position_ids": maybe_squeeze(state.position_ids),
        "pre_rope_keys": maybe_squeeze(keys),
        "values": maybe_squeeze(values),
    }


def prefix_state_from_record(record: dict[str, Any], device: str | torch.device) -> PrefixKVState:
    embeddings = torch.as_tensor(record["prefix_embeddings"], device=device)
    pad_mask = torch.as_tensor(record["prefix_pad_mask"], device=device)
    attention = torch.as_tensor(record["prefix_attention_pattern"], device=device)
    positions = torch.as_tensor(record["prefix_position_ids"], device=device)
    keys = torch.as_tensor(record["pre_rope_keys"], device=device)
    values = torch.as_tensor(record["values"], device=device)
    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)
        pad_mask = pad_mask.unsqueeze(0)
        attention = attention.unsqueeze(0)
        positions = positions.unsqueeze(0)
        keys = keys.unsqueeze(0)
        values = values.unsqueeze(0)
    if keys.ndim != 5:
        raise ValueError("serialized pre-RoPE keys must be [B,L,H,S,D]")
    state = PrefixKVState(
        embeddings=embeddings,
        pad_mask=pad_mask,
        attention_pattern=attention,
        position_ids=positions,
        pre_rope_keys=tuple(keys[:, layer] for layer in range(keys.shape[1])),
        values=tuple(values[:, layer] for layer in range(values.shape[1])),
    )
    state.validate()
    return state


def prefix_embedding_from_record(record: dict[str, Any], device: str | torch.device) -> PrefixEmbeddingState:
    state = prefix_state_from_record(record, device)
    return PrefixEmbeddingState(
        embeddings=state.embeddings,
        pad_mask=state.pad_mask,
        attention_pattern=state.attention_pattern,
        position_ids=state.position_ids,
    )


def load_adapter_checkpoint(
    checkpoint: str,
    device: str | torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    descriptor = payload["config"]
    adapter_type = descriptor.get("adapter_type", "openpi_variable_time_latentloop")
    if adapter_type == "openpi_variable_time_latentloop":
        config = adapter_config_from_dict(descriptor["adapter"])
        adapter = OpenPIKVLatentLoop(config)
    elif adapter_type in {"latent_bridge_style", "latent_bridge_style_kv_baseline"}:
        from .latent_bridge_baseline import LocalLatentBridgeAdapter, latent_bridge_config_from_dict

        config = latent_bridge_config_from_dict(descriptor["adapter"])
        adapter = LocalLatentBridgeAdapter(config)
    else:
        raise ValueError(f"unknown adapter_type in checkpoint: {adapter_type}")
    adapter.load_state_dict(payload["adapter"], strict=True)
    adapter.to(device).eval()
    return adapter, payload
