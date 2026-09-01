"""Latent Bridge adaptation for frozen SimVLA."""

from .checkpoint import load_bridge_checkpoint, save_bridge_checkpoint
from .condition_hook import SimVLAConditionWithStableHook
from .model import SimVLALatentBridge, SimVLALatentBridgeConfig
from .provenance import (
    latent_bridge_source_manifest,
    simvla_latent_bridge_integration_manifest,
)

__all__ = [
    "SimVLAConditionWithStableHook",
    "SimVLALatentBridge",
    "SimVLALatentBridgeConfig",
    "latent_bridge_source_manifest",
    "simvla_latent_bridge_integration_manifest",
    "load_bridge_checkpoint",
    "save_bridge_checkpoint",
]
