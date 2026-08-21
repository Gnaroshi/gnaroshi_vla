"""External pi0.5 LatentLoop adapter; OpenPI upstream remains unchanged.

The LIBERO client runs in OpenPI's separate Python 3.8 environment. Keep this
package import lightweight so contract-only modules do not pull server-side
dependencies such as JAX into that client process.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BudgetedDynamicPolicy": (".dynamic_policy", "BudgetedDynamicPolicy"),
    "LayerSharedKVCodec": (".kv_codec", "LayerSharedKVCodec"),
    "LatentBridgeConfig": (".latent_bridge_baseline", "LatentBridgeConfig"),
    "LocalLatentBridgeAdapter": (".latent_bridge_baseline", "LocalLatentBridgeAdapter"),
    "OpenPIKVLatentLoop": (".transition_core", "OpenPIKVLatentLoop"),
    "OpenPIKVLatentLoopConfig": (".transition_core", "OpenPIKVLatentLoopConfig"),
    "OpenPILatentLoopPolicy": (".recurrent_policy", "OpenPILatentLoopPolicy"),
    "LatentLoopServingPolicy": (".online_policy", "LatentLoopServingPolicy"),
    "PrefixEmbeddingState": (".prefix_kv_hook", "PrefixEmbeddingState"),
    "PrefixKVHook": (".prefix_kv_hook", "PrefixKVHook"),
    "PrefixKVState": (".prefix_kv_hook", "PrefixKVState"),
}

__all__ = [
    "BudgetedDynamicPolicy",
    "LayerSharedKVCodec",
    "LatentBridgeConfig",
    "LocalLatentBridgeAdapter",
    "OpenPIKVLatentLoop",
    "OpenPIKVLatentLoopConfig",
    "OpenPILatentLoopPolicy",
    "LatentLoopServingPolicy",
    "PrefixEmbeddingState",
    "PrefixKVHook",
    "PrefixKVState",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
