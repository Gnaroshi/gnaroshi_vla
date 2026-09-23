"""External capture of SimVLA final and middle-layer action conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConditionWithStable:
    condition: Tensor
    stable: Tensor
    stable_layer_index: int
    layer_path: str


def resolve_text_layers(model: Any) -> tuple[Any, str]:
    roots = (
        (getattr(model.vlm.model, "text_model", None), "vlm.model.text_model"),
    )
    for root, prefix in roots:
        if root is None:
            continue
        candidates = (
            (getattr(root, "layers", None), f"{prefix}.layers"),
            (getattr(getattr(root, "model", None), "layers", None), f"{prefix}.model.layers"),
            (
                getattr(getattr(root, "decoder", None), "layers", None),
                f"{prefix}.decoder.layers",
            ),
        )
        for layers, path in candidates:
            if layers is not None and len(layers):
                return layers, path
    raise AttributeError("cannot resolve SmolVLM text transformer layers")


class SimVLAConditionWithStableHook:
    """Capture a frozen middle-layer output during exact `forward_vlm_efficient`."""

    def __init__(self, model: Any, *, stable_layer_index: int = 10) -> None:
        self.model = model
        self.layers, self.layer_path = resolve_text_layers(model)
        if not 0 <= stable_layer_index < len(self.layers):
            raise ValueError(
                f"stable layer index {stable_layer_index} outside [0,{len(self.layers) - 1}]"
            )
        self.stable_layer_index = int(stable_layer_index)
        self._captured: Tensor | None = None
        self._handle = self.layers[self.stable_layer_index].register_forward_hook(
            self._capture
        )

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise RuntimeError("unexpected SmolVLM middle-layer hook output")
        self._captured = hidden

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "SimVLAConditionWithStableHook":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def encode(
        self,
        *,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
    ) -> ConditionWithStable:
        self._captured = None
        encoded = self.model.forward_vlm_efficient(
            image_input, image_mask, input_ids
        )
        condition = encoded.get("vlm_features")
        stable = self._captured
        if condition is None or stable is None:
            raise RuntimeError(
                "exact SimVLA forward did not produce both final and stable features"
            )
        if stable.shape != condition.shape:
            raise RuntimeError(
                f"middle/final feature shapes differ: {tuple(stable.shape)} vs {tuple(condition.shape)}"
            )
        return ConditionWithStable(
            condition=condition,
            stable=stable,
            stable_layer_index=self.stable_layer_index,
            layer_path=self.layer_path,
        )
