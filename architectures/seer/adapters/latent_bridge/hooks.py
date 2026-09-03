"""External hooks for Seer representation and action-head boundaries."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import AbstractContextManager

import torch


def _tensor_from_output(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"hook output does not contain a tensor: {type(output)!r}")


class SeerBoundaryCapture(AbstractContextManager):
    """Capture GPT-2 block outputs, final LayerNorm, and action-head input."""

    def __init__(
        self,
        model,
        *,
        detach: bool = True,
        clone: bool = True,
        layer_indices: tuple[int, ...] | None = None,
    ):
        self.model = model.module if hasattr(model, "module") else model
        self.detach = detach
        self.clone = clone
        total_layers = len(self.model.transformer_backbone.h)
        self.layer_indices = (
            tuple(range(total_layers)) if layer_indices is None else tuple(layer_indices)
        )
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices contains duplicates")
        if any(index < 0 or index >= total_layers for index in self.layer_indices):
            raise ValueError(f"invalid layer_indices={self.layer_indices} for {total_layers} blocks")
        self.layer_outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.final_output: torch.Tensor | None = None
        self.action_head_input: torch.Tensor | None = None
        self._handles = []

    def _save(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.detach:
            tensor = tensor.detach()
        if self.clone:
            tensor = tensor.clone()
        return tensor

    def __enter__(self) -> "SeerBoundaryCapture":
        backbone = self.model.transformer_backbone
        for index in self.layer_indices:
            block = backbone.h[index]
            def capture_layer(_module, _inputs, output, *, name=f"block_{index:02d}"):
                self.layer_outputs[name] = self._save(_tensor_from_output(output))

            self._handles.append(block.register_forward_hook(capture_layer))

        def capture_final(_module, _inputs, output):
            self.final_output = self._save(_tensor_from_output(output))

        def capture_action_input(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise TypeError("Seer action decoder pre-hook did not receive a tensor")
            self.action_head_input = self._save(inputs[0])

        self._handles.append(backbone.ln_f.register_forward_hook(capture_final))
        self._handles.append(self.model.action_decoder.register_forward_pre_hook(capture_action_input))
        return self

    def clear(self) -> None:
        self.layer_outputs.clear()
        self.final_output = None
        self.action_head_input = None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def require_complete(self) -> None:
        expected_layers = len(self.layer_indices)
        if len(self.layer_outputs) != expected_layers:
            raise RuntimeError(
                f"captured {len(self.layer_outputs)} GPT-2 blocks, expected {expected_layers}"
            )
        if self.final_output is None:
            raise RuntimeError("final GPT-2 LayerNorm output was not captured")
        if self.action_head_input is None:
            raise RuntimeError("action decoder input was not captured")
