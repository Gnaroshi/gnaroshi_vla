"""Auditable compilation of callables, including non-forward entry points."""

from __future__ import annotations

import copy
import time

import torch
from torch import nn


class Compiler:
    def __init__(self, enabled: bool, *, backend=None):
        self.enabled = enabled
        self.backend = backend
        self.records = {}

    def wrap(self, name, function):
        record = self.records.setdefault(name, {"graphs": 0, "compile_seconds": 0.0})
        if not self.enabled:
            return function

        def counting_backend(graph, example_inputs):
            started = time.perf_counter()
            if self.backend is None:
                compiled = torch._inductor.compile(
                    graph, example_inputs,
                    options={"max_autotune": True, "triton.cudagraphs": False},
                )
            else:
                compiled = self.backend(graph, example_inputs)
            record["graphs"] += 1
            record["compile_seconds"] += time.perf_counter() - started
            return compiled

        return torch.compile(function, backend=counting_backend, fullgraph=False, dynamic=False)

    def graph_count(self):
        return sum(record["graphs"] for record in self.records.values())


def compile_bridge_predict_next(bridge, compiler):
    # Compile the bound entry point, not a Module wrapper that delegates this
    # method back to the original instance through __getattr__.
    bridge.predict_next = compiler.wrap("bridge_predict_next", bridge.predict_next)


class ActionStep(nn.Module):
    """Expose the existing concat transformer's hidden without dynamic hooks.

    All frozen layers/parameters remain shared. Only the copied module registry
    substitutes Identity for the readout, which is called once explicitly.
    """

    def __init__(self, transformer):
        super().__init__()
        if getattr(transformer, "use_adaln", False):
            raise ValueError("Only the released concat-mode action head is supported")
        self.hidden_model = copy.copy(transformer)
        self.hidden_model._modules = transformer._modules.copy()
        self.decoder = transformer.action_decoder
        self.hidden_model.action_decoder = nn.Identity()

    def forward(self, vlm_features, action_with_noise, proprio, t):
        hidden = self.hidden_model(
            vlm_features=vlm_features, action_with_noise=action_with_noise,
            proprio=proprio, t=t,
        )
        return hidden, self.decoder(hidden)
