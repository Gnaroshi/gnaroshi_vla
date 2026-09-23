"""SimVLA-facing exact-q2 model construction and adapter-only checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from methods.simvla_exact_q2.direct_model import DirectExactQ2
from methods.simvla_exact_q2.recurrent_model import (
    ExactQ2ModelConfig,
    ExactQ2Prediction,
    RecurrentExactQ2,
)


ExactQ2Candidate = Literal["recurrent_exact_q2", "direct_exact_q2"]
CHECKPOINT_TYPE = "simvla_r5_exact_q2_adapter_v1"


class SimVLAExactQ2Adapter(nn.Module):
    """Expose one common batch API while preserving candidate-specific semantics."""

    def __init__(self, candidate: ExactQ2Candidate, config: ExactQ2ModelConfig | None = None) -> None:
        super().__init__()
        self.candidate = candidate
        self.config = config or ExactQ2ModelConfig()
        if candidate == "recurrent_exact_q2":
            self.model: nn.Module = RecurrentExactQ2(self.config)
        elif candidate == "direct_exact_q2":
            self.model = DirectExactQ2(self.config)
        else:
            raise ValueError(f"unsupported exact-q2 candidate: {candidate}")

    def forward(self, batch: dict[str, Any]) -> ExactQ2Prediction:
        fields = {
            name: batch[name]
            for name in (
                "c0_full",
                "q0_raw_rgb",
                "q0_proprio",
                "q1_raw_rgb",
                "q1_proprio",
                "q2_raw_rgb",
                "q2_proprio",
                "x0_executed",
                "x1_executed",
                "elapsed_q0_to_q1",
                "elapsed_q1_to_q2",
            )
        }
        return self.model(**fields)


def build_exact_q2_adapter(candidate: ExactQ2Candidate) -> SimVLAExactQ2Adapter:
    return SimVLAExactQ2Adapter(candidate)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def parameter_budget_audit() -> dict[str, Any]:
    """Require the two candidate counts to match within the frozen +/-5% criterion."""

    modules = {
        candidate: build_exact_q2_adapter(candidate)
        for candidate in ("recurrent_exact_q2", "direct_exact_q2")
    }
    counts = {name: count_trainable_parameters(module) for name, module in modules.items()}
    relative = abs(counts["recurrent_exact_q2"] - counts["direct_exact_q2"]) / float(
        counts["recurrent_exact_q2"]
    )
    return {
        "schema_version": "simvla_exact_q2_parameter_audit_v1",
        "capacity": "historical_1x_lightweight",
        "candidates": {
            name: {
                "trainable_parameters": count,
                "config": modules[name].config.to_dict(),
            }
            for name, count in counts.items()
        },
        "relative_parameter_difference": relative,
        "required_tolerance": 0.05,
        "parameter_match_pass": relative <= 0.05,
    }


def trainable_parameter_names(module: nn.Module) -> list[str]:
    return sorted(name for name, parameter in module.named_parameters() if parameter.requires_grad)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def save_exact_q2_checkpoint(
    path: str | Path,
    *,
    adapter: SimVLAExactQ2Adapter,
    step: int,
    metadata: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically serialize only the lightweight candidate and reproducibility state."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "candidate": adapter.candidate,
        "config": adapter.config.to_dict(),
        "step": int(step),
        "adapter_state_dict": adapter.state_dict(),
        "metadata": metadata,
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if training_state is not None:
        payload["training_state"] = training_state
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_exact_q2_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SimVLAExactQ2Adapter, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(f"expected checkpoint type {CHECKPOINT_TYPE}")
    config = ExactQ2ModelConfig(**payload["config"])
    adapter = SimVLAExactQ2Adapter(payload["candidate"], config).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    return adapter, payload


def same_information_contract(batch: dict[str, Tensor]) -> dict[str, tuple[int, ...]]:
    """Return the candidate-independent input shapes recorded by smoke tests."""

    names = (
        "c0_full",
        "q0_raw_rgb",
        "q0_proprio",
        "q1_raw_rgb",
        "q1_proprio",
        "q2_raw_rgb",
        "q2_proprio",
        "x0_executed",
        "x1_executed",
        "elapsed_q0_to_q1",
        "elapsed_q1_to_q2",
    )
    return {name: tuple(batch[name].shape) for name in names}
