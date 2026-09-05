"""Shared deterministic-inference controls for paired Seer evaluation."""

from __future__ import annotations

import os

import torch


def configure_deterministic_inference() -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    requested = os.environ.get("SEER_LATENT_BRIDGE_DETERMINISTIC", "1").strip().lower()
    if requested not in {"0", "1", "false", "true"}:
        raise ValueError("SEER_LATENT_BRIDGE_DETERMINISTIC must be a boolean flag")
    enabled = requested in {"1", "true"}
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled
    return {
        "requested": enabled,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
