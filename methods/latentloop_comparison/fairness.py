"""Auditable matching and training-isolation utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


def parameter_count(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Count module parameters with an explicit trainability policy."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def named_trainable_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    """Return every trainable parameter keyed by its stable module name."""

    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def assert_parameter_match(
    reference_count: int,
    candidate_count: int,
    *,
    tolerance: float = 0.10,
) -> float:
    """Fail when a candidate exceeds the predeclared relative tolerance."""

    if reference_count <= 0 or candidate_count <= 0:
        raise ValueError("parameter counts must be positive")
    relative_error = abs(candidate_count - reference_count) / float(reference_count)
    if relative_error > tolerance:
        raise RuntimeError(
            f"Parameter mismatch {relative_error:.3%} exceeds {tolerance:.3%}"
        )
    return relative_error


def assert_optimizer_exactly_matches_trainable(
    module: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    """Verify that the optimizer contains every and only trainable parameter."""

    expected = {id(parameter) for parameter in named_trainable_parameters(module).values()}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if actual != expected:
        raise RuntimeError(
            "Optimizer/trainable mismatch: "
            f"missing={len(expected - actual)}, unexpected={len(actual - expected)}"
        )


def assert_zero_or_missing_gradients(modules: Iterable[nn.Module]) -> None:
    """Verify frozen modules received no nonzero gradient."""

    offenders: list[str] = []
    for module_index, module in enumerate(modules):
        for name, parameter in module.named_parameters():
            if parameter.grad is not None and torch.count_nonzero(parameter.grad):
                offenders.append(f"module{module_index}.{name}")
    if offenders:
        raise RuntimeError(f"Frozen parameters received gradients: {offenders[:20]}")


def stable_json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible manifest with canonical serialization."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_refuse_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON artifact without replacing prior experiment evidence."""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_fairness_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate required fields for all three comparison methods."""

    required_methods = {
        "canonical_latentloop",
        "matched_action_space_correction",
        "nonrecurrent_anchor_to_current_latent",
    }
    methods = manifest.get("methods", {})
    if set(methods) != required_methods:
        raise ValueError(
            f"Fairness manifest methods must be {sorted(required_methods)}"
        )
    required = {
        "trainable_parameters",
        "parameter_groups",
        "training_dataset_manifest_sha256",
        "training_examples",
        "optimizer",
        "learning_rate",
        "batch_size",
        "precision",
        "optimizer_steps",
        "checkpoint_frequency",
        "validation_split",
        "validation_metric",
        "checkpoint_selection_rule",
        "wall_clock_training_seconds",
        "gpu_count",
        "environment",
        "source_sha256",
    }
    for name, row in methods.items():
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{name} is missing fairness fields: {missing}")


def ensure_same_manifest_keys(
    rows: Mapping[str, Sequence[tuple[int, int]]]
) -> list[tuple[int, int]]:
    """Return the shared ordered keys or fail on any episode mismatch."""

    if not rows:
        raise ValueError("No episode manifests were supplied")
    iterator = iter(rows.items())
    reference_name, reference = next(iterator)
    reference_list = list(reference)
    for name, keys in iterator:
        if list(keys) != reference_list:
            raise RuntimeError(
                f"Episode manifest mismatch: {reference_name} versus {name}"
            )
    return reference_list
