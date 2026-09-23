"""Representation-matched metrics for the Stage-A action surrogate audit."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor


def action_from_arm_and_logit(arm: Tensor, gripper_logit: Tensor) -> Tensor:
    """Return the executed continuous action representation [arm, probability]."""

    if arm.shape[:-1] != gripper_logit.shape[:-1]:
        raise ValueError("Arm and gripper tensors have different leading dimensions")
    if arm.shape[-1] != 6 or gripper_logit.shape[-1] != 1:
        raise ValueError("Expected arm[...,6] and gripper_logit[...,1]")
    return torch.cat((arm, torch.sigmoid(gripper_logit)), dim=-1)


def per_sample_l1(predicted: Tensor, target: Tensor) -> list[float]:
    """Mean absolute error per example over every non-batch dimension."""

    if predicted.shape != target.shape:
        raise ValueError(
            f"Action shapes differ: predicted={tuple(predicted.shape)} "
            f"target={tuple(target.shape)}"
        )
    if predicted.ndim < 2:
        raise ValueError("Per-sample action metric requires a batch dimension")
    values = (predicted - target).abs().float().reshape(predicted.shape[0], -1)
    return values.mean(dim=1).detach().cpu().tolist()


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Metric distribution must be non-empty and finite")
    quantiles = np.quantile(array, [0.50, 0.90, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(array.max()),
    }


def build_amended_gate(
    *,
    metrics: dict[str, dict[str, float | int]],
    recursive_reference_mean: float,
    expected_examples: int,
    split_identity_match: bool,
    checkpoint_identity_match: bool,
) -> dict[str, object]:
    """Apply the pre-run v2 amendment contract to matched action-space metrics."""

    required = (
        "joint_to_exact_first_token_l1",
        "hold_to_exact_first_token_l1",
        "joint_to_teacher_first_token_l1",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise KeyError(f"Missing amended gate metrics: {missing}")
    finite = math.isfinite(float(recursive_reference_mean)) and all(
        int(metrics[name]["count"]) == expected_examples
        and math.isfinite(float(metrics[name]["mean"]))
        for name in required
    )
    joint_exact = float(metrics[required[0]]["mean"])
    hold_exact = float(metrics[required[1]]["mean"])
    joint_teacher = float(metrics[required[2]]["mean"])
    checks = {
        "finite": bool(finite),
        "expected_example_count": all(
            int(metrics[name]["count"]) == expected_examples for name in required
        ),
        "validation_split_identity": bool(split_identity_match),
        "checkpoint_identity": bool(checkpoint_identity_match),
        "decoder_fidelity_better_than_hold": bool(joint_exact < hold_exact),
        "teacher_fidelity_better_than_recursive_reference": bool(
            joint_teacher < recursive_reference_mean
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "summary": {
            "joint_to_exact_first_token_l1_mean": joint_exact,
            "hold_to_exact_first_token_l1_mean": hold_exact,
            "joint_to_teacher_first_token_l1_mean": joint_teacher,
            "recursive_to_teacher_first_token_l1_mean": float(
                recursive_reference_mean
            ),
        },
    }
