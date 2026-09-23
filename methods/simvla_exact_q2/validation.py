"""Validation summaries and validation-only checkpoint selection."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        raise ValueError("cannot summarize an empty metric")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("metric contains NaN or Inf")
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
        "max": float(tensor.max().item()),
    }


def paired_bootstrap_ci95(
    differences: Iterable[float],
    *,
    seed: int = 20260815,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Return a paired percentile-bootstrap CI over per-example differences."""

    values = torch.tensor(list(differences), dtype=torch.float64)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("paired differences must be nonempty and finite")
    generator = torch.Generator().manual_seed(int(seed))
    means: list[torch.Tensor] = []
    remaining = int(samples)
    while remaining > 0:
        count = min(remaining, 512)
        indices = torch.randint(
            0,
            values.numel(),
            (count, values.numel()),
            generator=generator,
        )
        means.append(values[indices].mean(dim=1))
        remaining -= count
    bootstrapped = torch.cat(means)
    return (
        float(torch.quantile(bootstrapped, 0.025).item()),
        float(torch.quantile(bootstrapped, 0.975).item()),
    )


def validation_selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    """Lexicographic validation rule with earlier step as the final tie-breaker."""

    metrics = row["metrics"]
    return (
        float(metrics["q2_prefix_l1"]["mean"]),
        float(metrics["q2_prefix_l1"]["p95"]),
        float(metrics["q2_gripper_command_l1"]["mean"]),
        float(metrics["q2_chunk_l1"]["mean"]),
        float(metrics["q2_condition_normalized_mse"]["mean"]),
        int(row["step"]),
    )


def select_validation_checkpoint(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidate: str,
) -> dict[str, Any]:
    """Select a checkpoint using validation metrics only."""

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("no validation checkpoint rows")
    if any(row.get("partition") != "validation" for row in candidates):
        raise ValueError("checkpoint selection received a non-validation row")
    selected = min(candidates, key=validation_selection_key)
    return {
        "schema_version": "simvla_exact_q2_selection_v1",
        "candidate": candidate,
        "selection_data": "validation_only",
        "selection_rule": [
            "q2_prefix_l1_mean",
            "q2_prefix_l1_p95",
            "q2_gripper_command_l1_mean",
            "q2_chunk_l1_mean",
            "q2_condition_normalized_mse_mean",
            "earlier_checkpoint",
        ],
        "selected_checkpoint": selected["checkpoint"],
        "selected_step": int(selected["step"]),
        "selected_metrics": selected["metrics"],
        "evaluated_checkpoints": candidates,
    }
