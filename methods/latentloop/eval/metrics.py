"""Efficiency and action diagnostics shared by offline and online evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor


def distribution_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return count and p50/p90/p95/p99/max without third-party dependencies."""

    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
        "max": float(tensor.max().item()),
    }


def action_diagnostics(actions: Tensor, *, chunk_boundaries: Tensor | None = None) -> dict[str, float]:
    """Measure normalized second differences, gripper switches, and boundaries."""

    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError("actions must be [N,7]")
    actions = actions.float()
    if actions.shape[0] >= 3:
        second = actions[2:] - 2 * actions[1:-1] + actions[:-2]
        translation_second = torch.linalg.vector_norm(second[:, :3], dim=-1).mean()
        rotation_second = torch.linalg.vector_norm(second[:, 3:6], dim=-1).mean()
    else:
        translation_second = actions.new_zeros(())
        rotation_second = actions.new_zeros(())
    if actions.shape[0] >= 2:
        gripper_switches = ((actions[1:, 6] >= 0) != (actions[:-1, 6] >= 0)).sum()
        first_diff = torch.linalg.vector_norm(actions[1:, :6] - actions[:-1, :6], dim=-1)
        within_variation = first_diff.mean()
    else:
        gripper_switches = actions.new_zeros((), dtype=torch.long)
        within_variation = actions.new_zeros(())
    boundary_discontinuity = actions.new_zeros(())
    if chunk_boundaries is not None and actions.shape[0] >= 2:
        boundaries = chunk_boundaries.to(dtype=torch.bool, device=actions.device)
        if boundaries.shape != (actions.shape[0],):
            raise ValueError("chunk_boundaries must be [N]")
        boundary_indices = torch.nonzero(boundaries[1:], as_tuple=False).flatten() + 1
        if boundary_indices.numel():
            boundary_discontinuity = torch.linalg.vector_norm(
                actions[boundary_indices, :6] - actions[boundary_indices - 1, :6],
                dim=-1,
            ).mean()
    return {
        "translation_second_difference": float(translation_second.item()),
        "rotation_second_difference": float(rotation_second.item()),
        "gripper_switches": float(gripper_switches.item()),
        "chunk_boundary_discontinuity": float(boundary_discontinuity.item()),
        "within_chunk_action_variation": float(within_variation.item()),
    }


@dataclass
class LatentLoopCounters:
    """Counters required to audit query reduction and hidden compute."""

    values: Counter[str] = field(default_factory=Counter)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment one nonnegative counter."""

        if amount < 0:
            raise ValueError("counter increments must be nonnegative")
        self.values[name] += int(amount)

    def summary(self, *, execution_horizon: int, full_query_interval: int) -> dict[str, float | int]:
        """Return raw counters and both query/env-step reduction definitions."""

        policy_queries = self.values["full_condition_calls"] + self.values["condition_updater_calls"]
        env_actions = self.values["environment_actions"]
        full_calls = self.values["full_condition_calls"]
        return {
            **{key: int(value) for key, value in sorted(self.values.items())},
            "prediction_horizon": 10,
            "execution_horizon": int(execution_horizon),
            "full_condition_interval": int(full_query_interval),
            "full_condition_environment_action_gap": int(execution_horizon) * int(full_query_interval),
            "full_condition_reduction_per_policy_query": float(1.0 - full_calls / max(policy_queries, 1)),
            "full_condition_calls_per_environment_step": float(full_calls / max(env_actions, 1)),
        }
