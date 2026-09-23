"""CPU/GPU-safe metrics for latent-filter traces and paired outcomes."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


ARM_TRANSLATION = slice(0, 3)
ARM_ROTATION = slice(3, 6)
GRIPPER_INDEX = 6


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    """Return the two-sided Wilson score interval for a Bernoulli proportion."""
    if total <= 0:
        return math.nan, math.nan
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_outcome_counts(
    baseline: Mapping[Tuple[int, int], int],
    candidate: Mapping[Tuple[int, int], int],
) -> Dict[str, int]:
    """Count paired success/failure transitions on common episode keys."""
    keys = sorted(set(baseline) & set(candidate))
    counts = {
        "paired": len(keys),
        "fail_to_success": 0,
        "success_to_fail": 0,
        "both_success": 0,
        "both_fail": 0,
    }
    labels = {
        (0, 1): "fail_to_success",
        (1, 0): "success_to_fail",
        (1, 1): "both_success",
        (0, 0): "both_fail",
    }
    for key in keys:
        counts[labels[(int(baseline[key]), int(candidate[key]))]] += 1
    counts["net_flip"] = counts["fail_to_success"] - counts["success_to_fail"]
    return counts


def hierarchical_paired_bootstrap_ci(
    paired_rows: Sequence[Mapping[str, int]],
    samples: int = 10_000,
    seed: int = 20260727,
) -> Tuple[float, float]:
    """Bootstrap paired SR differences by resampling tasks, then episodes."""
    by_task: Dict[int, list[float]] = defaultdict(list)
    for row in paired_rows:
        by_task[int(row["task_id"])].append(
            float(row["candidate_success"]) - float(row["baseline_success"])
        )
    if not by_task:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    task_ids = np.asarray(sorted(by_task), dtype=np.int64)
    draws = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
        values = []
        for task_id in sampled_tasks:
            episodes = np.asarray(by_task[int(task_id)], dtype=np.float64)
            values.extend(rng.choice(episodes, size=len(episodes), replace=True))
        draws[sample_index] = float(np.mean(values))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def checkpoint_hierarchical_paired_bootstrap_ci(
    paired_rows: Sequence[Mapping[str, int]],
    samples: int = 10_000,
    seed: int = 20260727,
) -> Tuple[float, float]:
    """Bootstrap checkpoints, tasks, then paired episodes for a mean SR delta.

    Checkpoint 36 and checkpoint 38 are the independent top-level units in the
    held-out confirmation. This routine deliberately does not flatten their
    episodes into one 400-episode Bernoulli sample.
    """
    by_checkpoint_task: Dict[int, Dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in paired_rows:
        by_checkpoint_task[int(row["checkpoint_id"])][int(row["task_id"])].append(
            float(row["candidate_success"]) - float(row["baseline_success"])
        )
    if not by_checkpoint_task:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    checkpoint_ids = np.asarray(sorted(by_checkpoint_task), dtype=np.int64)
    draws = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_checkpoints = rng.choice(
            checkpoint_ids,
            size=len(checkpoint_ids),
            replace=True,
        )
        checkpoint_means = []
        for checkpoint_id in sampled_checkpoints:
            by_task = by_checkpoint_task[int(checkpoint_id)]
            task_ids = np.asarray(sorted(by_task), dtype=np.int64)
            sampled_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
            episode_values = []
            for task_id in sampled_tasks:
                episodes = np.asarray(by_task[int(task_id)], dtype=np.float64)
                episode_values.extend(
                    rng.choice(episodes, size=len(episodes), replace=True)
                )
            checkpoint_means.append(float(np.mean(episode_values)))
        draws[sample_index] = float(np.mean(checkpoint_means))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _flatten_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1)


def latent_pair_metrics(
    z_prior: Optional[torch.Tensor],
    z_full: torch.Tensor,
    z_filter: torch.Tensor,
    alpha: float,
) -> Dict[str, float]:
    """Compute aggregate and token-wise latent disagreement statistics."""
    full = z_full.detach().float()
    filtered = z_filter.detach().float()
    metrics: Dict[str, float] = {
        "latent_filter_vs_full_l2": float(
            torch.linalg.vector_norm(filtered - full).item()
        ),
    }
    if z_prior is not None:
        prior = z_prior.detach().float()
        prior_flat = _flatten_float(prior)
        full_flat = _flatten_float(full)
        denominator = (
            torch.linalg.vector_norm(prior_flat)
            * torch.linalg.vector_norm(full_flat)
        )
        cosine = (
            torch.dot(prior_flat, full_flat) / denominator
            if float(denominator.item()) > 0.0
            else torch.tensor(float("nan"), device=full.device)
        )
        metrics.update(
            {
                "latent_prior_vs_full_l2": float(
                    torch.linalg.vector_norm(prior - full).item()
                ),
                "latent_prior_vs_full_cosine": float(cosine.item()),
                "latent_correction_l2": float(
                    torch.linalg.vector_norm(float(alpha) * (full - prior)).item()
                ),
            }
        )
    if full.ndim >= 2:
        token_axis = full.ndim - 2
        token_count = int(full.shape[token_axis])
        for token_index in range(token_count):
            full_token = full.select(token_axis, token_index)
            filter_token = filtered.select(token_axis, token_index)
            metrics[f"latent_token{token_index}_full_norm"] = float(
                torch.linalg.vector_norm(full_token).item()
            )
            metrics[f"latent_token{token_index}_filter_norm"] = float(
                torch.linalg.vector_norm(filter_token).item()
            )
            metrics[f"latent_token{token_index}_filter_vs_full_l2"] = float(
                torch.linalg.vector_norm(filter_token - full_token).item()
            )
            if z_prior is not None:
                prior_token = z_prior.detach().float().select(
                    token_axis,
                    token_index,
                )
                metrics[f"latent_token{token_index}_prior_norm"] = float(
                    torch.linalg.vector_norm(prior_token).item()
                )
                metrics[f"latent_token{token_index}_prior_vs_full_l2"] = float(
                    torch.linalg.vector_norm(prior_token - full_token).item()
                )
    return metrics


@dataclass
class LatentPathTracker:
    """Track per-branch path length and second finite differences per episode."""

    previous: Dict[str, torch.Tensor] = field(default_factory=dict)
    previous_update: Dict[str, torch.Tensor] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear all episode-local trajectory state."""
        self.previous.clear()
        self.previous_update.clear()

    def update(self, **latents: Optional[torch.Tensor]) -> Dict[str, float]:
        """Observe named latent branches and return step/path statistics."""
        metrics: Dict[str, float] = {}
        for name, latent in latents.items():
            if latent is None:
                continue
            current = latent.detach().float()
            previous = self.previous.get(name)
            if previous is not None:
                delta = current - previous
                metrics[f"latent_{name}_step_l2"] = float(
                    torch.linalg.vector_norm(delta).item()
                )
                previous_delta = self.previous_update.get(name)
                if previous_delta is not None:
                    metrics[f"latent_{name}_second_difference_l2"] = float(
                        torch.linalg.vector_norm(delta - previous_delta).item()
                    )
                self.previous_update[name] = delta.detach().clone()
            self.previous[name] = current.detach().clone()
        return metrics


def continuous_action_metrics(actions: np.ndarray) -> Dict[str, float]:
    """Summarize normalized arm second differences; exclude discrete gripper."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"Expected normalized actions [T, 7], got {values.shape}")
    if len(values) < 3:
        second = np.empty((0, 6), dtype=np.float64)
    else:
        second = (
            values[2:, :6]
            - 2.0 * values[1:-1, :6]
            + values[:-2, :6]
        )
    translation = np.linalg.norm(second[:, ARM_TRANSLATION], axis=-1)
    rotation = np.linalg.norm(second[:, ARM_ROTATION], axis=-1)

    def summarize(prefix: str, samples_array: np.ndarray) -> Dict[str, float]:
        if len(samples_array) == 0:
            return {f"{prefix}_mean": 0.0, f"{prefix}_p95": 0.0}
        return {
            f"{prefix}_mean": float(np.mean(samples_array)),
            f"{prefix}_p95": float(np.percentile(samples_array, 95)),
        }

    return {
        **summarize("translation_second_difference", translation),
        **summarize("rotation_second_difference", rotation),
    }


def gripper_continuity_metrics(actions: np.ndarray) -> Dict[str, float]:
    """Count discrete gripper switches and short-horizon reversals."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"Expected normalized actions [T, 7], got {values.shape}")
    states = values[:, GRIPPER_INDEX] > 0.0
    switches = np.flatnonzero(states[1:] != states[:-1]) + 1
    metrics = {
        "gripper_switches_per_100_steps": (
            100.0 * len(switches) / max(1, len(values))
        ),
    }
    for horizon in (1, 2, 5):
        reversals = 0
        for index, switch in enumerate(switches[:-1]):
            next_switch = switches[index + 1]
            if next_switch - switch <= horizon:
                reversals += 1
        metrics[f"gripper_reverse_within_{horizon}_per_100_steps"] = (
            100.0 * reversals / max(1, len(values))
        )
    return metrics
