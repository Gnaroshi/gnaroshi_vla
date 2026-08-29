"""Shadow-evaluation tensor comparisons."""

from __future__ import annotations

import torch

from .logging_utils import summarize_tensor_diff


def _cosine_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_flat = pred.detach().float().flatten(start_dim=1)
    target_flat = target.detach().float().flatten(start_dim=1)
    return float(torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=-1).mean().item())


def compare_condition_and_action(
    *,
    pred_condition: torch.Tensor,
    target_condition: torch.Tensor,
    pred_action: torch.Tensor | None = None,
    target_action: torch.Tensor | None = None,
) -> dict[str, float]:
    metrics = summarize_tensor_diff(pred_condition, target_condition, prefix="condition")
    metrics["condition_cosine"] = _cosine_mean(pred_condition, target_condition)
    if pred_action is not None and target_action is not None:
        metrics.update(summarize_tensor_diff(pred_action, target_action, prefix="action"))
    return metrics
