"""Logging helpers for DCLD evaluation."""

from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


class LatencyAccumulator:
    """Collect named wall-clock latency segments."""

    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add(self, name: str, seconds: float) -> None:
        self.values.setdefault(name, []).append(float(seconds))

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for name, vals in self.values.items():
            out[name] = {
                "count": float(len(vals)),
                "mean": float(statistics.fmean(vals)) if vals else 0.0,
                "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                "min": float(min(vals)) if vals else 0.0,
                "max": float(max(vals)) if vals else 0.0,
                "p50": _percentile(vals, 0.50),
                "p90": _percentile(vals, 0.90),
                "p95": _percentile(vals, 0.95),
                "p99": _percentile(vals, 0.99),
            }
        return out


def summarize_tensor_diff(pred: torch.Tensor, target: torch.Tensor, prefix: str = "") -> dict[str, float]:
    diff = (pred - target).detach().float()
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}mean_abs": float(diff.abs().mean().item()),
        f"{key}max_abs": float(diff.abs().max().item()),
        f"{key}l2": float(diff.flatten(start_dim=1).norm(dim=-1).mean().item()),
    }


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
