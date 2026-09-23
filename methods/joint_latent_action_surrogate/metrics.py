"""Two-error reset metrics used by offline gates and evaluation aggregation."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor


def latent_state_error(light_latent: Tensor, full_latent: Tensor) -> Tensor:
    return (light_latent.float() - full_latent.float()).pow(2).mean(dim=(-2, -1)).sqrt()


def action_surrogate_error(surrogate: Tensor, exact: Tensor) -> Tensor:
    return (surrogate.float() - exact.float()).abs().mean(dim=(-2, -1))


class ErrorAccumulator:
    """CPU scalar accumulator with age-stratified quantiles."""

    def __init__(self) -> None:
        self._values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    def add(self, name: str, age: int, values: Tensor) -> None:
        flat = values.detach().float().cpu().reshape(-1)
        self._values[str(name)][int(age)].extend(float(value) for value in flat)

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        result: dict[str, dict[str, dict[str, float]]] = {}
        for name, ages in self._values.items():
            result[name] = {}
            for age, values in sorted(ages.items()):
                tensor = torch.tensor(values, dtype=torch.float64)
                quantiles = torch.quantile(tensor, torch.tensor([0.5, 0.9, 0.95, 0.99]))
                result[name][f"age{age}"] = {
                    "count": int(tensor.numel()),
                    "mean": float(tensor.mean()),
                    "p50": float(quantiles[0]),
                    "p90": float(quantiles[1]),
                    "p95": float(quantiles[2]),
                    "p99": float(quantiles[3]),
                }
        return result
