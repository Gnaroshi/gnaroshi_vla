"""Deterministic global-step batch sampling for exact training resume."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class DeterministicStepBatchSampler(Sampler[list[int]]):
    """Yield the same shuffled batches for a global step before and after resume."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        seed: int,
        start_step: int,
        max_steps: int,
    ) -> None:
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= start_step <= max_steps:
            raise ValueError("start_step must be in [0,max_steps]")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.max_steps = int(max_steps)
        self.batches_per_epoch = math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        current_epoch = -1
        permutation: list[int] = []
        for step in range(self.start_step, self.max_steps):
            epoch = step // self.batches_per_epoch
            batch_index = step % self.batches_per_epoch
            if epoch != current_epoch:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.seed + epoch)
                permutation = torch.randperm(
                    self.dataset_size,
                    generator=generator,
                ).tolist()
                current_epoch = epoch
            start = batch_index * self.batch_size
            stop = min(start + self.batch_size, self.dataset_size)
            yield permutation[start:stop]

    def __len__(self) -> int:
        """Return the number of remaining optimization steps."""

        return self.max_steps - self.start_step
