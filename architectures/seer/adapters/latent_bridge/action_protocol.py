"""Exact Seer action-sequence and temporal-ensemble semantics."""

from __future__ import annotations

import numpy as np
import torch


class SeerTemporalEnsembler:
    """Replicate ``ModelWrapper._action_sequence_to_env_action`` exactly."""

    def __init__(self, max_steps: int, action_pred_steps: int = 3, temperature: float = 0.01):
        self.max_steps = int(max_steps)
        self.action_pred_steps = int(action_pred_steps)
        self.temperature = float(temperature)
        self.buffer = torch.zeros(self.max_steps, self.max_steps + self.action_pred_steps, 7)

    def to(self, device) -> "SeerTemporalEnsembler":
        self.buffer = self.buffer.to(device)
        return self

    def reset(self) -> None:
        self.buffer.zero_()

    def continuous_action(self, action_sequence: torch.Tensor, timestep: int) -> torch.Tensor:
        if tuple(action_sequence.shape) != (1, self.action_pred_steps, 7):
            raise ValueError(
                f"expected [1, {self.action_pred_steps}, 7], got {tuple(action_sequence.shape)}"
            )
        self.buffer[timestep : timestep + 1, timestep : timestep + self.action_pred_steps] = action_sequence
        candidates = self.buffer[:, timestep]
        candidates = candidates[torch.all(candidates != 0, dim=1)]
        if candidates.shape[0] == 0:
            raise RuntimeError("Seer temporal ensemble found no populated candidate")
        weights = np.exp(-self.temperature * np.arange(len(candidates)))
        weights /= weights.sum()
        weights_t = torch.from_numpy(weights).to(candidates.device).unsqueeze(1)
        return (candidates * weights_t).sum(dim=0, keepdim=True)

    @staticmethod
    def to_environment_action(continuous_action: torch.Tensor) -> torch.Tensor:
        if tuple(continuous_action.shape) != (1, 7):
            raise ValueError(f"expected [1, 7], got {tuple(continuous_action.shape)}")
        result = torch.cat(
            [continuous_action[:, :6], continuous_action[:, 6:] > 0.5], dim=-1
        )
        result[:, -1] = (result[:, -1] - 0.5) * 2
        return result

    def step(self, action_sequence: torch.Tensor, timestep: int) -> torch.Tensor:
        return self.to_environment_action(self.continuous_action(action_sequence, timestep))
