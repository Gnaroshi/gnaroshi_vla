"""OpenPI dimensions around the unchanged SimVLA Generation updater."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .simvla_generation_loop import SimVLAGenerationHiddenUpdater


@dataclass(frozen=True)
class FlowHiddenConfig:
    hidden_dim: int
    action_dim: int
    condition_dim: int
    state_dim: int
    horizon: int = 10
    rank: int = 128
    gate_bias: float = -4.0


class FlowHiddenUpdater(nn.Module):
    def __init__(self, config: FlowHiddenConfig):
        super().__init__()
        self.config = config
        self.core = SimVLAGenerationHiddenUpdater(
            hidden_dim=config.hidden_dim, condition_dim=config.condition_dim,
            action_dim=config.action_dim, proprio_dim=config.state_dim,
            rank_dim=config.rank, max_generator_age=4, gate_bias=config.gate_bias)

    def descriptor(self):
        return asdict(self.config)

    def forward(self, hidden, previous_x, current_x, condition, state, previous_time, current_time, age):
        b, h, _ = hidden.shape
        if h != self.config.horizon:
            raise ValueError("invalid generation horizon")
        return self.core(
            hidden.float(), previous_x.float(), current_x.float(),
            tau_before=previous_time.expand(b), tau_after=current_time.expand(b),
            proprio=state.float(), condition=condition.float()[:, None], condition_valid_mask=None,
            condition_change_code=torch.zeros(b, 128, device=hidden.device), generator_age=age).hidden
