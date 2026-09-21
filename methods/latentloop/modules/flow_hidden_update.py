"""Shape-parametric residual transition between action-token hidden states."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


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
        r = config.rank
        self.hidden = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, r))
        self.action = nn.Linear(3 * config.action_dim, r)
        self.condition = nn.Linear(config.condition_dim, r)
        self.state = nn.Linear(config.state_dim, r)
        self.time = nn.Sequential(nn.Linear(4, r), nn.SiLU(), nn.Linear(r, r))
        self.age = nn.Embedding(10, r)
        self.token = nn.Embedding(config.horizon, r)
        self.mix = nn.Sequential(nn.LayerNorm(r), nn.Linear(r, 2 * r), nn.SiLU(), nn.Linear(2 * r, r))
        self.up = nn.Linear(r, config.hidden_dim)
        self.gate = nn.Parameter(torch.tensor(config.gate_bias))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def descriptor(self):
        return asdict(self.config)

    def forward(self, hidden, previous_x, current_x, condition, state, previous_time, current_time, age):
        b, h, _ = hidden.shape
        if h != self.config.horizon or not 1 <= age < 10:
            raise ValueError("invalid generation horizon or anchor age")
        times = torch.stack((previous_time, current_time, current_time - previous_time,
                             previous_time * current_time)).to(hidden.device).float()
        features = self.hidden(hidden.float())
        features = features + self.action(torch.cat((previous_x, current_x, current_x - previous_x), -1).float())
        features = features + self.condition(condition.float())[:, None] + self.state(state.float())[:, None]
        features = features + self.time(times)[None, None] + self.age.weight[age][None, None]
        features = features + self.token.weight[None, :h]
        return hidden.float() + self.gate.sigmoid() * self.up(self.mix(features))
