"""One frozen generation call followed by state-conditioned cheap refinements.

This experiment uses a coarse grid spanning the entire [1, 0] flow interval.
It is NOT the original ten-step, hidden-replacement Generation Loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


VARIANTS = ("independent_hidden", "global_code_hidden", "token_code_hidden", "token_code_action")


@dataclass
class RefinementContext:
    condition: Tensor
    valid_mask: Tensor
    proprio: Tensor
    global_code: Tensor
    token_code: Tensor
    updated: Tensor


class SharedRefiner(nn.Module):
    def __init__(self, variant: str, *, hidden_dim: int = 1024,
                 condition_dim: int = 960, token_code_dim: int = 65,
                 global_code_dim: int = 128, rank: int = 128, horizon: int = 10):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden = nn.Linear(hidden_dim, rank)
        self.action = nn.Linear(21, rank)
        self.condition = nn.Linear(condition_dim, rank)
        self.proprio = nn.Linear(8, rank)
        self.time = nn.Linear(4, rank)
        self.position = nn.Embedding(horizon, rank)
        if variant == "global_code_hidden":
            self.code = nn.Linear(global_code_dim, rank, bias=False)
        elif variant.startswith("token_code"):
            self.key = nn.Linear(token_code_dim, rank, bias=False)
            self.value = nn.Linear(token_code_dim, rank, bias=False)
        self.mix = nn.Sequential(nn.GELU(), nn.Linear(rank, 2 * rank), nn.GELU(), nn.Linear(2 * rank, rank))
        self.gate = nn.Linear(rank, 1)
        nn.init.constant_(self.gate.bias, -4.0)
        if variant == "token_code_action":
            # Similar *active* parameter budget to the rank->hidden decoder,
            # but a learned action-space head instead of the frozen decoder.
            self.output = nn.Sequential(nn.Linear(rank, 960), nn.GELU(), nn.Linear(960, 7))
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)
        else:
            self.output = nn.Linear(rank, hidden_dim)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, hidden: Tensor, previous_action: Tensor, current_action: Tensor,
                context: RefinementContext, *, tau: float, dt: float, index: int) -> Tensor:
        valid = context.valid_mask.to(context.condition.dtype).unsqueeze(-1)
        mean_c = (context.condition * valid).sum(1) / valid.sum(1).clamp_min(1)
        b, horizon, _ = current_action.shape
        time = hidden.new_tensor([tau, dt, index / 2.0, 1.0]).expand(b, -1)
        z = self.hidden(self.norm(hidden))
        z = z + self.action(torch.cat((previous_action, current_action, current_action - previous_action), -1))
        z = z + self.condition(mean_c)[:, None] + self.proprio(context.proprio)[:, None]
        z = z + self.time(time)[:, None] + self.position.weight[:horizon][None]
        available = context.updated.to(z.dtype)[:, None, None]
        if self.variant == "global_code_hidden":
            z = z + available * self.code(context.global_code)[:, None]
        elif self.variant.startswith("token_code"):
            scores = torch.matmul(z, self.key(context.token_code).transpose(1, 2)) / math.sqrt(z.shape[-1])
            scores = scores.masked_fill(~context.valid_mask[:, None].bool(), -torch.inf)
            readout = torch.matmul(scores.softmax(-1), self.value(context.token_code))
            z = z + available * readout
        z = self.mix(z)
        return torch.sigmoid(self.gate(z)) * self.output(z)


def refine_from_anchor(refiner: SharedRefiner, decoder: nn.Module, *,
                       anchor_hidden: Tensor, anchor_velocity: Tensor,
                       noise: Tensor, context: RefinementContext, cheap_steps: int) -> Tensor:
    if cheap_steps not in (1, 2):
        raise ValueError("cheap_steps must be 1 or 2")
    dt = -1.0 / (cheap_steps + 1)
    previous_x = noise
    x = noise + dt * anchor_velocity
    hidden = anchor_hidden
    velocity = anchor_velocity
    for index in range(1, cheap_steps + 1):
        delta = refiner(hidden, previous_x, x, context,
                        tau=1.0 + index * dt, dt=dt, index=index)
        if refiner.variant == "token_code_action":
            velocity = velocity + delta
        else:
            hidden = hidden + delta
            velocity = decoder(hidden)
        previous_x, x = x, x + dt * velocity
    return x
