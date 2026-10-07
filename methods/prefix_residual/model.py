"""Predict changes in the expensive suffix, not the full visual representation."""
import math

import torch
from torch import nn


class PrefixResidual(nn.Module):
    def __init__(self, dim=960, rank=128):
        super().__init__()
        self.dim, self.rank = dim, rank
        self.anchor_norm = nn.LayerNorm(dim)
        self.prefix_norm = nn.LayerNorm(dim)
        self.anchor_down = nn.Linear(dim, rank)
        self.prefix_down = nn.Linear(dim, rank)
        self.delta_down = nn.Linear(dim, rank, bias=False)
        self.query = nn.Linear(rank, rank, bias=False)
        self.key = nn.Linear(rank, rank, bias=False)
        self.gate = nn.Linear(rank, rank)
        self.up = nn.Linear(2 * rank, dim, bias=False)
        # Start at measured prefix change plus the cached original suffix.
        nn.init.zeros_(self.up.weight)

    def forward(self, anchor, prefix_anchor, prefix_current, valid, learned=True):
        anchor, prefix_anchor, prefix_current = [x.float() for x in (anchor, prefix_anchor, prefix_current)]
        mask = valid.bool().unsqueeze(-1)
        delta = torch.where(mask, prefix_current - prefix_anchor, 0.)
        correction = torch.zeros_like(delta)
        if learned:
            scale = anchor.detach().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
            context = torch.tanh(self.anchor_down(self.anchor_norm(anchor))
                                 + self.prefix_down(self.prefix_norm(prefix_anchor)))
            value = self.delta_down(delta / scale)
            logits = self.query(context) @ self.key(context).transpose(-1, -2) / math.sqrt(self.rank)
            logits = logits.masked_fill(~valid[:, None, :], torch.finfo(logits.dtype).min)
            attention = logits.softmax(-1)
            local = value * self.gate(context).sigmoid()
            correction = self.up(torch.cat((local, attention @ value), -1)) * scale
        return torch.where(mask, anchor + delta + correction, anchor)
