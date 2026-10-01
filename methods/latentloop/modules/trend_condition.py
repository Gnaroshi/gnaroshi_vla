"""Anchor-conditioned trend plus causal, absolute residual correction."""
from dataclasses import dataclass

import torch
from torch import nn

from .native_simvla_v0 import NativeV0ObservationPair

ARMS = ("direct_anchor", "trend_residual", "trend_forecast", "trend_only")


class TrendHead(nn.Module):
    def __init__(self, dim=960, rank=64, proprio_dim=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.proprio = nn.Linear(proprio_dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, anchor, proprio):
        h = self.down(self.norm(anchor)) + self.proprio(proprio.float()).unsqueeze(1)
        return self.up(torch.nn.functional.gelu(h))


@dataclass
class TrendContext:
    anchor: torch.Tensor
    images: torch.Tensor
    proprio: torch.Tensor
    valid: torch.Tensor
    groups: torch.Tensor
    trend: torch.Tensor
    forecast: torch.Tensor | None


class TrendCondition(nn.Module):
    """All non-refresh predictions reference the immutable original anchor.

    Future observations/teacher conditions are absent from the inference API.
    Forecast batches the three residual ages into one residual-head call.
    """
    def __init__(self, parent, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.rank_dim = parent.rank_dim
        self.condition_dim = parent.condition_dim
        self.delta_dim = parent.delta_dim
        observed = arm in ("direct_anchor", "trend_residual")
        self.delta_encoder = parent.delta_encoder if observed else None
        self.condition_updater = parent.condition_updater if arm != "trend_only" else None
        self.trend_head = None if arm == "direct_anchor" else TrendHead(
            parent.condition_dim, parent.rank_dim, parent.proprio_dim)
        if self.trend_head is not None:
            self.trend_head.norm.load_state_dict(parent.condition_updater.norm.state_dict())
            self.trend_head.down.load_state_dict(parent.condition_updater.down.state_dict())

    def prepare(self, anchor, images, proprio, valid_mask, group_ids):
        if anchor.ndim != 3 or proprio.ndim != 2:
            raise ValueError("Expected batched condition and proprioception")
        trend = torch.zeros_like(anchor) if self.trend_head is None else self.trend_head(anchor, proprio)
        trend = torch.where(valid_mask.unsqueeze(-1), trend, torch.zeros_like(trend))
        forecast = None
        if self.arm == "trend_forecast":
            batch, tokens, dim = anchor.shape
            repeat = lambda x: x[:, None].expand(batch, 3, *x.shape[1:]).reshape(batch * 3, *x.shape[1:])
            copies = repeat(anchor)
            update = self.condition_updater(copies, anchor.new_zeros(batch * 3, self.delta_dim),
                valid_mask=repeat(valid_mask), group_ids=repeat(group_ids),
                age=torch.tensor([1, 2, 3], device=anchor.device).repeat(batch))
            forecast = (update.condition - copies).reshape(batch, 3, tokens, dim)
        return TrendContext(anchor, images, proprio, valid_mask, group_ids, trend, forecast)

    def predict(self, context, age, images=None, proprio=None):
        if age not in (1, 2, 3):
            raise ValueError("This pilot is trained only for K_C<=4")
        if self.arm == "trend_only":
            residual = torch.zeros_like(context.anchor)
        elif self.arm == "trend_forecast":
            residual = context.forecast[:, age - 1]
        else:
            if images is None or proprio is None:
                raise ValueError("Observed arms require the current query observation")
            code = self.delta_encoder(NativeV0ObservationPair(
                context.images, images, context.proprio, proprio))
            update = self.condition_updater(context.anchor, code,
                valid_mask=context.valid, group_ids=context.groups, age=age)
            residual = update.condition - context.anchor
        condition = context.anchor + age * context.trend + residual
        return condition, residual

    def sequence(self, sequence, age):
        context = self.prepare(sequence["anchor_condition"], sequence["image_sequence"][:, 0],
            sequence["proprio_sequence"][:, 0], sequence["valid_mask"].bool(), sequence["group_ids"])
        condition, residual = self.predict(context, age,
            sequence["image_sequence"][:, age], sequence["proprio_sequence"][:, age])
        return condition, context.trend, residual


def teacher_decomposition(anchor, teacher_conditions, age):
    if teacher_conditions.shape[1] != 3 or age not in (1, 2, 3):
        raise ValueError("Teacher window must contain ages 1,2,3")
    trend = (teacher_conditions[:, 2] - anchor) / 3.0
    residual = teacher_conditions[:, age - 1] - anchor - age * trend
    return trend.detach(), residual.detach()


def scaled_mse(predicted, target, anchor, valid):
    # A common, fixed teacher scale retains the magnitude of predicted deltas.
    scale = anchor.detach().float().var(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (((predicted.float() - target.detach().float()).square() / scale)[valid]).mean()


def decomposition_loss(model, sequence, age, condition, trend, residual):
    anchor, valid = sequence["anchor_condition"], sequence["valid_mask"].bool()
    if model.arm == "direct_anchor":
        return scaled_mse(condition, sequence["teacher_conditions"][:, age - 1], anchor, valid)
    target_b, target_r = teacher_decomposition(anchor, sequence["teacher_conditions"], age)
    trend_loss = scaled_mse(3 * trend, 3 * target_b, anchor, valid)
    if model.arm == "trend_only":
        return trend_loss
    return (trend_loss + scaled_mse(residual, target_r, anchor, valid)) / 2
