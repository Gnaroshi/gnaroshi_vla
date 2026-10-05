"""Causal recurrent prediction and an independently observed condition estimate."""
from dataclasses import dataclass

import torch
from torch import nn

from .native_simvla_v0 import NativeV0ObservationPair, TokenSharedConditionUpdater
from .trend_condition import TrendCondition

ARMS = ('recurrent', 'trend_recurrent', 'midpoint_recurrent', 'observed_recurrent')


@dataclass
class RecurrentContext:
    anchor: torch.Tensor
    images: torch.Tensor
    proprio: torch.Tensor
    valid: torch.Tensor
    groups: torch.Tensor
    trend: torch.Tensor
    previous: torch.Tensor
    previous_images: torch.Tensor
    previous_proprio: torch.Tensor
    interval: int
    age: int = 0


class ObservationGain(nn.Module):
    def __init__(self, dim, code_dim, rank):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.difference = nn.Linear(dim, rank)
        self.observation = nn.Linear(code_dim, rank)
        self.readout = nn.Linear(rank, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.constant_(self.readout.bias, -2.0)

    def forward(self, predicted, measured, code):
        difference = self.norm(measured) - self.norm(predicted)
        feature = self.difference(difference) + self.observation(code).unsqueeze(1)
        return torch.sigmoid(self.readout(torch.nn.functional.gelu(feature)))


class ObservationCorrection(TrendCondition):
    def __init__(self, parent, arm, max_age=7):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(parent, 'frozen_trend_residual', max_age=max_age)
        self.arm = arm
        if arm == 'recurrent':
            self.trend_head = None
        self.measurement = self.correction_gain = None
        if arm == 'observed_recurrent':
            u = self.condition_updater
            self.measurement = TokenSharedConditionUpdater(condition_dim=u.condition_dim,
                delta_dim=u.delta_dim, rank_dim=u.rank_dim, max_tokens=u.max_tokens,
                num_token_groups=u.num_token_groups, max_age=max_age)
            self.correction_gain = ObservationGain(u.condition_dim, u.delta_dim, u.rank_dim)

    def initialize_from_anchor_model(self, state):
        common = {k: v for k, v in state.items() if k in self.state_dict()}
        missing, unexpected = self.load_state_dict(common, strict=False)
        if unexpected or any(not k.startswith(('measurement.', 'correction_gain.')) for k in missing):
            raise RuntimeError((missing, unexpected))
        if self.measurement is not None:
            self.measurement.load_state_dict(self.condition_updater.state_dict(), strict=True)

    def prepare(self, anchor, images, proprio, valid_mask, group_ids, interval=8):
        if interval not in (4, 8) or interval - 1 > self.max_age:
            raise ValueError('Unsupported refresh interval')
        trend = torch.zeros_like(anchor) if self.trend_head is None else self.trend_head(anchor, proprio)
        trend = torch.where(valid_mask.unsqueeze(-1), trend, torch.zeros_like(trend))
        return RecurrentContext(anchor, images, proprio, valid_mask, group_ids, trend,
            anchor, images, proprio, interval)

    def observation_estimate(self, context, age, images, proprio):
        # This branch has no access to context.previous or its recurrent error.
        code = self.delta_encoder(NativeV0ObservationPair(
            context.images, images, context.proprio, proprio))
        base = context.anchor + age * context.trend
        measured = self.measurement(base, code, valid_mask=context.valid,
            group_ids=context.groups, age=age).condition
        return measured, code

    def predict(self, context, age, images, proprio):
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Recurrent queries must be consecutive after refresh')
        code = self.delta_encoder(NativeV0ObservationPair(
            context.previous_images, images, context.previous_proprio, proprio))
        base = context.previous + context.trend
        predicted = self.condition_updater(base, code, valid_mask=context.valid,
            group_ids=context.groups, age=age).condition
        condition = predicted
        diagnostics = {'predicted': predicted}
        if self.measurement is not None:
            measured, anchor_code = self.observation_estimate(context, age, images, proprio)
            gain = self.correction_gain(predicted, measured, anchor_code)
            condition = predicted + gain * (measured - predicted)
            diagnostics.update(measured=measured, gain=gain)
        condition = torch.where(context.valid.unsqueeze(-1), condition, context.anchor)
        context.previous = condition
        context.previous_images, context.previous_proprio = images, proprio
        context.age = age
        if self.arm == 'midpoint_recurrent' and age == context.interval // 2:
            # Re-estimate the slope for subsequent queries; this is a prediction,
            # not a new backbone evaluation or a ground-truth midpoint.
            trend = self.trend_head(condition, proprio)
            context.trend = torch.where(context.valid.unsqueeze(-1), trend, torch.zeros_like(trend))
        return condition, diagnostics


def expected_condition_counts(arm, queries, interval):
    full = (queries + interval - 1) // interval
    light = queries - full
    midpoint = sum(q % interval == interval // 2 for q in range(queries))
    return dict(full=full, light=light,
        trend=0 if arm == 'recurrent' else full + (midpoint if arm == 'midpoint_recurrent' else 0),
        observation=light * (2 if arm == 'observed_recurrent' else 1),
        measurement=light if arm == 'observed_recurrent' else 0)
