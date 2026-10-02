"""Observation-dependent progress along a frozen trend, with orthogonal correction."""
import torch
from torch import nn
from torch.nn import functional as F

from .native_simvla_v0 import NativeV0ObservationPair, _as_ordered_views, _float_chw
from .trend_condition import TrendCondition

ARMS = ('progress_only', 'progress_residual', 'progress_spatial')


def projection_coefficient(value, direction, anchor, valid):
    # Use exactly the metric of scaled_mse, excluding invalid tokens.
    scale = anchor.detach().float().var(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    weight = valid.unsqueeze(-1) / scale
    numerator = (value.float() * direction.float() * weight).sum(dim=(1, 2))
    denominator = (direction.float().square() * weight).sum(dim=(1, 2))
    safe = denominator.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.where(denominator > 0, numerator / safe, torch.zeros_like(numerator))[:, None, None]


def orthogonal_component(value, direction, anchor, valid):
    result = value - projection_coefficient(value, direction, anchor, valid) * direction
    return torch.where(valid.unsqueeze(-1), result, torch.zeros_like(result))


class ObservedProgressCondition(TrendCondition):
    def __init__(self, parent, arm, max_age=7):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(parent, 'frozen_trend_residual', max_age=max_age)
        self.arm = arm
        self.progress_head = nn.Sequential(nn.Linear(self.delta_dim + self.rank_dim + 1, 64),
            nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.progress_head[-1].weight)
        nn.init.zeros_(self.progress_head[-1].bias)
        if arm == 'progress_only':
            self.condition_updater = None
        if arm == 'progress_spatial':
            self.spatial_key = nn.Linear(192, self.rank_dim)
            self.spatial_value = nn.Linear(192, self.rank_dim)
            self.spatial_position = nn.Parameter(torch.empty(32, self.rank_dim))
            nn.init.normal_(self.spatial_position, std=.02)

    def initialize_frozen_trend(self, trend_state):
        self.trend_head.load_state_dict(trend_state, strict=True)
        self.trend_head.requires_grad_(False)
        if self.condition_updater is not None:
            nn.init.zeros_(self.condition_updater.up.weight)
            nn.init.zeros_(self.condition_updater.up.bias)
            nn.init.zeros_(self.condition_updater.gate_head.weight)
            nn.init.constant_(self.condition_updater.gate_head.bias, -4.0)

    def spatial_encode(self, pair, anchor):
        encoder = self.delta_encoder
        features, tokens = [], []
        for old, new in zip(_as_ordered_views(pair.previous_images), _as_ordered_views(pair.current_images)):
            old, new = _float_chw(old), _float_chw(new)
            old = F.interpolate(old, size=(64, 64), mode='bilinear', align_corners=False)
            new = F.interpolate(new, size=(64, 64), mode='bilinear', align_corners=False)
            h = torch.cat((old, new, new-old), dim=1)
            for layer in list(encoder.image_encoder)[:12]:
                h = layer(h)
            tokens.append(h.flatten(2).transpose(1, 2))
            for layer in list(encoder.image_encoder)[12:]:
                h = layer(h)
            features.append(h)
        old, new = pair.previous_proprio.float(), pair.current_proprio.float()
        proprio = encoder.proprio_encoder(torch.cat((old, new, new-old), dim=-1))
        code = encoder.fusion(torch.cat((*features, proprio), dim=-1))
        visual = torch.cat(tokens, dim=1)
        if visual.shape[1] != 32:
            raise ValueError('Spatial encoder requires two ordered 64x64 views')
        query = self.trend_head.down(self.trend_head.norm(anchor))
        key = self.spatial_key(visual) + self.spatial_position
        attention = torch.softmax(torch.bmm(query, key.transpose(1, 2)) / self.rank_dim**.5, dim=-1)
        local = torch.bmm(attention, self.spatial_value(visual))
        return code, local

    def predict(self, context, age, images=None, proprio=None):
        if not 1 <= age <= self.max_age or images is None or proprio is None:
            raise ValueError('Current observations and a trained condition age are required')
        pair = NativeV0ObservationPair(context.images, images, context.proprio, proprio)
        if self.arm == 'progress_spatial':
            code, local = self.spatial_encode(pair, context.anchor)
        else:
            code, local = self.delta_encoder(pair), None
        token_features = self.trend_head.down(self.trend_head.norm(context.anchor))
        mask = context.valid.unsqueeze(-1)
        summary = (token_features * mask).sum(1) / mask.sum(1).clamp_min(1)
        age_feature = code.new_full((code.shape[0], 1), age / self.max_age)
        alpha = age + self.progress_head(torch.cat((code, summary, age_feature), -1)).unsqueeze(1)
        base = context.anchor + alpha * context.trend
        residual = torch.zeros_like(base)
        if self.condition_updater is not None:
            update = self.condition_updater(base, code, valid_mask=context.valid,
                group_ids=context.groups, age=age, token_feature=local)
            residual = orthogonal_component(update.condition-base, context.trend, context.anchor, context.valid)
        condition = torch.where(mask, base + residual, context.anchor)
        self.last_alpha = alpha.detach()
        return condition, residual


def build_model(parent, arm, max_age=7):
    if arm in ARMS:
        return ObservedProgressCondition(parent, arm, max_age=max_age)
    return TrendCondition(parent, arm, max_age=max_age)
