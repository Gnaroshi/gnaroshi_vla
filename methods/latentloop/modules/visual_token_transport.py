"""Current-image correspondence for a fixed, trained visual Condition reader."""
import copy
import math

import torch
from torch import nn

from .visual_memory import SpatialPairEncoder
from .visual_midpoint import VisualMidpointCondition, VisualReference
from .visual_readout_distillation import ReadoutDistillationCondition

ARMS = ('fresh_vision', 'transport', 'transport_residual')


class TokenTransport(nn.Module):
    """Move reference K/V within each camera before predicting new content.

    Each camera has 36 reference tokens. Current/reference RGB give 36x64
    descriptors; 32D queries/keys give a row-stochastic 36x36 correspondence.
    A sigmoid blend with identity keeps the initial model close to retention.
    Reference reader K/V are [B,72,64]. The output is a delta to that fixed
    reference; previous predictions never enter this visual transport.
    """
    def __init__(self, residual=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AvgPool2d(4, stride=4))
        self.query = nn.Linear(64, 32)
        self.key = nn.Linear(192, 32)
        self.blend = nn.Linear(64, 1)
        nn.init.zeros_(self.blend.weight)
        nn.init.constant_(self.blend.bias, -4.)
        self.register_buffer('identity', torch.eye(36))
        self.appearance = None
        if residual:
            self.appearance = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128))
            nn.init.zeros_(self.appearance[-1].weight)
            nn.init.zeros_(self.appearance[-1].bias)

    def descriptors(self, images):
        x = self.encoder(SpatialPairEncoder.rgb(images))
        return x.flatten(2).transpose(1, 2).reshape(images.shape[0], 2, 36, 64)

    def correspondence(self, reference_key, reference_value, reference_images, current_images):
        ref, cur = self.descriptors(reference_images), self.descriptors(current_images)
        rk, rv = (x.reshape(x.shape[0], 2, 36, 64) for x in (reference_key, reference_value))
        query = self.query(cur)
        key = self.key(torch.cat((ref, rk, rv), dim=-1))
        scores = query @ key.transpose(-1, -2) / math.sqrt(32.) + 4. * self.identity
        blend = self.blend(cur).sigmoid()
        weights = (1. - blend) * self.identity + blend * scores.softmax(-1)
        return weights, ref, cur

    @staticmethod
    def apply_transport(weights, tokens):
        batch = tokens.shape[0]
        return (weights @ tokens.reshape(batch, 2, 36, 64)).reshape(batch, 72, 64)

    def forward(self, reference_key, reference_value, reference_images, current_images, age):
        weights, ref, cur = self.correspondence(
            reference_key, reference_value, reference_images, current_images)
        moved_key = self.apply_transport(weights, reference_key)
        moved_value = self.apply_transport(weights, reference_value)
        delta_key, delta_value = moved_key - reference_key, moved_value - reference_value
        if self.appearance is not None:
            inputs = torch.cat((ref.flatten(1, 2), cur.flatten(1, 2), moved_key, moved_value), dim=-1)
            dk, dv = self.appearance(inputs).chunk(2, dim=-1)
            delta_key, delta_value = delta_key + dk, delta_value + dv
        return delta_key, delta_value


class FreshVisionCondition(VisualMidpointCondition):
    """Use current vision at ages4..7 with the exact completed warm reader."""
    def __init__(self, warm):
        super().__init__(warm.early, 'late_every')
        self.late = copy.deepcopy(warm.late)
        self.visual = VisualReference(warm.early.condition_dim, warm.early.rank_dim)
        self.visual.load_state_dict(warm.visual.state_dict(), strict=True)
        self.arm = 'fresh_vision'
        self.requires_grad_(False)

    def uses_vision(self, age):
        return 4 <= age <= 7


class TransportCondition(ReadoutDistillationCondition):
    def __init__(self, warm, arm):
        if arm not in ('transport', 'transport_residual'):
            raise ValueError(arm)
        # The completed warm model shares all original reader tensor names.
        parent = copy.deepcopy(warm)
        parent.arm = 'late_every'
        super().__init__(parent, 'distilled_readout')
        self.visual.predictor = TokenTransport(residual=arm == 'transport_residual')
        self.arm = arm
        self.requires_grad_(False)
        self.visual.predictor.requires_grad_(True)
