"""Current frozen vision features for the late part of a Condition interval."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .age_specialized_condition import AgeSpecializedCondition
from .native_simvla_v0 import NativeV0ObservationPair


class VisualReference(nn.Module):
    def __init__(self, condition_dim=960, rank=64, max_visual_tokens=128):
        super().__init__()
        self.key = nn.Linear(condition_dim, rank)
        self.value = nn.Linear(condition_dim, rank)
        self.query = nn.Linear(rank, rank)
        self.position = nn.Embedding(max_visual_tokens, rank)
        self.output = nn.Linear(rank, rank)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, features, valid):
        positions = self.position(torch.arange(features.shape[1], device=features.device))[None]
        features = F.layer_norm(features, (features.shape[-1],))
        key = self.key(features) + positions
        value = self.value(features) + positions
        weights = (self.query(query) @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])).softmax(-1)
        return self.output(weights @ value) * valid.unsqueeze(-1)


class VisualMidpointCondition(AgeSpecializedCondition):
    def __init__(self, trained_short, arm):
        if arm not in ('midpoint', 'late_every'):
            raise ValueError(arm)
        super().__init__(trained_short, protected=True)
        self.arm = arm
        self.visual = VisualReference(trained_short.condition_dim, trained_short.rank_dim)
        self.feature_provider = None

    def uses_vision(self, age):
        return age == 4 or (self.arm == 'late_every' and age > 4)

    def predict(self, context, age, images, proprio):
        if age < 4:
            return self.early.predict(context, age, images, proprio)
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Queries must be consecutive')
        code = self.late.delta_encoder(NativeV0ObservationPair(
            context.previous_images, images, context.previous_proprio, proprio))
        updater = self.late.condition_updater
        token_feature = None
        if self.uses_vision(age):
            if self.feature_provider is None:
                raise RuntimeError('Current vision feature provider is required')
            features = self.feature_provider(age, images)
            positions = updater.token_embedding(torch.arange(context.previous.shape[1], device=images.device))[None]
            query = updater.down(updater.norm(context.previous)) + positions + updater.group_embedding(context.groups)
            query = query + updater.delta_projection(code)[:, None]
            token_feature = self.visual(query, features, context.valid)
        prediction = updater(context.previous, code, valid_mask=context.valid,
            group_ids=context.groups, age=age, token_feature=token_feature)
        candidate = context.previous + prediction.residual
        output = context.previous + prediction.gate * (candidate - context.previous)
        diagnostics = dict(base=output, carried=output, addition=output-context.previous,
            gate=prediction.gate, candidate=candidate)
        context.previous = output
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, diagnostics
