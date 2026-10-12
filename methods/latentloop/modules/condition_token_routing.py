"""Observation-conditioned access to the original refresh Condition tokens."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .age_specialized_condition import AgeSpecializedCondition

ARMS = ('attention', 'tokenwise')


class TokenReference(nn.Module):
    def __init__(self, width, arm, heads=4):
        super().__init__()
        if arm not in ARMS or width % heads:
            raise ValueError('Invalid reference arm or head width')
        self.arm, self.heads, self.width = arm, heads, width
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, current, reference, valid):
        batch, tokens, width = current.shape
        def split(x):
            return x.reshape(batch, tokens, self.heads, width // self.heads).transpose(1, 2)
        query = split(self.query(F.layer_norm(current, (width,))))
        key = split(self.key(F.layer_norm(reference, (width,))))
        value = split(self.value(F.layer_norm(reference, (width,))))
        scale = math.sqrt(width // self.heads)
        if self.arm == 'attention':
            scores = (query @ key.transpose(-1, -2)) / scale
            scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
            weights = scores.softmax(-1) * valid[:, None, None, :]
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            result = weights @ value
        else:
            weights = torch.sigmoid((query * key).sum(-1, keepdim=True) / scale)
            result = weights * value
        result = result.transpose(1, 2).reshape(batch, tokens, width)
        return self.output(result) * valid.unsqueeze(-1)


class ReferencedUpdater(nn.Module):
    def __init__(self, original, arm):
        super().__init__()
        self.original = original
        self.reference = TokenReference(original.rank_dim, arm)

    def forward(self, previous, code, *, valid_mask, group_ids, age, anchor):
        base = self.original
        count = previous.shape[1]
        positions = base.token_embedding(torch.arange(count, device=previous.device))[None]
        positions = positions + base.group_embedding(group_ids)
        age = torch.as_tensor(age, dtype=torch.long, device=previous.device)
        if age.ndim == 0:
            age = age.expand(previous.shape[0])
        current = base.down(base.norm(previous)) + positions
        current = current + base.delta_projection(code)[:, None] + base.age_embedding(age)[:, None]
        reference = base.down(base.norm(anchor)) + positions
        feature = self.reference(current, reference, valid_mask.bool())
        return base(previous, code, valid_mask=valid_mask, group_ids=group_ids,
                    age=age, token_feature=feature)


class TokenRoutedCondition(AgeSpecializedCondition):
    def __init__(self, trained_short, arm):
        if trained_short.variant != 'recurrent':
            raise ValueError('Use the registered recurrent K4 parent')
        super().__init__(trained_short, protected=True)
        self.arm = arm
        self.late.condition_updater = ReferencedUpdater(self.late.condition_updater, arm)

    def predict(self, context, age, images, proprio):
        if age < 4:
            return self.early.predict(context, age, images, proprio)
        from .native_simvla_v0 import NativeV0ObservationPair
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Queries must be consecutive within the refresh interval')
        code = self.late.delta_encoder(NativeV0ObservationPair(
            context.previous_images, images, context.previous_proprio, proprio))
        prediction = self.late.condition_updater(context.previous, code,
            valid_mask=context.valid, group_ids=context.groups, age=age, anchor=context.anchor)
        # Match BoundedHistoryCondition(recurrent)'s arithmetic at initialization.
        candidate = context.previous + prediction.residual
        output = context.previous + prediction.gate * (candidate - context.previous)
        diagnostics = dict(base=output, carried=output, addition=output-context.previous,
                           gate=prediction.gate, candidate=candidate)
        context.previous = output
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, diagnostics
