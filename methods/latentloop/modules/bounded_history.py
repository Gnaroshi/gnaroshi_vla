"""Condition updates with an explicitly bounded previous-prediction path."""
import torch
from torch import nn

from .condition_output_split import SplitContext
from .native_simvla_v0 import NativeV0ObservationPair

VARIANTS = ('bounded', 'recurrent')


class BoundedHistoryCondition(nn.Module):
    def __init__(self, parent, variant, max_age=7):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.arm, self.max_age = variant, 'carry_base', max_age
        self.delta_encoder, self.condition_updater = parent.delta_encoder, parent.condition_updater
        self.rank_dim, self.condition_dim, self.delta_dim = parent.rank_dim, parent.condition_dim, parent.delta_dim
        old = self.condition_updater.age_embedding
        extended = nn.Embedding(max_age + 1, old.embedding_dim).to(old.weight)
        with torch.no_grad():
            for age in range(max_age + 1):
                extended.weight[age].copy_(old.weight[min(age, old.num_embeddings - 1)])
        self.condition_updater.age_embedding = extended
        self.condition_updater.max_age = max_age

    def prepare(self, anchor, images, proprio, valid_mask, group_ids, interval=8):
        if type(interval) is not int or interval not in (2, 3, 4, 8) or interval - 1 > self.max_age:
            raise ValueError('Unsupported refresh interval')
        return SplitContext(anchor, anchor, images, proprio, valid_mask.bool(), group_ids, interval)

    def update(self, previous, anchor, code, *, valid_mask, group_ids, age):
        source = anchor if self.variant == 'bounded' else previous
        prediction = self.condition_updater(source, code, valid_mask=valid_mask, group_ids=group_ids, age=age)
        candidate = source + prediction.residual
        # In the bounded arm, candidate and gate are independent of previous.
        output = previous + prediction.gate * (candidate - previous)
        return output, dict(base=output, carried=output, addition=output-previous,
                            gate=prediction.gate, candidate=candidate)

    def predict(self, context, age, images, proprio):
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Queries must be consecutive within the refresh interval')
        code = self.delta_encoder(NativeV0ObservationPair(context.previous_images, images,
            context.previous_proprio, proprio))
        output, diagnostics = self.update(context.previous, context.anchor, code,
            valid_mask=context.valid, group_ids=context.groups, age=age)
        context.previous = output
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, diagnostics
