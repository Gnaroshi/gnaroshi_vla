"""Re-estimate Condition from the original anchor when current vision is read."""
import torch

from .native_simvla_v0 import NativeV0ObservationPair
from .visual_midpoint import VisualMidpointCondition


class VisualReanchorCondition(VisualMidpointCondition):
    def __init__(self, trained_short, schedule, contrast):
        super().__init__(trained_short, schedule)
        self.contrast = bool(contrast)

    def prepare(self, anchor, images, proprio, valid_mask, group_ids, interval=8):
        context = super().prepare(anchor, images, proprio, valid_mask, group_ids, interval)
        context.anchor_images = images
        context.anchor_proprio = proprio
        return context

    def predict(self, context, age, images, proprio):
        if not self.uses_vision(age):
            return super().predict(context, age, images, proprio)
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Queries must be consecutive')
        if self.feature_provider is None:
            raise RuntimeError('Vision feature provider is required')
        source = context.anchor
        code = self.late.delta_encoder(NativeV0ObservationPair(
            context.anchor_images, images, context.anchor_proprio, proprio))
        updater = self.late.condition_updater
        positions = updater.token_embedding(torch.arange(source.shape[1], device=source.device))[None]
        query = updater.down(updater.norm(source)) + positions + updater.group_embedding(context.groups)
        query = query + updater.delta_projection(code)[:, None]
        token_feature = self.visual(query, self.feature_provider(age, images), context.valid)
        if self.contrast:
            token_feature = token_feature - self.visual(
                query, self.feature_provider(0, images), context.valid)
        prediction = updater(source, code, valid_mask=context.valid, group_ids=context.groups,
            age=age, token_feature=token_feature)
        candidate = source + prediction.residual
        output = source + prediction.gate * (candidate - source)
        diagnostics = dict(base=output, carried=output, addition=output-context.previous,
            gate=prediction.gate, candidate=candidate)
        context.previous = output
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, diagnostics
