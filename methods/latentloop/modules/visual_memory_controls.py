"""Refresh-feature reuse and input controls for retained visual memory."""
import math

import torch
from torch.nn import functional as F

from .native_simvla_v0 import NativeV0ObservationPair
from .visual_memory import VisualMemoryCondition, MemoryReference


class ControlledReference(MemoryReference):
    def __init__(self, condition_dim, rank, spatial, control):
        super().__init__(condition_dim, rank, spatial)
        self.control = control

    def forward(self, query, features, valid, reference_images, current_images):
        if self.control == 'mean_vision':
            b, _, c = features.shape
            features = features.reshape(b, 2, 36, c).mean(2, keepdim=True).expand(-1, -1, 36, -1).reshape(b, 72, c)
        if self.control != 'permuted_grid':
            return super().forward(query, features, valid, reference_images, current_images)
        local = self.spatial(reference_images, current_images)
        local = local.reshape(local.shape[0], 2, 36, -1).flip(2).reshape_as(local)
        positions = self.position(torch.arange(72, device=features.device))[None]
        source = F.layer_norm(features, (features.shape[-1],))
        key = self.key(source) + positions + self.spatial_key(local)
        value = self.value(source) + positions + self.spatial_value(local)
        weights = (self.query(query) @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])).softmax(-1)
        return self.output(weights @ value) * valid.unsqueeze(-1)


class ReferenceMemoryCondition(VisualMemoryCondition):
    def __init__(self, trained_short, spatial=False, reference_age=0, control='none'):
        super().__init__(trained_short, spatial)
        if reference_age not in (0, 4):
            raise ValueError('reference_age must be0 or4')
        if control not in ('none', 'frozen_rgb', 'permuted_grid', 'mean_vision'):
            raise ValueError(control)
        if control == 'permuted_grid' and not spatial:
            raise ValueError('Spatial grid required')
        self.reference_age, self.control = reference_age, control
        original = self.visual
        self.visual = ControlledReference(trained_short.condition_dim, trained_short.rank_dim, spatial, control)
        self.visual.load_state_dict(original.state_dict(), strict=True)

    def prepare(self, anchor, images, proprio, valid, groups, interval):
        context = super().prepare(anchor, images, proprio, valid, groups, interval)
        if self.reference_age == 0:
            if self.feature_provider is None:
                raise RuntimeError('Original refresh vision capture required')
            context.visual_memory = self.feature_provider(0, images)
            context.visual_images = images
        return context

    def uses_vision(self, age):
        return self.reference_age == 4 and age == 4

    def predict(self, context, age, images, proprio):
        if age < 4:
            return self.early.predict(context, age, images, proprio)
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Queries must be consecutive')
        if self.uses_vision(age):
            context.visual_memory = self.feature_provider(age, images)
            context.visual_images = images
        previous_rgb, current_rgb = context.previous_images, images
        if self.control == 'frozen_rgb':
            previous_rgb = current_rgb = context.visual_images
        code = self.late.delta_encoder(NativeV0ObservationPair(
            previous_rgb, current_rgb, context.previous_proprio, proprio))
        updater = self.late.condition_updater
        positions = updater.token_embedding(torch.arange(context.previous.shape[1], device=images.device))[None]
        query = updater.down(updater.norm(context.previous)) + positions + updater.group_embedding(context.groups)
        query = query + updater.delta_projection(code)[:, None]
        feature = self.visual(query, context.visual_memory, context.valid, context.visual_images, current_rgb)
        prediction = updater(context.previous, code, valid_mask=context.valid, group_ids=context.groups,
            age=age, token_feature=feature)
        candidate = context.previous + prediction.residual
        output = context.previous + prediction.gate * (candidate - context.previous)
        diagnostics = dict(base=output, carried=output, addition=output-context.previous,
            gate=prediction.gate, candidate=candidate)
        context.previous = output
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, diagnostics
