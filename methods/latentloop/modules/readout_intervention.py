"""Frozen-checkpoint interventions on visual content and attention selection."""
import torch

from .attention_query_contract import AttentionQueryCondition, make_query
from .native_simvla_v0 import NativeV0ObservationPair
from .short_visual_training import build_model
from .visual_midpoint import VisualMidpointCondition


MODES = ('control', 'current_content', 'original_selector', 'both')


def uses_original_selector(mode):
    return mode in ('original_selector', 'both')


def needs_previous_reference(mode, interval, age):
    return uses_original_selector(mode) and interval == 8 and 3 <= age <= 6


class ReadoutIntervention(AttentionQueryCondition):
    """Retain the loaded weights and recurrent residual; change readout inputs.

    original_previous is z^F_(t-1), supplied by the preceding observation's
    original backbone evaluation. Only make_query reads it. The late updater
    and residual addition always use the predicted previous Condition.
    """
    def __init__(self, native, mode):
        if mode not in MODES:
            raise ValueError(mode)
        parent = build_model(native, 'fresh_joint', seed=7)
        teacher = VisualMidpointCondition(parent.early, 'late_every')
        super().__init__(parent, teacher, 'original_query_target')
        self.intervention_mode = mode
        self.original_previous = None
        self.original_previous_age = None
        self.requires_grad_(False)

    def prepare(self, *args, **kwargs):
        self.original_previous = self.original_previous_age = None
        return super().prepare(*args, **kwargs)

    def uses_vision(self, age):
        return age == 4 or (self.intervention_mode in ('current_content', 'both') and 5 <= age <= 7)

    def predict(self, context, age, images, proprio):
        if age < 4:
            return self.early.predict(context, age, images, proprio)
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Queries must be consecutive')
        if age == 4:
            context.visual_memory = self.feature_provider(age, images)
            context.visual_images = images
        pair = NativeV0ObservationPair(context.previous_images, images, context.previous_proprio, proprio)
        code = self.late.delta_encoder(pair)
        updater = self.late.condition_updater
        query_condition = context.previous
        if uses_original_selector(self.intervention_mode):
            if self.original_previous is None or self.original_previous_age != age - 1:
                raise RuntimeError('Expected original Condition from the preceding query')
            if self.original_previous.shape != context.previous.shape:
                raise RuntimeError('Original/predicted Condition shapes differ')
            query_condition = self.original_previous
        query = make_query(updater, query_condition, code, context.groups)
        if self.intervention_mode in ('current_content', 'both') and age >= 5:
            current = self.feature_provider(age, images)
            # Age4 takes the frozen key/value projections without the predictor.
            feature = self.visual(query, current, context.valid, images, images, 4)
        else:
            feature = self.visual(query, context.visual_memory, context.valid,
                                  context.visual_images, images, age)
        prediction = updater(context.previous, code, valid_mask=context.valid,
            group_ids=context.groups, age=age, token_feature=feature)
        candidate = context.previous + prediction.residual
        output = context.previous + prediction.gate * (candidate - context.previous)
        diagnostics = dict(base=output, carried=output, addition=output-context.previous,
                           gate=prediction.gate, candidate=candidate)
        context.previous = output
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, diagnostics
