"""Separate attention target quality from recurrent Condition query quality."""
import copy

import torch
from torch import nn

from .native_simvla_v0 import NativeV0ObservationPair
from .visual_readout_distillation import ReadoutDistillationCondition

ARMS = ("student_query_control", "original_query_target", "refresh_query_reader")


def make_query(updater, condition, code, groups):
    positions = updater.token_embedding(torch.arange(condition.shape[1], device=condition.device))[None]
    return (updater.down(updater.norm(condition)) + positions
            + updater.group_embedding(groups) + updater.delta_projection(code)[:, None])


class AttentionQueryCondition(ReadoutDistillationCondition):
    """Use the identical late updater and visual predictor for three arms.

    At query t, attention returns [B,T,64] information used in the update of
    Condition z_t [B,T,960]. Arms differ in the attention target/read query.
    Teacher previous Condition z^F_(t-1) is supplied only during training.
    The refresh-reader arm uses fixed z^F_r and refresh/current RGB for its
    attention query. Its Condition update still consumes and adds to z_(t-1).
    """
    def __init__(self, parent, teacher, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        nn.Module.__init__(self)
        self.arm, self.protected = arm, True
        self.early = copy.deepcopy(parent.early)
        self.late = copy.deepcopy(parent.late)
        self.visual = copy.deepcopy(parent.visual)
        # The parent's hook closes over the old model, so targets are explicit.
        self.visual._forward_hooks.clear()
        self.teacher_visual = copy.deepcopy(teacher.visual)
        self.teacher_late = copy.deepcopy(teacher.late)
        self.requires_grad_(False)
        self.late.requires_grad_(True)
        self.visual.predictor.requires_grad_(True)
        self.feature_provider = None
        self.distill_enabled = False
        self.teacher_previous = None
        self.attention_losses = []
        self.query_errors = []

    def prepare(self, anchor, images, proprio, valid_mask, group_ids, interval=8):
        context = self.early.prepare(anchor, images, proprio, valid_mask, group_ids, interval)
        context.refresh_images, context.refresh_proprio = images, proprio
        return context

    def predict(self, context, age, images, proprio):
        if age < 4:
            return self.early.predict(context, age, images, proprio)
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError("Queries must be consecutive")
        if age == 4:
            context.visual_memory = self.feature_provider(age, images)
            context.visual_images = images
        pair = NativeV0ObservationPair(context.previous_images, images, context.previous_proprio, proprio)
        code = self.late.delta_encoder(pair)
        updater = self.late.condition_updater
        query_condition, query_code = context.previous, code
        if self.arm == "refresh_query_reader":
            query_condition = context.anchor
            query_code = self.late.delta_encoder(NativeV0ObservationPair(
                context.refresh_images, images, context.refresh_proprio, proprio))
        query = make_query(updater, query_condition, query_code, context.groups)
        feature = self.visual(query, context.visual_memory, context.valid,
                              context.visual_images, images, age)
        if self.distill_enabled and age >= 5:
            with torch.no_grad():
                current_vision = self.feature_provider(age, images)
                if self.teacher_previous is None:
                    raise RuntimeError("Missing current training sequence's original previous Condition")
                original_query = make_query(self.teacher_late.condition_updater,
                    self.teacher_previous.detach(), self.teacher_late.delta_encoder(pair), context.groups)
                teacher_query = query.detach() if self.arm == "student_query_control" else original_query
                target = self.teacher_visual(teacher_query, current_vision, context.valid)
                same_query = self.teacher_visual(query.detach(), current_vision, context.valid)
                original_target = self.teacher_visual(original_query, current_vision, context.valid)
            mask = context.valid.unsqueeze(-1).expand_as(feature)
            scale = target.square().mean(-1, keepdim=True).clamp_min(1e-6)
            self.attention_losses.append(((feature-target).square()/scale).masked_select(mask).mean())
            self.query_errors.append(dict(age=age,
                same_vs_original=float((same_query-original_target).square().masked_select(mask).mean()),
                predicted_vs_original=float((feature.detach()-original_target).square().masked_select(mask).mean())))
        prediction = updater(context.previous, code, valid_mask=context.valid,
            group_ids=context.groups, age=age, token_feature=feature)
        candidate = context.previous + prediction.residual
        output = context.previous + prediction.gate * (candidate - context.previous)
        diagnostics = dict(base=output, carried=output, addition=output-context.previous,
                           gate=prediction.gate, candidate=candidate)
        context.previous = output
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, diagnostics
