"""Align the visual teacher, then distill its query-dependent attention output."""
import copy

import torch

from .visual_token_transport import FreshVisionCondition, TransportCondition


class AdaptedFreshCondition(FreshVisionCondition):
    def __init__(self, warm):
        super().__init__(warm)
        self.arm = 'adapt_fresh'
        self.late.requires_grad_(True)
        self.visual.requires_grad_(True)


class AttentionStudent(TransportCondition):
    """Predict age4 visual K/V but supervise what the current query reads.

    Teacher attention receives the identical detached query [B,T,64], current
    original vision [B,72,960], and fixed trained visual projections. Student
    attention uses predicted K/V from age4 vision and current RGB. Only the
    training driver enables teacher reads; deployment uses one vision call at4.
    """
    def __init__(self, teacher, joint=False):
        warm = copy.deepcopy(teacher)
        warm.arm = 'late_every'
        super().__init__(warm, 'transport_residual')
        self.arm = 'attention_joint' if joint else 'attention_distill'
        self.teacher_visual = copy.deepcopy(teacher.visual).requires_grad_(False)
        if joint:
            self.late.requires_grad_(True)
        self.distill_enabled = False
        self.attention_losses = []
        self._current_target = None
        self.visual.register_forward_hook(self._capture_attention)

    def _capture_attention(self, module, args, output):
        if not self.distill_enabled:
            return
        query, _, valid, _, _, age = args
        if age < 5:
            return
        with torch.no_grad():
            target = self.teacher_visual(query.detach(), self._current_target, valid)
        mask = valid.unsqueeze(-1).expand_as(output)
        scale = target.square().mean(-1, keepdim=True).clamp_min(1e-6)
        self.attention_losses.append(((output-target).square()/scale).masked_select(mask).mean())

    def predict(self, context, age, images, proprio):
        if self.distill_enabled and age >= 5:
            self._current_target = self.feature_provider(age, images)
        return super().predict(context, age, images, proprio)
