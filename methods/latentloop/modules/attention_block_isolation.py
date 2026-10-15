"""Keep the deployed attention computation fixed; isolate trainable blocks."""

from .attention_query_contract import AttentionQueryCondition

ARMS = ('visual_predictor_only', 'condition_updater_only')


def build_model(student, teacher, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    model = AttentionQueryCondition(student, teacher, 'student_query_control')
    model.requires_grad_(False)
    if arm == 'visual_predictor_only':
        model.visual.predictor.requires_grad_(True)
    else:
        model.late.requires_grad_(True)
    return model
