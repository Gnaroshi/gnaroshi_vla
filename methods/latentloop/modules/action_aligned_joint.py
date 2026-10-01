"""Matched parameter-factorization controls for executed-action distillation."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ARMS = ("condition_latent", "condition_action", "generation_action", "joint_action")


def trainable_groups(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return arm != "generation_action", arm in ("generation_action", "joint_action")


def query_schedule(step):
    if step < 1:
        raise ValueError(step)
    phase = (step - 1) % 4
    return ((step - 1) // 4 % 3 + 1, True) if phase == 3 else (phase + 1, False)


def condition_loss(condition, teacher, valid_mask):
    left, right = condition[valid_mask].float(), teacher[valid_mask].detach().float()
    return F.mse_loss(F.layer_norm(left, (left.shape[-1],)),
        F.layer_norm(right, (right.shape[-1],)))


def action_loss(prediction, target):
    # R=5 is the executed prefix. No thresholded gripper or hand-tuned weights.
    return F.l1_loss(prediction[:, :5].float(), target[:, :5].detach().float())


def differentiable_rollout(loop, step, condition, proprio, noise, *, recompute=True):
    def full(x, tau):
        # Frozen parameters do not imply detached inputs: C and earlier actions
        # must receive derivatives through all three full transformer calls.
        if recompute and torch.is_grad_enabled() and (condition.requires_grad or x.requires_grad):
            return checkpoint(step, condition, x, proprio, tau, use_reentrant=False)
        return step(condition, x, proprio, tau)
    return loop(noise, full_step=full, full_step_indices=(0, 4, 8),
        proprio=proprio, condition=condition, condition_valid_mask=None,
        condition_change_code=condition.new_zeros(condition.shape[0], loop.updater.condition_code_dim)).final_noisy_action
