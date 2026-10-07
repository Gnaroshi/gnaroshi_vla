"""Matched condition predictors differing only in next-query input routing."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .native_simvla_v0 import NativeV0ObservationPair, TokenSharedConditionUpdater

ARMS = ('carry_output', 'carry_base')


@dataclass
class SplitContext:
    anchor: torch.Tensor
    previous: torch.Tensor
    previous_images: torch.Tensor
    previous_proprio: torch.Tensor
    valid: torch.Tensor
    groups: torch.Tensor
    interval: int
    age: int = 0


class ConditionOutputSplit(nn.Module):
    def __init__(self, parent, arm, max_age=7):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.max_age = arm, max_age
        self.delta_encoder = parent.delta_encoder
        self.condition_updater = parent.condition_updater
        self.rank_dim = parent.rank_dim
        self.condition_dim = parent.condition_dim
        self.delta_dim = parent.delta_dim
        old = self.condition_updater.age_embedding
        extended = nn.Embedding(max_age + 1, old.embedding_dim).to(old.weight)
        with torch.no_grad():
            for age in range(max_age + 1):
                extended.weight[age].copy_(old.weight[min(age, old.num_embeddings - 1)])
        self.condition_updater.age_embedding = extended
        self.condition_updater.max_age = max_age
        # An independent input-dependent residual, exactly zero at initialization.
        u = self.condition_updater
        self.action_condition_updater = TokenSharedConditionUpdater(
            condition_dim=u.condition_dim, delta_dim=u.delta_dim, rank_dim=u.rank_dim,
            max_tokens=u.max_tokens, num_token_groups=u.num_token_groups, max_age=max_age).to(old.weight)
        self.action_condition_updater.load_state_dict(u.state_dict(), strict=True)
        nn.init.zeros_(self.action_condition_updater.up.weight)
        nn.init.zeros_(self.action_condition_updater.up.bias)

    def prepare(self, anchor, images, proprio, valid_mask, group_ids, interval=8):
        if interval not in (4, 8) or interval - 1 > self.max_age:
            raise ValueError('Unsupported refresh interval')
        return SplitContext(anchor, anchor, images, proprio, valid_mask.bool(), group_ids, interval)

    def update(self, previous, code, *, valid_mask, group_ids, age):
        kwargs = dict(valid_mask=valid_mask, group_ids=group_ids, age=age)
        base = self.condition_updater(previous, code, **kwargs).condition
        # Current action loss trains the extra head only. In carry_output,
        # later condition loss can also reach it through the carried output.
        output = self.action_condition_updater(base.detach(), code.detach(), **kwargs).condition
        carried = output if self.arm == 'carry_output' else base
        return output, base, carried

    def predict(self, context, age, images, proprio):
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Queries must be consecutive within the refresh interval')
        code = self.delta_encoder(NativeV0ObservationPair(context.previous_images, images,
            context.previous_proprio, proprio))
        output, base, carried = self.update(context.previous, code,
            valid_mask=context.valid, group_ids=context.groups, age=age)
        context.previous = carried
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, dict(base=base, carried=carried, addition=output-base.detach())


def geometry(predicted, target, valid):
    """Per-token exact squared-distance decomposition, averaged over valid tokens."""
    x, y = predicted.detach().float()[valid], target.detach().float()[valid]
    if not x.numel() or x.shape != y.shape:
        raise ValueError('Empty or incompatible token comparison')
    dim = x.shape[-1]
    nx, ny = x.norm(dim=-1), y.norm(dim=-1)
    dot = (x*y).sum(-1)
    radial = (nx-ny).square()/dim
    angular = 2*(nx*ny-dot)/dim
    mse = (x-y).square().mean(-1)
    centered_x, centered_y = x-x.mean(-1, keepdim=True), y-y.mean(-1, keepdim=True)
    return dict(raw_mse=float(mse.mean()), cosine=float(F.cosine_similarity(x,y,dim=-1).mean()),
        norm_difference_mse=float(radial.mean()), weighted_direction_mse=float(angular.mean()),
        decomposition_max_abs=float((mse-radial-angular).abs().max()),
        predicted_norm_mean=float(nx.mean()), target_norm_mean=float(ny.mean()),
        token_mean_mse=float((x.mean(-1)-y.mean(-1)).square().mean()),
        centered_mse=float((centered_x-centered_y).square().mean()),
        layernorm_mse=float((F.layer_norm(x,(dim,))-F.layer_norm(y,(dim,))).square().mean()))
