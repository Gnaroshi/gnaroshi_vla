"""Matched 16K-parameter cross-token and tokenwise Condition updates."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .native_simvla_v0 import NativeV0UpdateOutput, TokenSharedConditionUpdater


class TokenInteraction(nn.Module):
    """Map current-query [B,T,64] features to a masked residual of equal shape."""
    def __init__(self, kind):
        super().__init__()
        if kind not in ('attention', 'mlp'):
            raise ValueError(kind)
        self.kind = kind
        self.norm = nn.LayerNorm(64, elementwise_affine=False)
        if kind == 'attention':
            self.query = nn.Linear(64, 64, bias=False)
            self.key = nn.Linear(64, 64, bias=False)
            self.value = nn.Linear(64, 64, bias=False)
            self.output = nn.Linear(64, 64, bias=False)
        else:
            self.input = nn.Linear(64, 128, bias=False)
            self.output = nn.Linear(128, 64, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden, valid):
        x = self.norm(hidden)
        if self.kind == 'attention':
            batch, tokens, _ = x.shape
            q, k, v = (layer(x).reshape(batch, tokens, 2, 32).transpose(1, 2)
                       for layer in (self.query, self.key, self.value))
            # Explicit math attention keeps deterministic backward and permits
            # masked all-padding examples without NaNs in either runtime.
            scores = (q @ k.transpose(-1, -2)) / (32 ** .5)
            scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
            weights = scores.softmax(-1) * valid[:, None, None, :]
            value = (weights @ v).transpose(1, 2).reshape(batch, tokens, 64)
        else:
            value = F.gelu(self.input(x))
        return self.output(value) * valid.unsqueeze(-1)


class InteractingUpdater(TokenSharedConditionUpdater):
    """Preserve the parent update, inserting a residual before its GELU."""
    def __init__(self, parent, kind, seed=7):
        nn.Module.__init__(self)
        if parent.rank_dim != 64:
            raise ValueError('Expected the existing rank64 updater')
        for name in ('condition_dim', 'delta_dim', 'rank_dim', 'max_tokens', 'num_token_groups', 'max_age'):
            setattr(self, name, getattr(parent, name))
        for name in ('norm', 'down', 'delta_projection', 'token_embedding', 'group_embedding',
                     'age_embedding', 'up', 'gate_head'):
            setattr(self, name, copy.deepcopy(getattr(parent, name)))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.mixer = TokenInteraction(kind)
        self.mixer.to(parent.down.weight)

    def forward(self, previous_condition, delta_feature, *, valid_mask, group_ids, age, token_feature=None):
        if previous_condition.ndim != 3 or previous_condition.shape[-1] != self.condition_dim:
            raise ValueError('Condition dimensions changed')
        batch_size, token_count, _ = previous_condition.shape
        if token_count > self.max_tokens or delta_feature.shape != (batch_size, self.delta_dim):
            raise ValueError('Token/delta dimensions changed')
        valid_mask = valid_mask.to(device=previous_condition.device, dtype=torch.bool)
        group_ids = group_ids.to(device=previous_condition.device, dtype=torch.long)
        if valid_mask.shape != (batch_size, token_count) or group_ids.shape != valid_mask.shape:
            raise ValueError('Invalid token layout')
        if bool((group_ids < 0).any()) or bool((group_ids >= self.num_token_groups).any()):
            raise ValueError('Invalid group IDs')
        age_tensor = torch.as_tensor(age, device=previous_condition.device, dtype=torch.long)
        if age_tensor.ndim == 0:
            age_tensor = age_tensor.expand(batch_size)
        if age_tensor.shape != (batch_size,) or bool((age_tensor < 1).any()) or bool((age_tensor > self.max_age).any()):
            raise ValueError('Invalid query age')
        token_ids = torch.arange(token_count, device=previous_condition.device)
        hidden = self.down(self.norm(previous_condition))
        hidden = hidden + self.delta_projection(delta_feature).unsqueeze(1)
        hidden = hidden + self.token_embedding(token_ids).unsqueeze(0)
        hidden = hidden + self.group_embedding(group_ids)
        hidden = hidden + self.age_embedding(age_tensor).unsqueeze(1)
        if token_feature is not None:
            if token_feature.shape != hidden.shape:
                raise ValueError('Token observation features must match hidden shape')
            hidden = hidden + token_feature
        hidden = hidden + self.mixer(hidden, valid_mask)
        hidden = F.gelu(hidden)
        residual = self.up(hidden)
        gate = torch.sigmoid(self.gate_head(hidden))
        candidate = previous_condition + gate * residual
        return NativeV0UpdateOutput(
            condition=torch.where(valid_mask.unsqueeze(-1), candidate, previous_condition),
            residual=torch.where(valid_mask.unsqueeze(-1), residual, torch.zeros_like(residual)),
            gate=torch.where(valid_mask.unsqueeze(-1), gate, torch.zeros_like(gate)))


def install(model, kind, *, warm, seed=7):
    parent = model.late.condition_updater
    if isinstance(parent, InteractingUpdater):
        raise ValueError('Mixer already installed')
    model.late.condition_updater = InteractingUpdater(parent, kind, seed)
    model.requires_grad_(False)
    if not warm:
        model.late.requires_grad_(True)
        model.visual.requires_grad_(True)
    model.late.condition_updater.mixer.requires_grad_(True)
    # Reuse the established training audit for a frozen early updater.
    model.short_training_arm = 'protected_short'
    model.interaction_kind, model.interaction_warm = kind, warm
    return model


def gradient_blocks(model):
    mixer = model.late.condition_updater.mixer
    result = {'mixer.' + name: module for name, module in mixer.named_children()
              if any(p.requires_grad for p in module.parameters())}
    if not model.interaction_warm:
        from .short_visual_training import gradient_blocks as original_blocks
        result.update(original_blocks(model))
    return result


def gradient_coverage(model):
    result = {}
    for name, module in gradient_blocks(model).items():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('Nonfinite gradient: ' + name)
        result[name] = any(bool(g.count_nonzero()) for g in gradients)
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen weights received gradients')
    return result
