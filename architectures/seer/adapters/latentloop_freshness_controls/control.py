"""Explicit controls for feature freshness and gate scale, not new trained models."""
from dataclasses import dataclass

import torch

ROWS = ('cached_feature', 'gate_cached_residual_zero', 'gate_fixed_residual_zero')


@dataclass
class RefreshFeature:
    value: object = None
    step: object = None

    def commit(self, value, step):
        self.value = value.detach().clone()
        self.step = int(step)

    def read(self, timestep):
        if self.value is None or self.step is None:
            raise RuntimeError('Feature control used before refresh initialization')
        if int(timestep) - self.step not in (1, 2, 3):
            raise RuntimeError('Feature control must use the current K4 refresh segment')
        return self.value


def controlled_update(model, z, cached_feature, age, variant, fixed_gates=None, apply=None):
    if variant not in ROWS or float(age) not in (1., 2., 3.):
        raise ValueError(f'Unknown control or K4 age: {variant}, {age}')
    node = model.lrnode_dynamics
    if node.use_post_layernorm:
        raise ValueError('These controls require the retained post-LN=0 checkpoint')
    apply = apply or model.lrnode_apply_dynamics
    if variant == 'cached_feature':
        return apply(z_prev=z, u_delta=cached_feature, dt=1., age=age)
    if variant == 'gate_cached_residual_zero':
        apply(z_prev=z, u_delta=cached_feature, dt=1., age=age)
        gate = node.last_gate.clone()
    else:
        if fixed_gates is None:
            raise ValueError('Independent calibration must finish before fixed-gate evaluation')
        scalar = float(fixed_gates[str(int(age))])
        if not 0 < scalar < 1:
            raise ValueError(f'Invalid calibrated gate: {scalar}')
        gate = torch.full_like(z[..., :1], scalar)
    apply(z_prev=z, u_delta=torch.zeros_like(cached_feature), dt=1., age=age)
    residual = node.last_dz.clone()
    delta = gate * 1. * residual
    node.last_gate, node.last_dz, node.last_update = gate, residual, delta
    return z + delta


def calibration_means(episodes):
    """Equal weight per episode, never select a gate using success labels."""
    if not episodes:
        raise ValueError('No calibration episodes')
    result = {}
    for age in (1, 2, 3):
        values = [float(ep['gate_mean_by_age'][str(age)]) for ep in episodes]
        if any(not 0 < value < 1 for value in values):
            raise ValueError('Calibration requires finite sigmoid outputs at every age')
        result[str(age)] = sum(values) / len(values)
    return result
