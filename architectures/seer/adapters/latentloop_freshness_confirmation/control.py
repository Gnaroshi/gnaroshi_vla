"""One additional intervention: retain cached residual input, fix only the gate."""
import math

import torch

NEW_ROW = 'gate_fixed_residual_cached'
ROWS = ('full', 'normal', 'zero', 'gate_observed_residual_zero',
        'cached_feature', 'gate_fixed_residual_zero', NEW_ROW)
SEEDS = (42, 43, 44)


def fixed_gate_cached_update(model, z, feature, age, gates, apply=None):
    if float(age) not in (1., 2., 3.) or model.lrnode_dynamics.use_post_layernorm:
        raise ValueError('Retain K4 age=1,2,3 and post-LN=0')
    if gates is None:
        raise ValueError('Frozen calibration is required')
    scalar = float(gates[str(int(age))])
    if not math.isfinite(scalar) or not 0 < scalar < 1:
        raise ValueError('Invalid fixed gate')
    apply = apply or model.lrnode_apply_dynamics
    apply(z_prev=z, u_delta=feature, dt=1., age=age)
    node = model.lrnode_dynamics
    residual = node.last_dz.clone()
    gate = torch.full_like(z[..., :1], scalar)
    update = gate * 1. * residual
    node.last_gate, node.last_dz, node.last_update = gate, residual, update
    return z + update


def experiment_plan():
    # Six completed seed42 rows are references, not new work.
    return [(42, NEW_ROW)] + [(seed, row) for seed in SEEDS[1:] for row in ROWS]
