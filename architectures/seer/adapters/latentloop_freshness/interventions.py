"""Input interventions; the learned encoder and dynamics remain unchanged."""
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class Inputs:
    primary: torch.Tensor
    wrist: torch.Tensor
    state_history: torch.Tensor

    @property
    def q(self):
        # This is the actual upstream runtime contract, not the most recent q.
        return self.state_history[:, 0]

    @property
    def latest_q(self):
        return self.state_history[:, -1]


def feature(model, previous, current, variant='normal'):
    p, w, q0, q1 = current.primary, current.wrist, previous.q, current.q
    if variant in ('stale_visual', 'repeated_observation'):
        p, w = previous.primary, previous.wrist
    if variant in ('stale_proprio', 'repeated_observation'):
        q1 = q0
    if variant == 'current_proprio':
        q0, q1 = previous.latest_q, current.latest_q
    value = model.lrnode_encode_delta(
        key_image_primary=previous.primary, key_image_wrist=previous.wrist,
        cur_image_primary=p, cur_image_wrist=w, q_key=q0, q_cur=q1,
    )
    return torch.zeros_like(value) if variant == 'zero' else value


def update(model, z, d, age, variant='normal', dt=1.0, apply=None):
    apply = apply or model.lrnode_apply_dynamics
    node = model.lrnode_dynamics
    if node.use_post_layernorm:
        raise ValueError('This gate/residual protocol requires the trained post-LN=0 model')
    if variant == 'hold':
        node.last_gate = torch.zeros_like(z[..., :1])
        node.last_dz = torch.zeros_like(z)
        node.last_update = torch.zeros_like(z)
        return z.clone()
    if variant == 'zero':
        return apply(z_prev=z, u_delta=torch.zeros_like(d), dt=dt, age=age)
    observed = apply(z_prev=z, u_delta=d, dt=dt, age=age)
    if variant not in ('gate_observed_residual_zero', 'gate_zero_residual_observed'):
        return observed
    gate_observed, residual_observed = node.last_gate.clone(), node.last_dz.clone()
    apply(z_prev=z, u_delta=torch.zeros_like(d), dt=dt, age=age)
    if variant == 'gate_observed_residual_zero':
        gate, residual = gate_observed, node.last_dz.clone()
    else:
        gate, residual = node.last_gate.clone(), residual_observed
    # Use exactly the parent's multiplication order. dt is one in this campaign.
    delta = gate * dt * residual
    node.last_gate, node.last_dz, node.last_update = gate, residual, delta
    return z + delta


@contextmanager
def replaced_method(obj, name, replacement):
    previous = obj.__dict__.get(name)
    existed = name in obj.__dict__
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def tensor_metrics(z, target, actions, target_actions):
    zf, tf = z.float(), target.float()
    norm = tf.norm(dim=-1).clamp_min(1e-12)
    return {
        'latent_mse': (zf - tf).square().mean().item(),
        'latent_relative_l2': ((zf - tf).norm(dim=-1) / norm).mean().item(),
        'latent_cosine': torch.nn.functional.cosine_similarity(zf, tf, dim=-1).mean().item(),
        'arm_mse': (actions[..., :6].float() - target_actions[..., :6].float()).square().mean().item(),
        'gripper_probability_mae': (actions[..., 6:].float() - target_actions[..., 6:].float()).abs().mean().item(),
        'gripper_disagreement': ((actions[..., 6:] > .5) != (target_actions[..., 6:] > .5)).float().mean().item(),
    }
