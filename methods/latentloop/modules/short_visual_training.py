"""Fresh/protected short training with the existing residual transport reader."""

import torch

from .bounded_history import BoundedHistoryCondition
from .native_simvla_v0 import NativeSimVLAV0
from .visual_midpoint import VisualMidpointCondition
from .visual_readout_distillation import ReadoutDistillationCondition
from .visual_token_transport import TokenTransport


ARMS = ('fresh_joint', 'protected_short')


class ShortVisualCondition(ReadoutDistillationCondition):
    """Keep joint_readout inference and allow an updater-free K1 context.

    Ages 1..3 use early; age 4 reads original vision once; ages 5..7 use
    TokenTransport deltas added to fixed age-4 key/value projections. Previous
    predicted conditions enter both the recurrent updater and residual addition.
    Previous predicted key/value tensors never enter the transport predictor.
    """

    def __init__(self, short, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(VisualMidpointCondition(short, 'late_every'), 'joint_readout')
        self.visual.predictor = TokenTransport(residual=True).to(self.visual.key.weight)
        self.short_training_arm = arm
        self.requires_grad_(True)
        self.early.requires_grad_(arm == 'fresh_joint')

    def prepare(self, anchor, images, proprio, valid_mask, group_ids, interval=8):
        if type(interval) is int and interval == 1:
            return None
        return super().prepare(anchor, images, proprio, valid_mask, group_ids, interval)


def build_model(native, arm, *, seed=7, early_state=None):
    """Read only native's shape/device, never its parameter values.

    protected_short strictly loads fresh10K early before either expert is
    copied. Its late expert intentionally starts from those same trained weights.
    Fresh experts instead share a new seed-7 initialization, with separate storage.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    if (arm == 'protected_short') != (early_state is not None):
        raise ValueError('Only protected_short requires an early state dict')
    if native.rank_dim != 64 or native.num_views != 2:
        raise ValueError('Residual transport requires rank64 and two ordered cameras')
    # Constructor RNG is CPU-only; preserve the caller's RNG, including on CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        fresh = NativeSimVLAV0(
            num_views=native.num_views, proprio_dim=native.proprio_dim,
            condition_dim=native.condition_dim, delta_dim=native.delta_dim,
            rank_dim=native.rank_dim, max_tokens=native.condition_updater.max_tokens,
            num_token_groups=native.condition_updater.num_token_groups)
        fresh.delta_encoder.image_size = native.delta_encoder.image_size
        short = BoundedHistoryCondition(fresh, 'recurrent')
        if early_state is not None:
            short.load_state_dict(early_state, strict=True)
        model = ShortVisualCondition(short, arm)
    model.to(next(native.parameters()))
    # Preserve the exact input conversion used by the existing runtime loader.
    if hasattr(native.delta_encoder, '_exact_normalized_float_rgb'):
        from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import (
            install_exact_uint8_delta_path, use_normalized_float_delta_inputs,
        )
        for _, expert in model.experts():
            install_exact_uint8_delta_path(expert)
            if native.delta_encoder._exact_normalized_float_rgb:
                use_normalized_float_delta_inputs(expert)
    return model


def gradient_blocks(model):
    """Audited functional blocks; zero-head upstream tensors may activate later."""
    blocks = {
        'late.delta_encoder': model.late.delta_encoder,
        'late.condition_updater': model.late.condition_updater,
        **{'visual.' + name: getattr(model.visual, name)
           for name in ('key', 'value', 'query', 'position', 'output')},
        **{'visual.predictor.' + name: getattr(model.visual.predictor, name)
           for name in ('encoder', 'query', 'key', 'blend', 'appearance')},
    }
    if model.short_training_arm == 'fresh_joint':
        blocks.update({'early.delta_encoder': model.early.delta_encoder,
                       'early.condition_updater': model.early.condition_updater})
    return blocks


def gradient_coverage(model):
    result = {}
    for name, module in gradient_blocks(model).items():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('Nonfinite gradient: ' + name)
        result[name] = any(bool(g.count_nonzero()) for g in gradients)
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen early received parameter gradients')
    return result


def parameter_counts(model):
    def count(module):
        parameters = list(module.parameters())
        return dict(total=sum(p.numel() for p in parameters),
                    trainable=sum(p.numel() for p in parameters if p.requires_grad))
    return dict(**count(model), blocks={name: count(getattr(model, name))
                                       for name in ('early', 'late', 'visual')})
