"""Extend completed controls without changing their locked source files."""
from pathlib import Path

import numpy as np
import torch

from ..latentloop_freshness import runtime as base
from ..latentloop_freshness.interventions import replaced_method
from ..latentloop_freshness_controls import runtime as retained
from .control import NEW_ROW, fixed_gate_cached_update


class FixedGateCachedWrapper(retained.ControlWrapper):
    def _update_from_lrnode_cache(self, image_x, gripper, state, **kwargs):
        model = self._base_model()
        native = model.lrnode_apply_dynamics

        def dynamics(z_prev, u_delta, dt=1., age=1.):
            if dt != 1.:
                raise ValueError('Do not change trained dt=1')
            return fixed_gate_cached_update(model, z_prev, u_delta, age, self.fixed_gates, native)

        # The parent remains the cached-feature path, including refresh/reset/head.
        with replaced_method(model, 'lrnode_apply_dynamics', dynamics):
            actions, debug = super()._update_from_lrnode_cache(image_x, gripper, state, **kwargs)
        debug['feedback_source'] = NEW_ROW
        self.control_records[-1]['variant'] = NEW_ROW
        return actions, debug


def new_wrapper(args, model, processor, tokenizer, gates):
    result = base.build_wrapper(args, model, processor, tokenizer,
                                'cached_feature', FixedGateCachedWrapper)
    result.fixed_gates = gates
    return result


def smoke(args, model, processor, tokenizer, directory, config):
    result = retained.smoke(args, model, processor, tokenizer, directory)
    suite = base.original.benchmark.get_benchmark_dict()['libero_10']()
    task = suite.get_task(0)
    env, obs = base.make_env(args, task, 0, args.rank)
    policy = new_wrapper(args, model, processor, tokenizer, config['fixed_gates'])
    policy.reset()
    policy.set_episode_context(task, env)
    checks = 0
    try:
        for t in range(8):
            previous = None if policy.lrnode_cached_latent is None else policy.lrnode_cached_latent.clone()
            action = policy.step(obs, task.language, t)
            if not np.isfinite(action).all():
                raise AssertionError('Nonfinite fixed-gate cached action')
            if t % 4:
                native = policy._base_model()
                cached = policy.refresh_feature.read(t)
                native.lrnode_apply_dynamics(z_prev=previous, u_delta=cached, dt=1., age=float(t % 4))
                gate = torch.full_like(previous[..., :1], config['fixed_gates'][str(t % 4)])
                expected = previous + gate * 1. * native.lrnode_dynamics.last_dz
                if not torch.equal(expected, policy.lrnode_cached_latent):
                    raise AssertionError('Fixed gate / cached residual formula mismatch')
                record = policy.control_records[-1]
                if record['variant'] != NEW_ROW or record['feature_source_step'] != t-t%4:
                    raise AssertionError('Incorrect control record')
                if abs(record['gate_mean']-config['fixed_gates'][str(t%4)]) > 1e-7:
                    raise AssertionError('Fixed gate was not applied to runtime')
                checks += 1
            elif policy.refresh_feature.step != t:
                raise AssertionError('Refresh cache timestamp mismatch')
            obs, _, _, _ = env.step(action)
    finally:
        env.close()
    policy.reset()
    if policy.refresh_feature.value is not None or policy.control_records:
        raise AssertionError('Cache leaks between episodes')
    result.update(fixed_cached_formula_checks=checks, fixed_cached_reset_pass=True)
    return result


def evaluate_loaded(args, model, image_processor, tokenizer, config, stage, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    if stage == 'smoke':
        wrapped = base.IndependentInference(model.module).eval()
        wrapped.requires_grad_(False)
        torch.set_num_threads(config['threads'])
        with torch.no_grad():
            result = smoke(args, wrapped, image_processor, tokenizer, directory, config)
        base.atomic_json(Path(directory) / f'rank{args.rank}.json', result)
    elif stage == NEW_ROW:
        def factory(args, model, processor, tokenizer, variant, gates=None):
            if variant != NEW_ROW:
                raise ValueError('Unexpected variant in new control factory')
            return new_wrapper(args, model, processor, tokenizer, config['fixed_gates'])
        with replaced_method(retained, 'wrapper', factory):
            retained.evaluate_loaded(args, model, image_processor, tokenizer, config, stage, directory)
    elif stage in ('cached_feature', 'gate_fixed_residual_zero'):
        retained.evaluate_loaded(args, model, image_processor, tokenizer, config, stage, directory)
    else:
        base.evaluate_loaded(args, model, image_processor, tokenizer, config, stage, directory)
