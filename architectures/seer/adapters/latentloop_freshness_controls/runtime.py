"""Keep original Seer action protocol; intervene only in skipped latent updates."""
import copy
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ..latentloop_freshness import runtime as base
from ..latentloop_freshness.config import assigned_ids
from ..latentloop_freshness.interventions import Inputs, feature, replaced_method
from .control import ROWS, RefreshFeature, controlled_update


class ControlWrapper(base.FreshnessWrapper):
    def __init__(self, *args, **kwargs):
        self.refresh_feature = RefreshFeature()
        self.fixed_gates = None
        self.control_records = []
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.refresh_feature = RefreshFeature()
        self.control_records = []

    def _cache_full_forward_state(self, action_latent, selected_step, image_x, gripper, state, **kwargs):
        previous = None
        if self.variant in ROWS and self.lrnode_cached_image_primary is not None:
            previous = Inputs(self.lrnode_cached_image_primary[:, 0],
                              self.lrnode_cached_image_wrist[:, 0], self.lrnode_cached_state)
        super()._cache_full_forward_state(action_latent, selected_step, image_x, gripper, state, **kwargs)
        if self.variant in ROWS:
            current = Inputs(image_x[:, 0], gripper[:, 0], state)
            # At t=0, repeated current observation is available; no future frame is used.
            with base.original.preserve_rng_state(include_cuda=True):
                value = feature(self._base_model(), previous or current, current)
            if kwargs.get('timestep') is None:
                raise ValueError('Refresh feature requires an explicit timestep')
            self.refresh_feature.commit(value, kwargs['timestep'])

    def _update_from_lrnode_cache(self, image_x, gripper, state, **kwargs):
        if self.variant not in ROWS:
            return super()._update_from_lrnode_cache(image_x, gripper, state, **kwargs)
        if kwargs.get('use_zero_delta', False):
            raise ValueError('Do not combine independent feature controls with another ablation')
        timestep = kwargs['timestep']
        cached = self.refresh_feature.read(timestep)
        model = self._base_model()
        native = model.lrnode_apply_dynamics

        def encode(**unused):
            return cached if self.variant != 'gate_fixed_residual_zero' else torch.zeros_like(cached)

        def dynamics(z_prev, u_delta, dt=1., age=1.):
            if dt != 1.:
                raise ValueError('This campaign preserves trained dt=1')
            return controlled_update(model, z_prev, cached, age, self.variant, self.fixed_gates, native)

        # Call the native parent directly; the old variant hook is intentionally bypassed.
        with replaced_method(model, 'lrnode_encode_delta', encode):
            with replaced_method(model, 'lrnode_apply_dynamics', dynamics):
                actions, debug = base.original.ModelWrapper._update_from_lrnode_cache(
                    self, image_x, gripper, state, **kwargs)
        debug['feature_source_step'] = self.refresh_feature.step if self.variant != 'gate_fixed_residual_zero' else -1
        debug['feedback_source'] = self.variant
        debug['fast_encoder_called'] = 0  # cached lookup, not a fresh encoder evaluation
        record = {k: debug[k] for k in ('cache_age', 'gate_mean', 'update_norm', 'feature_source_step')}
        record.update(timestep=int(timestep), variant=self.variant, refresh_step=self.refresh_feature.step,
                      residual_norm=float(model.lrnode_dynamics.last_dz.float().norm().item()),
                      cached_feature_norm=float(cached.float().norm().item()))
        self.control_records.append(record)
        return actions, debug


def wrapper(args, model, processor, tokenizer, variant, gates=None):
    result = base.build_wrapper(args, model, processor, tokenizer, variant, ControlWrapper)
    result.fixed_gates = gates
    return result


def calibrate_episode(args, policy, task, case, directory):
    task = task._replace(init_states_file=case['init_states_file'])
    env, obs = base.make_env(args, task, case['task_id'], case['trial'])
    policy.reset()
    policy.set_episode_context(task, env)
    previous = None
    gate_samples = {str(age): [] for age in (1, 2, 3)}
    started = time.monotonic()
    try:
        for timestep in range(args.libero_eval_max_steps):
            action = policy.step(obs, task.language, timestep)
            current = policy.captured
            age = timestep % 4
            if age:
                with base.original.preserve_rng_state(include_cuda=True):
                    model = policy._base_model()
                    d = feature(model, previous, current)
                    model.lrnode_apply_dynamics(z_prev=policy.lrnode_cached_latent, u_delta=d, dt=1., age=float(age))
                    gate_samples[str(age)].append(float(model.lrnode_dynamics.last_gate.float().mean().item()))
            previous = current
            obs, _, done, _ = env.step(action)
            if timestep % 50 == 0 or done:
                base.heartbeat(directory, args.rank, {'task': case['task_id'], 'trial': case['trial'], 'step': timestep + 1})
            if done:
                break
    finally:
        env.close()
    if any(not values for values in gate_samples.values()):
        raise RuntimeError('Calibration episode did not cover all three K4 ages')
    return {'success': bool(done), 'env_steps': timestep + 1, 'seconds': time.monotonic() - started,
            'calibration_case': case, 'gate_samples_by_age': gate_samples,
            'gate_mean_by_age': {age: sum(values) / len(values) for age, values in gate_samples.items()}}


def smoke(args, model, processor, tokenizer, directory):
    suite = base.original.benchmark.get_benchmark_dict()['libero_10']()
    task = suite.get_task(0)
    env, obs = base.make_env(args, task, 0, args.rank)
    reference = base.build_wrapper(args, model, processor, tokenizer, wrapper_class=base.original.ModelWrapper)
    candidates = {row: wrapper(args, model, processor, tokenizer, row, {'1': .1, '2': .09, '3': .08})
                  for row in ('normal',) + ROWS}
    for policy in (reference, *candidates.values()):
        policy.reset()
        policy.set_episode_context(task, env)
    checks = 0
    maximum = 0.
    try:
        for t in range(8):
            rng = base.rng_state()
            action_reference = reference.step(obs, task.language, t)
            for row, policy in candidates.items():
                base.restore_rng(rng)
                z_prev = None if policy.lrnode_cached_latent is None else policy.lrnode_cached_latent.clone()
                action = policy.step(obs, task.language, t)
                if not np.isfinite(action).all():
                    raise AssertionError(f'Nonfinite smoke action: {row}')
                if row == 'normal':
                    maximum = max(maximum, float(np.max(np.abs(action - action_reference))))
                    if not np.array_equal(action, action_reference):
                        raise AssertionError('Normal action protocol changed')
                elif t % 4:
                    expected = controlled_update(policy._base_model(), z_prev,
                        policy.refresh_feature.read(t), t % 4, row, policy.fixed_gates)
                    if not torch.equal(expected, policy.lrnode_cached_latent):
                        raise AssertionError(f'Runtime does not implement specified update: {row}, t={t}')
                    checks += 1
                elif policy.refresh_feature.step != t:
                    raise AssertionError('Refresh feature timestamp mismatch')
            obs, _, _, _ = env.step(action_reference)
    finally:
        env.close()
    for policy in candidates.values():
        policy.reset()
        if policy.refresh_feature.value is not None or policy.control_records:
            raise AssertionError('Episode reset leaks feature state')
    payload = Path(directory) / f'payload_rank{args.rank}.pt'
    torch.save({'checks': checks, 'rng_after': base.rng_state()}, payload)
    if torch.load(payload, map_location='cpu')['checks'] != 18:
        raise AssertionError('Smoke artifact round-trip failed')
    return {'normal_action_max_abs_error': maximum, 'formula_checks': checks,
            'renderer': base.original.get_renderer_backend_metadata()}


def evaluate_loaded(args, ddp_model, processor, tokenizer, config, stage, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config['threads'])
    model = base.IndependentInference(ddp_model.module).eval()
    model.requires_grad_(False)
    base.atomic_json(directory / f'environment_rank{args.rank}.json', {
        'architecture': 'seer', 'method': 'latentloop_freshness_controls', 'stage': stage,
        'python': platform.python_version(), 'packages': {p: importlib.metadata.version(p)
          for p in ('torch', 'mujoco', 'robosuite', 'numpy', 'Pillow')},
        'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'policy_dtype': str(next(model.module.lrnode_dynamics.parameters()).dtype),
        'vision_dtype': str(next(model.module.vision_encoder.parameters()).dtype)})
    with torch.no_grad():
        if stage == 'smoke':
            base.atomic_json(directory / f'rank{args.rank}.json', smoke(args, model, processor, tokenizer, directory))
            return
        calibration = stage == 'calibration'
        cases = config['calibration_cases'] if calibration else None
        total = len(cases) if calibration else 500
        ids = assigned_ids(total, args.world_size, args.rank)
        gates = None
        if stage == 'gate_fixed_residual_zero':
            gates = json.loads((directory.parent / 'calibration.json').read_text())['gates']
        policy = wrapper(args, model, processor, tokenizer, 'full' if calibration else stage, gates)
        suite = base.original.benchmark.get_benchmark_dict()['libero_10']()
        seen_gap = False
        successes, completed = 0, 0
        for eval_id in ids:
            path = directory / f'episode_{eval_id:04d}.pt'
            marker = path.with_suffix('.json')
            if marker.exists():
                if seen_gap:
                    raise RuntimeError('Noncontiguous rank resume would change RNG history')
                value = json.loads(marker.read_text())
                if not path.is_file() or path.stat().st_size != value['artifact_bytes']:
                    raise RuntimeError(f'Damaged episode payload: {path}')
                result = torch.load(path, map_location='cpu')
                if result['eval_id'] != eval_id or result['stage'] != stage:
                    raise RuntimeError('Saved episode identity mismatch')
                base.restore_rng(result['rng_after'])
            else:
                seen_gap = True
                if calibration:
                    case = cases[eval_id]
                    task_id, trial = case['task_id'], case['trial']
                    result = calibrate_episode(args, policy, suite.get_task(task_id), case, directory)
                else:
                    task_id, trial = divmod(eval_id, 50)
                    task = suite.get_task(task_id)
                    env, obs = base.make_env(args, task, task_id, trial)
                    try:
                        result = base.episode(args, policy, env, obs, task, directory=directory)
                        result['control_records'] = list(policy.control_records)
                        expected = (result['env_steps'] + 3) // 4
                        if result['full_calls'] != expected or len(policy.control_records) != result['env_steps'] - expected:
                            raise AssertionError('Incorrect full/skip schedule')
                    finally:
                        env.close()
                result.update(eval_id=eval_id, stage=stage, task_id=task_id, trial=trial,
                    seed=args.seed, rank=args.rank, rng_after=base.rng_state(),
                    renderer=base.original.get_renderer_backend_metadata())
                tmp = path.with_suffix('.pt.tmp')
                torch.save(result, tmp)
                os.replace(tmp, path)
                summary = {k: result[k] for k in ('eval_id', 'stage', 'task_id', 'trial', 'seed', 'rank', 'success', 'env_steps', 'seconds')}
                if calibration:
                    summary['gate_mean_by_age'] = result['gate_mean_by_age']
                summary['artifact_bytes'] = path.stat().st_size
                base.atomic_json(marker, summary)
            completed += 1
            successes += int(result['success'])
            print(f'[EPISODE] {stage} rank={args.rank} completed={completed}/{len(ids)} SR={100*successes/completed:.2f}%', flush=True)
        base.atomic_json(directory / f'rank{args.rank}_complete.json', {'stage': stage, 'ids': ids})
