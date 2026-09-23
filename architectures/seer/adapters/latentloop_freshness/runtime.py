"""Reuse Seer's loader, preprocessing, decoder, cache schedule and action ensembling."""
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils import eval_utils_libero as original
from .config import EXTERNAL_ROWS, ROWS, VARIANTS, assigned_ids, proprio_source_steps
from .interventions import Inputs, feature, replaced_method, tensor_metrics, update


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())


def action_tokens(model, z):
    arm, grip = model.decode_action_from_latent(z)
    return torch.cat((arm, grip), dim=-1)


class IndependentInference(nn.Module):
    """Keep the .module interface without per-forward DDP collectives."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, **kwargs):
        return self.module(**kwargs)


class FreshnessWrapper(original.ModelWrapper):
    def __init__(self, *args, **kwargs):
        self.variant = 'normal'
        self.captured = None
        super().__init__(*args, **kwargs)

    def _cache_full_forward_state(self, action_latent, selected_step, image_x, gripper, state, **kwargs):
        super()._cache_full_forward_state(action_latent, selected_step, image_x, gripper, state, **kwargs)
        self.captured = Inputs(image_x[:, 0].detach(), gripper[:, 0].detach(), state.detach())

    def _update_from_lrnode_cache(self, image_x, gripper, state, **kwargs):
        if self.variant == 'normal':
            return super()._update_from_lrnode_cache(image_x, gripper, state, **kwargs)
        model = self._base_model()
        previous = Inputs(self.lrnode_cached_image_primary[:, 0],
                          self.lrnode_cached_image_wrist[:, 0], self.lrnode_cached_state)
        current = Inputs(image_x[:, 0], gripper[:, 0], state)
        native_encode, native_dynamics = model.lrnode_encode_delta, model.lrnode_apply_dynamics

        def encode(**unused):
            # Calling feature needs the native encoder, not this temporary hook.
            with replaced_method(model, 'lrnode_encode_delta', native_encode):
                return feature(model, previous, current, self.variant)

        def dynamics(z_prev, u_delta, dt=1., age=1.):
            return update(model, z_prev, u_delta, age, self.variant, dt, native_dynamics)

        with replaced_method(model, 'lrnode_encode_delta', encode):
            with replaced_method(model, 'lrnode_apply_dynamics', dynamics):
                return super()._update_from_lrnode_cache(image_x, gripper, state, **kwargs)


def build_wrapper(args, model, processor, tokenizer, variant='normal', wrapper_class=FreshnessWrapper):
    result = wrapper_class(
        model, tokenizer, processor, original.get_cast_dtype(args.precision),
        history_len=args.sequence_length, use_ensembling=True, ensembling_temp=.01,
        libero_eval_max_steps=args.libero_eval_max_steps, action_pred_steps=3,
        gripper_width=True, use_lrnode_latent_update=1,
        lrnode_eval_skip_full_forward=int(variant != 'full'), lrnode_query_interval=4,
        evaluation_seed=args.seed,
    )
    result.variant = 'normal' if variant == 'full' else variant
    return result


def configure_variant(wrapper, variant):
    wrapper.variant = 'normal' if variant == 'full' else variant
    wrapper.lrnode_eval_skip_full_forward = variant != 'full'


def make_env(args, task, task_id, trial):
    rank = torch.distributed.get_rank()
    settle = original._settle_steps(20.)
    gpu = original._renderer_gpu_device_id(rank)
    env = original.OffScreenRenderEnv(
        bddl_file_name=str(Path(args.libero_path) / 'libero/libero/bddl_files' / task.problem_folder / task.bddl_file),
        camera_heights=128, camera_widths=128, render_gpu_device_id=gpu,
        control_freq=20, horizon=original._env_horizon(args.libero_eval_max_steps, settle),
    )
    env.task_id, env.exp_id, env.task_name, env.task_suite_name = task_id, trial, task.name, 'libero_10'
    env.reset()
    original.verify_renderer_backend(env, gpu)
    env.seed(args.seed)
    path = Path(args.libero_path) / 'libero/libero/init_files' / task.problem_folder / task.init_states_file
    init = torch.load(path, map_location='cpu')[trial]
    obs = env.set_init_state(init)
    for _ in range(settle):
        env.step(np.zeros(7))
    # Deliberately retain Seer's legacy pre-settle first observation.
    return env, obs


def snapshot_wrapper(wrapper):
    return copy.deepcopy({k: v for k, v in vars(wrapper).items()
                          if k not in ('model', 'image_process_fn', 'text_process_fn')})


def restore_wrapper(wrapper, state):
    for key in list(vars(wrapper)):
        if key not in ('model', 'image_process_fn', 'text_process_fn'):
            delattr(wrapper, key)
    vars(wrapper).update(copy.deepcopy(state))


def post_action(wrapper, seq, timestep):
    probability, _candidate_count = wrapper._action_sequence_to_probability_action(
        seq, timestep, wrapper.all_time_actions.clone())
    return wrapper._threshold_probability_action(probability)


def compare_same_input(wrapper, previous, previous_z, recurrent, timestep):
    current, teacher = wrapper.captured, wrapper.lrnode_cached_latent
    model = wrapper._base_model()
    target_actions = action_tokens(model, teacher)
    records, saved = [], {}
    age = timestep % 4
    normal_feature = feature(model, previous, current)
    for mode in ('teacher_input_age1', 'teacher_input_runtime_age', 'recursive_runtime_age'):
        r = 1 if mode == 'teacher_input_age1' else age
        for variant in VARIANTS:
            z = recurrent[variant] if mode == 'recursive_runtime_age' else previous_z
            d = normal_feature if variant in ('normal', 'hold', 'zero',
                    'gate_observed_residual_zero', 'gate_zero_residual_observed') else feature(model, previous, current, variant)
            predicted = update(model, z, d, r, variant)
            actions = action_tokens(model, predicted)
            values = tensor_metrics(predicted, teacher, actions, target_actions)
            executed, target_executed = post_action(wrapper, actions, timestep), post_action(wrapper, target_actions, timestep)
            gate, residual = model.lrnode_dynamics.last_gate, model.lrnode_dynamics.last_dz
            q_previous_step, q_current_step = proprio_source_steps(timestep, wrapper.history_len, variant)
            values.update({
                'timestep': timestep, 'runtime_age': age, 'updater_age': r,
                'mode': mode, 'variant': variant,
                'proprio_current_source_step': q_current_step,
                'proprio_previous_source_step': q_previous_step,
                'feature_norm': (torch.zeros_like(d) if variant == 'zero' else d).float().norm().item(),
                'gate_mean': gate.float().mean().item(), 'gate_min': gate.min().item(), 'gate_max': gate.max().item(),
                'residual_norm': residual.float().norm().item(),
                'update_norm': (predicted - z).float().norm().item(),
                'executed_arm_l2_to_teacher': float(np.linalg.norm(executed[:6] - target_executed[:6])),
                'executed_gripper_disagreement': int(executed[-1] != target_executed[-1]),
                'raw_action': actions.float().cpu().tolist(), 'executed_action': executed.tolist(),
            })
            records.append(values)
            if mode == 'recursive_runtime_age':
                recurrent[variant] = predicted.detach()
                if variant in ('hold', 'zero', 'normal'):
                    saved[variant] = predicted.detach().cpu()
    # This decomposition is an identity for a COMMON z and age, not a causal label.
    normal = update(model, previous_z, normal_feature, age)
    zero = update(model, previous_z, normal_feature, age, 'zero')
    b, c = zero - previous_z, normal - zero
    if not torch.allclose(previous_z + b + c, normal, atol=1e-6, rtol=1e-5):
        raise AssertionError('Additive latent decomposition failed')
    saved.update({'teacher': teacher.cpu(), 'previous_teacher': previous_z.cpu(),
                  'b': b.cpu(), 'c': c.cpu(), 'timestep': timestep, 'age': age,
                  'd_observed': normal_feature.cpu(), 'q_legacy_current': current.q.cpu(),
                  'q_latest_current': current.latest_q.cpu()})
    return records, saved


def heartbeat(directory, rank, payload):
    atomic_json(Path(directory) / f'progress_rank{rank}.json', dict(payload, heartbeat=time.time()))


def episode(args, wrapper, env, obs, task, diagnostic=False, directory=None):
    wrapper.reset()
    wrapper.set_episode_context(task, env)
    previous, previous_z, recurrent = None, None, {}
    records, tensors, actions = [], [], []
    start, done = time.monotonic(), False
    for timestep in range(args.libero_eval_max_steps):
        action = wrapper.step(obs, task.language, timestep)
        if diagnostic:
            with original.preserve_rng_state(include_cuda=True):
                if timestep % 4 == 0:
                    recurrent = {v: wrapper.lrnode_cached_latent.clone() for v in VARIANTS}
                else:
                    values, saved = compare_same_input(wrapper, previous, previous_z, recurrent, timestep)
                    records.extend(values)
                    tensors.append(saved)
                previous, previous_z = wrapper.captured, wrapper.lrnode_cached_latent.clone()
        obs, _, done, _ = env.step(action)
        actions.append(np.asarray(action).tolist())
        if directory is not None and (timestep % 50 == 0 or done):
            heartbeat(directory, args.rank, {'task': env.task_id, 'trial': env.exp_id,
                      'step': timestep + 1, 'seconds': time.monotonic() - start})
        if done:
            break
    summary = {'success': bool(done), 'env_steps': len(actions), 'seconds': time.monotonic() - start,
               'full_calls': sum(r.get('mode') == 'full' for r in wrapper.current_step_records),
               'actions': actions, 'diagnostic_records': records, 'latent_records': tensors}
    return summary


def visible_obs(env):
    env.env._update_observables(force=True)
    return env.env._get_observations()


def choose_perturbation(env, sign, displacement):
    """Move a free task object only, rejecting robot contact and new deep penetration."""
    sim = env.sim
    state = sim.get_state().flatten().copy()
    def contacts():
        return {(int(x.geom1), int(x.geom2)): float(x.dist) for x in sim.data.contact[:sim.data.ncon]}
    before = contacts()
    rejected = []
    for name in sorted(env.obj_of_interest):
        obj = env.env.objects_dict.get(name)
        if obj is None or len(obj.joints) != 1:
            continue
        joint = obj.joints[0]
        qpos = np.asarray(sim.data.get_joint_qpos(joint)).copy()
        if qpos.shape != (7,):
            continue
        geom_ids = {sim.model.geom_name2id(x) for x in obj.contact_geoms}
        touching_robot = any(
            (g1 in geom_ids and str(sim.model.geom_id2name(g2)).startswith('robot')) or
            (g2 in geom_ids and str(sim.model.geom_id2name(g1)).startswith('robot'))
            for g1, g2 in before)
        if touching_robot:
            rejected.append({'object': name, 'reason': 'robot_contact'})
            continue
        shifted = qpos.copy()
        shifted[0] += sign * displacement
        sim.data.set_joint_qpos(joint, shifted)
        sim.forward()
        after = contacts()
        invalid = any((g1 in geom_ids or g2 in geom_ids) and distance < min(-.003, before.get((g1, g2), 0.) - .002)
                      for (g1, g2), distance in after.items())
        sim.set_state_from_flattened(state)
        sim.forward()
        if invalid:
            rejected.append({'object': name, 'reason': 'new_deep_penetration'})
            continue
        return {'object': name, 'joint': joint, 'before_qpos': qpos.tolist(),
                'after_qpos': shifted.tolist(), 'displacement_m': [sign * displacement, 0., 0.],
                'rejected_candidates': rejected}
    return {'skip_reason': 'no_eligible_free_task_object', 'rejected_candidates': rejected}


def external_episode(args, wrapper, task, task_id, trial, config, directory):
    event_step = config['event_step']
    env, obs = make_env(args, task, task_id, trial)
    wrapper.reset()
    wrapper.set_episode_context(task, env)
    configure_variant(wrapper, 'normal')
    prefix = []
    for timestep in range(event_step):
        action = wrapper.step(obs, task.language, timestep)
        obs, _, done, _ = env.step(action)
        prefix.append(action.copy())
        if done:
            env.close()
            return {'skip_reason': 'task_completed_before_event', 'prefix_steps': len(prefix)}
    perturb = choose_perturbation(env, 1 if (task_id + trial) % 2 == 0 else -1, config['displacement_m'])
    if 'skip_reason' in perturb:
        env.close()
        return perturb
    physical = env.get_sim_state().copy()
    saved_wrapper, saved_rng = snapshot_wrapper(wrapper), rng_state()
    # Generate the paired intervention inputs in ONE render context. Recreating
    # separate EGL contexts for these two images would add a renderer confound.
    env.sim.forward()
    scene_observations = {'nominal': copy.deepcopy(visible_obs(env))}
    env.sim.data.set_joint_qpos(perturb['joint'], np.asarray(perturb['after_qpos']))
    env.sim.forward()
    scene_observations['displaced'] = copy.deepcopy(visible_obs(env))
    env.sim.set_state_from_flattened(physical)
    env.sim.forward()
    nominal_repeat = copy.deepcopy(visible_obs(env))
    for key in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
        a, b = scene_observations['nominal'][key], scene_observations['displaced'][key]
        if not np.array_equal(a, b):
            raise AssertionError(f'External robot observation mismatch: {key}, max error={np.max(np.abs(a-b))}')
    env.close()
    result = {'event_step': event_step, 'perturbation': perturb, 'prefix_actions': np.asarray(prefix).tolist(),
              'prefix_state': physical.tolist(), 'branches': [], 'replay_state_max_error': [],
              'nominal_repeat_image_l1': {
                  key: float(np.abs(nominal_repeat[key].astype(float) - scene_observations['nominal'][key].astype(float)).mean())
                  for key in ('agentview_image', 'robot0_eye_in_hand_image')}}
    for scene in ('nominal', 'displaced'):
        for variant in EXTERNAL_ROWS:
            env, _ = make_env(args, task, task_id, trial)
            try:
                for action in prefix:
                    env.step(action)
                error = float(np.max(np.abs(env.get_sim_state() - physical)))
                result['replay_state_max_error'].append(error)
                if error > 1e-8:
                    raise RuntimeError(f'External branch prefix replay mismatch: {error}')
                if scene == 'displaced':
                    env.sim.data.set_joint_qpos(perturb['joint'], np.asarray(perturb['after_qpos']))
                # Recompute kinematics in BOTH scenes; post-integration site data
                # can otherwise lag behind qpos only in the nominal scene.
                env.sim.forward()
                # All policies within a scene receive byte-identical initial RGB/q
                # from the original shared render context, not this replay context.
                branch_obs = copy.deepcopy(scene_observations[scene])
                restore_wrapper(wrapper, saved_wrapper)
                configure_variant(wrapper, variant)
                restore_rng(saved_rng)
                before_refresh, first_raw = [], None
                done, start = False, time.monotonic()
                for timestep in range(event_step, args.libero_eval_max_steps):
                    action = wrapper.step(branch_obs, task.language, timestep)
                    if timestep < event_step + (4 - event_step % 4):
                        before_refresh.append(action.tolist())
                    if timestep == event_step:
                        first_raw = wrapper.all_time_actions[timestep, timestep:timestep + 3].float().cpu().tolist()
                    branch_obs, _, done, _ = env.step(action)
                    if timestep % 50 == 0 or done:
                        heartbeat(directory, args.rank, {'task': task_id, 'trial': trial,
                                  'scene': scene, 'variant': variant, 'step': timestep + 1})
                    if done:
                        break
                result['branches'].append({'scene': scene, 'variant': variant, 'success': bool(done),
                    'total_env_steps': timestep + 1, 'post_event_steps': timestep + 1 - event_step,
                    'seconds': time.monotonic() - start, 'before_refresh_actions': before_refresh,
                    'first_raw_horizon': first_raw})
            finally:
                env.close()
    for key in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
        if not np.array_equal(scene_observations['nominal'][key], scene_observations['displaced'][key]):
            raise AssertionError(f'External manipulation changed robot state: {key}')
    result['observation_image_l1'] = {
        k: float(np.abs(scene_observations['nominal'][k].astype(float) - scene_observations['displaced'][k].astype(float)).mean())
        for k in ('agentview_image', 'robot0_eye_in_hand_image')}
    result['paired_initial_observations'] = scene_observations
    lookup = {(b['scene'], b['variant']): b for b in result['branches']}
    for variant in ('zero', 'hold', 'stale_visual'):
        a = np.asarray(lookup['nominal', variant]['first_raw_horizon'])
        b = np.asarray(lookup['displaced', variant]['first_raw_horizon'])
        if not np.allclose(a, b, rtol=1e-6, atol=1e-6):
            raise AssertionError(f'Unobserved displacement changed first raw action: {variant}')
    restore_rng(saved_rng)
    return result


def smoke(args, model, processor, tokenizer, config, directory):
    suite = original.benchmark.get_benchmark_dict()['libero_10']()
    task = suite.get_task(0)
    env, obs = make_env(args, task, 0, args.rank)
    reference = build_wrapper(args, model, processor, tokenizer, wrapper_class=original.ModelWrapper)
    candidate = build_wrapper(args, model, processor, tokenizer)
    reference.reset()
    candidate.reset()
    reference.set_episode_context(task, env)
    candidate.set_episode_context(task, env)
    maximum = 0.
    try:
        for t in range(8):
            if t == 1:
                cache = snapshot_wrapper(candidate)
                native_rng = rng_state()
                for variant in VARIANTS:
                    restore_wrapper(candidate, cache)
                    configure_variant(candidate, variant)
                    restore_rng(native_rng)
                    previous = Inputs(candidate.lrnode_cached_image_primary[:, 0],
                                      candidate.lrnode_cached_image_wrist[:, 0], candidate.lrnode_cached_state)
                    z_previous = candidate.lrnode_cached_latent.clone()
                    candidate.step(obs, task.language, t)
                    current = Inputs(candidate.lrnode_cached_image_primary[:, 0],
                                     candidate.lrnode_cached_image_wrist[:, 0], candidate.lrnode_cached_state)
                    d = feature(candidate._base_model(), previous, current, variant)
                    expected = update(candidate._base_model(), z_previous, d, 1., variant)
                    if not torch.equal(candidate.lrnode_cached_latent, expected):
                        raise AssertionError(f'Wrapper intervention does not match module formula: {variant}')
                restore_wrapper(candidate, cache)
                restore_rng(native_rng)
            state = rng_state()
            action_ref = reference.step(obs, task.language, t)
            restore_rng(state)
            action = candidate.step(obs, task.language, t)
            maximum = max(maximum, float(np.max(np.abs(action_ref - action))))
            if not np.array_equal(action_ref, action):
                raise AssertionError(f'Normal wrapper parity failed at step {t}: {maximum}')
            obs, _, _, _ = env.step(action_ref)
    finally:
        env.close()
    # Exercise every input intervention and every age on actual loaded tensors.
    z = candidate.lrnode_cached_latent
    inputs = Inputs(candidate.lrnode_cached_image_primary[:, 0],
                    candidate.lrnode_cached_image_wrist[:, 0], candidate.lrnode_cached_state)
    core = candidate._base_model()
    for age in (1, 2, 3):
        for variant in VARIANTS:
            d = feature(core, inputs, inputs, variant)
            out = update(core, z, d, age, variant)
            if out.shape != z.shape or not torch.isfinite(out).all():
                raise AssertionError(f'Nonfinite or shape mismatch: {variant} age={age}')
    tiny = copy.copy(args)
    tiny.libero_eval_max_steps = 8
    env, obs = make_env(tiny, task, 0, args.rank)
    try:
        diagnostic = episode(tiny, build_wrapper(tiny, model, processor, tokenizer, 'full'),
                             env, obs, task, diagnostic=True)
        if not diagnostic['diagnostic_records']:
            raise AssertionError('No diagnostic records in smoke')
    finally:
        env.close()
    ext_config = dict(config, event_step=1)
    external = external_episode(tiny, build_wrapper(tiny, model, processor, tokenizer),
                                task, 0, args.rank, ext_config, directory)
    if 'skip_reason' in external:
        raise AssertionError(f'External smoke not exercised: {external}')
    payload = Path(directory) / f'validation_payload_rank{args.rank}.pt'
    torch.save({'diagnostic': diagnostic, 'external': external, 'rng_after': rng_state()}, payload)
    loaded = torch.load(payload, map_location='cpu')
    if len(loaded['external']['branches']) != len(external['branches']):
        raise AssertionError('Validation artifact round-trip failed')
    restore_rng(loaded['rng_after'])
    integration = Path(directory) / 'integration'
    small_config = dict(config, tasks=1, episodes_per_task=args.world_size, diagnostic_trials=args.world_size)
    for row in ('same_input',) + ROWS:
        row_root = integration / row
        evaluate_loaded(tiny, model, processor, tokenizer, small_config, row, row_root)
        artifact = row_root / f'episode_{args.rank:04d}.pt'
        before = artifact.stat().st_mtime_ns
        evaluate_loaded(tiny, model, processor, tokenizer, small_config, row, row_root)
        if artifact.stat().st_mtime_ns != before:
            raise AssertionError(f'Resume reran a completed episode: {row}')
    return {'normal_action_max_abs_error': maximum, 'same_input_records': len(diagnostic['diagnostic_records']),
            'external_branches': len(external['branches']), 'episode_save_resume_rows': 11,
            'renderer': original.get_renderer_backend_metadata()}


def evaluate_loaded(args, ddp_model, processor, tokenizer, config, stage, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config['threads'])
    model = IndependentInference(ddp_model.module).eval()
    model.requires_grad_(False)
    if not (directory / f'environment_rank{args.rank}.json').exists():
        import importlib.metadata
        import platform
        versions = {}
        for package in ('torch', 'mujoco', 'robosuite', 'numpy', 'Pillow'):
            versions[package] = importlib.metadata.version(package)
        atomic_json(directory / f'environment_rank{args.rank}.json', {
            'architecture': 'seer', 'method': 'latentloop_freshness_interventions',
            'stage': stage, 'python': platform.python_version(), 'packages': versions,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'policy_dtype': str(next(model.module.lrnode_dynamics.parameters()).dtype),
            'vision_dtype': str(next(model.module.vision_encoder.parameters()).dtype),
        })
    with torch.no_grad():
        if stage == 'smoke':
            result = smoke(args, model, processor, tokenizer, config, directory)
            atomic_json(directory / f'rank{args.rank}.json', result)
            return
        diagnostic = stage == 'same_input'
        external = stage == 'external'
        per_task = config['diagnostic_trials'] if diagnostic or external else config['episodes_per_task']
        ids = assigned_ids(config['tasks'] * per_task, args.world_size, args.rank)
        wrapper = build_wrapper(args, model, processor, tokenizer, 'full' if diagnostic else ('normal' if external else stage))
        suite = original.benchmark.get_benchmark_dict()['libero_10']()
        successes, done_count, started = 0, 0, time.monotonic()
        seen_gap = False
        for eval_id in ids:
            task_id, trial = divmod(eval_id, per_task)
            path = directory / f'episode_{eval_id:04d}.pt'
            marker = path.with_suffix('.json')
            if marker.is_file():
                if seen_gap:
                    raise RuntimeError('Noncontiguous saved rank sequence; do not silently change RNG history')
                saved = torch.load(path, map_location='cpu')
                if saved['eval_id'] != eval_id or saved['stage'] != stage:
                    raise RuntimeError('Saved episode identity mismatch')
                restore_rng(saved['rng_after'])
                done_count += 1
                successes += int(saved.get('success', False))
                continue
            seen_gap = True
            task = suite.get_task(task_id)
            if external:
                result = external_episode(args, wrapper, task, task_id, trial, config, directory)
            else:
                env, obs = make_env(args, task, task_id, trial)
                try:
                    result = episode(args, wrapper, env, obs, task, diagnostic, directory)
                finally:
                    env.close()
            result.update({'eval_id': eval_id, 'stage': stage, 'task_id': task_id, 'trial': trial,
                           'task_name': task.name, 'seed': args.seed, 'rank': args.rank,
                           'renderer': original.get_renderer_backend_metadata(), 'rng_after': rng_state()})
            tmp = path.with_suffix('.pt.tmp')
            torch.save(result, tmp)
            os.replace(tmp, path)
            summary = {k: result[k] for k in ('eval_id', 'stage', 'task_id', 'trial', 'seed', 'rank', 'task_name')}
            summary.update({k: result[k] for k in ('success', 'env_steps', 'seconds', 'skip_reason') if k in result})
            summary['artifact_bytes'] = path.stat().st_size
            atomic_json(marker, summary)
            done_count += 1
            successes += int(result.get('success', False))
            print(f'[EPISODE] {stage} rank={args.rank} {done_count}/{len(ids)} success={successes} '
                  f'wall={time.monotonic()-started:.1f}s task={task_id} trial={trial}', flush=True)
        atomic_json(directory / f'rank{args.rank}_complete.json', {'stage': stage, 'ids': ids})
