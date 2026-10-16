"""Long500 short-training evaluation, with K1 and optional protected K4 parity."""
import copy
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

from tools.simvla import short_visual_train as training
from tools.simvla import failure_interventions as prior
from tools.simvla import visual_readout_campaign as merge_tools
from tools.simvla.attention_query_eval import attach_visual, goal_progress
from tools.simvla.age_specialization_eval import check as check_counts
from tools.simvla.compile_benchmark import read_json, write_json


def row_name(arm, interval, seed_index):
    if arm not in training.ARMS or interval not in (4, 8) or seed_index not in (1, 2):
        raise ValueError('Unsupported short-training evaluation')
    return f'{arm}_k{interval}_seed{seed_index}'


def output_directory(arm, interval, seed_index, shard):
    if shard not in range(4):
        raise ValueError('Invalid shard')
    return training.OUTPUT/'shards'/row_name(arm, interval, seed_index)/f'part{shard}'


def check(policy):
    counts = check_counts(policy, 'protected')
    expected = sum(t % policy.k_c == 4 for t in range(counts['queries']))
    if policy._extra_vision_calls != expected:
        raise RuntimeError('Original vision invocation schedule changed')
    return dict(**counts, extra_vision_calls=expected)


def evaluate(arm, interval=8, seed_index=1, shard=0, smoke=False, training_smoke=False, bounded=False):
    from tools.simvla.refresh_allocation import setup
    c = training.checked()
    cfg = copy.deepcopy(c['model']['config'])
    seed = c['manifest']['action_noise_seed_base'] + seed_index - 1
    setup(cfg)
    import numpy as np
    import torch
    from libero.libero import benchmark
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
        build_env_obs, get_libero_env, video_frame_from_obs, save_episode_video)
    if prior.IS_RB2:
        raise RuntimeError('This validation campaign uses sd1 eager only')
    torch.set_num_threads(1)
    configure_strict_torch_determinism(20260815)
    row = row_name(arm, interval, seed_index)
    root = output_directory(arm, interval, seed_index, shard)
    if training_smoke:
        root = training.OUTPUT/'pretraining_runtime'/row/f'part{shard}'
    out = root/('eval_smoke' if smoke else 'rows')/row
    suite = benchmark.get_benchmark_dict()['libero_10']()
    episodes = merge_tools.selected_manifest(c['manifest'], shard, 4)['episodes']
    if bounded:
        if not (smoke and training_smoke):
            raise ValueError('Bounded episode belongs only to pretraining verification')
        episodes = episodes[:1]
    with torch.inference_mode():
        policy, compiler = prior.build_policy('recurrent_k4', cfg)
        policy.action_noise_seed_base = seed
        if compiler is not None or policy.action_noise_seed_base != seed:
            raise RuntimeError('Runtime compiler or noise seed changed')
        payload, checksum = training.model_payload(c, arm, smoke=training_smoke)
        env, prompt = get_libero_env(suite.get_task(9), 256, 7)
        try:
            inputs = build_env_obs(env.reset())
            policy.k_c = policy.refresh_every = 1
            policy.reset()
            reference = np.stack([policy.act(*inputs, prompt).action.copy() for _ in range(40)])
            protected_reference = None
            if arm == 'protected_short':
                policy.k_c = policy.refresh_every = 4
                policy.reset()
                protected_reference = np.stack([policy.act(*inputs, prompt).action.copy() for _ in range(80)])
            native, _ = load_native_v0_checkpoint(cfg['condition_checkpoint'], device='cuda', require_final_150k=True)
            install_exact_uint8_delta_path(native)
            model = training.initial_model(c, arm, native).to('cuda').eval()
            model.load_state_dict(payload['model'], strict=True)
            model.requires_grad_(False)
            policy = attach_visual(policy, model, 1, None)
            actual = np.stack([policy.act(*inputs, prompt).action.copy() for _ in range(40)])
            k1_diff = float(np.abs(reference-actual).max())
            if k1_diff != 0 or policy._extra_vision_calls or sum(policy._age_calls.values()):
                raise RuntimeError('K1 bypass parity failed')
            k4_diff = None
            if protected_reference is not None:
                policy.k_c = policy.refresh_every = 4
                policy.reset()
                actual = np.stack([policy.act(*inputs, prompt).action.copy() for _ in range(80)])
                k4_diff = float(np.abs(protected_reference-actual).max())
                if k4_diff != 0:
                    raise RuntimeError('Frozen early updater K4 parity failed')
            policy.k_c = policy.refresh_every = interval
            policy.reset()
            for _ in range(160):
                policy.act(*inputs, prompt)
            counts = check(policy)
            write_json(out/'runtime.json', dict(identity=c['identity'], model_sha256=checksum,
                k1_max_action_diff=k1_diff, protected_k4_max_action_diff=k4_diff,
                counts=counts, action_noise_seed_base=seed, gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__, command=sys.argv,
                environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_GL','PYOPENGL_PLATFORM')}))
        finally:
            env.close()
        if smoke and not bounded:
            write_json(out/'summary.json', dict(identity=c['identity'], arm=row, verdict='EVAL_SMOKE_PASS',
                k1_max_action_diff=k1_diff, protected_k4_max_action_diff=k4_diff, counts=counts))
            return
        started, rows = time.monotonic(), []
        for spec in episodes:
            task, trial = spec['task_id'], spec['trial_id']
            path = out/'episodes'/f'task{task}_trial{trial:02d}.json'
            if path.exists():
                record = read_json(path)
                if (record['identity'], record['model_sha256'], record['action_noise_seed_base']) != (c['identity'], checksum, seed):
                    raise RuntimeError('Mixed resumed episode')
                rows.append(record)
                continue
            configure_strict_torch_determinism(20260815)
            env, prompt = get_libero_env(suite.get_task(task), 256, 7)
            try:
                env.seed(7)
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task)[trial])
                for _ in range(10):
                    obs, _, _, _ = env.step([0.]*6+[-1.])
                policy.task_id, policy.trial_id = task, trial
                policy.reset()
                if shutil.disk_usage(out).free < 10*1024**3:
                    raise RuntimeError('Insufficient free space for episode evidence')
                actions, times, frames, observations, goals = [], [], [], [], []
                tick = last = time.monotonic()
                success = False
                for index in range(900):
                    inputs = build_env_obs(obs)
                    if index % 2 == 0:
                        frames.append(video_frame_from_obs(obs))
                    if not policy.action_queue:
                        goals.append(dict(action_index=index, **goal_progress(env)))
                    observations.append({k:np.asarray(v).copy() for k,v in obs.items()
                        if 'image' not in k and np.asarray(v).dtype.kind in 'biuf' and np.asarray(v).ndim <= 1})
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    step = policy.act(*inputs, prompt)
                    torch.cuda.synchronize()
                    times.append(1000*(time.perf_counter()-began))
                    if not np.isfinite(step.action).all():
                        raise RuntimeError('Nonfinite action')
                    actions.append(step.action.copy())
                    obs, _, success, _ = env.step(step.action.tolist())
                    if time.monotonic()-last > 30:
                        write_json(out/'progress.json', dict(completed=len(rows), total=len(episodes),
                            successes=sum(r['success'] for r in rows), task=task, trial=trial, actions=index+1))
                        last = time.monotonic()
                    if success:
                        break
                counts = check(policy)
                trajectory = {k:np.stack([v[k] for v in observations]) for k in observations[0]
                    if all(k in v and v[k].shape == observations[0][k].shape for v in observations)}
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path.with_suffix('.npz'), actions=np.stack(actions),
                    **{'obs_'+k:v for k,v in trajectory.items()})
                video = out/'videos'/f'task{task}_trial{trial}.mp4'
                save_episode_video(frames, video, fps=10)
                if not video.is_file() or video.stat().st_size == 0:
                    raise RuntimeError('Missing episode video')
                record = dict(identity=c['identity'], arm=row, model_sha256=checksum,
                    task_id=task, trial_id=trial, success=bool(success), episode_length=len(actions),
                    policy_ms_total=sum(times), timing_valid=True, counts=counts,
                    wall_seconds=time.monotonic()-tick, action_noise_seed_base=seed,
                    action_sha256=hashlib.sha256(np.stack(actions).tobytes()).hexdigest(),
                    query_goal_progress=goals, terminal_goal_progress=goal_progress(env),
                    video=str(video), video_stride=2, metrics=policy.metrics.summary())
                write_json(path, record)
                rows.append(record)
                write_json(out/'progress.json', dict(completed=len(rows),total=len(episodes),
                    successes=sum(r['success'] for r in rows), updated_unix=time.time()))
                print(f'{row} {len(rows)}/{len(episodes)} success={sum(r["success"] for r in rows)}', flush=True)
            finally:
                env.close()
        write_json(out/'summary.json', dict(identity=c['identity'], arm=row,
            verdict='EVAL_SMOKE_PASS' if bounded else 'EVAL_COMPLETE',
            k1_max_action_diff=k1_diff, protected_k4_max_action_diff=k4_diff,
            episodes=len(rows), successes=sum(r['success'] for r in rows),
            success_rate=sum(r['success'] for r in rows)/len(rows),
            policy_ms_per_action=sum(r['policy_ms_total'] for r in rows)/sum(r['episode_length'] for r in rows),
            timing_valid_episodes=len(rows), action_noise_seed_base=seed,
            elapsed_seconds=time.monotonic()-started, hardware='sd1 RTX3090 eager',
            timing_scope='Synchronized policy.act including vision/readout and action queue; excludes env/render/video.'))


def merge(arm, interval=8, seed_index=1):
    c = training.checked()
    row = row_name(arm, interval, seed_index)
    old = merge_tools.OUTPUT, merge_tools.ARMS
    try:
        merge_tools.OUTPUT, merge_tools.ARMS = training.OUTPUT, (row,)
        summary, records = merge_tools.merged_result(c, row)
    finally:
        merge_tools.OUTPUT, merge_tools.ARMS = old
    expected_seed = c['manifest']['action_noise_seed_base'] + seed_index - 1
    if any(r['action_noise_seed_base'] != expected_seed for r in records):
        raise RuntimeError('Mixed noise seeds')
    summary.update(action_noise_seed_base=expected_seed, source_arm=arm, interval=interval)
    write_json(training.OUTPUT/'rows'/row/'episode_outcomes.json', records)
    write_json(training.OUTPUT/'rows'/row/'summary.json', summary)
    return summary
