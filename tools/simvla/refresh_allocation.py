"""Equal-backbone-budget Long500 scheduling experiment with automatic follow-ups."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.gpu_followup_queue import run_queue
from methods.latentloop.modules.refresh_allocation import STRATEGIES

OUTPUT = prior.STORAGE/'results/simvla/refresh_allocation/two_calls_eight_queries_v1'
MODES = ('bounded_k8', 'recurrent_k8')
MODULE = 'tools.simvla.refresh_allocation'
FILES = ('methods/latentloop/modules/refresh_allocation.py',
         'architectures/simvla/adapters/latentloop/refresh_allocation_policy.py',
         'tools/simvla/refresh_allocation.py', 'tools/simvla/recover_failure_continuations.py',
         'architectures/simvla/wrappers/run_refresh_allocation.sh')


def prepare():
    parent = prior.checked_contract()
    manifest = dict(suite='libero_10', environment_seed=7, determinism_seed=20260815,
        action_noise_seed_base=6828326409295398833, action_horizon=10, execution_horizon=5,
        wait_actions=10, max_actions=900, episodes=[dict(task_id=t, trial_id=r)
            for t in reversed(range(10)) for r in range(50)])
    c = dict(parent_contract=str(prior.OUTPUT/'contract.json'),
        parent_contract_sha256=sha(prior.OUTPUT/'contract.json'),
        source_sha256={f:sha(ROOT/f) for f in FILES}, manifest=manifest,
        models={m:parent['models'][m] for m in MODES}, strategies=list(STRATEGIES),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        hardware='rb2 RTX5090 compiled' if prior.IS_RB2 else 'sd1 RTX3090 eager',
        protocol='Independent env construction and seed reset per episode; paired noise. Two original backbone calls per complete eight-query block. NFE1; unchanged H10/R5.',
        controls='Even0/4 versus consecutive0/1; random and causal overlapping-action score follow automatically. Same K8-trained checkpoint within each comparison.',
        overlap_score='Mean absolute normalized action difference: current chunk first5 versus previous chunk last5, same future execution times. Fresh query noise differs; random control tests usefulness of this signal.',
        overlap_rule='One mandatory refresh at phase0. Allocate the second using current-score empirical rank against previous <=64 scores and threshold 1-1/(8-phase). Fewer than7 past scores uses phase4; phase7 is the deadline.',
        budget='Exactly2 calls per complete block. An incomplete final block has1 or2 calls; report actual counts and latency, including discarded cheap candidates.',
        new_training=False, stopping_threshold=None)
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Allocation contract changed')
    write_json(path,c)
    return c


def checked():
    c = read_json(OUTPUT/'contract.json')
    prior.checked_contract()
    if sha(c['parent_contract']) != c['parent_contract_sha256']:
        raise RuntimeError('Parent provenance changed')
    for path, expected in c['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise RuntimeError('Allocation source changed: '+path)
    return c


def setup(c):
    if prior.IS_RB2:
        from tools.simvla.compile_benchmark import configure
        configure(c)
        sys.path.insert(0,c['libero_root'])
        os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    else:
        from tools.simvla.error_compensation_common import configure
        configure(c)


def worker(mode, strategy):
    c = checked()
    setup(c['models'][mode]['config'])
    import logging
    import numpy as np
    import torch
    from libero.libero import benchmark
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.adapters.latentloop.refresh_allocation_policy import attach, check
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs, get_libero_env, video_frame_from_obs, save_episode_video
    from tools.simvla.bounded_history_rb2 import compiler_checker
    from tools.simvla.compile_audit import WarningCapture
    torch.set_num_threads(1)
    configure_strict_torch_determinism(20260815)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total-2*1024**3)/total)
    row = mode+'_'+strategy
    out = OUTPUT/'rows'/row
    capture = WarningCapture()
    for logger in ('torch._dynamo', 'torch._inductor'):
        logging.getLogger(logger).addHandler(capture)
    suite = benchmark.get_benchmark_dict()['libero_10']()
    with torch.inference_mode():
        policy, compiler = prior.build_policy(mode,c['models'][mode]['config'])
        env,prompt = get_libero_env(suite.get_task(9),256,7)
        try:
            obs = env.reset()
            inputs = build_env_obs(obs)
            # The same K8-trained updater with periodic4 is the exact control.
            policy.k_c = policy.refresh_every = 4
            policy.reset()
            original = [policy.act(*inputs,prompt).action.copy() for _ in range(80)]
            policy.k_c = policy.refresh_every = 8
            policy = attach(policy,'even')
            actual = [policy.act(*inputs,prompt).action.copy() for _ in range(80)]
            if not np.array_equal(original, actual):
                raise RuntimeError('Periodic control action parity failed')
            check(policy)
            # Cover every trained age and both adjacent/external refresh resets.
            for schedule in ('burst',strategy):
                policy._allocation_strategy = schedule
                policy.reset()
                for _ in range(160):
                    policy.act(*inputs,prompt)
                check(policy)
            if compiler:
                compiler_checker(compiler,mode)
            write_json(out/'runtime.json',dict(parity_max_action_diff=0.0,
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                compiler=compiler.records if compiler else None,
                command=sys.argv, checkpoint_sha256=c['models'][mode]['sha256'],
                environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_GL','PYOPENGL_PLATFORM','LIBERO_CONFIG_PATH')}))
        finally:
            env.close()
        start = time.monotonic()
        rows = []
        for spec in c['manifest']['episodes']:
            task, trial = spec['task_id'],spec['trial_id']
            path = out/'episodes'/f'task{task}_trial{trial:02d}.json'
            if path.exists():
                record = read_json(path)
                if record['identity'] != c['identity'] or record['row'] != row:
                    raise RuntimeError('Mixed episode provenance')
                rows.append(record)
                continue
            configure_strict_torch_determinism(20260815)
            env,prompt = get_libero_env(suite.get_task(task),256,7)
            try:
                env.seed(7)
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task)[trial])
                for _ in range(10):
                    obs,_,_,_ = env.step([0.0]*6+[-1.0])
                policy.task_id,policy.trial_id = task,trial
                policy._allocation_strategy = strategy
                policy.reset()
                graphs = compiler.graph_count() if compiler else 0
                times, actions, frames = [], [], []
                tick = last_report = time.monotonic()
                success = False
                for i in range(900):
                    inputs = build_env_obs(obs)
                    if trial == 0 and i % 2 == 0:
                        frames.append(video_frame_from_obs(obs))
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    step = policy.act(*inputs,prompt)
                    torch.cuda.synchronize()
                    times.append(1000*(time.perf_counter()-began))
                    if not np.isfinite(step.action).all():
                        raise RuntimeError('Nonfinite action')
                    actions.append(step.action.copy())
                    obs,_,success,_ = env.step(step.action.tolist())
                    if time.monotonic()-last_report > 30:
                        write_json(out/'progress.json',dict(completed=len(rows),successes=sum(r['success'] for r in rows),
                            total=500,task=task,trial=trial,actions=i+1,phase='rollout'))
                        last_report = time.monotonic()
                    if success:
                        break
                counts = check(policy)
                if compiler:
                    compiler_checker(compiler,mode)
                fallback = any(any(s in msg for s in ('hit config.recompile_limit','hit config.cache_size_limit',"WON'T CONVERT")) for msg in capture.messages)
                valid = not compiler or (compiler.graph_count()==graphs and not fallback)
                record = dict(identity=c['identity'],row=row,task_id=task,trial_id=trial,
                    success=bool(success),episode_length=len(actions),policy_ms_total=sum(times),
                    timing_valid=valid,wall_seconds=time.monotonic()-tick,counts=counts,
                    counters=dict(policy.metrics.counters),allocation_trace=policy._allocation.records,
                    action_sha256=hashlib.sha256(np.stack(actions).tobytes()).hexdigest())
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path.with_suffix('.npz'),actions=np.stack(actions))
                write_json(path,record)
                if frames:
                    try:
                        save_episode_video(frames,out/'videos'/f'task{task}_trial0.mp4',fps=10)
                    except Exception as exc:
                        write_json(out/f'video_error_task{task}.json',dict(error=str(exc)))
                rows.append(record)
                progress = dict(completed=len(rows),total=500,successes=sum(r['success'] for r in rows),
                    recent_wall_seconds=record['wall_seconds'],updated_unix=time.time(),phase='rollout')
                write_json(out/'progress.json',progress)
                print(f"{row} {len(rows)}/500 success={progress['successes']}/{len(rows)} last={int(success)} wall={record['wall_seconds']:.1f}s",flush=True)
            finally:
                env.close()
        valid = [r for r in rows if r['timing_valid']]
        summary = dict(verdict='ALLOCATION_COMPLETE',identity=c['identity'],mode=mode,strategy=strategy,
            episodes=len(rows),successes=sum(r['success'] for r in rows),
            success_rate=sum(r['success'] for r in rows)/len(rows),timing_valid_episodes=len(valid),
            policy_ms_per_action=sum(r['policy_ms_total'] for r in valid)/sum(r['episode_length'] for r in valid) if valid else None,
            counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},
            per_task={str(t):sum(r['success'] for r in rows if r['task_id']==t) for t in range(10)},
            hardware=c['hardware'],timing_scope='Synchronized policy.act incl. routing and discarded cheap candidates; excludes env, rendering and video. Compiled graph changes excluded from latency, retained in SR.',
            elapsed_seconds=time.monotonic()-start)
        write_json(out/'summary.json',summary)
    summarize()


def summarize():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows = {p.parent.name:read_json(p) for p in (OUTPUT/'rows').glob('*/summary.json')}
        write_json(OUTPUT/'comparison_summary.json',dict(rows=rows,complete=len(rows)==8,
            interpretation='Compare schedules within the same checkpoint and host. Development seed01, 500 episodes each; final-block call counts and selector overhead are reported.'))


def jobs(c):
    result = []
    for strategy in STRATEGIES:
        for mode in MODES:
            key = mode+'_'+strategy
            deps = [] if strategy in ('even','burst') else [mode+'_even',mode+'_burst']
            result.append(dict(id=key,cmd=[sys.executable,'-u','-m',MODULE,'worker','--mode',mode,'--strategy',strategy],
                deps=deps,summary=str(OUTPUT/'rows'/key/'summary.json'),
                completion=dict(verdict='ALLOCATION_COMPLETE',identity=c['identity'],mode=mode,strategy=strategy,episodes=500)))
    if prior.IS_RB2:
        result.insert(0,dict(id='recover_previous_branches',cmd=[sys.executable,'-u','-m','tools.simvla.recover_failure_continuations'],
            summary=str(OUTPUT/'recovery/summary.json'),completion=dict(verdict='RECOVERY_AUDIT_COMPLETE')))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command',nargs='?',default='all',choices=('all','preflight','worker','summary'))
    p.add_argument('--mode',choices=MODES)
    p.add_argument('--strategy',choices=STRATEGIES)
    a = p.parse_args()
    if a.command == 'worker':
        worker(a.mode,a.strategy)
        return 0
    if a.command == 'summary':
        summarize()
        return 0
    c = prepare()
    plan = jobs(c)
    write_json(OUTPUT/'planned_jobs.json',plan)
    print('PREFLIGHT_PASS allocation jobs=',len(plan),'output=',OUTPUT,flush=True)
    if a.command == 'preflight':
        return 0
    try:
        return run_queue(OUTPUT/'queue',plan,gpus=(0,) if prior.IS_RB2 else (4,5,6,7),
            predecessor=[],environment=prior.environment,cwd=ROOT,timeout=16*3600)
    finally:
        summarize()


if __name__ == '__main__':
    raise SystemExit(main())
