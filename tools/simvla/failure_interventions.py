"""Task-state matched Condition/solver interventions, with resumable follow-ups."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import MethodType

from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.bounded_history_pipeline import MODES, ARM, OUTPUT as SD_SOURCE
from tools.simvla.gpu_followup_queue import run_queue

ARMS = ('predicted_1', 'original_1', 'predicted_10', 'original_10')
STAGES = ('capture', 'isolate', 'continue')
RB_STORAGE = Path('/home/mingyujung/private/gnaroshi_vla_storage')
IS_RB2 = socket.gethostname() == 'jbr-TRX50'
STORAGE = RB_STORAGE if IS_RB2 else SD_SOURCE.parents[3]
OUTPUT = STORAGE / 'results/simvla/failure_interventions/task_state_v1'
MODULE = 'tools.simvla.failure_interventions'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def config(mode):
    if IS_RB2:
        from tools.simvla.bounded_history_rb2 import base_config, ready_spec
        c = base_config()
        c['model_checkpoint'] = ready_spec(mode)
        return c
    return read_json(SD_SOURCE / mode / 'runtime_config.json')


def source_directory(mode):
    if IS_RB2:
        from tools.simvla.bounded_history_rb2 import OUTPUT as source
        return source / 'online' / mode / 'rows/libero_10/seed01' / mode
    return SD_SOURCE / mode / 'online' / f'kc{MODES[mode][1]}_{ARM}'


def select_cases(outcomes, tasks):
    """Predeclared contrasting outcomes, identical episode IDs across models."""
    result = []
    for task in tasks:
        ids = sorted(trial for t, trial in outcomes['bounded_k4'] if t == task)
        k8_loss = [r for r in ids if all(outcomes[m][task, r] for m in ('bounded_k4', 'recurrent_k4'))
                   and not outcomes['bounded_k8'][task, r] and not outcomes['recurrent_k8'][task, r]]
        k4_loss = [r for r in ids if not outcomes['bounded_k4'][task, r] or not outcomes['recurrent_k4'][task, r]]
        first = (k8_loss or ids)[0]
        remaining = [r for r in (k4_loss + ids) if r != first]
        for trial, reason in ((first, 'K4_success_K8_failure' if k8_loss else 'fallback_first'),
                              (remaining[0], 'K4_failure' if remaining[0] in k4_loss else 'additional_control')):
            result.append(dict(task_id=task, trial_id=trial, selection=reason,
                previous_success={m: outcomes[m][task, trial] for m in MODES}))
    return result


def select_queries(trace, k):
    """Two retrospective query locations; events are diagnostic, not safety labels."""
    eligible = [r for r in trace[1:-1] if r['query'] % k]
    if len(eligible) < 2:
        return []
    # The query preceding the largest object displacement in the early rollout.
    early = [r for r in eligible if r['query'] <= 30] or eligible
    event = max(early, key=lambda r: r['object_displacement'])
    event_query = min((r['query'] for r in eligible), key=lambda t: (abs(t - (event['query']-1)), t))
    # Longest interval without a newly attained task-goal predicate.
    seen, last, gaps = set(), 0, []
    for r in trace:
        active = {i for i, yes in enumerate(r['goals']) if yes}
        if active - seen:
            gaps.append((r['query'] - last, last, r['query']))
            last = r['query']
        seen |= active
    gaps.append((trace[-1]['query'] - last, last, trace[-1]['query']))
    _, start, end = max(gaps)
    target = start + min(k, max(1, (end-start)//2))
    others = [r['query'] for r in eligible if r['query'] != event_query]
    stall_query = min(others, key=lambda t: (abs(t-target), t))
    return [dict(query=event_query, reason='before_early_object_motion', event_query=event['query']),
            dict(query=stall_query, reason='inside_longest_no_new_goal_interval', interval=[start, end])]


def execution_plan(query, k, branch, arm, continuation):
    if arm not in ARMS or continuation not in ('isolate', 'continue', 'capture'):
        raise ValueError('Unknown intervention')
    after = branch is not None and query > branch and continuation == 'isolate'
    intervention = branch is not None and query == branch
    return dict(full=after or query % k == 0,
        nfe=10 if after or (intervention and arm.endswith('_10')) else 1,
        replace_condition=intervention and arm.startswith('original_'))


def environment(gpu):
    if IS_RB2:
        from tools.simvla.condition_output_split_rb2 import environment as env
        return env(gpu)
    from tools.simvla.error_compensation_common import environment as env
    return env(config('bounded_k4'), gpu)


def prepare():
    if socket.gethostname() not in ('jbrserver1', 'jbr-TRX50'):
        raise RuntimeError('Unsupported host')
    outcomes = {}
    sources = {}
    for mode in MODES:
        d = source_directory(mode)
        values = [read_json(p) for p in sorted((d/'episodes').glob('*.json'))]
        if len(values) != 500:
            raise RuntimeError(f'{mode}: expected completed 500-episode source')
        outcomes[mode] = {(r['task_id'], r['trial_id']): bool(r['success']) for r in values}
        c = config(mode)
        ckpt = Path(c['model_checkpoint']['path']) if IS_RB2 else Path(c['output'])/'train'/ARM/'latest.pt'
        sources[mode] = dict(checkpoint=str(ckpt), sha256=sha(ckpt), config=c,
            source_summary=str(d/'summary.json'), source_summary_sha256=sha(d/'summary.json'))
    cases = select_cases(outcomes, (6, 8, 9) if IS_RB2 else (0, 1, 7))
    files = [p for folder in ('tools/simvla', 'methods/latentloop', 'architectures/simvla/adapters')
             for p in (ROOT/folder).rglob('*.py')]
    contract = dict(host=socket.gethostname(), models=sources, cases=cases,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        action_horizon=10, execution_horizon=5, wait_actions=10, max_actions=900,
        environment_seed=7, intervention_arms=list(ARMS), stages=list(STAGES),
        scope='Outcome-selected causal diagnosis. Instrumented latency, not population SR or paper latency. No training.',
        condition_intervention='Replace current action-head input with original condition; retain identical updater history for subsequent cheap queries.',
        continuation='isolate: original backbone and NFE10 from the next query; continue: original cheap K/NFE1 schedule.',
        selection='Six episode IDs per host, shared by four models. sd1 tasks0/1/7; rb2 tasks6/8/9. Two hindsight-selected query locations per episode.')
    contract['identity'] = digest(contract)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != contract:
        raise RuntimeError('Experiment contract changed; preserve previous results')
    write_json(path, contract)
    return contract


def checked_contract():
    c = read_json(OUTPUT/'contract.json')
    for file, expected in c['source_sha256'].items():
        if sha(ROOT/file) != expected:
            raise RuntimeError('Source changed: '+file)
    return c


def attach_recorder(policy):
    decode = policy._decode
    policy._intervention = dict(branch=None, arm='predicted_1', continuation='capture')
    policy._pending_record = None
    expected = Counter()
    original_reset = policy.reset

    def reset(self):
        original_reset()
        expected.clear()
        self._pending_record = None

    def measured_decode(self, condition, proprio, *, policy_query_index):
        plan = execution_plan(policy_query_index, self.k_c, **self._intervention)
        predicted = condition
        self.nfe = plan['nfe']
        if plan['replace_condition']:
            batch = self._intervention_batch
            condition = self.condition_adapter.encode_condition(
                input_ids=batch['input_ids'], image_input=batch['image_input'], image_mask=batch['image_mask'])
            self.metrics.counters['num_full_vlm_calls'] += 1
            expected['num_full_vlm_calls'] += 1
        action, seed = decode(condition, proprio, policy_query_index=policy_query_index)
        self._pending_record = dict(condition=condition.detach(), predicted=predicted.detach(),
            chunk=action.detach(), seed=seed, query=policy_query_index, **plan)
        expected['num_action_transformer_calls'] += plan['nfe']
        return action, seed

    def refill(self, batch):
        query = int(self.query_index)
        self._intervention_batch = batch
        plan = execution_plan(query, self.k_c, **self._intervention)
        self.metrics.counters['num_policy_queries'] += 1
        expected['num_policy_queries'] += 1
        if plan['full']:
            _, chunk, seed = self._full_refresh(batch, policy_query_index=query)
            source = 'full_refresh'
            expected['num_full_vlm_calls'] += 1
        else:
            _, chunk, seed = self._v0_update(batch, age=query % self.k_c, policy_query_index=query)
            source = 'native_v0'
            expected['num_condition_updater_calls'] += 1
        if tuple(chunk.shape[1:]) != (10, 7):
            raise RuntimeError('Action chunk H10x7 changed')
        self.action_queue.clear()
        self.action_queue.extend((action.detach(), source) for action in chunk[0, :5])
        self.query_trace.append(dict(policy_query_index=query, age=query % self.k_c, source=source,
            full_vlm_called=plan['full'], condition_updater_called=not plan['full'],
            action_noise_seed=seed, action_horizon=10, execution_horizon=5, k_c=self.k_c))
        self.query_index += 1
        for key, count in expected.items():
            if self.metrics.counters[key] != count:
                raise RuntimeError(f'Intervention counter mismatch {key}')
        return dict(refreshed=plan['full'], age=query % self.k_c, queue_mode=source, action_noise_seed=seed)

    policy.reset = MethodType(reset, policy)
    policy._decode = MethodType(measured_decode, policy)
    policy._refill_action_queue = MethodType(refill, policy)
    policy.reset()
    return policy


def array_hash(*arrays):
    import numpy as np
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def environment_record(env, obs):
    import numpy as np
    base = env.env
    goals = base.parsed_problem['goal_state']
    objects = {name: np.asarray(base.sim.data.body_xpos[base.obj_body_id[name]]).copy().tolist()
               for name in sorted(base.objects_dict) if name in base.obj_body_id}
    contacts = [dict(geom1=int(x.geom1), geom2=int(x.geom2), distance=float(x.dist))
                for x in base.sim.data.contact[:base.sim.data.ncon]]
    numeric = [obs[key] for key in sorted(obs) if isinstance(obs[key], np.ndarray)]
    return dict(goals=[bool(base._eval_predicate(g)) for g in goals], goal_predicates=goals,
        objects=objects, contacts=contacts,
        state_hash=array_hash(base.sim.get_state().flatten(), *numeric))


def build_policy(mode, c):
    import torch
    if IS_RB2:
        from tools.simvla.compiled_campaign import move
        from tools.simvla.bounded_history_rb2 import policy_factory, replay_factory
        from tools.simvla.compile_runtime import Compiler
        samples = move(torch.load(Path(c['recorded_input'])/'recorded_inputs.pt', map_location='cpu', weights_only=False)[:1])
        compiler = Compiler(True)
        replay = replay_factory(c, mode, compiler, samples)
        manifest = dict(suite='libero_10', action_noise_seed_base=6828326409295398833)
        policy = policy_factory(replay, c, mode, manifest)
    else:
        from tools.simvla.condition_output_split_eval import make_policy
        policy = make_policy(c, ARM, k_c=MODES[mode][1])
        compiler = None
    return policy, compiler


def rollout(policy, compiler, suite, case, destination, identity, *, limit=900, branch=None,
            arm='predicted_1', continuation='capture', reference=None, save_video=True):
    import numpy as np
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
        get_libero_env, build_env_obs, video_frame_from_obs, save_episode_video)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination/'result.json'
    key = dict(identity=identity, case=case, branch=branch, arm=arm, continuation=continuation, limit=limit)
    if result_path.exists():
        result = read_json(result_path)
        if result['key'] != key:
            raise RuntimeError('Mixed intervention results')
        return result
    task, trial = case['task_id'], case['trial_id']
    configure_strict_torch_determinism(20260815)
    env, prompt = get_libero_env(suite.get_task(task), 256, 7)
    try:
        env.seed(7)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(task)[trial])
        initial_hash = environment_record(env, obs)['state_hash']
        for _ in range(10):
            obs, _, _, _ = env.step([0.0]*6+[-1.0])
        post_wait_hash = environment_record(env, obs)['state_hash']
        policy.reset()
        policy.task_id, policy.trial_id = task, trial
        policy._intervention = dict(branch=branch, arm=arm, continuation=continuation)
        trace, states, actions, times, frames = [], [], [], [], []
        prefix = hashlib.sha256()
        began = time.monotonic()
        graphs_before = compiler.graph_count() if compiler else 0
        success = False
        prefix_verified = branch is None
        for index in range(limit):
            query = not policy.action_queue
            state = environment_record(env, obs)
            state['action_index'] = index
            states.append(state)
            if query:
                t = int(policy.query_index)
                before = dict(query=t, state_hash=state['state_hash'], action_prefix_hash=prefix.hexdigest())
                if reference is not None and t <= branch:
                    ref = reference['trace'][t]
                    if any(before[x] != ref[x] for x in before):
                        raise RuntimeError(f'Unmatched pre-intervention prefix task{task} trial{trial} query{t}')
                    if t == branch:
                        prefix_verified = True
            if save_video and index % 2 == 0:
                frames.append(video_frame_from_obs(obs))
            inputs = build_env_obs(obs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            step = policy.act(*inputs, prompt)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
            if not np.isfinite(step.action).all():
                raise RuntimeError('Nonfinite action')
            if query:
                r = policy._pending_record
                z = r['condition'].float().cpu().numpy()
                pred = r['predicted'].float().cpu().numpy()
                chunk = r['chunk'].float().cpu().numpy()
                movement = max((float(np.linalg.norm(np.asarray(pos)-np.asarray(trace[-1]['objects'][name])))
                               for name, pos in state['objects'].items()), default=0.0) if trace else 0.0
                record = dict(**before, goals=state['goals'], objects=state['objects'],
                    object_displacement=movement, chunk=chunk.tolist(), chunk_hash=array_hash(chunk),
                    condition_hash=array_hash(z), condition_norm=float(np.linalg.norm(z)),
                    condition_shape=list(z.shape), seed=r['seed'], nfe=r['nfe'], full=r['full'],
                    replace_condition=r['replace_condition'], queue_length=len(policy.action_queue)+1,
                    measured_query_ms=times[-1])
                if r['replace_condition']:
                    record['predicted_vs_original'] = dict(mse=float(np.mean((pred-z)**2)),
                        cosine=float(np.sum(pred*z)/max(1e-30,float(np.linalg.norm(pred)*np.linalg.norm(z)))),
                        predicted_norm=float(np.linalg.norm(pred)), original_norm=float(np.linalg.norm(z)))
                trace.append(record)
                if reference is not None and t < branch and record['chunk_hash'] != reference['trace'][t]['chunk_hash']:
                    raise RuntimeError('Prefix action chunk changed')
            action = np.asarray(step.action)
            actions.append(action.copy())
            prefix.update(np.ascontiguousarray(action).tobytes())
            obs, _, success, _ = env.step(action.tolist())
            if index % 100 == 0:
                write_json(destination/'progress.json', dict(task=task, trial=trial, actions=index+1,
                    branch=branch, arm=arm, continuation=continuation, elapsed_seconds=time.monotonic()-began))
            if success:
                break
        states.append(dict(action_index=len(actions), **environment_record(env, obs)))
        if not prefix_verified:
            raise RuntimeError('Episode ended before the planned intervention')
        graphs_after = compiler.graph_count() if compiler else 0
        result = dict(key=key, success=bool(success), episode_length=len(actions), trace=trace,
            initial_hash=initial_hash, post_wait_hash=post_wait_hash, prefix_verified=prefix_verified,
            counters=dict(policy.metrics.counters), policy_ms_total=sum(times),
            policy_ms_per_action=sum(times)/len(actions), wall_seconds=time.monotonic()-began,
            compiler_graphs_before=graphs_before, compiler_graphs_after=graphs_after,
            timing_valid=graphs_before==graphs_after,
            timing_scope='instrumented policy.act incl. record assembly; env/state extraction/video excluded; not paper latency',
            success_check='LIBERO env.step done after every executed action', goal_predicates=states[-1]['goal_predicates'])
        np.savez_compressed(destination/'trajectory.npz', actions=np.asarray(actions), policy_ms=np.asarray(times))
        write_json(destination/'environment_trace.json', states)
        if frames:
            save_episode_video(frames, destination/'episode.mp4', fps=10)
        write_json(result_path, result)
        print(f"EPISODE {continuation} {arm} task={task} trial={trial} query={branch} success={int(success)} actions={len(actions)} wall={result['wall_seconds']:.1f}s", flush=True)
        return result
    finally:
        env.close()


def worker(mode, stage, smoke=False):
    contract = checked_contract()
    c = contract['models'][mode]['config']
    if IS_RB2:
        from tools.simvla.compile_benchmark import configure
        configure(c)
        sys.path.insert(0, c['libero_root'])
        os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    else:
        from tools.simvla.error_compensation_common import configure
        configure(c)
    import torch
    from libero.libero import benchmark
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import get_libero_env, build_env_obs
    torch.set_num_threads(1)
    configure_strict_torch_determinism(20260815)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total-2*1024**3)/total)
    out = OUTPUT/('smoke' if smoke else 'models')/mode
    cases = contract['cases'][:1] if smoke else contract['cases']
    suite = benchmark.get_benchmark_dict()['libero_10']()
    with torch.inference_mode():
        policy, compiler = build_policy(mode, c)
        # First prove recorder parity against the unmodified policy on a fixed observation.
        env, prompt = get_libero_env(suite.get_task(cases[0]['task_id']), 256, 7)
        try:
            obs = env.reset()
            inputs = build_env_obs(obs)
            expected = []
            policy.reset()
            for _ in range(5*policy.k_c):
                expected.append(policy.act(*inputs, prompt).action.copy())
            policy = attach_recorder(policy)
            actual = [policy.act(*inputs, prompt).action.copy() for _ in expected]
            import numpy as np
            if not np.array_equal(expected, actual):
                raise RuntimeError('Recorder altered original policy actions')
            # Warm NFE10 on both original and predicted conditions, outside episode timing.
            policy.reset()
            policy._intervention = dict(branch=1, arm='original_10', continuation='isolate')
            for _ in range(16):
                policy.act(*inputs, prompt)
            write_json(out/f'{stage}_runtime.json', dict(parity_max_action_diff=0.0,
                compiler=compiler.records if compiler else None, gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__, environment={k:os.environ.get(k) for k in
                    ('CUDA_VISIBLE_DEVICES','MUJOCO_GL','PYOPENGL_PLATFORM','LIBERO_CONFIG_PATH','LD_LIBRARY_PATH')},
                checkpoint_sha256=contract['models'][mode]['sha256']))
        finally:
            env.close()
        rows = []
        for case in cases:
            name = f"task{case['task_id']}_trial{case['trial_id']}"
            baseline_dir = out/'capture'/name
            limit = 46 if smoke else 900
            if stage == 'capture' or smoke:
                result = rollout(policy, compiler, suite, case, baseline_dir, contract['identity'], limit=limit)
                points = select_queries(result['trace'], policy.k_c)
                if len(points) != 2:
                    raise RuntimeError('Insufficient queries for two matched intervention locations')
                write_json(baseline_dir/'branches.json', points)
                rows.append(dict(case=case, arm='predicted_1', branch=None, result=str(baseline_dir/'result.json'),
                    success=result['success'], episode_length=result['episode_length'], wall_seconds=result['wall_seconds']))
            if stage != 'capture' or smoke:
                reference = read_json(baseline_dir/'result.json')
                points = read_json(baseline_dir/'branches.json')
                continuations = ('isolate','continue') if smoke else (stage,)
                for continuation in continuations:
                    for point in points:
                        for arm in ARMS:
                            # The unchanged cheap arm is already the capture trajectory.
                            if continuation == 'continue' and arm == 'predicted_1':
                                result, dest = reference, baseline_dir
                            else:
                                dest = out/continuation/name/f"query{point['query']}"/arm
                                result = rollout(policy, compiler, suite, case, dest, contract['identity'], limit=limit,
                                    branch=point['query'], arm=arm, continuation=continuation, reference=reference)
                            rows.append(dict(case=case, point=point, arm=arm, continuation=continuation,
                                result=str(dest/'result.json'), success=result['success'],
                                episode_length=result['episode_length'], wall_seconds=result['wall_seconds'],
                                policy_ms_per_action=result['policy_ms_per_action'], reused_capture=dest==baseline_dir))
        write_json(out/f'{stage}_summary.json', dict(verdict='SMOKE_PASS' if smoke else 'STAGE_COMPLETE',
            identity=contract['identity'], mode=mode, stage=stage, rows=rows, cases=len(cases)))
    summarize()


def summarize():
    # Several sd1 workers can finish together; serialize the shared report only.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summaries, paired = {}, []
        for p in (OUTPUT/'models').glob('*/*_summary.json'):
            summary = read_json(p)
            summaries[str(p.relative_to(OUTPUT))] = summary
            groups = {}
            for row in summary['rows']:
                if 'point' not in row:
                    continue
                key = (row['case']['task_id'], row['case']['trial_id'], row['point']['query'])
                groups.setdefault(key, {})[row['arm']] = row
            for (task, trial, query), arms in groups.items():
                if set(arms) != set(ARMS):
                    raise RuntimeError('Incomplete factorial comparison')
                paired.append(dict(mode=summary['mode'], stage=summary['stage'], task_id=task,
                    trial_id=trial, query=query,
                    success={a: arms[a]['success'] for a in ARMS},
                    episode_length={a: arms[a]['episode_length'] for a in ARMS},
                    condition_success_difference_nfe1=int(arms['original_1']['success'])-int(arms['predicted_1']['success']),
                    solver_success_difference_predicted=int(arms['predicted_10']['success'])-int(arms['predicted_1']['success']),
                    combined_success_difference=int(arms['original_10']['success'])-int(arms['predicted_1']['success']),
                    evidence={a: arms[a]['result'] for a in ARMS}))
        write_json(OUTPUT/'comparison_summary.json', dict(stages=summaries, paired=paired,
            complete=len(summaries)==12,
            interpretation='Outcome-selected paired diagnosis; compare arms within the same model, episode, branch and continuation. Instrumented latency only.'))


def jobs(smoke_only=False):
    result = []
    # Each model can progress independently; technical failures preserve unrelated work.
    modes = ('bounded_k8',) if smoke_only else tuple(MODES)
    for mode in modes:
        stages = ('capture',) if smoke_only else STAGES
        dependency = []
        for stage in stages:
            key = mode+'_'+stage
            cmd = [sys.executable, '-u', '-m', MODULE, 'worker', '--mode', mode, '--stage', stage]
            if smoke_only:
                cmd.append('--smoke')
            result.append(dict(id=key, cmd=cmd, deps=dependency,
                summary=str(OUTPUT/('smoke' if smoke_only else 'models')/mode/f'{stage}_summary.json'),
                completion=dict(verdict='SMOKE_PASS' if smoke_only else 'STAGE_COMPLETE',
                    identity=read_json(OUTPUT/'contract.json')['identity'], mode=mode, stage=stage)))
            dependency = [key]
    # Stage-order scheduling keeps all four sd1 devices busy before follow-ups begin.
    return sorted(result, key=lambda j: (STAGES.index(j['id'].rsplit('_',1)[-1]), j['id']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('preflight','smoke','all','worker','summary'), nargs='?', default='all')
    p.add_argument('--mode', choices=MODES)
    p.add_argument('--stage', choices=STAGES)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.command == 'worker':
        worker(a.mode, a.stage, a.smoke)
        return 0
    if a.command == 'summary':
        summarize()
        return 0
    contract = prepare()
    plan = jobs(a.command=='smoke')
    write_json(OUTPUT/('smoke_plan.json' if a.command=='smoke' else 'planned_jobs.json'), plan)
    print('PREFLIGHT_PASS', OUTPUT, 'cases=', contract['cases'], flush=True)
    if a.command == 'preflight':
        return 0
    if a.command == 'all':
        smoke_path = OUTPUT/'smoke/bounded_k8/capture_summary.json'
        if not smoke_path.exists() or read_json(smoke_path).get('identity') != contract['identity']:
            raise RuntimeError('Run and pass the bounded K8 all-arms smoke first')
    rc = run_queue(OUTPUT/('smoke_queue' if a.command=='smoke' else 'queue'), plan,
        gpus=(0,) if IS_RB2 else (4,5,6,7), predecessor=[], environment=environment, cwd=ROOT, timeout=12*3600)
    summarize()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
