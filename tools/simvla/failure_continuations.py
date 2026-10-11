"""Separate persistent Condition and solver effects, then restrict their duration."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT = prior.STORAGE/'results/simvla/failure_continuations/persistent_axis_v1'
MODES = ('bounded_k8', 'recurrent_k8')
AXES = ('condition', 'solver')
PHASES = ('suffix', 'window')
MODULE = 'tools.simvla.failure_continuations'
ORIGINAL_PLAN = prior.execution_plan


def window_end(branch, k):
    # Resume cheap prediction at a scheduled refresh, with a valid age-zero context.
    return (branch//k+2)*k


def execution_plan(query, k, branch, arm, continuation):
    if continuation in prior.STAGES:
        return ORIGINAL_PLAN(query, k, branch, arm, continuation)
    if continuation not in {f'{p}_{a}' for p in PHASES for a in AXES} or arm != 'predicted_1':
        raise ValueError('Unknown persistent intervention')
    phase, axis = continuation.split('_')
    active = branch is not None and query > branch and (phase == 'suffix' or query < window_end(branch, k))
    return dict(full=query % k == 0 or (active and axis == 'condition'),
        nfe=10 if active and axis == 'solver' else 1, replace_condition=False)


def prepare():
    parent = prior.checked_contract()
    dependencies = {}
    for mode in MODES:
        for case in parent['cases']:
            name = f"task{case['task_id']}_trial{case['trial_id']}"
            directory = prior.OUTPUT/'models'/mode/'capture'/name
            for file in ('result.json','branches.json'):
                path = directory/file
                dependencies[str(path)] = sha(path)
    c = dict(parent_contract=str(prior.OUTPUT/'contract.json'), parent_identity=parent['identity'],
        parent_contract_sha256=sha(prior.OUTPUT/'contract.json'), models=list(MODES), cases=parent['cases'],
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'architectures/simvla/wrappers/run_failure_continuations.sh':sha(ROOT/'architectures/simvla/wrappers/run_failure_continuations.sh')},
        recorded_prefix_sha256=dependencies,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        phases=list(PHASES), axes=list(AXES),
        condition='From query t+1: original backbone each query, NFE1; refresh the updater context with that original output.',
        solver='From query t+1: original K8 Condition schedule, NFE10; no additional backbone calls.',
        window='Stop the intervention immediately before query (floor(t/K)+2)*K; resume original K8/NFE1 at that scheduled refresh.',
        controls='Reuse prior capture (cheap suffix) and isolate/predicted_1 (original backbone+NFE10 suffix).',
        scope='Paired K8 task-state diagnosis, six selected episodes/two locations. No new training. Not benchmark population SR.',
        no_success_threshold=True)
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Continuation contract changed')
    write_json(path,c)
    return c


def verify():
    c = read_json(OUTPUT/'contract.json')
    for path, expected in c['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise RuntimeError('Source changed: '+path)
    for path, expected in c['recorded_prefix_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('Parent prefix changed: '+path)
    if sha(c['parent_contract']) != c['parent_contract_sha256']:
        raise RuntimeError('Parent contract changed')
    return c, prior.checked_contract()


def worker(mode, phase, axis):
    c, parent = verify()
    config = parent['models'][mode]['config']
    if prior.IS_RB2:
        from tools.simvla.compile_benchmark import configure
        configure(config)
        sys.path.insert(0,config['libero_root'])
        os.environ['LIBERO_CONFIG_PATH'] = config['libero_config']
    else:
        from tools.simvla.error_compensation_common import configure
        configure(config)
    import torch
    from libero.libero import benchmark
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs, get_libero_env
    from tools.simvla.bounded_history_rb2 import compiler_checker
    torch.set_num_threads(1)
    configure_strict_torch_determinism(20260815)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total-2*1024**3)/total)
    prior.execution_plan = execution_plan
    suite = benchmark.get_benchmark_dict()['libero_10']()
    out = OUTPUT/'models'/mode/f'{phase}_{axis}'
    with torch.inference_mode():
        policy, compiler = prior.build_policy(mode,config)
        # Exercise both the persistent path and the return to cheap prediction.
        env,prompt = get_libero_env(suite.get_task(parent['cases'][0]['task_id']),256,7)
        try:
            obs = env.reset()
            inputs = build_env_obs(obs)
            original = []
            policy.reset()
            for _ in range(40):
                original.append(policy.act(*inputs,prompt).action.copy())
            policy = prior.attach_recorder(policy)
            import numpy as np
            recorded = [policy.act(*inputs,prompt).action.copy() for _ in range(40)]
            if not np.array_equal(original,recorded):
                raise RuntimeError('Unmodified prefix parity failed')
            for kind in PHASES:
                policy.reset()
                policy._intervention = dict(branch=7,arm='predicted_1',continuation=f'{kind}_{axis}')
                for _ in range(86):
                    policy.act(*inputs,prompt)
            if compiler:
                compiler_checker(compiler,mode)
            write_json(out/'runtime.json',dict(parity_max_action_diff=0.0, gpu=torch.cuda.get_device_name(0),
                compiler=compiler.records if compiler else None, checkpoint_sha256=parent['models'][mode]['sha256'],
                command=sys.argv, environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_GL','LIBERO_CONFIG_PATH')}))
        finally:
            env.close()
        # One bounded physical replay validates simulator prefix equality and window exit.
        case = parent['cases'][0]
        name = f"task{case['task_id']}_trial{case['trial_id']}"
        reference = read_json(prior.OUTPUT/'models'/mode/'capture'/name/'result.json')
        smoke = prior.rollout(policy,compiler,suite,case,out/'runtime_smoke',c['identity'],
            branch=7,arm='predicted_1',continuation=f'window_{axis}',reference=reference,limit=86,save_video=False)
        if smoke['trace'][7]['chunk_hash'] != reference['trace'][7]['chunk_hash']:
            raise RuntimeError('The selected query must remain unchanged')
        rows = []
        for case in parent['cases']:
            name = f"task{case['task_id']}_trial{case['trial_id']}"
            directory = prior.OUTPUT/'models'/mode/'capture'/name
            reference = read_json(directory/'result.json')
            for point in read_json(directory/'branches.json'):
                t = point['query']
                dest = out/name/f'query{t}'
                result = prior.rollout(policy,compiler,suite,case,dest,c['identity'],branch=t,
                    arm='predicted_1',continuation=f'{phase}_{axis}',reference=reference)
                if result['trace'][t]['chunk_hash'] != reference['trace'][t]['chunk_hash']:
                    raise RuntimeError('Current action changed before continuation intervention')
                control_path = prior.OUTPUT/'models'/mode/'isolate'/name/f'query{t}'/'predicted_1/result.json'
                full = read_json(control_path)
                rows.append(dict(case=case,point=point,success=result['success'],
                    baseline_success=reference['success'],full10_success=full['success'],
                    episode_length=result['episode_length'],policy_ms_total=result['policy_ms_total'],
                    policy_ms_per_action=result['policy_ms_per_action'],timing_valid=result['timing_valid'],
                    counters=result['counters'],window_end=window_end(t,policy.k_c) if phase=='window' else None,
                    result=str(dest/'result.json'),baseline=str(directory/'result.json'),full10_reference=str(control_path)))
        write_json(out/'summary.json',dict(verdict='CONTINUATION_COMPLETE',identity=c['identity'],
            mode=mode,phase=phase,axis=axis,rows=rows,paired_locations=len(rows),
            successes=sum(r['success'] for r in rows),
            rescued=sum(r['success'] and not r['baseline_success'] for r in rows),
            harmed=sum(not r['success'] and r['baseline_success'] for r in rows)))
    summarize()


def summarize():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows = {str(p.relative_to(OUTPUT)):read_json(p) for p in (OUTPUT/'models').glob('*/*/summary.json')}
        write_json(OUTPUT/'comparison_summary.json',dict(complete=len(rows)==8,rows=rows,
            scope='Selected six-episode/two-location paired diagnosis; instrumented latency; no population SR estimate.'))


def jobs(c):
    result = []
    for phase in PHASES:
        for mode in MODES:
            for axis in AXES:
                key = f'{mode}_{phase}_{axis}'
                result.append(dict(id=key,cmd=[sys.executable,'-u','-m',MODULE,'worker',
                    '--mode',mode,'--phase',phase,'--axis',axis],
                    deps=[f'{mode}_suffix_{axis}'] if phase=='window' else [],
                    summary=str(OUTPUT/'models'/mode/f'{phase}_{axis}'/'summary.json'),
                    completion=dict(verdict='CONTINUATION_COMPLETE',identity=c['identity'],mode=mode,phase=phase,axis=axis)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command',nargs='?',default='all',choices=('all','preflight','worker','summary'))
    p.add_argument('--mode',choices=MODES)
    p.add_argument('--phase',choices=PHASES)
    p.add_argument('--axis',choices=AXES)
    a = p.parse_args()
    if a.command=='worker':
        worker(a.mode,a.phase,a.axis)
        return 0
    if a.command=='summary':
        summarize()
        return 0
    c = prepare()
    plan = jobs(c)
    write_json(OUTPUT/'planned_jobs.json',plan)
    print('PREFLIGHT_PASS',OUTPUT,'jobs=',len(plan),flush=True)
    if a.command=='preflight':
        return 0
    rc = run_queue(OUTPUT/'queue',plan,gpus=(0,) if prior.IS_RB2 else (4,5,6,7),
        predecessor=dict(path=str(prior.OUTPUT/'queue'),lock='queue.lock',allow_when_all_assigned=True),
        environment=prior.environment,cwd=ROOT,timeout=8*3600)
    summarize()
    return rc


if __name__=='__main__':
    raise SystemExit(main())
