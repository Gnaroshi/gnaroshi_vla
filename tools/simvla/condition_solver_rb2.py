"""Queued matched student-solver evaluation after the three-step controls."""
import argparse
import os
from pathlib import Path
import sys

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT, configure, read_json, write_json, sha
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.condition_initialization_rb2 import OUTPUT as PRIOR, ROWS, base_config as previous_config
from tools.simvla.condition_output_split_rb2 import STORAGE, environment, replay_factory, compiler_checker, reset_checker
from tools.simvla.condition_output_split_eval import load_payload, attach, check_policy
from tools.simvla.rollout_repair_rb2 import evaluate
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT = STORAGE / 'results/simvla/condition_output_split/solver_matched_nfe1_compiled_seed01_v1'
INCOMING = STORAGE / 'incoming/simvla_condition_solver'


def solver_paths(nfe):
    if type(nfe) is not int or nfe not in (1,2):
        raise ValueError('This evaluation supports NFE1 or NFE2')
    return (STORAGE / f'results/simvla/condition_output_split/solver_matched_nfe{nfe}_compiled_seed01_v1',
            INCOMING if nfe==1 else INCOMING.with_name(INCOMING.name+'_nfe2'))


def predecessor(nfe):
    return dict(path=str(PRIOR if nfe==1 else OUTPUT), lock='queue.lock')


def base_config(nfe=1):
    output, _ = solver_paths(nfe)
    c = previous_config()
    c.update(output=str(output), student_steps=nfe, campaign_module='tools.simvla.condition_solver_rb2',
        extra_source_files=c['extra_source_files'] + ['tools/simvla/condition_solver_rb2.py'],
        scope=f'Same 3K starts and next 7K samples/schedule as naive3 controls; train and deploy naive{nfe}. Original NFE10 teacher. Four models x K4/K8 x 500 LIBERO-Long seed01; compiled total policy latency, H10/R5 unchanged. No SR gate.')
    return c


def ready_spec(init, arm, nfe=1):
    _, incoming = solver_paths(nfe)
    directory = incoming / (init + '_10k') / arm
    if read_json(directory/'READY.json')['manifest_sha256'] != sha(directory/'manifest.json'):
        raise RuntimeError('Non-atomic transfer')
    m = read_json(directory/'manifest.json')
    if m['arm'] != arm or m['step'] != 7000 or sha(directory/'model.pt') != m['checkpoint_sha256']:
        raise RuntimeError('Transferred checkpoint mismatch')
    p = load_payload(directory/'model.pt', arm, m['source_identity'], steps=7000, action_mode=f'naive{nfe}')
    contract = p['contract']
    if (contract['total_training_steps'] != 10000 or contract['sample_step_offset'] != 3000
            or contract['initialization'] != 'continuation'
            or contract['solver_transition'] != f'naive3_to_naive{nfe}'
            or contract['continuation']['contract'].get('initialization', 'pretrained') != init):
        raise RuntimeError('Training initialization/solver/budget mismatch')
    control = read_json(PRIOR/'online'/f'{init}_{arm}_k4'/'runtime_config.json')['model_checkpoint']
    control_payload = load_payload(control['path'], arm, control['source_identity'], steps=7000)
    for key in ('initial_weights_sha256','data','heldout','batch_size','seed','optimizer',
                'condition_weight','current_action_gradient','future_condition_gradient'):
        if contract[key] != control_payload['contract'][key]:
            raise RuntimeError('Matched naive3 training control differs: '+key)
    return dict(path=str(directory/'model.pt'), sha256=m['checkpoint_sha256'], arm=arm, step=7000,
        total_training_steps=10000, initialization=init, source_identity=m['source_identity'],
        student_steps=nfe, matched_naive3_checkpoint_sha256=control['sha256'])


def policy_factory(replay, c, row, manifest):
    init, arm, k = ROWS[row]; spec = c['model_checkpoint']
    nfe=c['student_steps']
    if spec['student_steps']!=nfe: raise RuntimeError('Checkpoint/deployment solver mismatch')
    if sha(spec['path']) != spec['sha256']: raise RuntimeError('Checkpoint changed')
    payload = load_payload(spec['path'], arm, spec['source_identity'], steps=7000, action_mode=f'naive{nfe}')
    policy = attach_policy(replay, c, 'condition_naive3', manifest)
    policy.nfe = nfe
    return attach(policy, replay.native, payload, arm, k, replay.compiler)


def cell(c, row):
    nfe=c['student_steps']; output,_=solver_paths(nfe)
    init, arm, k = ROWS[row]; spec = ready_spec(init, arm, nfe)
    result = evaluate({**c, 'action_mode':f'condition_naive{nfe}', 'condition_interval':k}, row, spec, output=output)
    write_json(output/'completed'/f'{row}.json', dict(verdict='EVALUATION_COMPLETE', episodes=500,
        row=row, checkpoint_sha256=spec['sha256'], result=result))


def jobs(nfe=1):
    output,incoming=solver_paths(nfe)
    return [dict(id=row, cmd=[sys.executable,'-u','-m','tools.simvla.condition_solver_rb2','cell','--row',row,'--student-steps',str(nfe)],
        upstream_status_file=str(incoming/'pipeline_status.json'),
        ready_file=str(incoming/(init+'_10k')/arm/'READY.json'),
        summary=str(output/'completed'/f'{row}.json'),
        completion=dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row))
        for row,(init,arm,k) in ROWS.items()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('all','cell','smoke','worker'), default='all', nargs='?')
    p.add_argument('--preflight', action='store_true'); p.add_argument('--output', type=Path)
    p.add_argument('--row', choices=ROWS); p.add_argument('--suite', default='libero_10', choices=('libero_10',))
    p.add_argument('--student-steps', type=int, choices=(1,2), default=1)
    p.add_argument('--seed', default='seed01', choices=('seed01',)); a = p.parse_args()
    env = environment(0); os.environ.clear(); os.environ.update(env)
    c = read_json(a.output/'runtime_config.json') if a.output else base_config(a.student_steps)
    nfe=c['student_steps']; output,_=solver_paths(nfe)
    configure(c); sys.path.insert(0, c['libero_root']); os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    if a.command in ('smoke','worker'):
        campaign.worker(c, a.output, a.suite, a.seed, a.row, smoke=a.command=='smoke',
            replay_factory=replay_factory, policy_factory=policy_factory,
            policy_checker=lambda policy,row:check_policy(policy),
            compiler_checker=compiler_checker, reset_checker=reset_checker)
        return 0
    if a.command == 'cell': cell(c, a.row); return 0
    campaign.prepare(c, output); write_json(output/'runtime_config.json', c)
    plan = jobs(nfe); write_json(output/'planned_jobs.json', plan)
    if a.preflight:
        print(f'CPU_PREFLIGHT_PASS: eight naive{nfe} rows; predecessor={predecessor(nfe)}', flush=True)
        return 0
    rc = run_queue(output, plan, gpus=(0,), predecessor=predecessor(nfe),
        environment=environment, cwd=ROOT, timeout=24*3600)
    rows = {r:read_json(output/'completed'/f'{r}.json') for r in ROWS
        if (output/'completed'/f'{r}.json').exists()}
    write_json(output/'comparison_summary.json', dict(complete=len(rows)==len(ROWS),rows=rows,scope=c['scope']))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
