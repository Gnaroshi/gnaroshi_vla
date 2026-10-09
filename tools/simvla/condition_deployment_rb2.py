"""Prioritize missing LB/parent comparisons, then matched-interval models."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT, configure, read_json, sha, write_json
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.condition_deployment_pipeline import ARM, MODES, ROWS, PARENT_SHA
from tools.simvla.condition_output_split_eval import attach, check_policy, load_payload
from tools.simvla.condition_output_split_rb2 import environment, replay_factory, compiler_checker, reset_checker
from tools.simvla.condition_solver_rb2 import base_config as solver_config
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.rollout_repair_rb2 import STORAGE, evaluate

OUTPUT = STORAGE/'results/simvla/condition_output_split/deployment_interval_nfe1_compiled_seed01_v1'
PREDECESSOR = OUTPUT.parent/'joint_action_gradient_compiled_seed01_v1'
INCOMING = STORAGE/'incoming/simvla_condition_deployment'
COMPARISON_ROOT = ROOT.parent/'simvla_condition_interval_compare'
COMPARISON_OUTPUT = OUTPUT.parent/'bridge_interval_nfe1_compiled_seed01_v1'
PRIORITY = ('ours_k2', 'ours_k3', 'bridge_k4')
COMPARISON_REVISION = '6982a5a'


def base_config():
    c = solver_config()
    c.update(output=str(OUTPUT), long_rows=list(ROWS),
        campaign_module='tools.simvla.condition_deployment_rb2',
        extra_source_files=c['extra_source_files']+['tools/simvla/condition_deployment_rb2.py',
            'tools/simvla/condition_deployment_pipeline.py'], student_steps=1,
        scope='Same fresh10K parent plus5K: mixedK4/K8 versus K2/3/4-specific continuations. Six500-episode seed01 rows, NFE1, no Generation Loop, H10/R5. Compile all policy components. Reuse prior baseline/LB; original missing comparison gets first priority.')
    return c


def verify_comparison_source():
    commit = subprocess.check_output(['git', '-C', str(COMPARISON_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(COMPARISON_ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if not commit.startswith(COMPARISON_REVISION) or dirty.strip():
        raise RuntimeError('Priority comparison source is not the reviewed clean revision')
    return commit


def run_comparison(row=None):
    verify_comparison_source()
    env = environment(0); env['PYTHONPATH'] = str(COMPARISON_ROOT)
    fd = os.environ.get('GNAROSHI_GPU_LEASE_FD')
    cmd = [sys.executable, '-u', '-m', 'tools.simvla.condition_interval_compare']
    cmd += ['cell', '--row', row] if row else ['--preflight']
    subprocess.run(cmd, cwd=COMPARISON_ROOT, env=env, check=True,
        pass_fds=(int(fd),) if fd else ())


def validate_training_contract(contract, mode):
    if (contract['training_intervals'] != MODES[mode]
            or contract['total_training_steps'] != 15000 or contract['sample_step_offset'] != 10000
            or contract['action_mode'] != 'naive1' or contract['teacher_steps'] != 10
            or contract['initialization'] != 'continuation'
            or contract['continuation']['checkpoint_sha256'] != PARENT_SHA
            or contract['interval_transition'] != dict(source=[4,8], target=MODES[mode])):
        raise RuntimeError('Interval/solver/parent/training-budget mismatch')
    old = contract['continuation']['contract']
    for key in ('data', 'heldout', 'batch_size', 'seed', 'condition_weight',
                'current_action_gradient', 'future_condition_gradient', 'source_checkpoint_sha256'):
        if contract[key] != old[key]:
            raise RuntimeError('Matched objective changed: '+key)


def ready_spec(mode):
    directory = INCOMING/mode/ARM
    if read_json(directory/'READY.json')['manifest_sha256'] != sha(directory/'manifest.json'):
        raise RuntimeError('Non-atomic model transfer')
    m = read_json(directory/'manifest.json')
    if (m['arm'] != ARM or m['step'] != 5000 or m['parameters'] != 736130
            or sha(directory/'model.pt') != m['checkpoint_sha256']):
        raise RuntimeError('Transferred model changed')
    p = load_payload(directory/'model.pt', ARM, m['source_identity'], steps=5000, action_mode='naive1')
    validate_training_contract(p['contract'], mode)
    return dict(path=str(directory/'model.pt'), sha256=m['checkpoint_sha256'], arm=ARM,
        step=5000, total_training_steps=15000, source_identity=m['source_identity'],
        parent_sha256=PARENT_SHA, training_intervals=MODES[mode], student_steps=1)


def policy_factory(replay, c, row, manifest):
    mode, k = ROWS[row]; spec = c['model_checkpoint']
    if sha(spec['path']) != spec['sha256']:
        raise RuntimeError('Checkpoint changed')
    payload = load_payload(spec['path'], ARM, spec['source_identity'], steps=5000, action_mode='naive1')
    validate_training_contract(payload['contract'], mode)
    policy = attach_policy(replay, c, 'condition_naive3', manifest)
    policy.NFE = policy.nfe = 1
    policy = attach(policy, replay.native, payload, ARM, k, replay.compiler)
    policy.row_name = row
    return policy


def summarize():
    rows = {row: read_json(OUTPUT/'completed'/f'{row}.json') for row in ROWS
            if (OUTPUT/'completed'/f'{row}.json').exists()}
    reference = COMPARISON_OUTPUT/'comparison_summary.json'
    write_json(OUTPUT/'comparison_summary.json', dict(complete=len(rows)==6, rows=rows,
        fixed_parent_and_bridge=read_json(reference) if reference.exists() else None,
        parent_sha256=PARENT_SHA, hardware='rb2 RTX5090 compiled', episodes_per_row=500, seed='seed01'))


def cell(c, row):
    mode, k = ROWS[row]; spec = ready_spec(mode)
    result = evaluate({**c, 'action_mode': 'condition_naive1', 'condition_interval': k}, row, spec, output=OUTPUT)
    write_json(OUTPUT/'completed'/f'{row}.json', dict(verdict='EVALUATION_COMPLETE', episodes=500,
        row=row, checkpoint_sha256=spec['sha256'], result=result))
    summarize()


def jobs():
    plan = [dict(id='reference_'+row,
        cmd=[sys.executable, '-u', '-m', 'tools.simvla.condition_deployment_rb2', 'reference', '--row', row],
        summary=str(COMPARISON_OUTPUT/'completed'/f'{row}.json'),
        completion=dict(verdict='EVALUATION_COMPLETE', episodes=500, row=row)) for row in PRIORITY]
    # Ordered single-GPU dispatch prioritizes references; a technical failure
    # in a reference does not cancel independent trained-model evaluations.
    for row, (mode, _) in ROWS.items():
        plan.append(dict(id=row, cmd=[sys.executable, '-u', '-m', 'tools.simvla.condition_deployment_rb2', 'cell', '--row', row],
            ready_file=str(INCOMING/mode/ARM/'READY.json'), upstream_status_file=str(INCOMING/'pipeline_status.json'),
            summary=str(OUTPUT/'completed'/f'{row}.json'),
            completion=dict(verdict='EVALUATION_COMPLETE', episodes=500, row=row)))
    return plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('all', 'cell', 'reference', 'smoke', 'worker'), nargs='?', default='all')
    p.add_argument('--preflight', action='store_true'); p.add_argument('--output', type=Path)
    p.add_argument('--row', choices=tuple(ROWS)+PRIORITY)
    p.add_argument('--suite', choices=('libero_10',), default='libero_10')
    p.add_argument('--seed', choices=('seed01',), default='seed01'); a = p.parse_args()
    if a.command == 'reference':
        if a.row not in PRIORITY: p.error('Unknown priority row')
        run_comparison(a.row); summarize(); return 0
    if a.command != 'all' and a.row not in ROWS:
        p.error('A trained-model row is required')
    env = environment(0); os.environ.clear(); os.environ.update(env)
    c = read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c); sys.path.insert(0, c['libero_root']); os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    if a.command in ('smoke', 'worker'):
        campaign.worker(c, a.output, a.suite, a.seed, a.row, smoke=a.command=='smoke',
            replay_factory=replay_factory, policy_factory=policy_factory,
            policy_checker=lambda policy, row: check_policy(policy),
            compiler_checker=compiler_checker, reset_checker=reset_checker)
        return 0
    if a.command == 'cell': cell(c, a.row); return 0
    run_comparison()
    campaign.prepare(c, OUTPUT); write_json(OUTPUT/'runtime_config.json', c)
    plan = jobs(); write_json(OUTPUT/'planned_jobs.json', plan); summarize()
    if a.preflight:
        print('PREFLIGHT_PASS: three missing comparisons first, then six matched-interval rows', flush=True)
        return 0
    rc = run_queue(OUTPUT, plan, gpus=(0,), predecessor=dict(path=str(PREDECESSOR), lock='queue.lock'),
        environment=environment, cwd=ROOT, timeout=24*3600)
    summarize()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
