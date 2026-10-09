"""Compiled evaluation of fresh NFE1 noise-supervision models on rb2."""
import argparse
import os
from pathlib import Path
import sys

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT, configure, read_json, write_json, sha
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.condition_gradient_rb2 import base_config as previous_config, OUTPUT as PREDECESSOR
from tools.simvla.condition_noise_pipeline import VARIANTS, ARM
from tools.simvla.condition_output_split_rb2 import STORAGE, environment, replay_factory, compiler_checker, reset_checker
from tools.simvla.condition_output_split_eval import load_payload, attach, check_policy
from tools.simvla.rollout_repair_rb2 import evaluate
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT = STORAGE/'results/simvla/condition_output_split/fresh_noise_nfe1_compiled_seed01_v1'
INCOMING = STORAGE/'incoming/simvla_condition_noise'
ROWS = {f'{name}_k{k}': (name, k) for k in (4,8) for name in VARIANTS}


def base_config():
    c = previous_config()
    c.update(output=str(OUTPUT), long_rows=list(ROWS), campaign_module='tools.simvla.condition_noise_rb2',
        extra_source_files=sorted(set(c['extra_source_files']+['tools/simvla/condition_noise_rb2.py'])),
        scope='Fresh NFE1 for all10K; detached/joint action gradients x one/two training noises; 736130 params and same inference. Four models x K4/K8 x500 paired seed01 Long episodes; compiled RTX5090 latency.')
    return c


def ready_spec(name):
    directory = INCOMING/name/ARM
    if read_json(directory/'READY.json')['manifest_sha256'] != sha(directory/'manifest.json'):
        raise RuntimeError('Non-atomic transfer')
    m = read_json(directory/'manifest.json')
    if m['arm'] != ARM or m['step'] != 10000 or sha(directory/'model.pt') != m['checkpoint_sha256']:
        raise RuntimeError('Checkpoint transfer mismatch')
    p = load_payload(directory/'model.pt', ARM, m['source_identity'], steps=10000, action_mode='naive1')
    mode, count = VARIANTS[name]
    if (p['contract']['initialization'] != 'fresh' or p['contract']['sample_step_offset'] != 0
            or p['contract']['action_gradient_mode'] != mode or p['contract']['action_noise_samples'] != count):
        raise RuntimeError('Training objective/initialization mismatch')
    return dict(path=str(directory/'model.pt'), sha256=m['checkpoint_sha256'], arm=ARM, step=10000,
        source_identity=m['source_identity'], student_steps=1, variant=name)


def policy_factory(replay, c, row, manifest):
    name, k = ROWS[row]; spec = c['model_checkpoint']
    if spec['variant'] != name or sha(spec['path']) != spec['sha256']:
        raise RuntimeError('Wrong checkpoint')
    payload = load_payload(spec['path'], ARM, spec['source_identity'], steps=10000, action_mode='naive1')
    policy = attach_policy(replay, c, 'condition_naive3', manifest)
    policy.nfe = 1
    return attach(policy, replay.native, payload, ARM, k, replay.compiler)


def cell(c, row):
    name, k = ROWS[row]; spec = ready_spec(name)
    result = evaluate({**c, 'student_steps': 1, 'action_mode': 'condition_naive1',
        'condition_interval': k}, row, spec, output=OUTPUT)
    write_json(OUTPUT/'completed'/f'{row}.json', dict(verdict='EVALUATION_COMPLETE', episodes=500,
        row=row, checkpoint_sha256=spec['sha256'], result=result))


def jobs():
    return [dict(id=row, cmd=[sys.executable, '-u', '-m', 'tools.simvla.condition_noise_rb2', 'cell', '--row', row],
        upstream_status_file=str(INCOMING/'pipeline_status.json'), ready_file=str(INCOMING/name/ARM/'READY.json'),
        summary=str(OUTPUT/'completed'/f'{row}.json'),
        completion=dict(verdict='EVALUATION_COMPLETE', episodes=500, row=row))
        for row, (name, _) in ROWS.items()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('all','cell','smoke','worker'), default='all', nargs='?')
    p.add_argument('--preflight', action='store_true'); p.add_argument('--output', type=Path)
    p.add_argument('--row', choices=ROWS); p.add_argument('--suite', default='libero_10', choices=('libero_10',))
    p.add_argument('--seed', default='seed01', choices=('seed01',)); a = p.parse_args()
    env = environment(0); os.environ.clear(); os.environ.update(env)
    c = read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c); sys.path.insert(0,c['libero_root']); os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    if a.command in ('smoke','worker'):
        campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',
            replay_factory=replay_factory,policy_factory=policy_factory,
            policy_checker=lambda policy,row:check_policy(policy),compiler_checker=compiler_checker,reset_checker=reset_checker)
        return 0
    if a.command == 'cell':
        cell(c,a.row)
        return 0
    campaign.prepare(c,OUTPUT); write_json(OUTPUT/'runtime_config.json',c)
    plan = jobs(); write_json(OUTPUT/'planned_jobs.json',plan)
    if a.preflight:
        print('PREFLIGHT_PASS: eight new rows after existing rb2 joint-gradient queue',flush=True)
        return 0
    rc = run_queue(OUTPUT,plan,gpus=(0,),predecessor=dict(path=str(PREDECESSOR),lock='queue.lock'),
        environment=environment,cwd=ROOT,timeout=24*3600)
    rows = {r:read_json(OUTPUT/'completed'/f'{r}.json') for r in ROWS if (OUTPUT/'completed'/f'{r}.json').exists()}
    write_json(OUTPUT/'comparison_summary.json',dict(complete=len(rows)==len(ROWS),rows=rows,scope=c['scope']))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
