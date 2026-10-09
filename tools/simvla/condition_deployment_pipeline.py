"""Matched refresh-interval training from the same completed 10K model."""
import argparse
from pathlib import Path
import subprocess

from tools.simvla.condition_solver_pipeline import OUTPUT as SOLVER
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_common import ROOT, environment, identity, read_json, sha, write_json
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.observation_correction_pipeline import export_arm

ARM = 'carry_base'
MODES = {'mixed': [4, 8], 'k2': [2], 'k3': [3], 'k4': [4]}
ROWS = {f'{mode}_k{k}': (mode, k) for k in (2, 3, 4) for mode in ('mixed', f'k{k}')}
PARENT_SHA = '25d5336ce4ee4df7eae47e5cd1ea2bd6afd6f77d82262b2b0abb058a99b86938'
OUTPUT = SOLVER.parent/'deployment_interval_nfe1_seed01_v1'
PREDECESSOR = SOLVER.parent/'observation_feature_nfe1_seed01_v1'
DEST = 'rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_condition_deployment'
EXTRA = ['tools/simvla/condition_deployment_pipeline.py', 'tools/simvla/condition_deployment_rb2.py',
         'architectures/simvla/wrappers/run_condition_deployment.sh']


def configurations():
    source = SOLVER/'fresh_10k'
    previous = read_json(source/'runtime_config.json')
    summary_path = source/'train'/ARM/'summary.json'
    summary = read_json(summary_path)
    if (summary['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE' or summary['total_training_steps'] != 10000
            or summary['steps'] != 7000 or summary['checkpoint_sha256'] != PARENT_SHA
            or sha(summary['checkpoint']) != PARENT_SHA):
        raise RuntimeError('The common 92.8% fresh10K parent changed')
    configs = {}
    for mode, intervals in MODES.items():
        c = {**previous, 'output': str(OUTPUT/mode), 'steps': 5000, 'warmup_steps': 250,
            'sample_step_offset': 10000, 'initialization': 'continuation',
            'continuation_source_steps': 7000, 'continuation_source_action_mode': 'naive1',
            'initial_models': {ARM: dict(summary=str(summary_path), identity=summary['identity'])},
            'training_intervals': intervals, 'offline_validation_intervals': [2, 3, 4],
            'interval_transition': dict(source=[4, 8], target=intervals),
            'evaluation_condition_intervals': [2, 3, 4] if mode == 'mixed' else intervals,
            'training_condition_ages': list(range(1, max(intervals))), 'training_k_c': max(intervals),
            'run_label': 'condition_deployment_'+mode, 'rb2_destination': DEST+'/'+mode,
            'extra_source_files': sorted(set(previous['extra_source_files']+EXTRA)),
            'training_description': 'Same selected fresh10K carry_base parent, detached action-head inputs, original NFE10 teacher, NFE1 student. Four 5K continuations: mixedK4/K8 control or fixedK2/3/4. Same windows10001..15000, batch2, AdamW1e-4 warmup250 cosine. Unroll ages and thus training FLOPs differ; report wall time. Architecture736130 unchanged. No SR gate.',
            'evaluation_plan': 'Mixed model at K2/3/4 and each specialist at its own K: six paired Long500 seed01 rows, separately on sd1 eager and rb2 compiled. Preceding fixed-parent vs LB comparison is prioritized on rb2.'}
        c.pop('solver_transition', None)
        prepare(c)
        write_json(Path(c['output'])/'runtime_config.json', c)
        configs[mode] = c
    return configs


def jobs(configs):
    result = []
    for mode, c in configs.items():
        out = Path(c['output'])
        def add(kind, module, extra, summary, completion, deps):
            key = mode+'_'+kind
            result.append(dict(id=key, cmd=[c['python'], '-u', '-m', module,
                '--config', str(out/'runtime_config.json'), '--arm', ARM]+extra,
                summary=str(summary), completion=dict(identity=identity(c), **completion), deps=deps))
            return key
        smoke = add('smoke_train', 'tools.simvla.condition_output_split_train', ['--smoke'],
            out/'smoke'/ARM/'summary.json', dict(verdict='SMOKE_PASS', steps=14), [])
        k = max(MODES[mode])
        env = add('smoke_env', 'tools.simvla.condition_output_split_eval', ['--smoke', '--k-c', str(k)],
            out/'eval_smoke'/f'kc{k}_{ARM}'/'summary.json', dict(verdict='SMOKE_PASS', episodes=1), [smoke])
        trained = add('train', 'tools.simvla.condition_output_split_train', [],
            out/'train'/ARM/'summary.json', dict(verdict='TRAIN_AND_OFFLINE_COMPLETE', steps=5000), [env])
        add('export', 'tools.simvla.condition_deployment_pipeline', ['--export'],
            out/'exports'/f'{ARM}.json', dict(verdict='BUNDLE_EXPORTED'), [trained])
    for row, (mode, k) in ROWS.items():
        c = configs[mode]; out = Path(c['output'])
        result.append(dict(id=row, deps=[mode+'_train'],
            cmd=[c['python'], '-u', '-m', 'tools.simvla.condition_output_split_eval',
                '--config', str(out/'runtime_config.json'), '--arm', ARM, '--k-c', str(k)],
            summary=str(out/'online'/f'kc{k}_{ARM}'/'summary.json'),
            completion=dict(identity=identity(c), verdict='EVALUATION_COMPLETE', episodes=500)))
    return result


def transfer_status(status):
    host, remote = DEST.split(':', 1)
    for attempt in range(2):
        try:
            subprocess.run(['ssh', host, 'mkdir', '-p', remote], check=True, timeout=30)
            subprocess.run(['rsync', '-a', str(status), host+':'+remote+'/'], check=True, timeout=60)
            return
        except (subprocess.SubprocessError, OSError) as exc:
            print(f'STATUS_TRANSFER_WARNING attempt={attempt+1}: {exc}', flush=True)


def summarize(configs):
    rows, training = {}, {}
    for row, (mode, k) in ROWS.items():
        path = Path(configs[mode]['output'])/'online'/f'kc{k}_{ARM}'/'summary.json'
        if path.exists():
            rows[row] = read_json(path)
    for mode, c in configs.items():
        path = Path(c['output'])/'train'/ARM/'summary.json'
        if path.exists():
            training[mode] = read_json(path)
    write_json(OUTPUT/'comparison_summary.json', dict(complete=len(rows)==6, rows=rows,
        training=training, parent_sha256=PARENT_SHA, hardware='sd1 RTX3090 eager'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preflight', action='store_true')
    p.add_argument('--export', action='store_true'); p.add_argument('--config'); p.add_argument('--arm', choices=(ARM,))
    a = p.parse_args()
    if a.export:
        export_arm(read_json(a.config), a.arm)
        return 0
    configs = configurations(); plan = jobs(configs)
    write_json(OUTPUT/'planned_jobs.json', plan)
    if a.preflight:
        print('PREFLIGHT_PASS: common10K parent, four matched5K continuations, six Long500 rows', flush=True)
        return 0
    status = OUTPUT/'pipeline_status.json'; write_json(status, dict(phase='running')); transfer_status(status)
    rc = 1
    try:
        rc = run_queue(OUTPUT, plan, gpus=(4, 5, 6, 7),
            predecessor=dict(path=str(PREDECESSOR), lock='queue.lock', allow_when_all_assigned=True),
            environment=lambda gpu: environment(configs['mixed'], gpu), cwd=ROOT, timeout=12*3600)
    finally:
        summarize(configs)
        write_json(status, dict(phase='complete' if not rc else 'finished_with_failures')); transfer_status(status)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
