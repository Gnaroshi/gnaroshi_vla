"""Frozen SimVLA and accelerators on official PRO Long Position/Task variants."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

from tools.simvla import compiled_campaign as campaign, condition_nfe_sweep as nfe
from tools.simvla.compile_benchmark import ROOT, DEFAULT_CONFIG, Replay, configure, read_json, write_json
from tools.simvla.compiled_policy import attach_policy, check_policy, check_reset
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.libero_pro_assets import ASSETS, STORAGE, SUITES, verify

OUTPUT = STORAGE/'results/simvla/libero_pro/long_position_task_seed01_v1'
ROWS = ('baseline', 'naive_nfe3', 'condition_naive3', 'ours_kc2_ng3',
        'bridge_f2_nfe3', 'bridge_f3_nfe3')


def configuration():
    return {**read_json(DEFAULT_CONFIG), **read_json(campaign.CONFIG),
        'output': str(OUTPUT), 'manifest_root': str(OUTPUT/'manifests'),
        'libero_root': str(ASSETS/'upstream'), 'libero_config': str(ASSETS/'runtime_config'),
        'benchmark_suites': list(SUITES), 'long_rows': list(ROWS), 'other_rows': [], 'other_suites': [],
        'seeds': ['seed01'], 'smoke_episodes': 1, 'smoke_actions': 41, 'warmup_actions': 40,
        'campaign_module': 'tools.simvla.libero_pro_zero_shot', 'new_training': False,
        'extra_source_files': ['tools/simvla/libero_pro_zero_shot.py', 'tools/simvla/libero_pro_assets.py',
            'tools/simvla/condition_nfe_sweep.py', 'tools/simvla/gpu_followup_queue.py',
            'architectures/simvla/wrappers/run_libero_pro_zero_shot_rb2.sh',
            str(ASSETS/'asset_contract.json')]
            + [str(p) for p in sorted((ASSETS/'upstream/libero').rglob('*.py'))],
        'scope': 'Zero-shot Long Position and Task variants; six frozen methods; 500 paired episodes per cell; seed01; no PRO tuning. SimVLA H10/R5, wait10, max900 retained. Not a full PRO benchmark or an online-disturbance response test.'}


def make_manifest(template, suite, assets):
    m = {k: v for k, v in template.items() if k != 'manifest_sha256'}
    m.update(suite=suite, max_policy_actions=900, pro_tasks=assets['tasks'][suite],
        pro_code_commit=assets['code_commit'], pro_data_revision=assets['data_revision'])
    m['episodes'] = [{**e, 'suite': suite} for e in template['episodes']]
    m['manifest_sha256'] = campaign.digest(m)
    return m


def validate_manifest(m, suite, seed):
    if suite not in SUITES or seed != 'seed01':
        raise RuntimeError('Unexpected PRO suite/seed')
    if campaign.digest({k: v for k, v in m.items() if k != 'manifest_sha256'}) != m['manifest_sha256']:
        raise RuntimeError('PRO manifest hash changed')
    # Reuse every Long control/noise/trial check while preserving PRO identity.
    standard = {k: v for k, v in m.items() if k != 'manifest_sha256'}
    standard['suite'] = 'libero_10'
    standard['episodes'] = [{**e, 'suite': 'libero_10'} for e in m['episodes']]
    standard['manifest_sha256'] = campaign.digest(standard)
    campaign.validate_manifest(standard, 'libero_10', seed)
    if m['suite'] != suite or any(e['suite'] != suite for e in m['episodes']):
        raise RuntimeError('PRO episode suite mismatch')
    assets = verify()
    if (m['pro_tasks'] != assets['tasks'][suite] or m['pro_code_commit'] != assets['code_commit']
            or m['pro_data_revision'] != assets['data_revision']):
        raise RuntimeError('PRO prompt/state/source identity changed')
    return m


class NumericSuite:
    def __init__(self, manifest):
        from libero.libero import benchmark
        self.inner = benchmark.get_benchmark_dict()[manifest['suite']]()
        self.tasks = manifest['pro_tasks']

    def get_task(self, index):
        task, spec = self.inner.get_task(index), self.tasks[index]
        if task.name != spec['name']:
            raise RuntimeError('Official PRO task order differs from manifest')
        return task._replace(language=spec['language'])

    def get_task_init_states(self, index):
        import numpy as np
        return np.load(self.tasks[index]['numeric_states'], allow_pickle=False)


def replay_factory(c, row, compiler, samples):
    return nfe.replay_factory(c, row, compiler, samples) if row.startswith('bridge_') else Replay(c, row, compiler, samples)


def policy_factory(replay, c, row, manifest):
    return nfe.make_policy(replay, c, row, manifest) if row.startswith('bridge_') else attach_policy(replay, c, row, manifest)


def policy_checker(policy, row):
    return nfe.check_policy(policy, row) if row.startswith('bridge_') else check_policy(policy, row)


def compiler_checker(compiler, row):
    return nfe.check_compiler(compiler, row) if row.startswith('bridge_') else campaign.check_compiler(compiler, row)


def recover(suite, row, smoke=False):
    directory = OUTPUT/('smoke' if smoke else 'rows')/suite/'seed01'/row
    m = read_json(OUTPUT/'manifests'/suite/'seed01/episode_manifest.json')
    specs = sorted(m['episodes'], key=lambda e: (-e['task_id'], e['trial_id']))
    if smoke:
        specs = specs[:1]
    identity = campaign.digest(dict(campaign=campaign.digest(read_json(OUTPUT/'campaign_contract.json')),
        suite=suite, seed='seed01', row=row, smoke=smoke))
    return campaign.summarize_cell(directory, identity, [(e['task_id'], e['trial_id']) for e in specs])


def marker(suite, row):
    return OUTPUT/'completed'/f'{suite}__{row}.json'


def record(suite, row, result):
    write_json(marker(suite, row), dict(verdict='ROW_COMPLETE', suite=suite, row=row, episodes=500,
        result=result, source=str(OUTPUT/'rows'/suite/'seed01'/row/'summary.json')))


def run_cell(c, suite, row):
    for smoke in (True, False):
        if recover(suite, row, smoke):
            continue
        directory = OUTPUT/('smoke' if smoke else 'rows')/suite/'seed01'/row
        if list((directory/'episodes').glob('*.json')):
            archive = OUTPUT/'failed_attempts'/f'{suite}_{row}_{"smoke" if smoke else "worker"}'
            if archive.exists():
                raise RuntimeError('Two partial attempts preserved; inspect before retrying')
            archive.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(archive)
        lease = os.environ.get('GNAROSHI_GPU_LEASE_FD')
        proc = subprocess.run([c['python'], '-u', '-m', 'tools.simvla.libero_pro_zero_shot',
            'smoke' if smoke else 'worker', '--suite', suite, '--row', row], cwd=ROOT,
            pass_fds=(() if lease is None else (int(lease),)))
        if not recover(suite, row, smoke):
            raise RuntimeError(f'Episodes incomplete; child rc={proc.returncode}')
    record(suite, row, recover(suite, row))


def summarize(c):
    results = [read_json(marker(s, r)) for s in SUITES for r in ROWS if marker(s, r).exists()]
    write_json(OUTPUT/'comparison_summary.json', dict(complete=len(results)==12, rows=results,
        scope=c['scope'], episodes_requested=6000, new_training=False))
    with (OUTPUT/'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['suite', 'method', 'successes', 'episodes', 'SR_percent',
            'policy_ms_per_action', 'timing_valid_episodes', 'source'])
        for value in results:
            r = value['result']
            writer.writerow([value['suite'], value['row'], r['successes'], r['episodes'],
                100*r['success_rate'], r['pooled_policy_ms_per_action'], r['timing_valid_episodes'], value['source']])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('all', 'preflight', 'cell', 'smoke', 'worker'), nargs='?', default='all')
    p.add_argument('--suite', choices=SUITES)
    p.add_argument('--row', choices=ROWS)
    args = p.parse_args()
    child = args.command in ('cell', 'smoke', 'worker')
    if child and (not args.suite or not args.row):
        p.error('Child needs --suite and --row')
    c = read_json(OUTPUT/'runtime_config.json') if child else configuration()
    configure(c)
    sys.path.insert(0, c['libero_root'])
    os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    verify()
    if args.command in ('smoke', 'worker'):
        campaign.worker(c, OUTPUT, args.suite, 'seed01', args.row, smoke=args.command=='smoke',
            replay_factory=replay_factory, policy_factory=policy_factory, policy_checker=policy_checker,
            compiler_checker=compiler_checker, reset_checker=check_reset,
            manifest_validator=validate_manifest, suite_factory=NumericSuite)
        return 0
    if args.command == 'cell':
        run_cell(c, args.suite, args.row)
        summarize(c)
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = verify()
    original = read_json(campaign.CONFIG)
    template = campaign.validate_manifest(read_json(campaign.manifest_path(original, 'libero_10', 'seed01')), 'libero_10', 'seed01')
    for suite in SUITES:
        m = make_manifest(template, suite, assets)
        path = campaign.manifest_path(c, suite, 'seed01')
        if path.exists() and read_json(path) != m:
            raise RuntimeError('Existing PRO manifest differs')
        write_json(path, m)
    campaign.prepare(c, OUTPUT, manifest_validator=validate_manifest)
    write_json(OUTPUT/'runtime_config.json', c)
    for suite in SUITES:
        for row in ROWS:
            result = recover(suite, row)
            if result:
                record(suite, row, result)
            elif marker(suite, row).exists():
                raise RuntimeError('Unverifiable completion marker')
    summarize(c)
    if args.command == 'preflight':
        print('PRO_CPU_PREFLIGHT_PASS: 12 cells x 500 episodes; no training or GPU evaluation started', flush=True)
        return 0
    plan = [dict(id=f'{suite}__{row}', cmd=[c['python'], '-u', '-m', 'tools.simvla.libero_pro_zero_shot',
        'cell', '--suite', suite, '--row', row], summary=str(marker(suite, row)),
        completion=dict(verdict='ROW_COMPLETE', suite=suite, row=row, episodes=500)) for suite in SUITES for row in ROWS]
    def environment(gpu):
        return {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'MUJOCO_EGL_DEVICE_ID': str(gpu),
            'PYTHONHASHSEED': str(campaign.SEEDS['seed01'][0])}
    rc = run_queue(OUTPUT, plan, gpus=(0,), predecessor=nfe.OUTPUT,
        predecessor_lock_name='queue.lock', environment=environment, cwd=ROOT)
    summarize(c)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
