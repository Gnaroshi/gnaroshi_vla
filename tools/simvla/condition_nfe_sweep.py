"""Frozen condition methods with native one/two/three-step action generation."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType

from tools.simvla import compiled_campaign as campaign
from tools.simvla.bridge_interval_sweep import compatible_contract, SOURCE_EQUIVALENCE
from tools.simvla.compile_benchmark import (
    ROOT, DEFAULT_CONFIG, Replay, configure, read_json, write_json, sha,
)
from tools.simvla.compile_checks import required_components
from tools.simvla.compiled_policy import attach_policy, check_reset
from tools.simvla.gpu_followup_queue import run_queue

STORAGE = Path('/home/mingyujung/private/gnaroshi_vla_storage')
OUTPUT = STORAGE / 'results/simvla/condition_nfe/compiled_seed01_v1'
PREDECESSOR = STORAGE / 'results/simvla/observation_correction/mixed_k4_k8_compiled_seed01_v1'
GROUPS = {
    'condition_k2': ('condition_naive3', 2),
    'baseline': ('naive_nfe3', 1),
    'bridge_f2': ('latent_bridge_f2', 2),
    'bridge_f3': ('latent_bridge_f2', 3),
}
ROWS = tuple(f'{group}_nfe{nfe}' for nfe in (1, 2, 3) for group in GROUPS)
REFERENCES = {
    'baseline_nfe3': ('compiled_paper/three_seed_v2', 'naive_nfe3'),
    'condition_k2_nfe3': ('compiled_paper/three_seed_v2', 'condition_naive3'),
    'bridge_f2_nfe3': ('latent_bridge/compiled_naive3_seed01_v1', 'bridge_f2_naive3'),
    'bridge_f3_nfe3': ('latent_bridge/compiled_naive3_seed01_v1', 'bridge_f3_naive3'),
}


def specification(row):
    if row not in ROWS:
        raise ValueError(row)
    group, suffix = row.rsplit('_nfe', 1)
    base, interval = GROUPS[group]
    return group, base, interval, int(suffix)


def expected_counts(row, queries):
    group, _, interval, nfe = specification(row)
    full = (queries + interval - 1) // interval
    return dict(num_full_vlm_calls=full, num_condition_updater_calls=queries-full,
        num_latent_bridge_calls=queries-full if group.startswith('bridge') else 0,
        num_action_transformer_calls=nfe*queries,
        num_action_transformer_decodes=queries, num_generation_decoder_only_steps=0)


def replay_factory(c, row, compiler, samples):
    return Replay(c, specification(row)[1], compiler, samples)


def make_policy(replay, c, row, manifest):
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval import SynchronizedNaiveNFE3Policy
    group, base, interval, nfe = specification(row)
    if replay.loop is not None:
        raise RuntimeError('Generation Loop must not be loaded')
    policy = attach_policy(replay, c, base, manifest)
    policy.row_name = row
    if group.startswith('bridge'):
        policy.refresh_every = interval
    policy.NFE = nfe
    policy.nfe = nfe
    # Change only the original Euler step count, not the paired-noise key.
    policy._decode = MethodType(SynchronizedNaiveNFE3Policy._decode, policy)
    if policy.flow_steps != 10:
        raise RuntimeError('Original paired-noise key must retain flow_steps=10')
    return policy


def check_policy(policy, row):
    q = int(policy.metrics.counters['num_policy_queries'])
    for name, expected in expected_counts(row, q).items():
        if int(policy.metrics.counters.get(name, 0)) != expected:
            raise RuntimeError(f'{row}: {name} != {expected}')
    if q != (policy.step_index + 4)//5 or policy.flow_steps != 10:
        raise RuntimeError('H10/R5 queue or original paired-noise key changed')


def check_compiler(compiler, row):
    missing = [name for name in required_components(specification(row)[1])
        if not compiler.records.get(name, {}).get('graphs', 0)]
    if missing:
        raise RuntimeError('Compile bypass: ' + str(missing))
    if compiler.records.get('generation_updater', {}).get('graphs', 0):
        raise RuntimeError('Generation Loop unexpectedly executed')


def configuration():
    return {**read_json(DEFAULT_CONFIG), **read_json(campaign.CONFIG),
        'output': str(OUTPUT), 'long_rows': list(ROWS), 'other_rows': [], 'other_suites': [],
        'seeds': ['seed01'], 'smoke_episodes': 1, 'smoke_actions': 41, 'warmup_actions': 40,
        'campaign_module': 'tools.simvla.condition_nfe_sweep',
        'extra_source_files': ['tools/simvla/condition_nfe_sweep.py',
            'tools/simvla/bridge_interval_sweep.py', 'tools/simvla/gpu_followup_queue.py',
            'architectures/simvla/wrappers/run_condition_nfe_sweep_rb2.sh'],
        'scope': 'Frozen checkpoints; no Generation Loop; native Euler NFE=1,2,3; original condition, Ours K2 condition, Large Latent Bridge f2/f3; LIBERO-Long 500 paired episodes per cell; development seed01; not retrained for each solver.',
        'paired_noise_key_flow_steps': 10, 'new_training': False}


def validate_reference(directory, old_contract, old_row):
    result = read_json(directory/'summary.json')
    identity = campaign.digest(dict(campaign=campaign.digest(old_contract), suite='libero_10',
        seed='seed01', row=old_row, smoke=False))
    if (result.get('identity') != identity or result.get('episodes') != 500
            or result.get('verdict') != 'EPISODES_COMPLETE'
            or result.get('timing_valid_episodes') != 500):
        raise RuntimeError('Reference identity/completion/timing mismatch')
    with (directory/'outcomes.csv').open() as stream:
        records = list(csv.DictReader(stream))
    pairs = {(int(r['task_id']), int(r['trial_id'])) for r in records}
    if len(records) != 500 or pairs != {(t, i) for t in range(10) for i in range(50)}:
        raise RuntimeError('Reference trial IDs differ')
    successes = sum(int(r['success']) for r in records)
    actions = sum(int(r['episode_length']) for r in records)
    ms = sum(float(r['policy_ms_total']) for r in records)
    if (successes != result['successes'] or result['success_rate'] != successes/500
            or actions != result['executed_actions']
            or any(r['timing_valid'] != 'True' for r in records)
            or abs(ms/actions - result['pooled_policy_ms_per_action']) > 1e-9):
        raise RuntimeError('Reference aggregation differs from episode table')
    return result


def reuse_reference(row, contract):
    if row not in REFERENCES:
        return None, ['no identical completed reference']
    run, old_row = REFERENCES[row]
    old_root = STORAGE/'results/simvla'/run
    directory = old_root/'rows/libero_10/seed01'/old_row
    if not (directory/'summary.json').exists():
        return None, ['reference missing']
    old = read_json(old_root/'campaign_contract.json')
    issues = compatible_contract(old, contract)
    if issues:
        return None, issues
    # SOURCE_EQUIVALENCE also accepts an added optional token_feature argument.
    # Its default is None; the frozen native adapter never supplies it. The
    # original arithmetic and weights are unchanged for this Condition row.
    result = validate_reference(directory, old, old_row)
    return dict(result=result, reused=True, source=str(directory/'summary.json'),
        source_sha256=sha(directory/'summary.json'), outcomes_sha256=sha(directory/'outcomes.csv')), []


def recover_cell(row, smoke):
    root = OUTPUT/('smoke' if smoke else 'rows')/'libero_10/seed01'/row
    manifest = read_json(OUTPUT/'manifests/libero_10/seed01/episode_manifest.json')
    specs = sorted(manifest['episodes'], key=lambda x: (-x['task_id'], x['trial_id']))
    if smoke:
        specs = specs[:1]
    identity = campaign.digest(dict(campaign=campaign.digest(read_json(OUTPUT/'campaign_contract.json')),
        suite='libero_10', seed='seed01', row=row, smoke=smoke))
    return campaign.summarize_cell(root, identity, [(x['task_id'], x['trial_id']) for x in specs])


def record_completion(row, report):
    write_json(OUTPUT/'completed'/f'{row}.json', dict(verdict='ROW_COMPLETE', row=row,
        episodes=500, nfe=specification(row)[3], **report))


def run_cell(c, row):
    for smoke in (True, False):
        if recover_cell(row, smoke):
            continue
        directory = OUTPUT/('smoke' if smoke else 'rows')/'libero_10/seed01'/row
        if list((directory/'episodes').glob('*.json')):
            archive = OUTPUT/'failed_attempts'/f'{row}_{"smoke" if smoke else "worker"}'
            if archive.exists():
                raise RuntimeError('Two partial attempts preserved; inspect before rerunning')
            archive.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(archive)
        lease = os.environ.get('GNAROSHI_GPU_LEASE_FD')
        result = subprocess.run([c['python'], '-u', '-m', 'tools.simvla.condition_nfe_sweep',
            'smoke' if smoke else 'worker', '--row', row], cwd=ROOT,
            pass_fds=(() if lease is None else (int(lease),)))
        if not recover_cell(row, smoke):
            raise RuntimeError(f'Episode completion missing; returncode={result.returncode}')
    record_completion(row, dict(result=recover_cell(row, False), reused=False,
        source=str(OUTPUT/'rows/libero_10/seed01'/row/'summary.json')))


def summarize(c):
    rows = {row: read_json(OUTPUT/'completed'/f'{row}.json') for row in ROWS
        if (OUTPUT/'completed'/f'{row}.json').exists()}
    write_json(OUTPUT/'comparison_summary.json', dict(complete=len(rows)==len(ROWS), rows=rows,
        seed='seed01', generation_loop=False, scope=c['scope']))
    with (OUTPUT/'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['condition', 'refresh_interval', 'NFE', 'successes', 'episodes',
            'SR_percent', 'policy_ms_per_action', 'timing_valid_episodes', 'reused', 'source'])
        for row, value in rows.items():
            group, _, k, nfe = specification(row)
            r = value['result']
            writer.writerow([group, k, nfe, r['successes'], r['episodes'], 100*r['success_rate'],
                r['pooled_policy_ms_per_action'], r['timing_valid_episodes'], value['reused'], value['source']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('all', 'preflight', 'cell', 'smoke', 'worker'), nargs='?', default='all')
    parser.add_argument('--row', choices=ROWS)
    args = parser.parse_args()
    child = args.command in ('cell', 'smoke', 'worker')
    if child and args.row is None:
        parser.error('--row is required')
    c = read_json(OUTPUT/'runtime_config.json') if child else configuration()
    configure(c)
    sys.path.insert(0, c['libero_root'])
    os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    if args.command in ('smoke', 'worker'):
        campaign.worker(c, OUTPUT, 'libero_10', 'seed01', args.row, smoke=args.command=='smoke',
            replay_factory=replay_factory, policy_factory=make_policy, policy_checker=check_policy,
            compiler_checker=check_compiler, reset_checker=check_reset)
        return 0
    if args.command == 'cell':
        run_cell(c, args.row)
        summarize(c)
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = campaign.prepare(c, OUTPUT)
    write_json(OUTPUT/'runtime_config.json', c)
    reuse = {}
    for row in ROWS:
        reference, issues = reuse_reference(row, contract)
        reuse[row] = dict(reused=reference is not None, issues=issues)
        if reference:
            record_completion(row, reference)
            print(f'REUSE {row}: {reference["result"]["successes"]}/500', flush=True)
        else:
            # A stale completion marker never bypasses episode validation.
            existing = recover_cell(row, False)
            if existing:
                record_completion(row, dict(result=existing, reused=False,
                    source=str(OUTPUT/'rows/libero_10/seed01'/row/'summary.json')))
            elif (OUTPUT/'completed'/f'{row}.json').exists():
                raise RuntimeError(f'Unverifiable completion marker: {row}')
    write_json(OUTPUT/'reference_validation.json', dict(rows=reuse,
        audited_source_equivalence=SOURCE_EQUIVALENCE,
        condition_equivalence='Optional token_feature=None added; not supplied by the frozen native adapter. Existing native forward arithmetic remains unchanged.'))
    summarize(c)
    plan = [dict(id=row, cmd=[c['python'], '-u', '-m', 'tools.simvla.condition_nfe_sweep', 'cell', '--row', row],
        summary=str(OUTPUT/'completed'/f'{row}.json'),
        completion=dict(verdict='ROW_COMPLETE', row=row, episodes=500, nfe=specification(row)[3])) for row in ROWS]
    if args.command == 'preflight':
        print('CPU_PREFLIGHT_PASS: native NFE=1,2,3; no Generation Loop; 500 episodes per cell', flush=True)
        return 0
    def environment(gpu):
        return {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'MUJOCO_EGL_DEVICE_ID': str(gpu),
            'PYTHONHASHSEED': str(campaign.SEEDS['seed01'][0])}
    rc = run_queue(OUTPUT, plan, gpus=(0,), predecessor=PREDECESSOR, environment=environment, cwd=ROOT)
    summarize(c)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
