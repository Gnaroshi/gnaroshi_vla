"""Frozen Large Latent Bridge with the same paired noise and a 3-step solver."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
from types import MethodType

from tools.simvla import compiled_campaign as campaign
from tools.simvla.bridge_interval_sweep import replay_factory, check_compiler as bridge_compiler
from tools.simvla.compile_benchmark import ROOT, DEFAULT_CONFIG, configure, read_json, write_json
from tools.simvla.compiled_policy import attach_policy, check_reset
from tools.simvla.gpu_followup_queue import run_queue

STORAGE = Path('/home/mingyujung/private/gnaroshi_vla_storage')
OUTPUT = STORAGE / 'results/simvla/latent_bridge/compiled_naive3_seed01_v1'
PREDECESSOR = STORAGE / 'results/simvla/observation_correction/mixed_k4_k8_compiled_seed01_v1'
ROWS = tuple(f'bridge_f{k}_naive3' for k in (2, 3, 4))


def interval(row):
    if row not in ROWS:
        raise ValueError(row)
    return int(row.split('_')[1][1:])


def expected_counts(row, queries):
    k = interval(row)
    full = (queries + k - 1) // k
    return dict(num_full_vlm_calls=full, num_condition_updater_calls=queries-full,
        num_latent_bridge_calls=queries-full, num_action_transformer_calls=3*queries,
        num_generation_decoder_only_steps=0)


def make_policy(replay, c, row, manifest):
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval import SynchronizedNaiveNFE3Policy
    policy = attach_policy(replay, c, 'latent_bridge_f2', manifest)
    policy.row_name = row
    policy.refresh_every = interval(row)
    policy.NFE = 3
    # Keep flow_steps=10: it is part of the original paired-noise key.
    # Use the existing baseline NFE3 decoder, not a new solver implementation.
    policy._decode = MethodType(SynchronizedNaiveNFE3Policy._decode, policy)
    if policy.flow_steps != 10 or replay.loop is not None:
        raise RuntimeError('Noise contract changed or Generation Loop was loaded')
    return policy


def check_policy(policy, row):
    q = int(policy.metrics.counters['num_policy_queries'])
    for key, value in expected_counts(row, q).items():
        if int(policy.metrics.counters.get(key, 0)) != value:
            raise RuntimeError(f'{row}: {key} != {value}')
    if q != (policy.step_index + 4)//5 or policy.flow_steps != 10:
        raise RuntimeError('Action queue/noise contract changed')


def check_compiler(compiler, row):
    bridge_compiler(compiler, f'latent_bridge_f{interval(row)}')


def configuration():
    return {**read_json(DEFAULT_CONFIG), **read_json(campaign.CONFIG),
        'output': str(OUTPUT), 'long_rows': list(ROWS), 'other_rows': [], 'other_suites': [],
        'seeds': ['seed01'], 'smoke_episodes': 1, 'smoke_actions': 41, 'warmup_actions': 40,
        'campaign_module': 'tools.simvla.bridge_coarse_followup',
        'extra_source_files': ['tools/simvla/bridge_coarse_followup.py',
            'tools/simvla/bridge_interval_sweep.py', 'tools/simvla/gpu_followup_queue.py'],
        'scope': 'Frozen Large Latent Bridge f2/f3/f4 + original naive3 Euler decoder; same original10 noise key; H10/R5; 500 paired episodes each; no retraining; development seed01.',
        'solver_steps': 3, 'paired_noise_key_flow_steps': 10}


def recover_cell(c, row, smoke):
    root = OUTPUT / ('smoke' if smoke else 'rows') / 'libero_10/seed01' / row
    manifest = read_json(OUTPUT/'manifests/libero_10/seed01/episode_manifest.json')
    specs = sorted(manifest['episodes'], key=lambda x: (-x['task_id'], x['trial_id']))
    if smoke:
        specs = specs[:1]
    key = campaign.digest(dict(campaign=campaign.digest(read_json(OUTPUT/'campaign_contract.json')),
        suite='libero_10', seed='seed01', row=row, smoke=smoke))
    return campaign.summarize_cell(root, key, [(s['task_id'], s['trial_id']) for s in specs])


def run_cell(c, row):
    for smoke in (True, False):
        if recover_cell(c, row, smoke):
            continue
        directory = OUTPUT / ('smoke' if smoke else 'rows') / 'libero_10/seed01' / row
        if list((directory/'episodes').glob('*.json')):
            archive = OUTPUT/'failed_attempts'/f'{row}_{"smoke" if smoke else "worker"}'
            if archive.exists():
                raise RuntimeError('Two partial attempts preserved; inspect before rerunning')
            archive.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(archive)
        # Separate CUDA/compiler lifetimes for the short smoke and production.
        lease = os.environ.get('GNAROSHI_GPU_LEASE_FD')
        result = subprocess.run([c['python'], '-u', '-m', 'tools.simvla.bridge_coarse_followup',
            'smoke' if smoke else 'worker', '--row', row], cwd=ROOT,
            pass_fds=(() if lease is None else (int(lease),)))
        # Recover a complete episode table even if shutdown logging failed.
        if not recover_cell(c, row, smoke):
            raise RuntimeError(f'Missing episode completion; returncode={result.returncode}')
    result = recover_cell(c, row, False)
    write_json(OUTPUT/'completed'/f'{row}.json', dict(verdict='ROW_COMPLETE', row=row,
        episodes=500, solver_steps=3, **{'result': result}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('all', 'preflight', 'cell', 'smoke', 'worker'), nargs='?', default='all')
    p.add_argument('--row', choices=ROWS)
    a = p.parse_args()
    child = a.command in ('cell', 'smoke', 'worker')
    if child and a.row is None:
        p.error('--row is required for a child command')
    c = read_json(OUTPUT/'runtime_config.json') if child else configuration()
    configure(c)
    sys.path.insert(0, c['libero_root'])
    os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    if a.command in ('smoke', 'worker'):
        campaign.worker(c, OUTPUT, 'libero_10', 'seed01', a.row, smoke=a.command=='smoke',
            replay_factory=replay_factory, policy_factory=make_policy, policy_checker=check_policy,
            compiler_checker=check_compiler, reset_checker=check_reset)
        return 0
    if a.command == 'cell':
        run_cell(c, a.row)
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    campaign.prepare(c, OUTPUT)
    write_json(OUTPUT/'runtime_config.json', c)
    plan = [dict(id=row, cmd=[c['python'], '-u', '-m', __spec__.name, 'cell', '--row', row],
        summary=str(OUTPUT/'completed'/f'{row}.json'),
        completion=dict(verdict='ROW_COMPLETE', row=row, episodes=500, solver_steps=3)) for row in ROWS]
    if a.command == 'preflight':
        print('CPU_PREFLIGHT_PASS: f2/f3/f4, 500 episodes each, original noise, no retraining', flush=True)
        return 0
    def environment(gpu):
        return {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'MUJOCO_EGL_DEVICE_ID': str(gpu),
            'PYTHONHASHSEED': str(campaign.SEEDS['seed01'][0])}
    rc = run_queue(OUTPUT, plan, gpus=(0,), predecessor=PREDECESSOR, environment=environment, cwd=ROOT)
    reports = {row: read_json(OUTPUT/'completed'/f'{row}.json')['result'] for row in ROWS
               if (OUTPUT/'completed'/f'{row}.json').exists()}
    write_json(OUTPUT/'comparison_summary.json', dict(complete=len(reports)==3, rows=reports,
        solver_steps=3, seed='seed01', scope=c['scope']))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
