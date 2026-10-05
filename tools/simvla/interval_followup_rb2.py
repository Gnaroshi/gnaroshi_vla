"""Fixed-checkpoint interval and action-head controls after the bridge sweep."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import sys
import time

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import Replay, configure, read_json, sha, write_json
from tools.simvla.rollout_repair_rb2 import base_config as previous_config
from tools.simvla.rollout_repair_rb2 import evaluate, make_policy, STORAGE
from tools.simvla.rollout_round2_rb2 import bundle, wait_idle
from tools.simvla.trend_compiled_rb2 import reset_policy

OUTPUT = STORAGE / 'results/simvla/trend_condition/post_bridge_controls_seed01_v1'
PREDECESSOR = STORAGE / 'results/simvla/latent_bridge/compiled_k1_k8_seed01_v1'
INCOMING = STORAGE / 'incoming/simvla_rollout_round2'
CHECKPOINT_SHA256 = 'f010bbb00ebed8a3bfef7f0a768a6b50f34e5650abf9624c88f27d75c011823e'
ROWS = {
    'previous_joint_k4_generation3': (4, 'ours_kc2_ng3'),
    'previous_joint_k4_naive3': (4, 'condition_naive3'),
    'previous_joint_k4_full10': (4, 'condition_nfe10'),
    'previous_joint_k5_generation3': (5, 'ours_kc2_ng3'),
    'previous_joint_k6_generation3': (6, 'ours_kc2_ng3'),
}


def selected_checkpoint():
    ready = bundle(INCOMING)
    if not ready:
        raise RuntimeError('Second-round checkpoint bundle missing')
    item = ready['checkpoints']['previous_joint']
    if item['sha256'] != CHECKPOINT_SHA256 or ready['selected_arm'] != 'frozen_trend_residual':
        raise RuntimeError('Preselected checkpoint identity changed')
    return dict(path=str(INCOMING / item['file']), sha256=CHECKPOINT_SHA256,
                arm=ready['selected_arm'], step=3000)


def base_config():
    c = previous_config()
    c.update(output=str(OUTPUT), long_rows=list(ROWS), seeds=['seed01'],
             other_suites=[], other_rows=[], smoke_episodes=1, smoke_actions=41,
             warmup_actions=40, campaign_module='tools.simvla.interval_followup_rb2',
             extra_source_files=c['extra_source_files'] + [
                 'tools/simvla/interval_followup_rb2.py', 'tools/simvla/rollout_round2_rb2.py'],
             selection=dict(variant='previous_joint', checkpoint_sha256=CHECKPOINT_SHA256,
                 evidence='sd1 K4: 469/500; best of four completed round2 development candidates',
                 reused_k8_summary=str(STORAGE / 'results/simvla/trend_condition/'
                     'rollout_round2_compiled_seed01_v1/online/previous_joint_k8/'
                     'rows/libero_10/seed01/previous_joint_k8/summary.json')),
             scope='Fixed condition checkpoint trained with Generation3. K4/K5/K6 test refresh spacing; '
                   'K4 naive3/full10 are inference interventions, not separately optimized training. '
                   'Development seed01 selected using sd1 results; not independent confirmation. '
                   'No new training, no success-rate cutoff; H10/R5, identical 500 episode manifest.')
    return c


def expected_counts(row, queries, *, rows=ROWS):
    k, mode = rows[row]
    full = (queries + k - 1) // k
    return dict(num_full_vlm_calls=full, num_condition_updater_calls=queries-full,
                num_action_transformer_calls=queries*(10 if mode == 'condition_nfe10' else 3),
                num_generation_decoder_only_steps=7*queries if mode == 'ours_kc2_ng3' else 0,
                num_trend_head_calls=full, num_observation_encoder_calls=queries-full)


def check_policy(policy, row, *, rows=ROWS):
    queries = int(policy.metrics.counters['num_policy_queries'])
    for name, value in expected_counts(row, queries, rows=rows).items():
        if int(policy.metrics.counters.get(name, 0)) != value:
            raise RuntimeError(f'{row}: {name} mismatch')
    if queries != (policy.step_index + 4) // 5:
        raise RuntimeError('H10/R5 query cadence changed')


def check_compiler(compiler, row, *, rows=ROWS):
    required = {'vlm', 'action_transformer', 'trend_head', 'observation_encoder', 'condition_updater'}
    if rows[row][1] == 'ours_kc2_ng3':
        required |= {'action_decoder', 'generation_updater'}
    missing = [name for name in required if not compiler.records.get(name, {}).get('graphs', 0)]
    if missing:
        raise RuntimeError('Compile bypass: ' + str(missing))


def replay_factory(c, row, compiler, samples, *, rows=ROWS):
    if c['action_mode'] != rows[row][1] or c['condition_interval'] != rows[row][0]:
        raise RuntimeError('Worker config and declared row differ')
    return Replay(c, rows[row][1], compiler, samples)


def predecessor_busy():
    # The bridge pipeline holds this lock during all its rows AND its own wait.
    with (PREDECESSOR / 'pipeline.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def preflight(c):
    spec = selected_checkpoint()
    import torch
    payload = torch.load(spec['path'], map_location='cpu', weights_only=False)
    if payload['arm'] != spec['arm'] or payload['step'] != 3000 or payload['contract']['k_c'] != 8:
        raise RuntimeError('Checkpoint training contract differs')
    del payload
    campaign.prepare(c, OUTPUT)
    write_json(OUTPUT / 'preflight.json', dict(verdict='CPU_PREFLIGHT_PASS', gpu_jobs_started=False,
        checkpoint=spec, rows=list(ROWS), predecessor=str(PREDECESSOR),
        predecessor_busy=predecessor_busy()))
    print('CPU_PREFLIGHT_PASS; no GPU allocation', flush=True)
    return spec


def save_summary(c, results, failures):
    write_json(OUTPUT / 'combined_summary.json', dict(complete=len(results) == len(ROWS),
        results=results, failures=failures, scope=c['scope'], selection=c['selection'],
        suite='libero_10', seed='seed01', episodes_per_row=500))
    with (OUTPUT / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row', 'K_C', 'action_mode', 'successes', 'episodes',
                         'SR_percent', 'policy_ms_per_action', 'timing_valid_episodes'])
        for result in results:
            writer.writerow([result['row'], *ROWS[result['row']], result['successes'],
                result['episodes'], 100*result['success_rate'], result['pooled_policy_ms_per_action'],
                result['timing_valid_episodes']])


def run_all(c):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = preflight(c)
        while predecessor_busy():
            write_json(OUTPUT / 'pipeline_status.json', dict(phase='waiting_for_predecessor',
                gpu_used=False, predecessor=str(PREDECESSOR), remaining_rows=list(ROWS)))
            print('WAIT: complete Latent Bridge K1..8 queue; no GPU allocation', flush=True)
            time.sleep(30)
        write_json(OUTPUT / 'predecessor_completion.json', read_json(PREDECESSOR / 'pipeline_status.json'))
        results, failures = [], []
        for row, (k, mode) in ROWS.items():
            try:
                write_json(OUTPUT / 'pipeline_status.json', dict(phase='waiting_for_gpu', row=row))
                wait_idle()
                write_json(OUTPUT / 'pipeline_status.json', dict(phase='evaluation', row=row))
                result = evaluate({**c, 'action_mode': mode, 'condition_interval': k}, row, spec, output=OUTPUT)
                results.append(dict(row=row, **result))
                print(f"DONE {row}: {result['successes']}/{result['episodes']}; "
                      f"{result['pooled_policy_ms_per_action']:.4f} ms/action", flush=True)
            except Exception as exc:
                failures.append(dict(row=row, error=str(exc)))
                print(f'ROW_FAILED {row}: {exc}', flush=True)
            save_summary(c, results, failures)
        write_json(OUTPUT / 'pipeline_status.json', dict(
            phase='complete' if not failures else 'finished_with_failures', failures=failures))
        return int(bool(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('all', 'preflight', 'smoke', 'worker'), default='all', nargs='?')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--suite', default='libero_10', choices=('libero_10',))
    parser.add_argument('--seed', default='seed01', choices=('seed01',))
    parser.add_argument('--row', choices=ROWS)
    args = parser.parse_args()
    c = read_json(args.output / 'runtime_config.json') if args.output else base_config()
    configure(c)
    sys.path.insert(0, c['libero_root'])
    os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    if args.command == 'preflight':
        preflight(c)
        return 0
    if args.command == 'all':
        try:
            return run_all(c)
        except BlockingIOError:
            print('ALREADY_RUNNING; existing queue unchanged', flush=True)
            return 1
        except BaseException as exc:
            write_json(OUTPUT / 'pipeline_status.json', dict(phase='failed', error=str(exc)))
            raise
    campaign.worker(c, args.output, args.suite, args.seed, args.row, smoke=args.command == 'smoke',
        replay_factory=replay_factory, policy_factory=make_policy, policy_checker=check_policy,
        compiler_checker=check_compiler, reset_checker=reset_policy)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
