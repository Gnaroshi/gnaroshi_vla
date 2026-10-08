"""Matched sd1 efficacy queue; rb2 retains the paper-latency confirmation."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time

from methods.latentloop.modules.condition_output_split import ARMS
from tools.simvla.condition_gradient_pipeline import OUTPUT as TRAINING
from tools.simvla.condition_output_split_eval import attach, check_policy, load_payload
from tools.simvla.condition_output_split_train import check_gradient_control
from tools.simvla.condition_solver_pipeline import solver_paths
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_common import ROOT, environment, identity, read_json, sha, write_json
from tools.simvla.error_compensation_eval import make_policy as original_policy, run
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT = TRAINING.parent / 'matched_gradient_sd1_seed01_v1'
KST = timezone(timedelta(hours=9))
REVIEW_AT = '2026-10-09T10:00:00+09:00'


def row_specs():
    return {f'{mode}_nfe{nfe}_{arm}': dict(mode=mode, nfe=nfe, arm=arm)
        for nfe in (1, 2) for arm in ARMS for mode in ('detached', 'joint')}


def configuration():
    c = read_json(TRAINING / 'nfe1/runtime_config.json')
    models = {}
    payloads = {}
    for row, spec in row_specs().items():
        source = (TRAINING / f'nfe{spec["nfe"]}' if spec['mode'] == 'joint'
            else solver_paths(spec['nfe'])[0] / 'pretrained_10k')
        summary = read_json(source / 'train' / spec['arm'] / 'summary.json')
        if (summary['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE'
                or summary['total_training_steps'] != 10000
                or sha(summary['checkpoint']) != summary['checkpoint_sha256']):
            raise RuntimeError('Incomplete or changed checkpoint: ' + row)
        payload = load_payload(summary['checkpoint'], spec['arm'], summary['identity'],
            steps=7000, action_mode=f'naive{spec["nfe"]}')
        if payload['contract'].get('action_gradient_mode', 'detached') != spec['mode']:
            raise RuntimeError('Wrong gradient mode: ' + row)
        payloads[row] = payload['contract']
        models[row] = dict(**spec, path=summary['checkpoint'], sha256=summary['checkpoint_sha256'],
            source_identity=summary['identity'], training_summary=str(source / 'train' / spec['arm'] / 'summary.json'))
    for nfe in (1, 2):
        for arm in ARMS:
            check_gradient_control(payloads[f'joint_nfe{nfe}_{arm}'], payloads[f'detached_nfe{nfe}_{arm}'])
    c.update(output=str(OUTPUT), models=models, run_label='matched_gradient_sd1',
        smoke_policy_actions=41, evaluation_condition_intervals=[4, 8],
        extra_source_files=sorted(set(c['extra_source_files'] + ['tools/simvla/condition_overnight.py'])),
        training_description='Evaluation only: reuse eight completed matched 10K models (shared 3K plus 7K continuation). No new training or cache.',
        evaluation_plan='16 rows x 500 paired Long episodes; detached/joint x carry_output/carry_base x NFE1/2 x K4/8. Same sd1 eager protocol. RTX5090 confirmation remains in existing rb2 queues.')
    prepare(c)
    write_json(OUTPUT / 'runtime_config.json', c)
    return c


def policy_factory(c, row, *, smoke=False, k_c=4):
    m = c['models'][row]
    if sha(m['path']) != m['sha256']:
        raise RuntimeError('Checkpoint changed: ' + row)
    payload = load_payload(m['path'], m['arm'], m['source_identity'], steps=7000,
        action_mode=f'naive{m["nfe"]}')
    if payload['contract'].get('action_gradient_mode', 'detached') != m['mode']:
        raise RuntimeError('Gradient mode changed: ' + row)
    policy = original_policy(c, f'condition_naive{m["nfe"]}', k_c=min(k_c, 4))
    return attach(policy, policy.native_v0, payload, m['arm'], k_c)


def counter_checker(policy, row, actual, k_c):
    expected = check_policy(policy)
    if (actual.get('condition', 0) != expected['condition_updater']
            or actual.get('transformer', 0) != expected['transformer']):
        raise RuntimeError('Independent module hooks disagree')
    return expected


def jobs(c):
    plan = []
    for row in row_specs():
        plan.append(dict(id='smoke_' + row, deps=[], cmd=[c['python'], '-m',
            'tools.simvla.condition_overnight', '--worker', row, '--k-c', '8', '--smoke'],
            summary=str(OUTPUT / 'eval_smoke' / f'kc8_{row}' / 'summary.json'),
            completion=dict(verdict='SMOKE_PASS', episodes=1, identity=identity(c))))
    for k in (4, 8):
        for row in row_specs():
            plan.append(dict(id=f'kc{k}_{row}', deps=['smoke_' + row],
                cmd=[c['python'], '-m', 'tools.simvla.condition_overnight', '--worker', row, '--k-c', str(k)],
                summary=str(OUTPUT / 'online' / f'kc{k}_{row}' / 'summary.json'),
                completion=dict(verdict='EVALUATION_COMPLETE', episodes=500, identity=identity(c))))
    return plan


def estimate_remaining(rows, active, pending, now, gpus=4):
    """Use empirical per-episode wall time, retaining a cold-start allowance."""
    def cost(name):
        r = rows.get(name, {})
        count = r.get('episodes', 0)
        # Prior sd1 K4 rows took 1.8-2.0h; K8 failures took 3.6-4.2h.
        seconds = r.get('wall_seconds', 0) / count if count >= 20 else (15 if name.startswith('kc4') else 29)
        return max(0, 500 - count) * seconds + (90 if count == 0 else 0)
    loads = [cost(v['job']) for v in active.values() if not v['job'].startswith('smoke_')]
    loads += [0] * max(0, gpus - len(loads))
    for name in pending:
        if name.startswith('smoke_'):
            continue
        i = min(range(len(loads)), key=loads.__getitem__)
        loads[i] += cost(name)
    remain = max(loads, default=0)
    return dict(remaining_hours=remain / 3600,
        estimated_finish_kst=datetime.fromtimestamp(now + remain, KST).isoformat(timespec='minutes'),
        estimate_scope='Episode wall-time extrapolation, >=20 episodes or historical K4/K8 prior; excludes future GPU contention and retries.')


def report(c):
    state_path = OUTPUT / 'queue_status.json'
    state = read_json(state_path) if state_path.exists() else {}
    run_id = read_json(OUTPUT / 'contract.json')['identity']
    rows = {}
    for k in (4, 8):
        for row in row_specs():
            name = f'kc{k}_{row}'
            directory = OUTPUT / 'online' / name
            episodes = [read_json(p) for p in sorted((directory / 'episodes').glob('*.json'))]
            if any(e['identity'] != run_id or e['row'] != row or e['k_c'] != k for e in episodes):
                raise RuntimeError('Mixed episode identity: ' + name)
            actions = sum(e['episode_length'] for e in episodes)
            rows[name] = dict(episodes=len(episodes), successes=sum(e['success'] for e in episodes),
                success_rate=sum(e['success'] for e in episodes) / len(episodes) if episodes else None,
                policy_ms_per_action=sum(e['policy_ms_total'] for e in episodes) / actions if actions else None,
                wall_seconds=sum(e['wall_seconds'] for e in episodes),
                complete=(directory / 'summary.json').is_file())
    comparisons = {}
    for k in (4, 8):
        for nfe in (1, 2):
            for arm in ARMS:
                a = rows[f'kc{k}_detached_nfe{nfe}_{arm}']
                b = rows[f'kc{k}_joint_nfe{nfe}_{arm}']
                if a['complete'] and b['complete']:
                    comparisons[f'kc{k}_nfe{nfe}_{arm}'] = dict(
                        success_difference_percentage_points=100 * (b['success_rate'] - a['success_rate']),
                        latency_difference_ms_per_action=b['policy_ms_per_action'] - a['policy_ms_per_action'])
    now = time.time()
    data = dict(updated_kst=datetime.fromtimestamp(now, KST).isoformat(timespec='seconds'),
        review_at=REVIEW_AT, rows=rows, comparisons=comparisons, queue=state,
        host_axis='sd1 RTX3090 eager, single seed, H10/R5, Long 500; keep separate from rb2 compiled paper latency',
        **estimate_remaining(rows, state.get('active', {}), state.get('pending', []), now))
    write_json(OUTPUT / 'live_comparison.json', data)
    if now >= datetime.fromisoformat(REVIEW_AT).timestamp() and not (OUTPUT / 'morning_snapshot.json').exists():
        write_json(OUTPUT / 'morning_snapshot.json', data)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--worker', choices=tuple(row_specs()))
    p.add_argument('--k-c', type=int, choices=(4, 8))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--status', action='store_true')
    a = p.parse_args()
    if a.worker:
        if a.k_c is None:
            p.error('--worker requires --k-c')
        run(read_json(OUTPUT / 'runtime_config.json'), a.worker, smoke=a.smoke, k_c=a.k_c,
            policy_factory=policy_factory, counter_checker=counter_checker)
        return 0
    if a.status:
        print(report(read_json(OUTPUT / 'runtime_config.json')), flush=True)
        return 0
    c = configuration()
    plan = jobs(c)
    if a.preflight:
        print('PREFLIGHT_PASS: 8 matched checkpoints, 8 short smokes, 16 x 500 paired evaluations', flush=True)
        return 0
    stop = threading.Event()
    def monitor():
        while not stop.wait(60):
            try:
                d = report(c)
                n = sum(r['complete'] for r in d['rows'].values())
                print(f'ETA complete={n}/16 finish={d["estimated_finish_kst"]} remaining={d["remaining_hours"]:.2f}h', flush=True)
            except Exception as exc:
                print('REPORT_WARNING ' + repr(exc), flush=True)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        return run_queue(OUTPUT, plan, gpus=(4, 5, 6, 7),
            predecessor=dict(path=str(TRAINING), lock='queue.lock', allow_when_all_assigned=True),
            environment=lambda gpu: environment(c, gpu), cwd=ROOT, timeout=12 * 3600)
    finally:
        stop.set(); thread.join(timeout=30)
        report(c)


if __name__ == '__main__':
    raise SystemExit(main())
