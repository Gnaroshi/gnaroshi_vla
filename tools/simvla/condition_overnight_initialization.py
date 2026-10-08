"""Reuse fresh 10K models after matched gradients to isolate initialization."""
import argparse
from pathlib import Path
import threading
import time

from tools.simvla import condition_overnight as matched
from tools.simvla.condition_output_split_eval import load_payload
from tools.simvla.condition_solver_pipeline import solver_paths
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_common import ROOT, environment, identity, read_json, sha, write_json
from tools.simvla.error_compensation_eval import run
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT = matched.OUTPUT.parent / 'fresh_initialization_sd1_seed01_v1'
ROWS = {f'fresh_nfe{n}_{a}': dict(nfe=n, arm=a, mode='detached')
    for n in (1, 2) for a in matched.ARMS}


def check_origin(contract):
    if contract['continuation']['contract'].get('initialization') != 'fresh':
        raise RuntimeError('Initial 3K model was not freshly initialized')


def configuration():
    c = read_json(solver_paths(1)[0] / 'fresh_10k/runtime_config.json')
    models = {}
    for row, spec in ROWS.items():
        source = solver_paths(spec['nfe'])[0] / 'fresh_10k'
        summary = read_json(source / 'train' / spec['arm'] / 'summary.json')
        if (summary['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE' or summary['total_training_steps'] != 10000
                or summary['initialization'] != 'continuation' or sha(summary['checkpoint']) != summary['checkpoint_sha256']):
            raise RuntimeError('Fresh checkpoint contract mismatch: ' + row)
        payload = load_payload(summary['checkpoint'], spec['arm'], summary['identity'], steps=7000,
            action_mode=f'naive{spec["nfe"]}')
        check_origin(payload['contract'])
        control = read_json(solver_paths(spec['nfe'])[0] / 'pretrained_10k/train' / spec['arm'] / 'training_contract.json')
        for key in ('arm', 'training_intervals', 'action_mode', 'teacher_steps', 'data', 'heldout',
                'batch_size', 'seed', 'condition_loss', 'action_loss', 'condition_weight',
                'sample_step_offset', 'steps', 'total_training_steps', 'optimizer'):
            if payload['contract'][key] != control[key]:
                raise RuntimeError('Initialization control differs: ' + key)
        models[row] = dict(**spec, path=summary['checkpoint'], sha256=summary['checkpoint_sha256'],
            source_identity=summary['identity'])
    c.update(output=str(OUTPUT), models=models, run_label='fresh_initialization_sd1',
        smoke_policy_actions=41, evaluation_condition_intervals=[4],
        extra_source_files=sorted(set(c['extra_source_files'] + [
            'tools/simvla/condition_overnight.py', 'tools/simvla/condition_overnight_initialization.py'])),
        training_description='Evaluation only. Four completed fresh 10K NFE1/2 models; reuse pretrained control evaluations from matched_gradient_sd1_seed01_v1.',
        evaluation_plan='Four K4 x 500 Long paired episodes; same sd1 protocol. Isolate initial weights under equal new-training budget. Historical pretrained 150K cost reported separately.')
    prepare(c)
    write_json(OUTPUT / 'runtime_config.json', c)
    return c


def jobs(c):
    plan = []
    for smoke in (True, False):
        for row in ROWS:
            name = ('smoke_' if smoke else 'kc4_') + row
            plan.append(dict(id=name, deps=[] if smoke else ['smoke_' + row],
                cmd=[c['python'], '-m', 'tools.simvla.condition_overnight_initialization', '--worker', row]
                    + (['--smoke'] if smoke else []),
                summary=str(OUTPUT / ('eval_smoke' if smoke else 'online') / f'kc{8 if smoke else 4}_{row}' / 'summary.json'),
                completion=dict(verdict='SMOKE_PASS' if smoke else 'EVALUATION_COMPLETE',
                    episodes=1 if smoke else 500, identity=identity(c))))
    return plan


def report():
    rows = {}
    comparisons = {}
    run_id = read_json(OUTPUT / 'contract.json')['identity']
    for row, spec in ROWS.items():
        folder = OUTPUT / 'online' / f'kc4_{row}'
        episodes = [read_json(p) for p in (folder / 'episodes').glob('*.json')]
        if any(e['identity'] != run_id or e['row'] != row or e['k_c'] != 4 for e in episodes):
            raise RuntimeError('Mixed episode provenance: ' + row)
        actions = sum(e['episode_length'] for e in episodes)
        rows[row] = dict(episodes=len(episodes), successes=sum(e['success'] for e in episodes),
            policy_ms_per_action=sum(e['policy_ms_total'] for e in episodes) / actions if actions else None,
            wall_seconds=sum(e['wall_seconds'] for e in episodes))
        reference = matched.OUTPUT / 'online' / f'kc4_detached_nfe{spec["nfe"]}_{spec["arm"]}' / 'summary.json'
        if (folder / 'summary.json').is_file() and reference.is_file():
            x, y = read_json(folder / 'summary.json'), read_json(reference)
            comparisons[row] = dict(fresh=x, pretrained=y,
                fresh_minus_pretrained_sr_pp=100 * (x['success_rate'] - y['success_rate']),
                fresh_minus_pretrained_ms_per_action=x['policy_ms_per_action'] - y['policy_ms_per_action'])
    state_path = OUTPUT / 'queue_status.json'
    state = read_json(state_path) if state_path.exists() else {}
    now = time.time()
    start = now
    parent = matched.OUTPUT / 'live_comparison.json'
    if state.get('phase') == 'waiting_for_predecessor' and parent.exists():
        start = max(now, matched.datetime.fromisoformat(read_json(parent)['estimated_finish_kst']).timestamp())
    eta = matched.estimate_remaining({'kc4_' + r: d for r, d in rows.items()},
        state.get('active', {}), state.get('pending', []), start)
    eta['remaining_hours'] += (start - now) / 3600
    data = dict(updated_kst=matched.datetime.fromtimestamp(now, matched.KST).isoformat(timespec='seconds'),
        rows=rows, comparisons=comparisons, queue=state, **eta,
        timing_scope='sd1 eager only; keep rb2 compiled results separate')
    write_json(OUTPUT / 'live_comparison.json', data)
    print(f'FOLLOWUP_ETA finish={eta["estimated_finish_kst"]} remaining={eta["remaining_hours"]:.2f}h', flush=True)
    if now >= matched.datetime.fromisoformat(matched.REVIEW_AT).timestamp() and not (OUTPUT / 'morning_snapshot.json').exists():
        write_json(OUTPUT / 'morning_snapshot.json', data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--worker', choices=tuple(ROWS)); p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.worker:
        run(read_json(OUTPUT / 'runtime_config.json'), a.worker, smoke=a.smoke, k_c=8 if a.smoke else 4,
            policy_factory=matched.policy_factory, counter_checker=matched.counter_checker)
        return 0
    c = configuration()
    if a.preflight:
        print('PREFLIGHT_PASS: four fresh models, matched pretrained controls reused', flush=True)
        return 0
    stop = threading.Event()
    def monitor():
        while not stop.wait(60):
            try: report()
            except Exception as exc: print('REPORT_WARNING ' + repr(exc), flush=True)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    try:
        return run_queue(OUTPUT, jobs(c), gpus=(4, 5, 6, 7),
            predecessor=dict(path=str(matched.OUTPUT), lock='queue.lock', allow_when_all_assigned=True),
            environment=lambda gpu: environment(c, gpu), cwd=ROOT, timeout=12 * 3600)
    finally:
        stop.set(); thread.join(timeout=30); report()


if __name__ == '__main__':
    raise SystemExit(main())
