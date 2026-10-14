"""Ordered Long500 evaluation of preserved-reader distillation checkpoints."""
import argparse
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from tools.simvla import failure_interventions as prior
from tools.simvla import gpu_followup_queue as queue
from tools.simvla import visual_memory_followup as predecessor
from tools.simvla import visual_readout_train as training
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

OUTPUT, INCOMING, ARMS = training.OUTPUT, training.INCOMING, training.ARMS
MODULE = 'tools.simvla.visual_readout_campaign'


def manifest_ids(manifest):
    episodes = manifest['episodes']
    ids = [(e['task_id'], e['trial_id']) for e in episodes]
    expected = {(task, trial) for task in range(10) for trial in range(50)}
    if (len(ids) != 500 or len(set(ids)) != 500 or set(ids) != expected
            or any(type(i) is not int for pair in ids for i in pair)):
        raise RuntimeError('Expected the unchanged Long500 task/trial manifest')
    return ids


def validate_contract(c):
    manifest_ids(c['manifest'])
    if tuple(c['arms']) != ARMS or c['steps'] != 10000:
        raise RuntimeError('Unexpected training arms or step count')
    if not set(training.FILES).issubset(c['source_sha256']):
        raise RuntimeError('Missing shared training source hashes')
    return c


def prepare():
    return validate_contract(training.prepare())


def checked():
    return validate_contract(training.checked())


def gpu_ids():
    return (0,) if prior.IS_RB2 else (4, 5, 6, 7)


def shard_count(arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm: ' + str(arm))
    return 1 if prior.IS_RB2 else 4


def selected_manifest(manifest, shard, count):
    manifest_ids(manifest)
    if count not in (1, 4) or not 0 <= shard < count:
        raise ValueError('Invalid shard')
    return dict(manifest, episodes=manifest['episodes'][shard::count])


def shard_output(arm, shard):
    return OUTPUT / 'shards' / arm / f'part{shard}'


def evaluate(arm, shard=0, smoke=False):
    from tools.simvla import visual_midpoint_eval as evaluation

    c = checked()
    local = dict(c, manifest=selected_manifest(c['manifest'], shard, shard_count(arm)))
    old_study, old_write = evaluation.study, evaluation.write_json

    def write(path, value):
        if Path(path).name == 'progress.json':
            value = dict(value, total=len(local['manifest']['episodes']))
        old_write(path, value)

    # Keep the evaluator's parity, vision schedule and compile checks unchanged.
    evaluation.study = SimpleNamespace(
        OUTPUT=shard_output(arm, shard), checked=lambda: local,
        initial_model=training.initial_model, model_payload=training.model_payload,
        previous=training.previous)
    evaluation.write_json = write
    try:
        evaluation.evaluate(arm, smoke=smoke)
    finally:
        evaluation.study, evaluation.write_json = old_study, old_write


def merged_result(c, arm):
    """The visual_memory merge protocol, with explicit manifest and arm checks."""
    expected = manifest_ids(c['manifest'])
    records, parts, paths = [], [], []
    for shard in range(shard_count(arm)):
        out = shard_output(arm, shard) / 'rows' / arm
        summary = read_json(out / 'summary.json')
        if (summary.get('verdict') != 'EVAL_COMPLETE'
                or summary.get('identity') != c['identity'] or summary.get('arm') != arm):
            raise RuntimeError('Incomplete or mixed shard')
        rows = [read_json(p) for p in sorted((out / 'episodes').glob('*.json'))]
        ids = [(r['task_id'], r['trial_id']) for r in rows]
        wanted = set(expected[shard::shard_count(arm)])
        if len(ids) != len(set(ids)) or set(ids) != wanted or len(ids) != summary['episodes']:
            raise RuntimeError('Shard episode mismatch')
        if any(r['identity'] != c['identity'] or r['arm'] != arm for r in rows):
            raise RuntimeError('Mixed episode contract or arm')
        if any(type(r['success']) is not bool for r in rows):
            raise RuntimeError('Invalid episode outcome')
        successes = sum(r['success'] for r in rows)
        if successes != summary['successes'] or summary['success_rate'] != successes / len(rows):
            raise RuntimeError('Shard success count mismatch')
        if any(type(r['timing_valid']) is not bool or type(r['episode_length']) is not int
               or r['episode_length'] <= 0 or not math.isfinite(r['policy_ms_total'])
               or r['policy_ms_total'] < 0 for r in rows):
            raise RuntimeError('Invalid episode timing')
        if parts and any(summary[k] != parts[0][k] for k in ('hardware', 'timing_scope')):
            raise RuntimeError('Mixed shard runtime')
        records.extend(rows)
        parts.append(summary)
        paths.append(str(out / 'summary.json'))
    ids = [(r['task_id'], r['trial_id']) for r in records]
    models = {r['model_sha256'] for r in records}
    if len(ids) != 500 or len(set(ids)) != 500 or set(ids) != set(expected) or len(models) != 1:
        raise RuntimeError('Incomplete Long500 or mixed models')
    by_id = dict(zip(ids, records))
    records = [by_id[key] for key in expected]
    valid = [r for r in records if r['timing_valid']]
    successes = sum(r['success'] for r in records)
    summary = dict(
        verdict='EVAL_COMPLETE', identity=c['identity'], arm=arm, episodes=500,
        successes=successes, success_rate=successes / 500,
        policy_ms_per_action=(sum(r['policy_ms_total'] for r in valid)
                              / sum(r['episode_length'] for r in valid)) if valid else None,
        timing_valid_episodes=len(valid), model_sha256=models.pop(),
        per_task={str(t): sum(r['success'] for r in records if r['task_id'] == t) for t in range(10)},
        hardware=parts[0]['hardware'], shards=shard_count(arm), shard_summaries=paths,
        elapsed_seconds_sum=sum(p['elapsed_seconds'] for p in parts),
        timing_scope=parts[0]['timing_scope'])
    return summary, records


def merge(arm):
    summary, records = merged_result(checked(), arm)
    out = OUTPUT / 'rows' / arm
    write_json(out / 'episode_outcomes.json', records)
    write_json(out / 'summary.json', summary)
    print(summary, flush=True)
    return summary


def jobs(c):
    manifest_ids(c['manifest'])
    result = []

    def add(name, args, path, verdict, deps=(), completion=None, **extra):
        result.append(dict(
            id=name, cmd=[sys.executable, '-u', '-m', MODULE] + args, deps=list(deps),
            summary=str(path), completion=dict(identity=c['identity'], verdict=verdict,
                                               **(completion or {})), **extra))

    if not prior.IS_RB2:
        for arm in ARMS:
            add('train_smoke_' + arm, ['train-smoke', '--arm', arm],
                OUTPUT / 'train_smoke' / arm / 'summary.json', 'TRAIN_SMOKE_PASS',
                completion=dict(steps=2))
            add('train_' + arm, ['train', '--arm', arm],
                OUTPUT / 'train' / arm / 'summary.json', 'TRAIN_COMPLETE',
                ['train_smoke_' + arm], completion=dict(steps=10000))
            add('export_' + arm, ['export', '--arm', arm],
                OUTPUT / 'exports' / f'{arm}.json', 'EXPORTED', ['train_' + arm])
    for index, arm in enumerate(ARMS):
        deps = ['merge_' + ARMS[index - 1]] if index else []
        if not prior.IS_RB2:
            deps.append('train_' + arm)
        extra = dict(ready_file=str(INCOMING / arm / 'READY.json')) if prior.IS_RB2 else {}
        add('eval_smoke_' + arm, ['eval-smoke', '--arm', arm, '--shard', '0'],
            shard_output(arm, 0) / 'eval_smoke' / arm / 'summary.json', 'EVAL_SMOKE_PASS', deps, **extra)
        for shard in range(shard_count(arm)):
            add(f'eval_{arm}_{shard}', ['eval', '--arm', arm, '--shard', str(shard)],
                shard_output(arm, shard) / 'rows' / arm / 'summary.json', 'EVAL_COMPLETE',
                ['eval_smoke_' + arm], completion=dict(episodes=len(c['manifest']['episodes'][shard::shard_count(arm)])),
                **extra)
        add('merge_' + arm, ['merge', '--arm', arm], OUTPUT / 'rows' / arm / 'summary.json',
            'EVAL_COMPLETE', [f'eval_{arm}_{shard}' for shard in range(shard_count(arm))],
            completion=dict(episodes=500))
    return result


def audit(c=None):
    """CPU-only contract/plan/artifact checks, without checkpoint deserialization."""
    c = checked() if c is None else validate_contract(c)
    for source, expected in c['source_sha256'].items():
        if sha(ROOT / source) != expected:
            raise RuntimeError('Source changed: ' + source)
    plan = jobs(c)
    ids = {job['id'] for job in plan}
    if len(ids) != len(plan) or len({job['summary'] for job in plan}) != len(plan):
        raise RuntimeError('Duplicate job IDs or output paths')
    done = set()
    while len(done) < len(plan):
        ready = {job['id'] for job in plan if set(job['deps']).issubset(done)} - done
        if not ready:
            raise RuntimeError('Unknown or cyclic job dependencies')
        done.update(ready)
    rows, transfers = {}, {}
    for arm in ARMS:
        path = OUTPUT / 'rows' / arm / 'summary.json'
        rows[arm] = dict(path=str(path), complete=path.exists())
        if path.exists():
            summary, records = merged_result(c, arm)
            if read_json(path) != summary or read_json(path.parent / 'episode_outcomes.json') != records:
                raise RuntimeError('Merged row differs from shard evidence: ' + arm)
        if prior.IS_RB2:
            directory = INCOMING / arm
            marker = directory / 'READY.json'
            transfers[arm] = 'waiting_for_READY'
            if marker.exists():
                if read_json(marker)['manifest_sha256'] != sha(directory / 'manifest.json'):
                    raise RuntimeError('Incomplete READY manifest: ' + arm)
                spec = read_json(directory / 'manifest.json')
                if (spec['source_sha256'] != c['source_sha256']
                        or sha(directory / 'model.pt') != spec['checkpoint_sha256']):
                    raise RuntimeError('Transferred source or checkpoint changed: ' + arm)
                transfers[arm] = 'hashes_verified'
    result = dict(verdict='AUDIT_PASS', identity=c['identity'], jobs=len(plan),
                  gpus=list(gpu_ids()), evaluation_order=list(ARMS), rows=rows, transfers=transfers,
                  scope='CPU contract, shared sources, job DAG and completed-row evidence; runtime checks run in eval-smoke.')
    write_json(OUTPUT / 'audit' / 'summary.json', result)
    return result


def wait_for_predecessor(timeout=24 * 3600):
    """Require actual followup completion, not merely its last GPU assignment."""
    spec = dict(path=str(predecessor.OUTPUT), lock='queue.lock')
    began, reported = time.monotonic(), None
    while True:
        path = predecessor.OUTPUT / 'queue_status.json'
        if path.exists():
            state = read_json(path)
            if state.get('phase') == 'finished_with_failures':
                raise RuntimeError('visual_memory_followup finished with failures')
            if state.get('phase') == 'complete':
                if (state.get('failed') or state.get('active') or state.get('pending')
                        or not state.get('total_jobs')
                        or len(set(state.get('completed', []))) != state['total_jobs']):
                    raise RuntimeError('Invalid predecessor completion')
                if not queue.predecessor_pending(spec):
                    return spec
        now = time.monotonic()
        if now - began >= timeout:
            raise TimeoutError('Waiting for visual_memory_followup completion')
        if reported is None or now - reported >= 60:
            print('WAIT visual_memory_followup completion', flush=True)
            reported = now
        time.sleep(5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='all', choices=(
        'all', 'preflight', 'audit', 'train-smoke', 'train', 'export', 'eval-smoke', 'eval', 'merge'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--shard', type=int, choices=range(4), default=0)
    args = parser.parse_args(argv)
    if args.command not in ('all', 'preflight', 'audit') and args.arm is None:
        parser.error('--arm is required for ' + args.command)
    if args.command in ('eval-smoke', 'eval'):
        if args.shard >= shard_count(args.arm):
            parser.error('rb2 has only shard 0')
        evaluate(args.arm, args.shard, smoke=args.command == 'eval-smoke')
        return 0
    if args.command in ('train-smoke', 'train', 'export'):
        if prior.IS_RB2:
            parser.error('Training and export are assigned to sd1')
        if args.command == 'export':
            training.export(args.arm)
        else:
            training.train(args.arm, smoke=args.command == 'train-smoke')
        return 0
    if args.command == 'merge':
        merge(args.arm)
        return 0
    if args.command == 'audit':
        print(audit(), flush=True)
        return 0
    c = prepare()
    audit(c)
    plan = jobs(c)
    write_json(OUTPUT / 'planned_jobs.json', plan)
    print('PREFLIGHT_PASS', len(plan), 'jobs', OUTPUT, flush=True)
    if args.command == 'preflight':
        return 0
    previous = wait_for_predecessor()
    return queue.run_queue(OUTPUT, plan, gpus=gpu_ids(), predecessor=[previous],
                           environment=prior.environment, cwd=ROOT, timeout=24 * 3600)


if __name__ == '__main__':
    raise SystemExit(main())
