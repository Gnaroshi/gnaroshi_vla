"""Queue two block-isolation experiments after the current query-source study."""
import os
import time
from pathlib import Path
from types import SimpleNamespace

from tools.simvla import attention_query_campaign as template
from tools.simvla import attention_query_train as predecessor
from tools.simvla import attention_block_train as training

base = template.base
MODULE = 'tools.simvla.attention_block_campaign'


def configure():
    template.training, template.MODULE = training, MODULE
    base.training = training
    base.OUTPUT, base.INCOMING, base.ARMS = training.OUTPUT, training.INCOMING, training.ARMS
    base.MODULE, base.jobs, base.evaluate = MODULE, template.jobs, evaluate


def evaluate(arm, shard=0, smoke=False):
    from tools.simvla import attention_query_eval as evaluation
    c = base.checked()
    local = dict(c, manifest=base.selected_manifest(c['manifest'], shard, base.shard_count(arm)))
    old_study, old_write = evaluation.study, evaluation.write_json

    def write(path, value):
        if Path(path).name == 'progress.json':
            value = dict(value, total=len(local['manifest']['episodes']))
        if Path(path).name == 'comparison.json':
            value = dict(value, comparison=c['comparison'])
        old_write(path, value)

    evaluation.study = SimpleNamespace(OUTPUT=base.shard_output(arm, shard), checked=lambda: local,
        initial_model=training.initial_model, model_payload=training.model_payload,
        previous=training.previous)
    evaluation.write_json = write
    try:
        evaluation.evaluate(arm, smoke=smoke)
    finally:
        evaluation.study, evaluation.write_json = old_study, old_write


def predecessor_complete():
    path = predecessor.OUTPUT / 'queue_status.json'
    if not path.exists():
        return False
    state = base.read_json(path)
    if state.get('phase') not in ('complete', 'finished_with_failures'):
        return False
    return not base.queue.predecessor_pending(dict(path=str(predecessor.OUTPUT), lock='queue.lock'))


def wait_for_predecessor():
    began = time.monotonic()
    while not predecessor_complete():
        if time.monotonic() - began > 48 * 3600:
            raise TimeoutError('Current attention-query queue did not finish within48h')
        base.write_json(training.OUTPUT / 'queue_status.json', dict(
            phase='waiting_for_predecessor', predecessor=str(predecessor.OUTPUT),
            pid=os.getpid(), total_jobs=16, updated_unix=time.time()))
        print('WAIT current attention-query training and evaluations', flush=True)
        time.sleep(60)
    return dict(path=str(predecessor.OUTPUT), lock='queue.lock')


def paired_changes(control, current):
    def indexed(rows):
        result = {(r['task_id'], r['trial_id']): r for r in rows}
        expected = {(t, n) for t in range(10) for n in range(50)}
        if len(result) != len(rows) or set(result) != expected:
            raise RuntimeError('Paired comparison requires500 unique task/trial IDs')
        return result
    old, new = indexed(control), indexed(current)
    lost = [list(key) for key in old if old[key]['success'] and not new[key]['success']]
    gained = [list(key) for key in old if not old[key]['success'] and new[key]['success']]
    return dict(lost_successes=lost, gained_successes=gained,
                net_success_change=len(gained) - len(lost))


def summarize():
    c = base.checked()
    reference = Path(c['matched_control'])
    control = base.read_json(reference / 'summary.json')
    outcomes = base.read_json(reference / 'episode_outcomes.json')
    if control.get('verdict') != 'EVAL_COMPLETE' or control.get('episodes') != 500:
        raise RuntimeError('Completed matched joint control is missing')
    rows = {}
    for arm in training.ARMS:
        path = training.OUTPUT / 'rows' / arm
        if not (path / 'summary.json').exists():
            continue
        summary = base.read_json(path / 'summary.json')
        if summary['identity'] != c['identity'] or summary['verdict'] != 'EVAL_COMPLETE':
            raise RuntimeError('Mixed comparison output')
        changes = paired_changes(outcomes, base.read_json(path / 'episode_outcomes.json'))
        rows[arm] = dict(summary=summary, paired_against_joint=changes)
    result = dict(identity=c['identity'], complete=len(rows) == len(training.ARMS),
        matched_joint_control=control, rows=rows,
        comparison=c['comparison'], hardware='sd1 RTX3090 eager',
        source_parent_sha256=c['matched_parent_sha256'], updated_unix=time.time())
    base.write_json(training.OUTPUT / 'comparison_summary.json', result)
    return result


def run_ordered():
    owner = base.queue.acquire_lock(training.OUTPUT / 'queue.lock')
    if owner is None:
        raise RuntimeError('This follow-up already has an owner')
    try:
        c = base.prepare()
        base.audit(c)
        plan = template.jobs(c)
        base.write_json(training.OUTPUT / 'planned_jobs.json', plan)
        dependency = wait_for_predecessor()
        completed, failed = [], {}
        for arm in training.ARMS:
            local = [j for j in plan if j['id'] in
                     {'smoke_' + arm, 'train_' + arm, 'eval_smoke_' + arm, 'merge_' + arm}
                     or j['id'].startswith('eval_' + arm + '_')]
            base.write_json(training.OUTPUT / 'queue_status.json', dict(phase='running',
                active_arm=arm, detail=str(training.OUTPUT / 'queues' / arm / 'queue_status.json'),
                completed=completed, failed=failed, total_jobs=16,
                pid=os.getpid(), updated_unix=time.time()))
            base.queue.run_queue(training.OUTPUT / 'queues' / arm, local,
                gpus=base.gpu_ids(), predecessor=[dependency],
                environment=training.prior.environment, cwd=base.ROOT, timeout=24 * 3600)
            state = base.read_json(training.OUTPUT / 'queues' / arm / 'queue_status.json')
            completed.extend(state['completed'])
            failed.update(state['failed'])
            summarize()
        base.write_json(training.OUTPUT / 'queue_status.json', dict(
            phase='finished_with_failures' if failed else 'complete', completed=completed,
            failed=failed, active={}, pending=[], total_jobs=16, updated_unix=time.time()))
        return int(bool(failed))
    finally:
        owner.close()


def main(argv=None):
    import sys
    if training.prior.IS_RB2:
        raise RuntimeError('This follow-up is assigned to sd1 only')
    configure()
    args = sys.argv[1:] if argv is None else argv
    if not args or args == ['all']:
        return run_ordered()
    if args == ['summarize']:
        summarize()
        return 0
    return base.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
