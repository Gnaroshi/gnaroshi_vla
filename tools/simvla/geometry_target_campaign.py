"""Matched geometry/target experiments followed automatically by seed confirmation."""
import os
import sys
import time

from tools.simvla import attention_query_campaign as template
from tools.simvla import attention_block_campaign as reuse
from tools.simvla import geometry_target_train as training

base = template.base
MODULE = 'tools.simvla.geometry_target_campaign'


def configure():
    template.training, template.MODULE = training, MODULE
    reuse.training = training
    base.training = training
    base.OUTPUT, base.INCOMING, base.ARMS = training.OUTPUT, training.INCOMING, training.ARMS
    base.MODULE, base.jobs, base.evaluate = MODULE, jobs, reuse.evaluate


def jobs(c):
    plan = template.jobs(c)
    for arm in training.ARMS:
        runtime = 'runtime_' + arm
        plan.append(dict(id=runtime,
            cmd=[sys.executable, '-u', '-m', MODULE, 'runtime-smoke', '--arm', arm],
            deps=['smoke_'+arm], summary=str(training.OUTPUT/'pretraining_runtime'/arm/'shards'/arm/'part0/eval_smoke'/arm/'summary.json'),
            completion=dict(identity=c['identity'], verdict='EVAL_SMOKE_PASS')))
        next(j for j in plan if j['id'] == 'train_'+arm)['deps'] = [runtime]
    return plan


def summarize():
    c = training.checked()
    rows, outcomes = {}, {}
    for arm, item in c['reused_controls'].items():
        path = base.Path(item['path'])
        rows[arm] = base.read_json(path)
        outcomes[arm] = base.read_json(path.parent/'episode_outcomes.json')
    for arm in training.ARMS:
        path = training.OUTPUT/'rows'/arm/'summary.json'
        if path.exists():
            summary, records = base.merged_result(c, arm)
            if base.read_json(path) != summary:
                raise RuntimeError('Merged result differs from episode evidence')
            rows[arm], outcomes[arm] = summary, records
    control = rows['mean_pool_control']
    paired = {arm: dict(
        success_rate_difference_pp=100*(s['success_rate']-control['success_rate']),
        policy_ms_per_action_difference=s['policy_ms_per_action']-control['policy_ms_per_action'],
        **reuse.paired_changes(outcomes['mean_pool_control'], outcomes[arm]))
        for arm, s in rows.items() if arm != 'mean_pool_control'}
    result = dict(identity=c['identity'], complete=all(a in rows for a in training.ARMS),
        rows=rows, paired_against_mean_control=paired,
        hardware='sd1 RTX3090 eager', initial_checkpoint_sha256=c['matched_parent_sha256'],
        comparison=c['comparison'], updated_unix=time.time())
    base.write_json(training.OUTPUT/'comparison_summary.json', result)
    return result


def phases(plan):
    yield 'training', [j for j in plan if j['id'].startswith(('smoke_', 'runtime_', 'train_'))]
    for arm in training.ARMS:
        yield arm, [j for j in plan if j['id'] in {
            'smoke_'+arm, 'runtime_'+arm, 'train_'+arm, 'eval_smoke_'+arm, 'merge_'+arm}
            or j['id'].startswith('eval_'+arm+'_')]


def run():
    owner = base.queue.acquire_lock(training.OUTPUT/'queue.lock')
    if owner is None:
        raise RuntimeError('This campaign already has an owner')
    try:
        c = base.prepare()
        base.audit(c)
        plan = jobs(c)
        base.write_json(training.OUTPUT/'planned_jobs.json', plan)
        completed, failed = set(), {}

        def run_phase(name, local, root=training.OUTPUT):
            detail = root/'queues'/name
            base.write_json(training.OUTPUT/'queue_status.json', dict(phase='running',
                active_phase=name, detail=str(detail/'queue_status.json'),
                completed=sorted(completed), failed=failed, total_jobs=len(plan),
                pending_followup='selected_vs_mean_control_second_action_noise_seed',
                pid=os.getpid(), updated_unix=time.time()))
            base.queue.run_queue(detail, local, gpus=(4,5,6,7), predecessor=[],
                environment=training.prior.environment, cwd=base.ROOT, timeout=24*3600)
            state = base.read_json(detail/'queue_status.json')
            completed.update(state['completed'])
            failed.update(state['failed'])
            for key in completed:
                failed.pop(key, None)

        for name, local in phases(plan):
            run_phase(name, local)
            if name != 'training':
                summarize()
        from tools.simvla import geometry_target_confirmation as followup
        confirmation = followup.prepare(summarize())
        second = followup.jobs(confirmation)
        plan += second
        base.write_json(training.OUTPUT/'planned_jobs.json', plan)
        for arm in followup.ARMS:
            local = [j for j in second if j['arm'] == arm]
            run_phase('confirmation_'+arm, local, followup.OUTPUT)
        followup.summarize()
        base.write_json(training.OUTPUT/'queue_status.json', dict(
            phase='finished_with_failures' if failed else 'complete',
            completed=sorted(completed), failed=failed, active={}, pending=[],
            total_jobs=len(plan), updated_unix=time.time()))
        return int(bool(failed))
    except Exception as exc:
        base.write_json(training.OUTPUT/'queue_error.json', dict(error=repr(exc), updated_unix=time.time()))
        raise
    finally:
        owner.close()


def main(argv=None):
    if training.prior.IS_RB2:
        raise RuntimeError('sd1 only')
    configure()
    args = sys.argv[1:] if argv is None else argv
    if not args or args == ['all']:
        return run()
    if args == ['summarize']:
        print(summarize(), flush=True)
        return 0
    if args and args[0] == 'runtime-smoke':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--arm', required=True, choices=training.ARMS)
        arm = parser.parse_args(args[1:]).arm
        base.OUTPUT = training.OUTPUT/'pretraining_runtime'/arm
        original = training.model_payload
        training.model_payload = lambda c, a: original(c, a, smoke=True)
        try:
            reuse.evaluate(arm, smoke=True)
        finally:
            training.model_payload = original
        return 0
    return base.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
