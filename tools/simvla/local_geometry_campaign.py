"""Parallel matched training, then three sequential full Long500 evaluations."""
import os
import time
from tools.simvla import attention_query_campaign as template
from tools.simvla import attention_block_campaign as reuse
from tools.simvla import local_geometry_train as training

base = template.base
MODULE = 'tools.simvla.local_geometry_campaign'


def configure():
    template.training, template.MODULE = training, MODULE
    reuse.training = training
    base.training = training
    base.OUTPUT, base.INCOMING, base.ARMS = training.OUTPUT, training.INCOMING, training.ARMS
    base.MODULE, base.jobs, base.evaluate = MODULE, template.jobs, reuse.evaluate


def phases(plan):
    result = [('training', [j for j in plan if j['id'].startswith(('smoke_', 'train_'))])]
    for arm in training.ARMS:
        result.append((arm, [j for j in plan if j['id'] in {
            'smoke_'+arm, 'train_'+arm, 'eval_smoke_'+arm, 'merge_'+arm}
            or j['id'].startswith('eval_'+arm+'_')]))
    return result


def run():
    owner = base.queue.acquire_lock(training.OUTPUT / 'queue.lock')
    if owner is None:
        raise RuntimeError('This campaign already has an owner')
    try:
        c = base.prepare()
        base.audit(c)
        plan = template.jobs(c)
        base.write_json(training.OUTPUT/'planned_jobs.json',plan)
        completed, failed = set(), {}
        for phase, local in phases(plan):
            detail = training.OUTPUT/'queues'/phase
            base.write_json(training.OUTPUT/'queue_status.json',dict(phase='running',
                active_phase=phase, detail=str(detail/'queue_status.json'),
                completed=sorted(completed), failed=failed, total_jobs=len(plan),
                pid=os.getpid(),updated_unix=time.time()))
            base.queue.run_queue(detail,local,gpus=base.gpu_ids(),
                predecessor=[dict(path=str(training.parent.OUTPUT),lock='queue.lock')],
                environment=training.prior.environment,cwd=base.ROOT,timeout=24*3600)
            state = base.read_json(detail/'queue_status.json')
            completed.update(state['completed'])
            failed.update(state['failed'])
            # Successful recovery supersedes the previous phase's failure.
            failed = {key:value for key,value in failed.items() if key not in completed}
            if phase != 'training':
                reuse.summarize()
        base.write_json(training.OUTPUT/'queue_status.json',dict(
            phase='finished_with_failures' if failed else 'complete',
            completed=sorted(completed),failed=failed,active={},pending=[],
            total_jobs=len(plan),updated_unix=time.time()))
        return int(bool(failed))
    finally:
        owner.close()


def main(argv=None):
    import sys
    if training.prior.IS_RB2:
        raise RuntimeError('This campaign is assigned to sd1 only')
    configure()
    args = sys.argv[1:] if argv is None else argv
    if not args or args == ['all']:
        return run()
    if args == ['summarize']:
        reuse.summarize()
        return 0
    return base.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
