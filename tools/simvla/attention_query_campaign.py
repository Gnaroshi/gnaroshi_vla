"""Three matched attention-query interventions, followed by full Long500 rows."""
import sys
import os
import time
from pathlib import Path
from types import SimpleNamespace

from tools.simvla import visual_readout_campaign as base
from tools.simvla import attention_query_train as training

MODULE = 'tools.simvla.attention_query_campaign'


def jobs(c):
    out=training.OUTPUT
    result=[]

    def add(name,args,path,verdict,deps=(),**completion):
        result.append(dict(id=name,cmd=[sys.executable,'-u','-m',MODULE]+args,
            summary=str(path),deps=list(deps),completion=dict(identity=c['identity'],verdict=verdict,**completion)))

    for i,arm in enumerate(training.ARMS):
        # Every arm loads the same fixed parent, never the preceding arm.
        deps=[]
        add('smoke_'+arm,['train-smoke','--arm',arm],out/'train_smoke'/arm/'summary.json',
            'TRAIN_SMOKE_PASS',deps,steps=2)
        add('train_'+arm,['train','--arm',arm],out/'train'/arm/'summary.json',
            'TRAIN_COMPLETE',['smoke_'+arm],steps=10000)
        deps=['train_'+arm]
        add('eval_smoke_'+arm,['eval-smoke','--arm',arm],base.shard_output(arm,0)/'eval_smoke'/arm/'summary.json',
            'EVAL_SMOKE_PASS',deps)
        for shard in range(4):
            add(f'eval_{arm}_{shard}',['eval','--arm',arm,'--shard',str(shard)],
                base.shard_output(arm,shard)/'rows'/arm/'summary.json','EVAL_COMPLETE',
                ['eval_smoke_'+arm],episodes=125)
        add('merge_'+arm,['merge','--arm',arm],out/'rows'/arm/'summary.json',
            'EVAL_COMPLETE',[f'eval_{arm}_{s}' for s in range(4)],episodes=500)
    return result


def evaluate(arm, shard=0, smoke=False):
    from tools.simvla import attention_query_eval as evaluation
    c=base.checked()
    local=dict(c,manifest=base.selected_manifest(c['manifest'],shard,base.shard_count(arm)))
    old_study,old_write=evaluation.study,evaluation.write_json

    def write(path,value):
        if Path(path).name=='progress.json':
            value=dict(value,total=len(local['manifest']['episodes']))
        old_write(path,value)

    evaluation.study=SimpleNamespace(OUTPUT=base.shard_output(arm,shard),checked=lambda:local,
        initial_model=training.initial_model,model_payload=training.model_payload,previous=training.previous)
    evaluation.write_json=write
    try:
        evaluation.evaluate(arm,smoke=smoke)
    finally:
        evaluation.study,evaluation.write_json=old_study,old_write


def main(argv=None):
    if training.prior.IS_RB2:
        raise RuntimeError('This matched training campaign is assigned to sd1 only')
    base.training=training
    base.OUTPUT,base.INCOMING,base.ARMS=training.OUTPUT,training.INCOMING,training.ARMS
    base.MODULE,base.jobs,base.evaluate=MODULE,jobs,evaluate
    base.predecessor=SimpleNamespace(OUTPUT=training.parent.OUTPUT)
    args=sys.argv[1:] if argv is None else argv
    if not args or args==['all']:
        return run_ordered()
    return base.main(argv)


def wait_for_parent():
    spec=dict(path=str(training.parent.OUTPUT),lock='queue.lock')
    start=time.monotonic()
    while True:
        p=training.parent.OUTPUT/'queue_status.json'
        state=base.read_json(p) if p.exists() else {}
        if state.get('phase') in ('complete','finished_with_failures') and not base.queue.predecessor_pending(spec):
            training.parent.model_payload(training.parent.checked(),'attention_joint')
            return spec
        if time.monotonic()-start>24*3600:raise TimeoutError('Parent attention campaign did not finish')
        base.write_json(training.OUTPUT/'queue_status.json',dict(phase='waiting_for_predecessor',
            predecessor=str(training.parent.OUTPUT),pid=os.getpid(),total_jobs=24,updated_unix=time.time()))
        print('WAIT attention_joint evaluation and parent queue completion',flush=True)
        time.sleep(60)


def run_ordered():
    owner=base.queue.acquire_lock(training.OUTPUT/'queue.lock')
    if owner is None:raise RuntimeError('This campaign already has an owner')
    try:
        c=base.prepare();base.audit(c);plan=jobs(c)
        base.write_json(training.OUTPUT/'planned_jobs.json',plan)
        predecessor=wait_for_parent()
        completed=[];failed={}
        for arm in training.ARMS:
            local=[j for j in plan if j['id'] in {'smoke_'+arm,'train_'+arm,'eval_smoke_'+arm,'merge_'+arm}
                   or j['id'].startswith('eval_'+arm+'_')]
            base.write_json(training.OUTPUT/'queue_status.json',dict(phase='running',active_arm=arm,
                detail=str(training.OUTPUT/'queues'/arm/'queue_status.json'),completed=completed,
                failed=failed,total_jobs=24,pid=os.getpid(),updated_unix=time.time()))
            base.queue.run_queue(training.OUTPUT/'queues'/arm,local,gpus=base.gpu_ids(),
                predecessor=[predecessor],environment=training.prior.environment,cwd=base.ROOT,timeout=24*3600)
            state=base.read_json(training.OUTPUT/'queues'/arm/'queue_status.json')
            completed.extend(state['completed']);failed.update(state['failed'])
        base.write_json(training.OUTPUT/'queue_status.json',dict(phase='finished_with_failures' if failed else 'complete',
            completed=completed,failed=failed,active={},pending=[],total_jobs=24,updated_unix=time.time()))
        return int(bool(failed))
    finally:
        owner.close()


if __name__=='__main__':
    raise SystemExit(main())
