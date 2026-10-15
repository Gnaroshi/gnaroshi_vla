"""Three matched attention-query interventions, followed by full Long500 rows."""
import sys
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
        deps=['train_'+training.ARMS[i-1]] if i else []
        add('smoke_'+arm,['train-smoke','--arm',arm],out/'train_smoke'/arm/'summary.json',
            'TRAIN_SMOKE_PASS',deps,steps=2)
        add('train_'+arm,['train','--arm',arm],out/'train'/arm/'summary.json',
            'TRAIN_COMPLETE',['smoke_'+arm],steps=10000)
        deps=['train_'+arm]+(['merge_'+training.ARMS[i-1]] if i else [])
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
    return base.main(argv)


if __name__=='__main__':
    raise SystemExit(main())
