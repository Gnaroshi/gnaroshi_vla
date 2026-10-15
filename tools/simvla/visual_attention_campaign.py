"""sd1-only ordered teacher adaptation, attention distillation, recursive training."""
import sys
from types import SimpleNamespace
from tools.simvla import visual_readout_campaign as base
from tools.simvla import visual_attention_train as training

MODULE='tools.simvla.visual_attention_campaign'


def jobs(c):
    out=training.OUTPUT; result=[]
    def add(name,args,path,verdict,deps=(),**completion):
        result.append(dict(id=name,cmd=[sys.executable,'-u','-m',MODULE]+args,summary=str(path),deps=list(deps),
            completion=dict(identity=c['identity'],verdict=verdict,**completion)))
    for i,arm in enumerate(training.ARMS):
        deps=['train_'+training.ARMS[i-1]] if i else []
        add('smoke_'+arm,['train-smoke','--arm',arm],out/'train_smoke'/arm/'summary.json','TRAIN_SMOKE_PASS',deps,steps=2)
        add('train_'+arm,['train','--arm',arm],out/'train'/arm/'summary.json','TRAIN_COMPLETE',['smoke_'+arm],steps=10000)
        deps=['train_'+arm]+(['merge_'+training.ARMS[i-1]] if i else [])
        add('eval_smoke_'+arm,['eval-smoke','--arm',arm],base.shard_output(arm,0)/'eval_smoke'/arm/'summary.json','EVAL_SMOKE_PASS',deps)
        for s in range(4):
            add(f'eval_{arm}_{s}',['eval','--arm',arm,'--shard',str(s)],base.shard_output(arm,s)/'rows'/arm/'summary.json',
                'EVAL_COMPLETE',['eval_smoke_'+arm],episodes=125)
        add('merge_'+arm,['merge','--arm',arm],out/'rows'/arm/'summary.json','EVAL_COMPLETE',[f'eval_{arm}_{s}' for s in range(4)],episodes=500)
    return result


def main():
    if training.prior.IS_RB2: raise RuntimeError('This campaign is assigned to sd1 only')
    base.training=training;base.OUTPUT=training.OUTPUT;base.INCOMING=training.INCOMING;base.ARMS=training.ARMS
    base.MODULE=MODULE;base.jobs=jobs;base.predecessor=SimpleNamespace(OUTPUT=training.source.OUTPUT)
    return base.main()


if __name__=='__main__':raise SystemExit(main())
