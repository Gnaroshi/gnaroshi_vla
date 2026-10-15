"""Run fresh-vision control, then two learned correspondence rows, without repeats."""
import sys
from types import SimpleNamespace

from tools.simvla import visual_readout_campaign as base
from tools.simvla import visual_transport_train as training

MODULE = 'tools.simvla.visual_transport_campaign'


def jobs(c):
    output = training.OUTPUT
    result = []
    def add(name, args, path, verdict, deps=(), completion=None, **extra):
        result.append(dict(id=name, cmd=[sys.executable,'-u','-m',MODULE]+args,
            summary=str(path), deps=list(deps),
            completion=dict(identity=c['identity'], verdict=verdict, **(completion or {})), **extra))
    # The direct-information control starts immediately; training uses other idle GPUs.
    for i, arm in enumerate(training.ARMS):
        deps = ['merge_' + training.ARMS[i-1]] if i else []
        if arm != 'fresh_vision' and not training.prior.IS_RB2:
            add('train_smoke_'+arm, ['train-smoke','--arm',arm],
                output/'train_smoke'/arm/'summary.json', 'TRAIN_SMOKE_PASS', completion=dict(steps=2))
            add('train_'+arm, ['train','--arm',arm], output/'train'/arm/'summary.json',
                'TRAIN_COMPLETE', ['train_smoke_'+arm], dict(steps=10000))
            add('export_'+arm, ['export','--arm',arm], output/'exports'/f'{arm}.json',
                'EXPORTED', ['train_'+arm])
            deps.append('train_'+arm)
        extra = dict(ready_file=str(training.INCOMING/arm/'READY.json')) if training.prior.IS_RB2 and arm!='fresh_vision' else {}
        add('eval_smoke_'+arm, ['eval-smoke','--arm',arm], base.shard_output(arm,0)/'eval_smoke'/arm/'summary.json',
            'EVAL_SMOKE_PASS', deps, **extra)
        count = base.shard_count(arm)
        for shard in range(count):
            add(f'eval_{arm}_{shard}', ['eval','--arm',arm,'--shard',str(shard)],
                base.shard_output(arm,shard)/'rows'/arm/'summary.json', 'EVAL_COMPLETE',
                ['eval_smoke_'+arm], dict(episodes=len(c['manifest']['episodes'][shard::count])), **extra)
        add('merge_'+arm, ['merge','--arm',arm], output/'rows'/arm/'summary.json',
            'EVAL_COMPLETE', [f'eval_{arm}_{shard}' for shard in range(count)], dict(episodes=500))
    return result


def configure():
    base.training = training
    base.OUTPUT, base.INCOMING, base.ARMS = training.OUTPUT, training.INCOMING, training.ARMS
    base.MODULE, base.jobs = MODULE, jobs
    base.predecessor = SimpleNamespace(OUTPUT=training.previous.OUTPUT)


def main():
    configure()
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
