"""Exercise actual LIBERO inference on smoke checkpoints during full training."""
import argparse
import sys

from tools.simvla import local_geometry_campaign as campaign
from tools.simvla import local_geometry_train as training
from tools.simvla.compile_benchmark import read_json, sha

OUTPUT = training.OUTPUT / 'pretraining_runtime'
MODULE = 'tools.simvla.local_geometry_inference_smoke'


def model_payload(c,arm):
    import torch
    summary = read_json(training.OUTPUT/'train_smoke'/arm/'summary.json')
    if (summary['verdict'],summary['identity'],summary['steps']) != ('TRAIN_SMOKE_PASS',c['identity'],2):
        raise RuntimeError('Expected a completed matching two-step smoke checkpoint')
    if sha(summary['checkpoint']) != summary['checkpoint_sha256']:
        raise RuntimeError('Smoke checkpoint changed')
    value = torch.load(summary['checkpoint'],map_location='cpu',weights_only=False)
    if (value['identity'],value['arm'],value['step']) != (c['identity'],arm,2):
        raise RuntimeError('Smoke checkpoint contract mismatch')
    return value, summary['checkpoint_sha256']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['all','check'])
    parser.add_argument('--arm',choices=training.ARMS)
    args = parser.parse_args(argv)
    if training.prior.IS_RB2:
        raise RuntimeError('sd1 only')
    campaign.configure()
    c = campaign.base.checked()
    campaign.base.OUTPUT = OUTPUT
    if args.command == 'check':
        if args.arm is None:
            parser.error('--arm required')
        original = training.model_payload
        training.model_payload = model_payload
        try:
            campaign.reuse.evaluate(args.arm,smoke=True)
        finally:
            training.model_payload = original
        return 0
    jobs = []
    for arm in training.ARMS:
        jobs.append(dict(id='runtime_'+arm,
            cmd=[sys.executable,'-u','-m',MODULE,'check','--arm',arm],
            deps=[],summary=str(OUTPUT/'shards'/arm/'part0/eval_smoke'/arm/'summary.json'),
            completion=dict(identity=c['identity'],verdict='EVAL_SMOKE_PASS')))
    return campaign.base.queue.run_queue(OUTPUT,jobs,gpus=(6,),predecessor=[],
        environment=training.prior.environment,cwd=campaign.base.ROOT,timeout=3600)


if __name__ == '__main__':
    raise SystemExit(main())
