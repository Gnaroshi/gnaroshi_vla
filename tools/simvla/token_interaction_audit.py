"""GPU equality audit for the frozen70K readout intervention control."""
import argparse
from pathlib import Path

from tools.simvla.compile_benchmark import write_json
from tools.simvla import readout_causality as source

OUTPUT = source.prior.STORAGE / 'results/simvla/token_interaction/matched_nfe1_v1'


def audit(zero=False):
    import torch
    from tools.simvla.error_compensation_common import configure, snapshots
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from tools.simvla import visual_midpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
    c = source.checked()
    cfg = c['model']['config']
    configure(cfg)
    torch.set_num_threads(1)
    device, native, _, _, _, heldout = load_runtime(cfg, snapshots(cfg))
    value, checksum = source.payload(c, 'control')
    models = [source.initial_model(c, 'control', native), source.portable.construct(native)]
    features = CachedFeatures(visual_midpoint.OUTPUT / 'features', visual_midpoint.checked()['identity'])
    for model in models:
        model.to(device).eval()
        model.load_state_dict(value['model'], strict=True)
        model.feature_provider = features
    indices = _balanced_indices(heldout.identities, limit=30, seed=7)
    records = []
    for frozen in (False, True):
        if frozen:
            for model in models:
                model.requires_grad_(False)
        with torch.no_grad():
            for index in indices if frozen else indices[:1]:
                seq = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
                features.bind(seq)
                first = sample_from_sequence(seq, 1)
                contexts = [model.prepare(first['anchor'], first['anchor_images'], first['anchor_proprio'],
                    first['valid'], first['groups'], 8) for model in models]
                for age in range(1, 8):
                    sample = sample_from_sequence(seq, age)
                    outputs = [model.predict(context, age, sample['images'], sample['proprio'])[0]
                               for model, context in zip(models, contexts)]
                    difference = (outputs[0] - outputs[1]).abs()
                    records.append(dict(frozen=frozen, window=index, age=age,
                        maximum=float(difference.max()), mean=float(difference.mean()),
                        nonzero=int(difference.count_nonzero())))
    report = dict(checkpoint_sha256=checksum, records=records, windows=indices,
                  device=str(device), torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    report['verdict'] = ('GPU_CONTROL_PARITY_PASS' if all(r['maximum'] == 0 for r in records if r['frozen'])
                         else 'GPU_CONTROL_PARITY_FAIL')
    write_json(OUTPUT / 'audit/control_parity.json', report)
    print(report['verdict'], 'first differences=', [r for r in records if r['maximum']][:8], flush=True)
    if report['verdict'] != 'GPU_CONTROL_PARITY_PASS':
        raise RuntimeError('GPU control mismatch; inspect audit/control_parity.json')
    if zero:
        from tools.simvla import token_interaction_campaign as campaign
        from tools.simvla import short_visual_train as training
        configuration = campaign.checked()
        zero_records = []
        for arm in campaign.ARMS:
            candidate = campaign.initial_model(configuration, arm, native).to(device).eval()
            candidate.requires_grad_(False)
            if arm.startswith('warm_'):
                reference = models[1]
            else:
                reference = training.build_model(native, 'protected_short', seed=7,
                    early_state=training._protected_payload(configuration)['model']).to(device).eval()
                reference.requires_grad_(False)
            candidate.feature_provider = reference.feature_provider = features
            with torch.no_grad():
                for index in indices:
                    seq = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
                    features.bind(seq)
                    first = sample_from_sequence(seq, 1)
                    pair = (reference, candidate)
                    contexts = [model.prepare(first['anchor'],first['anchor_images'],first['anchor_proprio'],
                        first['valid'],first['groups'],8) for model in pair]
                    for age in range(1,8):
                        sample = sample_from_sequence(seq,age)
                        outputs = [model.predict(context,age,sample['images'],sample['proprio'])[0]
                                   for model,context in zip(pair,contexts)]
                        maximum = float((outputs[0]-outputs[1]).abs().max())
                        zero_records.append(dict(arm=arm,window=index,age=age,maximum=maximum))
                        if maximum != 0:
                            write_json(OUTPUT/'audit/zero_initialization.json',dict(verdict='ZERO_PARITY_FAIL',
                                identity=configuration['identity'],records=zero_records))
                            raise RuntimeError(f'Zero mixer changed output: {arm} window{index} age{age} diff{maximum}')
            del candidate
        write_json(OUTPUT/'audit/zero_initialization.json',dict(verdict='ZERO_PARITY_PASS',
            identity=configuration['identity'],records=zero_records,windows=indices))
        print('ZERO_PARITY_PASS',len(zero_records),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'queue', 'zero', 'queue-zero'))
    a = p.parse_args()
    if a.command in ('run','zero'):
        audit(a.command=='zero')
        return 0
    import sys
    zero = a.command == 'queue-zero'
    job = dict(id='zero_initialization' if zero else 'control_parity',
        cmd=[sys.executable,'-u','-m','tools.simvla.token_interaction_audit','zero' if zero else 'run'],deps=[],
        summary=str(OUTPUT / ('audit/zero_initialization.json' if zero else 'audit/control_parity.json')),
        completion=dict(verdict='ZERO_PARITY_PASS' if zero else 'GPU_CONTROL_PARITY_PASS'))
    return source.queue.run_queue(OUTPUT / ('zero_audit_queue' if zero else 'audit_queue'), [job], gpus=(4,5,6,7), predecessor=[],
        environment=source.prior.environment, cwd=source.ROOT, timeout=1200)


if __name__ == '__main__':
    raise SystemExit(main())
