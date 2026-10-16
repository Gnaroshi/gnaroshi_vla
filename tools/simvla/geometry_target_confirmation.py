"""Full paired confirmation using a held-out action-noise seed and fixed checkpoints."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.simvla import geometry_target_train as training
from tools.simvla import local_geometry_train as geometry
from tools.simvla import visual_readout_campaign as base
from tools.simvla.compile_benchmark import read_json, write_json, sha

OUTPUT = training.OUTPUT/'confirmation'
ARMS = ('control', 'selected')
MODULE = 'tools.simvla.geometry_target_confirmation'


def select_candidate(rows):
    candidates = [arm for arm in rows if arm != 'mean_pool_control']
    if not candidates:
        raise RuntimeError('No completed candidate')
    return min(candidates, key=lambda arm: (-rows[arm]['successes'],
        rows[arm]['policy_ms_per_action'], arm))


def prepare(result):
    parent = training.checked()
    selected = select_candidate(result['rows'])
    sources = {}
    for arm, source_arm in [('control', 'mean_pool_control'), ('selected', selected)]:
        provider = geometry if source_arm in ('mean_pool_control', 'subpatch_projection') else training
        pc = provider.checked()
        summary = read_json(provider.OUTPUT/'train'/source_arm/'summary.json')
        if summary['identity'] != pc['identity'] or summary['verdict'] != 'TRAIN_COMPLETE':
            raise RuntimeError('Incomplete selected checkpoint')
        sources[arm] = dict(arm=source_arm, provider='geometry' if provider is geometry else 'followup',
            contract=str(provider.OUTPUT/'contract.json'), contract_sha256=sha(provider.OUTPUT/'contract.json'),
            checkpoint_sha256=summary['checkpoint_sha256'], seed01_result=result['rows'][source_arm])
    c = copy.deepcopy(parent)
    seed = parent['confirmation']['action_noise_seed_base']
    c.update(arms=list(ARMS), sources=sources, seed01_complete=result['complete'])
    c['manifest']['action_noise_seed_base'] = seed
    c['model']['config']['action_noise_seed_base'] = seed
    c.pop('identity', None)
    c['identity'] = training.prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Confirmation selection or seed changed')
    write_json(path, c)
    return c


def checked():
    training.checked()
    c = read_json(OUTPUT/'contract.json')
    if c['model']['config']['action_noise_seed_base'] != c['manifest']['action_noise_seed_base']:
        raise RuntimeError('Configured policy seed differs from episode manifest')
    for record in c['sources'].values():
        if sha(record['contract']) != record['contract_sha256']:
            raise RuntimeError('Selected source contract changed')
    return c


def provider_for(c, arm):
    record = c['sources'][arm]
    return (geometry if record['provider'] == 'geometry' else training), record['arm']


def initial_model(c, arm, native):
    provider, source = provider_for(c, arm)
    return provider.initial_model(provider.checked(), source, native)


def model_payload(c, arm):
    provider, source = provider_for(c, arm)
    payload, checksum = provider.model_payload(provider.checked(), source)
    if checksum != c['sources'][arm]['checkpoint_sha256']:
        raise RuntimeError('Selected checkpoint changed')
    return payload, checksum


def configure():
    base.OUTPUT, base.ARMS = OUTPUT, ARMS
    base.training = SimpleNamespace(checked=checked, FILES=training.FILES)


def jobs(c):
    result = []
    for arm in ARMS:
        def add(name, args, path, deps, **completion):
            result.append(dict(id='confirm_'+arm+'_'+name, arm=arm,
                cmd=[sys.executable, '-u', '-m', MODULE]+args+['--arm',arm],
                deps=deps, summary=str(path), completion=dict(identity=c['identity'], **completion)))
        shard = OUTPUT/'shards'/arm
        add('smoke', ['smoke'], shard/'part0/eval_smoke'/arm/'summary.json', [], verdict='EVAL_SMOKE_PASS')
        for i in range(4):
            add(str(i), ['eval','--shard',str(i)], shard/f'part{i}/rows'/arm/'summary.json',
                ['confirm_'+arm+'_smoke'], verdict='EVAL_COMPLETE', episodes=125)
        add('merge', ['merge'], OUTPUT/'rows'/arm/'summary.json',
            ['confirm_'+arm+'_'+str(i) for i in range(4)], verdict='EVAL_COMPLETE', episodes=500)
    return result


def evaluate(arm, shard, smoke):
    from tools.simvla import attention_query_eval as evaluation
    c = checked()
    local = dict(c, manifest=base.selected_manifest(c['manifest'], shard, 4))
    old_study, old_write, old_builder = evaluation.study, evaluation.write_json, evaluation.prior.build_policy
    expected_seed = c['manifest']['action_noise_seed_base']

    def builder(mode, config):
        policy, compiler = old_builder(mode, config)
        if policy.action_noise_seed_base != expected_seed:
            raise RuntimeError('Runtime policy action-noise seed mismatch')
        return policy, compiler

    def writer(path, value):
        if Path(path).name == 'progress.json':
            value = dict(value, total=len(local['manifest']['episodes']))
        if Path(path).name in ('runtime.json', 'summary.json') or Path(path).parent.name == 'episodes':
            value = dict(value, action_noise_seed_base=expected_seed,
                         source_arm=c['sources'][arm]['arm'])
        old_write(path, value)

    evaluation.study = SimpleNamespace(OUTPUT=base.shard_output(arm,shard), checked=lambda:local,
        initial_model=initial_model, model_payload=model_payload, previous=training.previous)
    evaluation.write_json, evaluation.prior.build_policy = writer, builder
    try:
        evaluation.evaluate(arm, smoke=smoke)
    finally:
        evaluation.study, evaluation.write_json, evaluation.prior.build_policy = old_study, old_write, old_builder


def merge(arm):
    c = checked()
    for shard in range(4):
        path = base.shard_output(arm,shard)/'rows'/arm
        summary = read_json(path/'summary.json')
        if summary['action_noise_seed_base'] != c['manifest']['action_noise_seed_base']:
            raise RuntimeError('Mixed shard noise seeds')
        for p in (path/'episodes').glob('*.json'):
            if read_json(p)['action_noise_seed_base'] != c['manifest']['action_noise_seed_base']:
                raise RuntimeError('Mixed episode noise seeds')
    return base.merge(arm)


def summarize():
    from tools.simvla.attention_block_campaign import paired_changes
    rows = {arm: read_json(OUTPUT/'rows'/arm/'summary.json') for arm in ARMS
            if (OUTPUT/'rows'/arm/'summary.json').exists()}
    result = dict(complete=len(rows)==2, rows=rows, contract=str(OUTPUT/'contract.json'))
    if len(rows) == 2:
        result['paired'] = paired_changes(*[read_json(OUTPUT/'rows'/a/'episode_outcomes.json') for a in ARMS])
    write_json(OUTPUT/'comparison_summary.json', result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke','eval','merge','summarize'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--shard', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if training.prior.IS_RB2:
        raise RuntimeError('sd1 only')
    configure()
    if args.command == 'summarize':
        print(summarize())
    elif args.arm is None:
        parser.error('--arm required')
    elif args.command == 'merge':
        merge(args.arm)
    else:
        evaluate(args.arm,args.shard,args.command=='smoke')


if __name__ == '__main__':
    main()
