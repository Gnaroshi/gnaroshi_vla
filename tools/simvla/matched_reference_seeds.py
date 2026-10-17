"""Complete missing second-noise reference rows for the token-interaction study."""
import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from tools.simvla import failure_interventions as prior
from tools.simvla import gpu_followup_queue as queue
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

RESULTS = prior.STORAGE / 'results/simvla'
OUTPUT = RESULTS / 'matched_references/second_action_noise_v1'
MODULE = 'tools.simvla.matched_reference_seeds'
ROWS = ('recurrent_parent_k4', 'bridge_f4', 'bridge_f8')
ORIGINS = {
    'recurrent_parent_k4': 'age_specialization/preserve_short_nfe1_v1',
    'bridge_f4': 'visual_midpoint/current_vision_nfe1_v1',
    'bridge_f8': 'visual_memory_followup/reference_age_and_controls_v1',
}
FILES = ('tools/simvla/matched_reference_seeds.py',
         'tests/test_matched_reference_seeds.py',
         'architectures/simvla/wrappers/run_matched_reference_seeds.sh')


def second_manifest(original):
    result = copy.deepcopy(original)
    result['action_noise_seed_base'] += 1
    return result


def validate_records(summary, records, manifest, identity, checksum, seed=None):
    pairs = {(x['task_id'], x['trial_id']) for x in manifest['episodes']}
    if (len(records) != 500 or len(pairs) != 500
            or {(x['task_id'], x['trial_id']) for x in records} != pairs
            or summary['episodes'] != 500 or summary['identity'] != identity
            or any(x['identity'] != identity or x['model_sha256'] != checksum for x in records)
            or summary['successes'] != sum(x['success'] for x in records)
            or summary['success_rate'] != summary['successes'] / 500):
        raise RuntimeError('Incomplete/mixed reference episodes')
    if seed is not None and any(x.get('action_noise_seed_base') != seed for x in records):
        raise RuntimeError('Action-noise seed mismatch')
    valid = [x for x in records if x['timing_valid']]
    latency = sum(x['policy_ms_total'] for x in valid) / sum(x['episode_length'] for x in valid) if valid else None
    if (summary['timing_valid_episodes'] != len(valid)
            or (latency is None) != (summary['policy_ms_per_action'] is None)
            or (latency is not None and abs(summary['policy_ms_per_action'] - latency) > 1e-8)):
        raise RuntimeError('Latency aggregation mismatch')
    return summary


def prepare():
    if not prior.IS_RB2:
        raise RuntimeError('Reference evaluation belongs to rb2')
    cfg = prior.config('recurrent_k4')
    sources, manifest = {}, None
    for row in ROWS:
        root = RESULTS / ORIGINS[row]
        old = read_json(root / 'contract.json')
        m = old['manifest']
        if manifest is not None and m != manifest:
            raise RuntimeError('First-seed reference manifests differ')
        manifest = m
        model = old.get('model') or old['models']['recurrent']
        for key in ('checkpoint_revision', 'norm_stats', 'bridge_checkpoint', 'libero_root', 'libero_config'):
            if model['config'][key] != cfg[key]:
                raise RuntimeError('Reference config changed: ' + key)
        expected_hash = model['sha256'] if row == ROWS[0] else sha(cfg['bridge_checkpoint'])
        directory = root / 'rows' / row
        records = [read_json(p) for p in (directory / 'episodes').glob('*.json')]
        summary = validate_records(read_json(directory / 'summary.json'), records, m, old['identity'], expected_hash)
        sources[row] = dict(path=str(directory), model_sha256=expected_hash, summary=summary,
                            summary_sha256=sha(directory / 'summary.json'),
                            contract=str(root / 'contract.json'), contract_sha256=sha(root / 'contract.json'))
    # Match the same two noise seeds as the selected new model. Torch/env seeds stay fixed.
    portable = read_json(prior.STORAGE / 'transfers/simvla_geometry_comparison/bundle.json')
    if manifest != portable['manifest'] or cfg['model_checkpoint']['sha256'] != portable['parent_fresh_sha256']:
        raise RuntimeError('New-model comparison protocol differs')
    paths = set(FILES)
    for folder in ('tools/simvla', 'methods/latentloop', 'architectures/simvla/adapters',
                   'architectures/simvla/wrappers/dcld_eval'):
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*.py'))
    c = dict(model=dict(config=cfg, sha256=cfg['model_checkpoint']['sha256']),
             manifest=second_manifest(manifest), first_manifest=manifest, sources=sources,
             source_sha256={f:sha(ROOT / f) for f in sorted(paths)},
             artifacts={cfg[key]:sha(cfg[key]) for key in ('norm_stats', 'bridge_checkpoint', 'condition_checkpoint')},
             scope='Same independent episode reset, compiled RTX5090,NFE1,H10/R5,Long500. Only action-noise seed changes. No training.',
             seed_scope='Two action-noise seeds with fixed environment/Torch seeds and fixed weights; not independent training seeds.')
    c['artifacts'][cfg['model_checkpoint']['path']] = cfg['model_checkpoint']['sha256']
    c['metadata'] = dict(python=sys.version, executable=sys.executable,
        packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines(),
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                     '--format=csv,noheader'], text=True).strip())
    c['identity'] = prior.digest(c)
    path = OUTPUT / 'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Reference campaign contract changed')
    write_json(path, c)
    write_json(OUTPUT / 'source.json', dict(commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT / 'contract.json')
    if c['identity'] != prior.digest({k:v for k,v in c.items() if k != 'identity'}):
        raise RuntimeError('Contract identity changed')
    for name, digest in c['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Source changed: ' + name)
    for name, digest in c['artifacts'].items():
        if sha(name) != digest:
            raise RuntimeError('Artifact changed: ' + name)
    for source in c['sources'].values():
        if (sha(source['contract']) != source['contract_sha256']
                or sha(Path(source['path']) / 'summary.json') != source['summary_sha256']):
            raise RuntimeError('Completed reference changed')
    return c


@contextmanager
def runtime_scope(row, c):
    from tools.simvla import age_specialization_eval as recurrent
    from tools.simvla import visual_memory_followup_eval as bridge
    from tools.simvla import condition_interval_compare as interval
    evaluation = recurrent if row == ROWS[0] else bridge
    original_study, original_writer = evaluation.study, evaluation.write_json
    original_builder, original_bridge = prior.build_policy, interval.policy_factory
    seed = c['manifest']['action_noise_seed_base']
    def builder(mode, cfg):
        policy, compiler = original_builder(mode, cfg)
        policy.action_noise_seed_base = seed
        return policy, compiler
    def bridge_builder(replay, cfg, name, manifest):
        policy = original_bridge(replay, cfg, name, manifest)
        if policy.action_noise_seed_base != seed:
            raise RuntimeError('Bridge action-noise seed did not propagate')
        return policy
    def writer(path, value):
        if Path(path).name in ('runtime.json', 'summary.json') or Path(path).parent.name == 'episodes':
            value = dict(value, action_noise_seed_base=seed)
        original_writer(path, value)
    local = copy.deepcopy(c)
    local['models'] = {'recurrent': c['model']}
    try:
        prior.build_policy, interval.policy_factory = builder, bridge_builder
        evaluation.study = SimpleNamespace(OUTPUT=OUTPUT, checked=lambda:local,
            row_spec=lambda name:dict(kind='bridge', interval=int(name.removeprefix('bridge_f'))))
        evaluation.write_json = writer
        yield evaluation
    finally:
        evaluation.study, evaluation.write_json = original_study, original_writer
        prior.build_policy, interval.policy_factory = original_builder, original_bridge


def evaluate(row, smoke=False):
    c = checked()
    with runtime_scope(row, c) as evaluation:
        if row == ROWS[0]:
            evaluation.evaluate('recurrent', 'parent', 4, smoke=smoke)
        else:
            evaluation.evaluate(row, smoke=smoke)
    if not smoke:
        summarize()


def summarize():
    c = checked()
    rows = {}
    for row in ROWS:
        directory = OUTPUT / 'rows' / row
        if not (directory / 'summary.json').exists():
            continue
        records = [read_json(p) for p in (directory / 'episodes').glob('*.json')]
        summary = validate_records(read_json(directory / 'summary.json'), records, c['manifest'],
            c['identity'], c['sources'][row]['model_sha256'], c['manifest']['action_noise_seed_base'])
        first = c['sources'][row]['summary']
        rows[row] = dict(first=first, second=summary,
            mean_success_rate=(first['success_rate'] + summary['success_rate']) / 2,
            mean_policy_ms_per_action=(first['policy_ms_per_action'] + summary['policy_ms_per_action']) / 2
                if first['policy_ms_per_action'] is not None and summary['policy_ms_per_action'] is not None else None)
    write_json(OUTPUT / 'comparison_summary.json', dict(identity=c['identity'], rows=rows,
        complete=len(rows) == len(ROWS), seed_scope=c['seed_scope'],
        other_references='Bridge f2/f3 first-seed results remain available; no two-seed claim for them.'))


def jobs(c):
    result = []
    for row in ROWS:
        for smoke in (True, False):
            result.append(dict(id=row + ('_smoke' if smoke else '_eval'),
                cmd=[sys.executable, '-u', '-m', MODULE, 'smoke' if smoke else 'eval', '--row', row],
                deps=[] if smoke else [row + '_smoke'],
                summary=str(OUTPUT / ('eval_smoke' if smoke else 'rows') / row / 'summary.json'),
                completion=dict(identity=c['identity'], verdict='EVAL_SMOKE_PASS' if smoke else 'EVAL_COMPLETE',
                                **({} if smoke else dict(episodes=500)))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('all', 'preflight', 'smoke', 'eval', 'summarize'))
    parser.add_argument('--row', choices=ROWS)
    args = parser.parse_args()
    if not prior.IS_RB2:
        parser.error('rb2 only')
    if args.command in ('smoke', 'eval'):
        if not args.row:parser.error('--row required')
        evaluate(args.row, args.command == 'smoke');return 0
    if args.command == 'summarize':summarize();return 0
    c = prepare();plan = jobs(c)
    write_json(OUTPUT / 'planned_jobs.json', plan)
    print('PREFLIGHT_PASS', len(plan), 'jobs; three missing Long500 second-noise rows', flush=True)
    if args.command == 'preflight':return 0
    predecessors = [dict(path=str(RESULTS / 'token_interaction/matched_nfe1_v1' / phase), lock='queue.lock')
                    for phase in ('preselection', 'evaluation')]
    rc = queue.run_queue(OUTPUT, plan, gpus=(0,), predecessor=predecessors,
                        environment=prior.environment, cwd=ROOT, timeout=12*3600)
    summarize();return rc


if __name__ == '__main__':
    raise SystemExit(main())
