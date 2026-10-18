"""Compiled confirmation of the completed current-action intervention."""
import argparse
from types import SimpleNamespace

from methods.latentloop.modules.condition_routing import install
from tools.simvla import condition_routing_campaign as routing
from tools.simvla import geometry_comparison_rb2 as portable
from tools.simvla import failure_interventions as prior
from tools.simvla import gpu_followup_queue as queue
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

OUTPUT = prior.STORAGE / 'results/simvla/current_condition/original_reference_v1'
MODULE = 'tools.simvla.current_condition_reference'
ROW = 'selected_seed1'


def prepare():
    c = dict(portable.checked())
    c.pop('identity')
    c['source_sha256'] = dict(c['source_sha256'])
    for name in (MODULE.replace('.', '/') + '.py',
                 'tools/simvla/condition_routing_campaign.py',
                 'methods/latentloop/modules/condition_routing.py'):
        c['source_sha256'][name] = sha(ROOT / name)
    c.update(rows=[ROW], intervention='Original Condition at age4 goes to current action only; frozen70K recurrent input unchanged.',
             actual_backbone_schedule='Original at ages0,4; extra original vision at age4 retained; include all calls in latency.')
    c['identity'] = prior.digest(c)
    path = OUTPUT / 'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Reference contract changed')
    write_json(path, c)
    return c


def checked():
    c = read_json(OUTPUT / 'contract.json')
    if prior.digest({k: v for k, v in c.items() if k != 'identity'}) != c['identity']:
        raise RuntimeError('Contract identity changed')
    if portable.validate_bundle() != c['bundle']:
        raise RuntimeError('Checkpoint bundle changed')
    for f, h in c['source_sha256'].items():
        if sha(ROOT / f) != h:
            raise RuntimeError('Source changed: ' + f)
    return c


def evaluate(smoke):
    from tools.simvla import attention_query_eval as evaluation
    evaluation.study = SimpleNamespace(OUTPUT=OUTPUT, checked=checked, model_payload=portable.payload,
        initial_model=lambda c, arm, native: install(portable.construct(native), 'current_action'),
        previous=SimpleNamespace(OUTPUT=OUTPUT / 'unused_reference'))
    evaluation.attach_visual = routing.attach
    evaluation.check = lambda policy, arm: routing.check(policy)
    evaluation.evaluate(ROW, smoke=smoke)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('all', 'preflight', 'smoke', 'eval'))
    a = p.parse_args()
    if not prior.IS_RB2:
        p.error('rb2 only')
    if a.command in ('smoke', 'eval'):
        evaluate(a.command == 'smoke')
        return 0
    c = prepare()
    print('PREFLIGHT_PASS one Long500 current-action intervention, all calls timed', flush=True)
    if a.command == 'preflight':
        return 0
    import sys
    jobs = [dict(id=phase, cmd=[sys.executable, '-u', '-m', MODULE, phase],
        deps=[] if phase == 'smoke' else ['smoke'],
        summary=str(OUTPUT / ('eval_smoke' if phase == 'smoke' else 'rows') / ROW / 'summary.json'),
        completion=dict(identity=c['identity'], verdict='EVAL_SMOKE_PASS' if phase == 'smoke' else 'EVAL_COMPLETE'))
        for phase in ('smoke', 'eval')]
    write_json(OUTPUT / 'planned_jobs.json', jobs)
    return queue.run_queue(OUTPUT, jobs, gpus=(0,), predecessor=[],
        environment=prior.environment, cwd=ROOT, timeout=12 * 3600)


if __name__ == '__main__':
    raise SystemExit(main())
