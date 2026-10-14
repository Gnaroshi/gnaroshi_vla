"""Validate original sd1 training provenance separately from rb2 runtime paths."""
import sys
from types import SimpleNamespace
from tools.simvla import visual_readout_train as training
from tools.simvla import visual_readout_campaign as campaign
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

ORIGINAL_LOADER = training.model_payload
ORIGINAL_JOBS = campaign.jobs
FIELDS = ('arms', 'steps', 'warmup', 'learning_rate', 'seed', 'batch_size',
          'condition_weight', 'readout_weight', 'student_nfe', 'teacher_nfe',
          'manifest', 'source_sha256', 'trained_reader_sha256', 'objective',
          'target_definition', 'change', 'comparison')


def validate_origin(source, destination, marker):
    for field in FIELDS:
        if source[field] != destination[field]:
            raise RuntimeError('Transferred scientific contract differs: ' + field)
    if source['identity'] != marker['training_identity']:
        raise RuntimeError('Original training identity differs')
    if source['model']['sha256'] != destination['model']['sha256']:
        raise RuntimeError('Original K4 checkpoint differs')
    cfg = destination['model']['config']
    if source['model']['config']['checkpoint_revision'] != cfg['checkpoint_revision']:
        raise RuntimeError('Base SimVLA revision differs')
    for field in ('norm_stats', 'condition_checkpoint'):
        if sha(cfg[field]) != marker['artifact_sha256'][field]:
            raise RuntimeError('Base artifact differs: ' + field)
    return source


def source_contract(c):
    path = training.INCOMING / 'training_origin.json'
    marker = read_json(training.INCOMING / 'training_origin_ready.json')
    if sha(path) != marker['source_contract_sha256']:
        raise RuntimeError('Original training contract transfer incomplete')
    return validate_origin(read_json(path), c, marker)


def model_payload(c, arm):
    return ORIGINAL_LOADER(source_contract(c), arm)


def jobs(c):
    plan = ORIGINAL_JOBS(c)
    prefix = []
    for smoke in (True, False):
        name = 'bridge_f5_smoke' if smoke else 'bridge_f5_eval'
        folder = 'eval_smoke' if smoke else 'rows'
        prefix.append(dict(id=name,
            cmd=[sys.executable, '-u', '-m', 'tools.simvla.visual_readout_remote',
                 'bridge-smoke' if smoke else 'bridge-eval'],
            deps=[] if smoke else ['bridge_f5_smoke'],
            summary=str(training.OUTPUT / 'comparison' / folder / 'bridge_f5' / 'summary.json'),
            completion=dict(identity=c['identity'], verdict='EVAL_SMOKE_PASS' if smoke else 'EVAL_COMPLETE')))
    for job in plan:
        if job['id'] == 'eval_smoke_warm_retain':
            job['deps'].append('bridge_f5_eval')
    return prefix + plan


def bridge_evaluate(smoke):
    from tools.simvla import visual_memory_followup_eval as evaluator
    evaluator.study = SimpleNamespace(OUTPUT=training.OUTPUT / 'comparison',
        checked=training.checked,
        row_spec=lambda row: dict(kind='bridge', interval=5))
    evaluator.evaluate('bridge_f5', smoke=smoke)


def main():
    if not training.prior.IS_RB2:
        raise RuntimeError('Remote dispatcher is rb2 only')
    c = training.checked()
    source = source_contract(c)
    write_json(training.OUTPUT / 'transfer_validation.json', dict(
        verdict='TRANSFER_CONTRACT_PASS', source_training_identity=source['identity'],
        runtime_identity=c['identity'], matched_scientific_fields=list(FIELDS),
        loader_sha256=sha(ROOT / 'tools/simvla/visual_readout_remote.py')))
    training.model_payload = model_payload
    campaign.MODULE = 'tools.simvla.visual_readout_remote'
    campaign.jobs = jobs
    if len(sys.argv) > 1 and sys.argv[1] in ('bridge-smoke', 'bridge-eval'):
        bridge_evaluate(sys.argv[1] == 'bridge-smoke')
        return 0
    return campaign.main()


if __name__ == '__main__':
    raise SystemExit(main())
