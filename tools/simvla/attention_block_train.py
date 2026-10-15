"""Matched block-freezing continuations from the completed attention student."""
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

from methods.latentloop.modules.attention_block_isolation import ARMS
from tools.simvla import attention_query_train as driver
from tools.simvla import visual_attention_train as parent
from tools.simvla import visual_readout_train as previous
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

OUTPUT = prior.STORAGE / 'results/simvla/attention_block_isolation/matched_k8_nfe1_v1'
INCOMING = OUTPUT / 'unused_incoming'
FILES = (
    'methods/latentloop/modules/attention_block_isolation.py',
    'tools/simvla/attention_block_train.py',
    'tools/simvla/attention_block_campaign.py',
    'architectures/simvla/wrappers/run_attention_block_isolation.sh',
    'methods/latentloop/modules/attention_query_contract.py',
    'tools/simvla/attention_query_train.py',
    'tools/simvla/attention_query_campaign.py',
    'tools/simvla/attention_query_eval.py',
    'tools/simvla/visual_readout_campaign.py',
    'tools/simvla/gpu_followup_queue.py',
)


def source_checkpoint():
    path = parent.OUTPUT / 'train/attention_distill/summary.json'
    value = read_json(path)
    if value['verdict'] != 'TRAIN_COMPLETE' or value['steps'] != 10000:
        raise RuntimeError('Expected the completed 50K attention student')
    if sha(value['checkpoint']) != value['checkpoint_sha256']:
        raise RuntimeError('Parent checkpoint changed')
    return value


def prepare():
    old = parent.checked()
    source = source_checkpoint()
    c = dict(old, parent_contract=str(parent.OUTPUT / 'contract.json'),
             parent_sha256=sha(parent.OUTPUT / 'contract.json'), arms=list(ARMS),
             source_sha256={f: sha(ROOT / f) for f in FILES},
             matched_parent_checkpoint=source['checkpoint'],
             matched_parent_sha256=source['checkpoint_sha256'],
             matched_control=str(parent.OUTPUT / 'rows/attention_joint'),
             objective='Unchanged first5 original NFE10 action L1 +0.05 scaled Condition MSE + relative same-query attention MSE. Only trainable parameter blocks change.',
             target_definition='Frozen adapted reader, current original vision, identical detached student attention query; original same-noise NFE10 action and original Condition.',
             comparison='Two independent10K continuations from the fixed50K attention_distill final checkpoint; reuse completed joint60K as matched control. Same batches,seed,optimizer,LR,schedule,earlyK4 freeze,K8,NFE1,H10/R5,Long500.',
             video_contract='All500 episodes, two cameras every2 actions, numeric observations/actions and per-query goal predicates.')
    c.pop('identity', None)
    c['identity'] = prior.digest(c)
    path = OUTPUT / 'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Block isolation contract changed')
    write_json(path, c)
    write_json(OUTPUT / 'source.json', dict(git_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT / 'contract.json')
    parent.checked()
    if sha(c['parent_contract']) != c['parent_sha256']:
        raise RuntimeError('Parent contract changed')
    for name, expected in c['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Source changed: ' + name)
    return c


def initial_model(c, arm, native):
    from methods.latentloop.modules.attention_block_isolation import build_model
    pc = parent.checked()
    student = parent.initial_model(pc, 'attention_joint', native)
    payload, checksum = parent.model_payload(pc, 'attention_distill')
    if checksum != c['matched_parent_sha256']:
        raise RuntimeError('Initial checkpoint differs from fixed comparison parent')
    student.load_state_dict(payload['model'], strict=True)
    teacher = parent.initial_model(pc, 'adapt_fresh', native)
    payload, _ = parent.model_payload(pc, 'adapt_fresh')
    teacher.load_state_dict(payload['model'], strict=True)
    return build_model(student, teacher, arm)


def model_payload(c, arm):
    import torch
    summary = read_json(OUTPUT / 'train' / arm / 'summary.json')
    if summary['identity'] != c['identity'] or summary['verdict'] != 'TRAIN_COMPLETE':
        raise RuntimeError('Incomplete block training')
    if sha(summary['checkpoint']) != summary['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    p = torch.load(summary['checkpoint'], map_location='cpu', weights_only=False)
    if (p['identity'], p['arm'], p['step']) != (c['identity'], arm, c['steps']):
        raise RuntimeError('Checkpoint contract mismatch')
    return p, summary['checkpoint_sha256']


@contextmanager
def training_scope():
    # Reuse the exact sampler, objective and optimizer of the current campaign.
    replacements = dict(OUTPUT=OUTPUT, checked=checked, initial_model=initial_model,
        parent=SimpleNamespace(checked=parent.checked,
            model_payload=lambda _c, _arm: parent.model_payload(parent.checked(), 'attention_distill')))
    old = {name: getattr(driver, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(driver, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(driver, name, value)


def train(arm, smoke=False):
    if prior.IS_RB2:
        raise RuntimeError('This follow-up is assigned to sd1 only')
    if arm not in ARMS:
        raise ValueError(arm)
    with training_scope():
        driver.train(arm, smoke=smoke)
    path = OUTPUT / ('train_smoke' if smoke else 'train') / arm / 'summary.json'
    value = read_json(path)
    value.update(total_training_steps=50000 + value['steps'],
                 matched_control=str(parent.OUTPUT / 'rows/attention_joint'),
                 online_forward_unchanged=True)
    write_json(path, value)
