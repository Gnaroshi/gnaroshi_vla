"""Matched visual-only continuations with position-sensitive local pooling."""
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

from methods.latentloop.modules.local_geometry_input import ARMS, configure_model
from tools.simvla import attention_query_train as driver
from tools.simvla import attention_block_train as parent
from tools.simvla import visual_readout_train as previous
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

OUTPUT = prior.STORAGE / 'results/simvla/local_geometry_input/matched_k8_nfe1_v1'
INCOMING = OUTPUT / 'unused_incoming'
FILES = tuple(dict.fromkeys(parent.FILES + (
    'methods/latentloop/modules/local_geometry_input.py',
    'tools/simvla/local_geometry_train.py',
    'tools/simvla/local_geometry_campaign.py',
    'architectures/simvla/wrappers/run_local_geometry_input.sh',
)))


def prepare():
    old = parent.checked()
    source = read_json(parent.OUTPUT / 'train/visual_predictor_only/summary.json')
    if source['verdict'] != 'TRAIN_COMPLETE' or source['total_training_steps'] != 60000:
        raise RuntimeError('Expected the completed 60K visual-only parent')
    if sha(source['checkpoint']) != source['checkpoint_sha256']:
        raise RuntimeError('Parent checkpoint changed')
    c = dict(old, parent_contract=str(parent.OUTPUT / 'contract.json'),
        parent_sha256=sha(parent.OUTPUT / 'contract.json'), arms=list(ARMS),
        source_sha256={f: sha(ROOT / f) for f in FILES},
        matched_parent_checkpoint=source['checkpoint'],
        matched_parent_sha256=source['checkpoint_sha256'],
        matched_control=str(parent.OUTPUT / 'rows/visual_predictor_only'),
        objective='Unchanged first5 original NFE10 action L1 +0.05 scaled Condition MSE + relative same-query attention MSE. Only visual predictor is trainable.',
        target_definition='Frozen adapted reader, current original vision, identical detached student query; original same-noise NFE10 action and original Condition.',
        comparison='Three independent10K continuations from identical60K visual-only parent. Mean pooling versus mean plus zero-initialized spatial moments/full centered subpatch projection. Same samples,seed,optimizer,schedule,K8,NFE1,H10/R5,Long500; Condition updater frozen.',
        geometry=dict(input=[64,24,24], output=[64,6,6], cell=[4,4],
                      extra_parameters=dict(mean_pool_control=0,spatial_moments=8192,subpatch_projection=65536)),
        video_contract='All500 episodes, two cameras every2 actions, numeric observations/actions and per-query goal predicates.')
    c.pop('identity', None)
    c['identity'] = prior.digest(c)
    path = OUTPUT / 'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Local geometry contract changed')
    write_json(path, c)
    write_json(OUTPUT / 'source.json', dict(git_head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), files=c['source_sha256']))
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
    pc = parent.checked()
    model = parent.initial_model(pc, 'visual_predictor_only', native)
    payload, checksum = parent.model_payload(pc, 'visual_predictor_only')
    if checksum != c['matched_parent_sha256']:
        raise RuntimeError('Initial checkpoint differs from fixed comparison parent')
    model.load_state_dict(payload['model'], strict=True)
    return configure_model(model, arm)


def model_payload(c, arm):
    import torch
    summary = read_json(OUTPUT / 'train' / arm / 'summary.json')
    if summary['identity'] != c['identity'] or summary['verdict'] != 'TRAIN_COMPLETE':
        raise RuntimeError('Incomplete local geometry training')
    if sha(summary['checkpoint']) != summary['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    value = torch.load(summary['checkpoint'],map_location='cpu',weights_only=False)
    if (value['identity'],value['arm'],value['step']) != (c['identity'],arm,c['steps']):
        raise RuntimeError('Checkpoint contract mismatch')
    return value, summary['checkpoint_sha256']


@contextmanager
def training_scope():
    replacements = dict(OUTPUT=OUTPUT,checked=checked,initial_model=initial_model,
        parent=SimpleNamespace(checked=parent.checked,
            model_payload=lambda _c,_arm: parent.model_payload(parent.checked(),'visual_predictor_only')))
    old = {name:getattr(driver,name) for name in replacements}
    try:
        for name,value in replacements.items():
            setattr(driver,name,value)
        yield
    finally:
        for name,value in old.items():
            setattr(driver,name,value)


def train(arm, smoke=False):
    if prior.IS_RB2 or arm not in ARMS:
        raise ValueError('sd1 local geometry arm required')
    with training_scope():
        driver.train(arm,smoke=smoke)
