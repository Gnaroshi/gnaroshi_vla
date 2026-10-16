"""Three independent 10K continuations from the fixed 60K visual predictor."""
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

from methods.latentloop.modules.geometry_target_followup import ARMS, configure_model
from tools.simvla import local_geometry_train as geometry
from tools.simvla import attention_query_train as driver
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

parent, previous, prior = geometry.parent, geometry.previous, geometry.prior
OUTPUT = prior.STORAGE / 'results/simvla/geometry_target_followup/matched_k8_nfe1_v1'
INCOMING = OUTPUT / 'unused_incoming'
FILES = tuple(dict.fromkeys(geometry.FILES + (
    'methods/latentloop/modules/geometry_target_followup.py',
    'tools/simvla/geometry_target_train.py',
    'tools/simvla/geometry_target_campaign.py',
    'tools/simvla/geometry_target_confirmation.py',
    'architectures/simvla/wrappers/run_geometry_target_followup.sh',
)))


def prepare():
    old = geometry.checked()
    controls = {}
    for arm in ('mean_pool_control', 'subpatch_projection'):
        path = geometry.OUTPUT / 'rows' / arm / 'summary.json'
        result = read_json(path)
        if result['verdict'] != 'EVAL_COMPLETE' or result['episodes'] != 500:
            raise RuntimeError('Missing completed equal-budget control: ' + arm)
        controls[arm] = dict(path=str(path), sha256=sha(path))
    c = dict(old, arms=list(ARMS), parent_contract=str(geometry.OUTPUT/'contract.json'),
             parent_sha256=sha(geometry.OUTPUT/'contract.json'),
             source_sha256={f: sha(ROOT/f) for f in FILES}, reused_controls=controls,
             objective='First5 original NFE10 action L1 +0.05 scaled Condition MSE +relative attention MSE; visual predictor only, all Condition/action/teacher weights frozen.',
             target_definition='Capacity control uses detached student attention query; two original-target arms use frozen teacher projection of original previous Condition. Online query stays student in all arms.',
             comparison='Independent10K from identical60K parent; reuse completed70K mean/subpatch controls. Capacity-only adds65536 weights without within-cell geometry. Mean/subpatch original-target arms complete target-by-geometry factorial.',
             confirmation=dict(action_noise_seed_base=6828326409295398834,
                 selection='Highest seed01 SR among four noncontrol arms; ties use lowest ms/action then arm name. Compare selected checkpoint and mean70K control on500 paired episodes each.',
                 environment_seed=7, determinism_seed=20260815,
                 scope='Second action-noise seed only; unchanged environment initial states and training seed.'))
    c.pop('identity', None)
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Followup contract changed')
    write_json(path, c)
    write_json(OUTPUT/'source.json', dict(git_head=subprocess.check_output(
        ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT/'contract.json')
    geometry.checked()
    if sha(c['parent_contract']) != c['parent_sha256']:
        raise RuntimeError('Parent contract changed')
    for path, expected in c['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise RuntimeError('Source changed: ' + path)
    for record in c['reused_controls'].values():
        if sha(record['path']) != record['sha256']:
            raise RuntimeError('Reused control changed')
    return c


def initial_model(c, arm, native):
    model = geometry.initial_model(geometry.checked(), 'mean_pool_control', native)
    if c['matched_parent_sha256'] != geometry.checked()['matched_parent_sha256']:
        raise RuntimeError('Wrong initialization')
    return configure_model(model, arm)


def model_payload(c, arm, smoke=False):
    import torch
    steps, verdict = (2, 'TRAIN_SMOKE_PASS') if smoke else (c['steps'], 'TRAIN_COMPLETE')
    s = read_json(OUTPUT/('train_smoke' if smoke else 'train')/arm/'summary.json')
    if (s['identity'], s['verdict'], s['steps']) != (c['identity'], verdict, steps):
        raise RuntimeError('Incomplete or mixed checkpoint')
    if sha(s['checkpoint']) != s['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    value = torch.load(s['checkpoint'], map_location='cpu', weights_only=False)
    if (value['identity'], value['arm'], value['step']) != (c['identity'], arm, steps):
        raise RuntimeError('Checkpoint metadata mismatch')
    return value, s['checkpoint_sha256']


@contextmanager
def training_scope():
    replacements = dict(OUTPUT=OUTPUT, checked=checked, initial_model=initial_model,
        parent=SimpleNamespace(checked=parent.checked,
            model_payload=lambda _c, _arm: parent.model_payload(parent.checked(), 'visual_predictor_only')))
    old = {name: getattr(driver, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(driver, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(driver, name, value)


def train(arm, smoke=False):
    if prior.IS_RB2 or arm not in ARMS:
        raise ValueError('sd1 followup arm required')
    with training_scope():
        driver.train(arm, smoke=smoke)
