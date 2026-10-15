"""Distill spatial correspondence while holding the successful K8 updater fixed."""
import random
import shutil
import subprocess
import time
from pathlib import Path

from tools.simvla import failure_interventions as prior
from tools.simvla import visual_readout_train as previous
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

OUTPUT = prior.STORAGE / 'results/simvla/visual_token_transport/warm_reader_nfe1_v1'
INCOMING = Path('/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_visual_token_transport')
ARMS = ('fresh_vision', 'transport', 'transport_residual')
FILES = ('methods/latentloop/modules/visual_token_transport.py',
         'tools/simvla/visual_transport_train.py', 'tools/simvla/visual_transport_campaign.py',
         'architectures/simvla/wrappers/run_visual_token_transport.sh')


def parent_payload(c):
    if prior.IS_RB2:
        from tools.simvla.visual_readout_remote import model_payload
        return model_payload(c, 'warm_retain')
    return previous.model_payload(c, 'warm_retain')


def prepare():
    old = previous.checked()
    _, checkpoint_hash = parent_payload(old)
    c = dict(old)
    c.update(parent_contract=str(previous.OUTPUT / 'contract.json'),
             parent_sha256=sha(previous.OUTPUT / 'contract.json'),
             source_sha256={f: sha(ROOT / f) for f in FILES},
             warm_reader_sha256=checkpoint_hash, arms=list(ARMS), steps=10000,
             objective='Mean relative MSE of current key/value under fixed warm-reader projections. Predictor only trains, same seed7,batch2,10K and samples. Fresh-vision control has no training.',
             change='Full backbone every8; current original vision at age4. Transport age4 K/V across camera-local36-token positions at ages5..7; optional zero-initialized appearance residual. Warm Condition weights remain frozen.',
             comparison='Same warm_retain checkpoint. Fresh original vision ages4..7 tests current-information benefit at higher cost. Two predictors isolate camera-local correspondence and new-content prediction. Long500/NFE1/H10/R5.',
             target_definition='Same current72x960 original cached vision, fixed warm-reader LN/key/value. Positions added only after transport. No optical-flow labels, evaluation outcomes or future images are training targets.')
    c.pop('identity', None)
    # Identity excludes host paths; exact scientific inputs and sources match across hosts.
    cfg = c['model']['config']
    c['scientific_identity'] = dict(
        sources=c['source_sha256'], warm_reader_sha256=checkpoint_hash,
        original_k4_sha256=c['model']['sha256'], checkpoint_revision=cfg['checkpoint_revision'],
        norm_sha256=sha(cfg['norm_stats']), native_sha256=sha(cfg['condition_checkpoint']),
        manifest=c['manifest'], steps=c['steps'], batch=c['batch_size'], seed=c['seed'],
        lr=c['learning_rate'], warmup=c['warmup'], arms=list(ARMS), objective=c['objective'])
    c['identity'] = prior.digest(c['scientific_identity'])
    path = OUTPUT / 'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Transport contract changed')
    write_json(path, c)
    write_json(OUTPUT / 'source.json', dict(git_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT / 'contract.json')
    previous.checked()
    if sha(c['parent_contract']) != c['parent_sha256']:
        raise RuntimeError('Parent contract changed')
    for f, expected in c['source_sha256'].items():
        if sha(ROOT / f) != expected:
            raise RuntimeError('Source changed: ' + f)
    if prior.digest(c['scientific_identity']) != c['identity']:
        raise RuntimeError('Scientific identity mismatch')
    return c


def initial_model(c, arm, native):
    import torch
    from methods.latentloop.modules.visual_token_transport import FreshVisionCondition, TransportCondition
    old = previous.checked()
    warm = previous.initial_model(old, 'warm_retain', native)
    payload, checkpoint_hash = parent_payload(old)
    if checkpoint_hash != c['warm_reader_sha256']:
        raise RuntimeError('Warm reader changed')
    warm.load_state_dict(payload['model'], strict=True)
    torch.manual_seed(c['seed'])
    if arm == 'fresh_vision':
        return FreshVisionCondition(warm)
    return TransportCondition(warm, arm)


def train(arm, smoke=False):
    if prior.IS_RB2 or arm == 'fresh_vision':
        raise RuntimeError('Train only transport predictors on sd1')
    import torch
    from tqdm import trange
    from tools.simvla.error_compensation_common import configure, snapshots
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, append_jsonl
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime, lr_factor
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    c = checked(); cfg = c['model']['config']; configure(cfg)
    device, native, frozen, action, data, heldout = load_runtime(cfg, snapshots(cfg))
    model = initial_model(c, arm, native).to(device).eval()
    features = CachedFeatures(previous.parent.OUTPUT / 'features', previous.parent.checked()['identity'])
    model.feature_provider = features
    frozen_before, base_before = previous.frozen_hash(model), state_hash(frozen)
    parameters = [p for p in model.parameters() if p.requires_grad]
    out = OUTPUT / ('train_smoke' if smoke else 'train') / arm
    steps = 2 if smoke else c['steps']
    contract = dict(identity=c['identity'], arm=arm, data=data.contract(), heldout=heldout.contract(),
        steps=c['steps'], run_steps=steps, seed=c['seed'], batch_size=c['batch_size'],
        warm_reader_sha256=c['warm_reader_sha256'], objective=c['objective'],
        trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters), frozen_parameters_sha256=frozen_before,
        optimizer=dict(name='AdamW', lr=c['learning_rate'], weight_decay=0, warmup=c['warmup'], schedule='cosine to0.1x', clip=1))
    write_json(out / 'training_contract.json', contract)
    optimizer = torch.optim.AdamW(parameters, lr=c['learning_rate'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: lr_factor(n, c['steps'], c['warmup']))
    latest = out / 'latest.pt'; start, elapsed = 0, 0.
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved['contract'] != contract:
            raise RuntimeError('Resume contract changed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start, elapsed = saved['step'], saved['seconds']
    if not (out / 'initial_validation.json').exists():
        previous.validate(c, model, action, heldout, features, out / 'initial_validation.json', smoke)
    began = time.monotonic()
    for n in trange(start + 1, steps + 1, desc=arm, mininterval=2):
        rng = random.Random(c['seed'] * 100000 + n)
        age = 5 + (n - 1) % 3
        if smoke: age = 5 if n == 1 else 7
        optimizer.zero_grad(set_to_none=True)
        error = 0.
        for _ in range(c['batch_size']):
            seq = move_batch(collate_exact_teacher_sequences([data[rng.randrange(len(data))]]), device)
            features.bind(seq)
            pred, target = previous.readout_pair(model, features, seq, age)
            loss = previous.relative_readout_loss(pred, target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            (loss / c['batch_size']).backward()
            error += float(loss.detach()) / c['batch_size']
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if not norm > 0: raise RuntimeError('Missing gradient')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen model received gradients')
        optimizer.step(); scheduler.step()
        seconds = elapsed + time.monotonic() - began
        if n % 50 == 0 or n in (1, steps):
            metrics = dict(step=n, age=age, readout_relative_mse=error, seconds=seconds,
                           gradient_norm=float(norm), lr=optimizer.param_groups[0]['lr'])
            append_jsonl(out / 'metrics.jsonl', metrics); write_json(out / 'progress.json', metrics)
        if n % 500 == 0 or n == steps:
            atomic_torch_save(dict(format='simvla_visual_transport_v1', contract=contract, step=n,
                model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), seconds=seconds), latest)
    if previous.frozen_hash(model) != frozen_before or state_hash(frozen) != base_before:
        raise RuntimeError('Frozen parameters changed')
    previous.validate(c, model, action, heldout, features, out / 'validation.json', smoke)
    write_json(out / 'summary.json', dict(verdict='TRAIN_SMOKE_PASS' if smoke else 'TRAIN_COMPLETE',
        identity=c['identity'], arm=arm, checkpoint=str(latest), checkpoint_sha256=sha(latest),
        steps=steps, training_seconds=seconds, parent_training_steps=30000, total_training_steps=30000+steps,
        parameters=contract['total_parameters'], trainable_parameters=contract['trainable_parameters'],
        protected_frozen_parameters_unchanged=True))


def model_payload(c, arm):
    import torch
    if arm == 'fresh_vision':
        return parent_payload(previous.checked())
    if prior.IS_RB2:
        folder = INCOMING / arm
        if read_json(folder / 'READY.json')['manifest_sha256'] != sha(folder / 'manifest.json'):
            raise RuntimeError('Incomplete transfer')
        spec, path = read_json(folder / 'manifest.json'), folder / 'model.pt'
        if spec['source_sha256'] != c['source_sha256']:
            raise RuntimeError('Transferred source differs')
    else:
        spec = read_json(OUTPUT / 'train' / arm / 'summary.json'); path = Path(spec['checkpoint'])
    if spec['identity'] != c['identity'] or spec['verdict'] != 'TRAIN_COMPLETE' or sha(path) != spec['checkpoint_sha256']:
        raise RuntimeError('Invalid checkpoint identity or completion')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['format'] != 'simvla_visual_transport_v1' or payload['step'] != c['steps'] or payload['contract']['arm'] != arm or payload['contract']['identity'] != c['identity']:
        raise RuntimeError('Invalid checkpoint training provenance')
    return payload, spec['checkpoint_sha256']


def export(arm):
    c = checked(); model_payload(c, arm)
    summary = read_json(OUTPUT / 'train' / arm / 'summary.json')
    stage = OUTPUT / 'bundles' / arm; stage.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(summary['checkpoint'], stage / 'model.pt')
    write_json(stage / 'manifest.json', dict(**summary, source_sha256=c['source_sha256']))
    write_json(stage / 'READY.json', dict(manifest_sha256=sha(stage / 'manifest.json')))
    dest = str(INCOMING / arm)
    subprocess.run(['ssh','rb2','mkdir','-p',dest], check=True, timeout=30)
    subprocess.run(['rsync','-a','--partial','--exclude=READY.json',str(stage)+'/',f'rb2:{dest}/'], check=True, timeout=120)
    subprocess.run(['rsync','-a',str(stage/'READY.json'),f'rb2:{dest}/'], check=True, timeout=30)
    write_json(OUTPUT / 'exports' / f'{arm}.json', dict(verdict='EXPORTED', identity=c['identity']))
