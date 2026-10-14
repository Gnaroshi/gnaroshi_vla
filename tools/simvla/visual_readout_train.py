"""Preserve a trained Condition reader and distill its current visual inputs."""
import random
import shutil
import subprocess
import time
from pathlib import Path

from tools.simvla import visual_midpoint as parent
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

OUTPUT = prior.STORAGE / 'results/simvla/visual_readout_distillation/preserved_reader_nfe1_v1'
INCOMING = Path('/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_visual_readout_distillation')
ARMS = ('warm_retain', 'distilled_readout', 'joint_readout')
previous = parent.previous
FILES = ('methods/latentloop/modules/visual_readout_distillation.py',
         'tools/simvla/visual_readout_train.py', 'tools/simvla/visual_readout_campaign.py',
         'architectures/simvla/wrappers/run_visual_readout_distillation.sh')


def prepare():
    old = parent.checked()
    _, source_hash = parent.model_payload(old, 'late_every')
    c = dict(old)
    c.update(parent_contract=str(parent.OUTPUT / 'contract.json'),
             parent_sha256=sha(parent.OUTPUT / 'contract.json'),
             source_sha256={f: sha(ROOT / f) for f in FILES},
             trained_reader_sha256=source_hash, arms=list(ARMS),
             steps=10000, warmup=500, learning_rate=1e-4, seed=7, batch_size=2,
             condition_weight=0.05, readout_weight=1.0,
             change='Keep the completed late_every Condition reader. Warm retention tests preserving its trained initialization. Compare feature-only and joint action training from identical predictor initialization and matched10K samples.',
             objective={
                 'warm_retain': 'First5 normalized original NFE10 same-noise action L1 +0.05 unrolled Condition scaled MSE.',
                 'distilled_readout': 'Mean of fixed-reader key and value relative MSE; only the RGB/readout predictor trains. Ages5,6,7.',
                 'joint_readout': 'Same initialization and10K samples as distilled_readout: first5 action L1 +0.05 Condition scaled MSE +1.0 fixed-reader key/value relative MSE. Late Condition updater and predictor train; original key/value projections remain fixed.'},
             comparison='All Long500 seed01, NFE1,H10/R5,K8,one extra vision call at age4. Reuse completed hold and fresh-vision rows. Feature-only and joint both10K from the same parent,seed andages5..7.',
             target_definition='Current original vision/connector72x960 mapped by the frozen completed late_every key/value projections after layer norm, excluding position embeddings. Per-camera36 tokens; each key/value64 dimensions. Targets only used in training/offline validation.')
    c.pop('identity', None)
    c.pop('git_head', None)
    c['identity'] = prior.digest(c)
    path = OUTPUT / 'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Readout contract changed')
    write_json(path, c)
    write_json(OUTPUT / 'source.json', dict(git_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT / 'contract.json')
    parent.checked()
    if sha(c['parent_contract']) != c['parent_sha256']:
        raise RuntimeError('Parent contract changed')
    for f, expected in c['source_sha256'].items():
        if sha(ROOT / f) != expected:
            raise RuntimeError('Source changed: ' + f)
    return c


def initial_model(c, arm, native):
    import torch
    from methods.latentloop.modules.visual_readout_distillation import ReadoutDistillationCondition
    if arm not in ARMS:
        raise ValueError(arm)
    old = parent.checked()
    trained = parent.initial_model(old, 'late_every', native)
    payload, checkpoint_hash = parent.model_payload(old, 'late_every')
    if checkpoint_hash != c['trained_reader_sha256']:
        raise RuntimeError('Trained reader changed')
    trained.load_state_dict(payload['model'], strict=True)
    torch.manual_seed(c['seed'])
    return ReadoutDistillationCondition(trained, arm)


def relative_readout_loss(predicted, target):
    import torch
    return torch.stack([
        ((p - t).square().mean(-1) / t.square().mean(-1).clamp_min(1e-6)).mean()
        for p, t in zip(predicted, target)
    ]).mean()


def frozen_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def readout_pair(model, features, seq, age):
    reference = features(4, seq['image_sequence'][:, 4])
    target_features = features(age, seq['image_sequence'][:, age])
    predicted = model.visual.predicted_readout(reference, seq['image_sequence'][:, 4],
        seq['image_sequence'][:, age], age)
    target = model.visual.target_readout(target_features)
    return predicted, target


def validate(c, model, action, heldout, features, path, smoke):
    import torch
    from torch.nn import functional as F
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from methods.latentloop.modules.condition_output_split import geometry
    from tools.simvla.condition_output_split_train import unroll, action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    records = []
    with torch.no_grad():
        for index in _balanced_indices(heldout.identities, limit=2 if smoke else 30, seed=c['seed']):
            seq = move_batch(collate_exact_teacher_sequences([heldout[index]]), 'cuda')
            features.bind(seq)
            for age in (4, 5, 6, 7):
                z, _, _ = unroll(model, seq, age, 8)
                s = sample_from_sequence(seq, age)
                pred = action_prediction(action, z, s, steps=1, grad=False)
                original1 = action_prediction(action, s['target_condition'], s, steps=1, grad=False)
                row = dict(window=index, age=age,
                    action_l1=float(action_loss(pred, action.action_space.normalize_action(s['target_action']))),
                    action_l1_vs_original_nfe1=float(action_loss(pred, original1)),
                    condition=geometry(z, s['target_condition'], s['valid']))
                if model.arm != 'warm_retain':
                    pkv, tkv = readout_pair(model, features, seq, age)
                    reference = features(4, seq['image_sequence'][:, 4])
                    held = model.visual.target_readout(reference)
                    stale = model.visual.predicted_readout(reference, seq['image_sequence'][:, 4],
                        seq['image_sequence'][:, 4], age)
                    row.update(readout_relative_mse=float(relative_readout_loss(pkv, tkv)),
                        held_readout_relative_mse=float(relative_readout_loss(held, tkv)),
                        readout_cosine=[float(F.cosine_similarity(p, t, dim=-1).mean()) for p, t in zip(pkv, tkv)],
                        held_readout_cosine=[float(F.cosine_similarity(p, t, dim=-1).mean()) for p, t in zip(held, tkv)],
                        frozen_rgb_relative_mse=float(relative_readout_loss(stale, tkv)),
                        rgb_effect_rms=float(torch.stack([(p - h).square().mean() for p, h in zip(pkv, stale)]).mean().sqrt()))
                records.append(row)
    write_json(path, dict(records=records, reference='Same heldout windows and same noise, original NFE10 actions and original Conditions. Additional original NFE1 action distance isolates solver target mismatch. Key/value targets are frozen-reader current-vision outputs.',
                         no_selection_gate=True))


def train(arm, smoke=False):
    if prior.IS_RB2:
        raise RuntimeError('Training is assigned to sd1 GPUs4..7')
    import torch
    from tqdm import trange
    from tools.simvla.error_compensation_common import configure, snapshots
    from tools.simvla.condition_output_split_train import unroll, action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from methods.latentloop.modules.trend_condition import scaled_mse
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, append_jsonl
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime, lr_factor, assert_frozen
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    c = checked()
    cfg = c['model']['config']
    configure(cfg)
    device, native, frozen, action, data, heldout = load_runtime(cfg, snapshots(cfg))
    model = initial_model(c, arm, native).to(device).eval()
    init_hash = c['trained_reader_sha256']
    features = CachedFeatures(parent.OUTPUT / 'features', parent.checked()['identity'])
    model.feature_provider = features
    frozen_before, base_before = frozen_hash(model), state_hash(frozen)
    parameters = [p for p in model.parameters() if p.requires_grad]
    out = OUTPUT / ('train_smoke' if smoke else 'train') / arm
    out.mkdir(parents=True, exist_ok=True)
    steps = 2 if smoke else c['steps']
    contract = dict(identity=c['identity'], arm=arm, data=data.contract(), heldout=heldout.contract(),
        steps=c['steps'], run_steps=steps, seed=c['seed'], batch_size=c['batch_size'],
        trained_reader_sha256=c['trained_reader_sha256'], initialization_sha256=init_hash,
        objective=c['objective'][arm], target=c['target_definition'],
        trainable_names=[n for n, p in model.named_parameters() if p.requires_grad],
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters), frozen_parameters_sha256=frozen_before,
        optimizer=dict(name='AdamW', lr=c['learning_rate'], weight_decay=0, warmup=c['warmup'], schedule='cosine to0.1x', clip=1))
    write_json(out / 'training_contract.json', contract)
    optimizer = torch.optim.AdamW(parameters, lr=c['learning_rate'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: lr_factor(n, c['steps'], c['warmup']))
    latest = out / 'latest.pt'
    start, elapsed = 0, 0.
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved['contract'] != contract:
            raise RuntimeError('Resume contract changed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start, elapsed = saved['step'], saved['seconds']
    if not (out / 'initial_validation.json').exists():
        validate(c, model, action, heldout, features, out / 'initial_validation.json', smoke)
    began = time.monotonic()
    tracker = None
    if cfg.get('wandb_project') and not smoke:
        try:
            import wandb
            tracker = wandb.init(project=cfg['wandb_project'], name='visual_readout_' + arm,
                id=c['identity'][:12] + '_' + arm, resume='allow', config=contract, dir=str(out),
                settings=wandb.Settings(init_timeout=20))
        except Exception as exc:
            print('WANDB_WARNING', exc, flush=True)
    def save(n):
        atomic_torch_save(dict(format='simvla_visual_readout_v1', contract=contract, step=n,
            model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            seconds=elapsed + time.monotonic() - began), latest)
    try:
        progress = trange(start + 1, steps + 1, desc=arm, mininterval=2)
        for n in progress:
            rng = random.Random(c['seed'] * 100000 + n)
            age = (4 + (n - 1) % 4) if arm == 'warm_retain' else (5 + (n - 1) % 3)
            if smoke:
                age = (4 if n == 1 else 7) if arm == 'warm_retain' else (5 if n == 1 else 7)
            optimizer.zero_grad(set_to_none=True)
            metrics = dict(step=n, age=age, action_l1=0., condition_mse=0., readout_relative_mse=0.)
            for _ in range(c['batch_size']):
                seq = move_batch(collate_exact_teacher_sequences([data[rng.randrange(len(data))]]), device)
                features.bind(seq)
                loss_v = torch.zeros((), device=device)
                if arm != 'warm_retain':
                    pkv, tkv = readout_pair(model, features, seq, age)
                    loss_v = relative_readout_loss(pkv, tkv)
                loss_a, loss_c = torch.zeros((), device=device), torch.zeros((), device=device)
                if arm != 'distilled_readout':
                    s = sample_from_sequence(seq, age)
                    if n <= 4:
                        with torch.no_grad():
                            target = action.decode_action_from_condition(s['target_condition'], s['proprio'], steps=10, initial_noise=s['noise'])
                        if float((target - s['target_action']).abs().max()) > 2e-4:
                            raise RuntimeError('Teacher/cache mismatch')
                    z, bases, _ = unroll(model, seq, age, 8)
                    loss_c = sum(scaled_mse(b, seq['teacher_conditions'][:, j], s['anchor'], s['valid'])
                        for j, b in enumerate(bases)) / len(bases)
                    pred = action_prediction(action, z, s, steps=1)
                    loss_a = action_loss(pred, action.action_space.normalize_action(s['target_action']))
                loss = loss_a + c['condition_weight'] * loss_c + c['readout_weight'] * loss_v
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                (loss / c['batch_size']).backward()
                for k, value in [('action_l1', loss_a), ('condition_mse', loss_c), ('readout_relative_mse', loss_v)]:
                    metrics[k] += float(value.detach()) / c['batch_size']
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if not norm > 0:
                raise RuntimeError('Missing gradient')
            assert_frozen(frozen)
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen reader received parameter gradients')
            optimizer.step(); scheduler.step()
            metrics.update(seconds=elapsed + time.monotonic() - began,
                           lr=optimizer.param_groups[0]['lr'], gradient_norm=float(norm))
            progress.set_postfix(action=f"{metrics['action_l1']:.4f}", visual=f"{metrics['readout_relative_mse']:.4f}")
            if n % 50 == 0 or n in (1, steps):
                append_jsonl(out / 'metrics.jsonl', metrics)
                write_json(out / 'progress.json', metrics)
                if tracker:
                    try:
                        tracker.log(metrics, step=n)
                    except Exception as exc:
                        print('WANDB_WARNING', exc, flush=True)
            if n % 500 == 0 or n == steps:
                save(n)
    finally:
        if tracker:
            try:
                tracker.finish()
            except Exception:
                pass
    if frozen_hash(model) != frozen_before or state_hash(frozen) != base_before:
        raise RuntimeError('Frozen parameters changed')
    validate(c, model, action, heldout, features, out / 'validation.json', smoke)
    saved = torch.load(latest, map_location='cpu', weights_only=False)
    write_json(out / 'summary.json', dict(verdict='TRAIN_SMOKE_PASS' if smoke else 'TRAIN_COMPLETE',
        identity=c['identity'], arm=arm, checkpoint=str(latest), checkpoint_sha256=sha(latest),
        steps=steps, training_seconds=saved['seconds'],
        parent_training_steps=20000, total_training_steps=20000 + steps,
        parameters=contract['total_parameters'], trainable_parameters=contract['trainable_parameters'],
        protected_frozen_parameters_unchanged=True))


def export(arm):
    c = checked()
    summary = read_json(OUTPUT / 'train' / arm / 'summary.json')
    if summary['verdict'] != 'TRAIN_COMPLETE' or summary['identity'] != c['identity']:
        raise RuntimeError('Unfinished training')
    if sha(summary['checkpoint']) != summary['checkpoint_sha256']:
        raise RuntimeError('Model changed')
    stage = OUTPUT / 'bundles' / arm
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(summary['checkpoint'], stage / 'model.pt')
    write_json(stage / 'manifest.json', dict(**summary, source_sha256=c['source_sha256']))
    write_json(stage / 'READY.json', dict(manifest_sha256=sha(stage / 'manifest.json')))
    dest = str(INCOMING / arm)
    subprocess.run(['ssh', 'rb2', 'mkdir', '-p', dest], check=True, timeout=30)
    subprocess.run(['rsync', '-a', '--partial', '--exclude=READY.json', str(stage) + '/', f'rb2:{dest}/'], check=True, timeout=120)
    subprocess.run(['rsync', '-a', str(stage / 'READY.json'), f'rb2:{dest}/'], check=True, timeout=30)
    write_json(OUTPUT / 'exports' / f'{arm}.json', dict(verdict='EXPORTED', identity=c['identity']))


def model_payload(c, arm):
    import torch
    if prior.IS_RB2:
        directory = INCOMING / arm
        if read_json(directory / 'READY.json')['manifest_sha256'] != sha(directory / 'manifest.json'):
            raise RuntimeError('Incomplete transfer')
        spec, path = read_json(directory / 'manifest.json'), directory / 'model.pt'
        if spec['source_sha256'] != c['source_sha256']:
            raise RuntimeError('Transferred code differs')
    else:
        spec = read_json(OUTPUT / 'train' / arm / 'summary.json')
        path = Path(spec['checkpoint'])
    if spec['verdict'] != 'TRAIN_COMPLETE' or spec['identity'] != c['identity']:
        raise RuntimeError('Invalid model completion')
    if sha(path) != spec['checkpoint_sha256']:
        raise RuntimeError('Checkpoint checksum differs')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload['format'] != 'simvla_visual_readout_v1' or payload['step'] != c['steps']
            or payload['contract']['identity'] != c['identity'] or payload['contract']['arm'] != arm):
        raise RuntimeError('Training provenance mismatch')
    return payload, spec['checkpoint_sha256']
