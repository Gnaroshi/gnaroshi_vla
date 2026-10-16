"""Matched 10K action/condition training, with no learned visual teacher.

Only prepare() consults geometry_target_train.checked(). Subsequent operations
validate copied input records and this implementation's source lock directly.
"""

import argparse
import copy
from pathlib import Path
import random
import time

from methods.latentloop.modules.short_visual_training import (
    ARMS, build_model, gradient_coverage, parameter_counts,
)
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha


OUTPUT = prior.STORAGE / 'results/simvla/short_visual_training/fresh_and_protected_nfe1_v1'
FORMAT = 'simvla_short_visual_training_v1'
FILES = (
    'methods/latentloop/modules/short_visual_training.py',
    'tools/simvla/short_visual_train.py',
    'tests/test_short_visual_training.py',
    'tools/simvla/short_visual_campaign.py',
    'tools/simvla/short_visual_eval.py',
    'tools/simvla/short_visual_reference_repair.py',
    'tools/simvla/geometry_target_confirmation.py',
    'architectures/simvla/wrappers/run_short_visual_training.sh',
    'tools/simvla/failure_interventions.py',
    'tools/simvla/gpu_followup_queue.py',
    'tools/simvla/visual_readout_campaign.py',
    'tools/simvla/attention_query_eval.py',
    'tools/simvla/age_specialization_eval.py',
    'tools/simvla/observation_correction_eval.py',
    'tools/simvla/condition_interval_compare.py',
    'tools/simvla/condition_output_split_eval.py',
    'tools/simvla/error_compensation_eval.py',
    'tools/simvla/trend_condition_eval.py',
    'tools/simvla/refresh_allocation.py',
    'architectures/simvla/wrappers/dcld_eval/rollout_runner.py',
    'architectures/simvla/adapters/dcld/simvla_action_adapter.py',
    'architectures/simvla/adapters/dcld/simvla_delta_obs_adapter.py',
    'architectures/simvla/adapters/latentloop/native_v0_policy.py',
    'architectures/simvla/adapters/latentloop/native_v0_condition_hook.py',
    'architectures/simvla/adapters/latentloop/source_lock.py',
    'methods/latentloop/modules/native_simvla_v0.py',
    'methods/latentloop/modules/bounded_history.py',
    'methods/latentloop/modules/age_specialized_condition.py',
    'methods/latentloop/modules/visual_midpoint.py',
    'methods/latentloop/modules/visual_memory.py',
    'methods/latentloop/modules/visual_readout_distillation.py',
    'methods/latentloop/modules/visual_token_transport.py',
    'methods/latentloop/modules/condition_output_split.py',
    'methods/latentloop/modules/trend_condition.py',
    'methods/latentloop/modules/action_aligned_joint.py',
    'tools/simvla/condition_output_split_train.py',
    'tools/simvla/rollout_state_repair.py',
    'tools/simvla/visual_midpoint_features.py',
    'tools/simvla/error_compensation_common.py',
    'tools/simvla/compile_benchmark.py',
    'architectures/simvla/adapters/latentloop/native_v0_runtime.py',
    'architectures/simvla/adapters/latentloop/native_v0_checkpoint.py',
    'architectures/simvla/adapters/latentloop/native_v0_dataset.py',
    'architectures/simvla/adapters/latentloop/efficient_multirate/shared_refinement_train.py',
    'architectures/simvla/adapters/latentloop/efficient_multirate/condition_mechanism.py',
    'architectures/simvla/adapters/latentloop/efficient_multirate/exact_teacher_cache.py',
    'architectures/simvla/adapters/latentloop/efficient_multirate/efficient_delta.py',
    'architectures/simvla/adapters/latentloop/efficient_multirate/action_aligned_train.py',
    'architectures/simvla/adapters/latentloop/efficient_multirate/latent_fidelity_analysis.py',
)


def _record(path):
    return dict(path=str(path), sha256=sha(path))


def _verify_record(record):
    if sha(record['path']) != record['sha256']:
        raise RuntimeError('Input changed: ' + record['path'])


def _protected_payload(c):
    import torch
    spec = c['model']
    if sha(spec['checkpoint']) != spec['sha256']:
        raise RuntimeError('Fresh10K parent changed')
    saved = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    contract = saved['contract']
    expected = dict(training_intervals=[4], action_mode='naive1',
                    bounded_history_variant='recurrent', initialization='fresh', seed=7)
    if (saved['step'] != 10000 or any(contract.get(k) != v for k, v in expected.items())
            or contract.get('total_training_steps', 10000) != 10000
            or contract.get('continuation') is not None):
        raise RuntimeError('Expected proven fresh10K recurrent K4, not a continuation')
    return saved


def prepare():
    from tools.simvla import geometry_target_train, visual_midpoint
    from tools.simvla.error_compensation_common import snapshots
    old = geometry_target_train.checked()
    model = copy.deepcopy(old['model'])
    _protected_payload(dict(model=model))
    parent_summary_path = Path(model['checkpoint']).parent / 'summary.json'
    parent_summary = read_json(parent_summary_path)
    if (parent_summary['steps'] != 10000
            or parent_summary['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE'
            or parent_summary['checkpoint_sha256'] != model['sha256']
            or parent_summary.get('prior_training_seconds', 0) != 0):
        raise RuntimeError('Fresh10K training summary mismatch')
    parent_seconds = parent_summary['training_seconds']
    cfg = model['config']
    if cfg.get('training_k_c') != 8:
        raise RuntimeError('Expected the original eight-query dataset configuration')
    vision_contract = visual_midpoint.OUTPUT / 'contract.json'
    vision = read_json(vision_contract)
    if vision['model']['config']['cache'] != cfg['cache']:
        raise RuntimeError('Vision and teacher cache differ')
    provenance = snapshots(cfg)
    assets = {str(p): sha(p) for directory in provenance.values()
              for p in sorted(Path(directory).rglob('*')) if p.is_file()}
    if not assets:
        raise RuntimeError('Missing frozen original snapshots')
    assets[str(cfg['norm_stats'])] = sha(cfg['norm_stats'])
    c = dict(model=model, manifest=copy.deepcopy(old['manifest']), arms=list(ARMS),
        steps=10000, batch_size=2, seed=7, warmup=500, learning_rate=1e-4,
        condition_weight=0.05, training_ages=[4, 5, 6, 7], student_nfe=1, teacher_nfe=10,
        training_interval=8, action_horizon=10, execution_horizon=5,
        source_sha256={f: sha(ROOT / f) for f in FILES},
        input_sources={
            'geometry_contract': _record(geometry_target_train.OUTPUT / 'contract.json'),
            'geometry_source': _record(ROOT / 'tools/simvla/geometry_target_train.py'),
            'vision_contract': _record(vision_contract),
            'teacher_cache_manifest': _record(Path(cfg['cache']) / 'manifest.json'),
            'runtime_metadata_checkpoint': _record(cfg['condition_checkpoint']),
            'fresh10k_training_summary': _record(parent_summary_path),
            'proven_k4_summary': dict(path=model['source_summary'],
                                     sha256=model['source_summary_sha256'])},
        frozen_original_sha256=assets, snapshots=provenance,
        features=dict(root=str(visual_midpoint.OUTPUT / 'features'), identity=vision['identity']),
        initialization={
            'fresh_joint': dict(parent_fresh_sha256='NONE', parent_training_steps=0,
                parent_training_seconds=0.,
                total_training_steps=10000, early='new NativeSimVLAV0 seed7',
                late='independent copy of the same fresh early initialization'),
            'protected_short': dict(parent_fresh_sha256=model['sha256'], parent_training_steps=10000,
                parent_training_seconds=parent_seconds,
                total_training_steps=20000, early='strict fresh10K recurrent K4 load, frozen',
                late='trainable copy of the proven fresh10K early, not a fresh initialization')},
        objective='Terminal first5 normalized action L1, student NFE1 versus cached original NFE10 with same noise, plus0.05 mean anchor-variance-scaled condition MSE over all unrolled ages.',
        gradient_routing='Fresh early receives ages1..3 condition losses and terminal action gradient through the full unroll. Protected early is frozen. No extra direct early action examples. Late and ALL visual parameters train in both arms.',
        inference='Early ages1..3; original vision age4; fresh TokenTransport(residual=True) predicts age4-relative KV deltas at ages5..7; NFE1. No teacher_visual or learned attention/readout targets.',
        gradient_audit='Two optimizer steps; then backward only on the same second batch. Nested zero up/output heads delay fresh transport gradients until after update2. Audit performs zero optimizer updates.',
        runtime_weight_ancestry='The shared loader may temporarily read native150K for shape/data metadata; discard it after construction. Neither student inherits its parameter values.',
        evaluation=dict(k1_exact_parity=list(ARMS), k4_parent_parity=['protected_short'],
                        k4_success_rate_required=list(ARMS), no_performance_gate=True))
    c['identity'] = prior.digest(c)
    path = OUTPUT / 'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Short-training contract changed')
    write_json(path, c)
    return c


def checked():
    c = read_json(OUTPUT / 'contract.json')
    if c['identity'] != prior.digest({k: v for k, v in c.items() if k != 'identity'}):
        raise RuntimeError('Contract identity mismatch')
    for path, expected in c['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Source changed: ' + path)
    for record in c['input_sources'].values():
        _verify_record(record)
    for path, expected in c['frozen_original_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen original asset changed: ' + path)
    if sha(c['model']['checkpoint']) != c['model']['sha256']:
        raise RuntimeError('Fresh10K parent changed')
    return c


def initial_model(c, arm, native):
    if arm not in ARMS:
        raise ValueError(arm)
    state = _protected_payload(c)['model'] if arm == 'protected_short' else None
    return build_model(native, arm, seed=c['seed'], early_state=state)


def sampling_step(c, step, smoke=False):
    if type(step) is not int or step < 1 or (smoke and step > 2):
        raise ValueError('Invalid training step')
    return random.Random(c['seed'] * 100000 + step), ((5, 7)[step - 1] if smoke else 4 + (step - 1) % 4)


def loss_for(c, model, action, seq, age, features):
    from tools.simvla.condition_output_split_train import unroll, action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from methods.latentloop.modules.trend_condition import scaled_mse
    from methods.latentloop.modules.action_aligned_joint import action_loss
    features.bind(seq)
    sample = sample_from_sequence(seq, age)
    condition, bases, _ = unroll(model, seq, age, 8)
    condition_loss = sum(scaled_mse(base, seq['teacher_conditions'][:, j],
                                   sample['anchor'], sample['valid'])
                         for j, base in enumerate(bases)) / len(bases)
    prediction = action_prediction(action, condition, sample, steps=1)
    action_l1 = action_loss(prediction, action.action_space.normalize_action(sample['target_action']))
    return action_l1 + c['condition_weight'] * condition_loss, action_l1, condition_loss


def frozen_audit(model, original):
    modules = [original] + ([model.early] if model.short_training_arm == 'protected_short' else [])
    if any(p.requires_grad or p.grad is not None for module in modules for p in module.parameters()):
        raise RuntimeError('Frozen original/early gradient contract changed')
    original_storage = {p.data_ptr() for p in original.parameters()}
    if any(p.data_ptr() in original_storage for p in model.parameters()):
        raise RuntimeError('Student aliases frozen original storage')
    return dict(parameters=sum(p.numel() for module in modules for p in module.parameters()),
                parameter_bytes=sum(p.numel() * p.element_size() for module in modules for p in module.parameters()),
                gradient_bytes=0, student_original_storage_disjoint=True)


def validate(c, model, action, heldout, features, path, smoke=False):
    import torch
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from methods.latentloop.modules.condition_output_split import geometry
    from tools.simvla.condition_output_split_train import unroll, action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    count = 2 if smoke else 30
    indices = _balanced_indices(heldout.identities, limit=count, seed=c['seed'])
    if len(indices) != count:
        raise RuntimeError('Insufficient heldout windows')
    device = next(model.parameters()).device
    records = []
    with torch.no_grad():
        for index in indices:
            seq = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
            features.bind(seq)
            for age in range(1, 8):
                condition, _, _ = unroll(model, seq, age, 8)
                sample = sample_from_sequence(seq, age)
                pred = action_prediction(action, condition, sample, steps=1, grad=False)
                original1 = action_prediction(action, sample['target_condition'], sample, steps=1, grad=False)
                target = action.action_space.normalize_action(sample['target_action'])
                records.append(dict(window=index, age=age, action_l1=float(action_loss(pred, target)),
                    action_l1_vs_original_nfe1=float(action_loss(pred, original1)),
                    original_nfe1_l1_vs_original_nfe10=float(action_loss(original1, target)),
                    condition=geometry(condition, sample['target_condition'], sample['valid'])))
    write_json(path, dict(identity=c['identity'], indices=indices, records=records,
        reference='Same heldout window, current query and noise; condition raw MSE/cosine versus original backbone; normalized first5 action L1 versus cached original NFE10.',
        no_selection_gate=True))


def _check_saved(saved, contract, steps):
    if (saved.get('format') != FORMAT or saved.get('identity') != contract['identity']
            or saved.get('arm') != contract['arm'] or saved.get('contract') != contract
            or saved.get('contract_sha256') != prior.digest(contract)
            or type(saved.get('step')) is not int or not 0 <= saved['step'] <= steps):
        raise RuntimeError('Resume/checkpoint contract mismatch')


def model_payload(c, arm, smoke=False):
    import torch
    if arm not in ARMS:
        raise ValueError(arm)
    steps, verdict = (2, 'TRAIN_SMOKE_PASS') if smoke else (c['steps'], 'TRAIN_COMPLETE')
    summary = read_json(OUTPUT / ('train_smoke' if smoke else 'train') / arm / 'summary.json')
    if (summary['identity'], summary['arm'], summary['verdict'], summary['steps']) != (c['identity'], arm, verdict, steps):
        raise RuntimeError('Incomplete or mixed checkpoint')
    if sha(summary['checkpoint']) != summary['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    saved = torch.load(summary['checkpoint'], map_location='cpu', weights_only=False)
    contract = saved['contract']
    _check_saved(saved, contract, steps)
    if (saved['identity'] != c['identity'] or saved['arm'] != arm or saved['step'] != steps
            or summary['contract_sha256'] != saved['contract_sha256']
            or contract['initialization'] != c['initialization'][arm]
            or contract['run_steps'] != steps):
        raise RuntimeError('Checkpoint provenance mismatch')
    return saved, summary['checkpoint_sha256']


def train(arm, smoke=False):
    if arm not in ARMS:
        raise ValueError(arm)
    if prior.IS_RB2:
        raise RuntimeError('Training is assigned to sd1')
    import torch
    from tqdm import trange
    from tools.simvla.error_compensation_common import configure
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, append_jsonl
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime, lr_factor
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    c = checked()
    cfg = c['model']['config']
    configure(cfg)
    device, native, original, action, data, heldout = load_runtime(cfg, c['snapshots'])
    model = initial_model(c, arm, native).to(device).eval()
    if {p.data_ptr() for p in native.parameters()} & {p.data_ptr() for p in model.parameters()}:
        raise RuntimeError('Student aliases temporary native weights')
    del native
    features = CachedFeatures(c['features']['root'], c['features']['identity'])
    model.feature_provider = features
    early_before, original_before = state_hash(model.early), state_hash(original)
    counts = parameter_counts(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    steps = 2 if smoke else c['steps']
    contract = dict(identity=c['identity'], arm=arm, data=data.contract(), heldout=heldout.contract(),
        steps=c['steps'], run_steps=steps, seed=c['seed'], batch_size=c['batch_size'],
        initialization=c['initialization'][arm], initial_model_sha256=state_hash(model),
        objective=c['objective'], gradient_routing=c['gradient_routing'],
        training_ages=c['training_ages'], smoke_ages=[5, 7] if smoke else None,
        gradient_audit=c['gradient_audit'], student_nfe=1, teacher_nfe=10,
        early_sha256=early_before, frozen_original_sha256=original_before,
        frozen_memory=frozen_audit(model, original), temporary_native_discarded=True,
        counts=counts, total_parameters=counts['total'], trainable_parameters=counts['trainable'],
        trainable_names=[n for n, p in model.named_parameters() if p.requires_grad],
        optimizer=dict(name='AdamW', lr=c['learning_rate'], weight_decay=0, warmup=c['warmup'],
                       schedule='cosine to0.1x', clip=1))
    out = OUTPUT / ('train_smoke' if smoke else 'train') / arm
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out / 'training_contract.json'
    if contract_path.exists() and read_json(contract_path) != contract:
        raise RuntimeError('Training contract changed')
    optimizer = torch.optim.AdamW(parameters, lr=c['learning_rate'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: lr_factor(n, c['steps'], c['warmup']))
    latest = out / 'latest.pt'
    start, elapsed, coverage = 0, 0., {name: False for name in gradient_coverage(model)}
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        _check_saved(saved, contract, steps)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, elapsed, coverage = saved['step'], saved['seconds'], saved['gradient_coverage']
        del saved
    write_json(contract_path, contract)
    if state_hash(original) != original_before or (arm == 'protected_short' and state_hash(model.early) != early_before):
        raise RuntimeError('Frozen weights changed at resume')
    began = time.monotonic()

    def sequence(index):
        return move_batch(collate_exact_teacher_sequences([data[index]]), device)

    def record_gradients():
        for name, nonzero in gradient_coverage(model).items():
            coverage[name] = coverage[name] or nonzero
        frozen_audit(model, original)

    def save(step):
        atomic_torch_save(dict(format=FORMAT, identity=c['identity'], arm=arm, contract=contract,
            contract_sha256=prior.digest(contract), step=step, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            gradient_coverage=coverage, seconds=elapsed + time.monotonic() - began), latest)

    for step in trange(start + 1, steps + 1, desc=arm, mininterval=2):
        rng, age = sampling_step(c, step, smoke)
        indices = [rng.randrange(len(data)) for _ in range(c['batch_size'])]
        optimizer.zero_grad(set_to_none=True)
        metrics = dict(step=step, age=age, indices=indices, action_l1=0., condition_mse=0.)
        for index in indices:
            seq = sequence(index)
            if step <= 4:
                sample = sample_from_sequence(seq, age)
                with torch.no_grad():
                    teacher = action.decode_action_from_condition(sample['target_condition'], sample['proprio'],
                        steps=10, initial_noise=sample['noise'])
                difference = float((teacher - sample['target_action']).abs().max())
                if difference > 2e-4 or not torch.isfinite(teacher).all():
                    raise RuntimeError(f'Teacher/cache mismatch: {difference}')
            loss, loss_a, loss_c = loss_for(c, model, action, seq, age, features)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite loss')
            (loss / c['batch_size']).backward()
            metrics['action_l1'] += float(loss_a.detach()) / c['batch_size']
            metrics['condition_mse'] += float(loss_c.detach()) / c['batch_size']
        record_gradients()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if not norm > 0:
            raise RuntimeError('Missing gradient')
        optimizer.step()
        scheduler.step()
        if step == 2:
            # Fresh up and visual.output are nested zero heads. Test the graph
            # AFTER update2, using the same batch and no third optimizer update.
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                loss, _, _ = loss_for(c, model, action, sequence(index), age, features)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite gradient audit loss')
                (loss / c['batch_size']).backward()
            record_gradients()
            missing = [name for name, observed in coverage.items() if not observed]
            if missing:
                raise RuntimeError('Missing post-update2 block gradients: ' + ', '.join(missing))
            write_json(out / 'gradient_audit.json', dict(after_optimizer_steps=2,
                additional_optimizer_steps=0, reused_indices=indices, age=age, coverage=coverage,
                frozen_memory=frozen_audit(model, original)))
            optimizer.zero_grad(set_to_none=True)
        metrics.update(seconds=elapsed + time.monotonic() - began,
            lr=optimizer.param_groups[0]['lr'], gradient_norm=float(norm))
        append_jsonl(out / 'progress.jsonl', metrics)
        write_json(out / 'progress.json', metrics)
        if step % 500 == 0 or step == steps:
            if state_hash(original) != original_before or (arm == 'protected_short' and state_hash(model.early) != early_before):
                raise RuntimeError('Frozen weights changed')
            save(step)
    if not all(coverage.values()):
        raise RuntimeError('Incomplete gradient coverage')
    optimizer.zero_grad(set_to_none=True)
    validation_began = time.monotonic()
    validate(c, model, action, heldout, features, out / 'validation.json', smoke)
    validation_seconds = time.monotonic() - validation_began
    frozen_audit(model, original)
    early_after, original_after = state_hash(model.early), state_hash(original)
    if original_after != original_before or (arm == 'protected_short' and early_after != early_before):
        raise RuntimeError('Frozen weights changed')
    saved = torch.load(latest, map_location='cpu', weights_only=False)
    parent_steps = c['initialization'][arm]['parent_training_steps']
    prior_seconds = c['initialization'][arm]['parent_training_seconds']
    write_json(out / 'summary.json', dict(verdict='TRAIN_SMOKE_PASS' if smoke else 'TRAIN_COMPLETE',
        identity=c['identity'], arm=arm, checkpoint=str(latest), checkpoint_sha256=sha(latest),
        contract_sha256=prior.digest(contract), steps=steps, training_seconds=saved['seconds'],
        prior_training_seconds=prior_seconds, total_training_seconds=prior_seconds + saved['seconds'],
        validation_seconds=validation_seconds,
        total_training_and_validation_seconds=prior_seconds + saved['seconds'] + validation_seconds,
        parent_fresh_sha256=c['initialization'][arm]['parent_fresh_sha256'],
        parent_training_steps=parent_steps, total_training_steps=parent_steps + steps,
        planned_total_training_steps=c['initialization'][arm]['total_training_steps'],
        parameters=counts['total'], trainable_parameters=counts['trainable'], counts=counts,
        early_sha256_before=early_before, early_sha256_after=early_after,
        protected_early_unchanged=(early_after == early_before) if arm == 'protected_short' else None,
        frozen_original_unchanged=True, frozen_original_sha256=original_after,
        gradient_coverage=coverage, no_performance_gate=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'preflight', 'train', 'train-smoke'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.command in ('prepare', 'preflight'):
        c = prepare()
        print('PREFLIGHT_PASS', c['identity'], OUTPUT, flush=True)
    else:
        if args.arm is None:
            parser.error('--arm is required for training')
        train(args.arm, smoke=args.command == 'train-smoke')


if __name__ == '__main__':
    main()
