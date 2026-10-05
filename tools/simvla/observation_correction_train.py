"""Matched causal sequence training with actual predecessor prediction errors."""
import argparse
import copy
from pathlib import Path
import random
import time

import torch
from tqdm import trange

from methods.latentloop.modules.observation_correction import ARMS, ObservationCorrection
from methods.latentloop.modules.observed_progress import build_model
from methods.latentloop.modules.trend_condition import scaled_mse
from methods.latentloop.modules.action_aligned_joint import action_loss
from tools.simvla.rollout_state_repair import load_selected, sample_from_sequence
from tools.simvla.error_compensation_common import configure, identity, snapshots, read_json, write_json, sha
from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, append_jsonl
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime, lr_factor, assert_frozen
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash


def unroll(model, sequence, age, interval, error_source=None):
    ctx = model.prepare(sequence['anchor_condition'], sequence['image_sequence'][:, 0],
        sequence['proprio_sequence'][:, 0], sequence['valid_mask'].bool(), sequence['group_ids'], interval)
    for j in range(1, age + 1):
        if j == age and age > 1 and error_source is not None:
            with torch.no_grad():
                # The frozen predecessor's error belongs to this SAME sequence,
                # same token layout and previous query. Never transplant task tokens.
                prior, _, _ = error_source.sequence(sequence, age - 1)
            ctx.previous = prior.detach()
        condition, diagnostics = model.predict(ctx, j, sequence['image_sequence'][:, j],
            sequence['proprio_sequence'][:, j])
    return condition, diagnostics


def action_prediction(action, condition, s, *, grad=True):
    return action.decode_action_from_condition(condition, s['proprio'], steps=3,
        initial_noise=s['noise'], requires_grad=grad, return_debug=True).final_action_latent


@torch.no_grad()
def validate(c, model, source, action, heldout, out, smoke):
    ids = _balanced_indices(heldout.identities, limit=2 if smoke else 30, seed=c['seed'])
    records = []
    for index in ids:
        sequence = move_batch(collate_exact_teacher_sequences([heldout[index]]), 'cuda')
        for interval in (4, 8):
            for age in range(1, interval):
                s = sample_from_sequence(sequence, age)
                for recovery in (False, True):
                    condition, d = unroll(model, sequence, age, interval, source if recovery else None)
                    predicted = action_prediction(action, condition, s, grad=False)
                    record = dict(window=index, interval=interval, age=age, recovery=recovery,
                        action_l1=float(action_loss(predicted, action.action_space.normalize_action(s['target_action']))),
                        condition_mse=float(scaled_mse(condition, s['target_condition'], s['anchor'], s['valid'])),
                        condition_cosine=float(torch.nn.functional.cosine_similarity(condition.float(),
                            s['target_condition'].float(), dim=-1)[s['valid']].mean()))
                    if 'measured' in d:
                        record.update(gain=float(d['gain'][s['valid']].mean()),
                            prediction_mse=float(scaled_mse(d['predicted'], s['target_condition'], s['anchor'], s['valid'])),
                            measurement_mse=float(scaled_mse(d['measured'], s['target_condition'], s['anchor'], s['valid'])))
                    records.append(record)
    groups = {}
    for interval in (4, 8):
        for recovery in (False, True):
            rows = [r for r in records if r['interval'] == interval and r['recovery'] == recovery]
            groups[f'k{interval}_' + ('recovery' if recovery else 'free_run')] = {
                key: sum(r[key] for r in rows) / len(rows)
                for key in rows[0] if key not in ('window', 'interval', 'age', 'recovery')}
    write_json(out/'validation.json', dict(groups=groups, queries=len(records), records=records,
        interpretation='Heldout latent/action fidelity and recovery, not online task success. No stopping threshold.'))


def train(c, arm, smoke=False):
    configure(c)
    out = Path(c['output'])/('smoke' if smoke else 'train')/arm
    out.mkdir(parents=True, exist_ok=True)
    run_id = identity(c)
    device, parent, frozen, action, data, heldout = load_runtime(c, snapshots(c))
    saved_source = load_selected(c)
    source = build_model(copy.deepcopy(parent), saved_source['arm'], max_age=7).to(device).eval()
    source.load_state_dict(saved_source['model'], strict=True)
    source.requires_grad_(False)
    model = ObservationCorrection(parent, arm).to(device).eval()
    model.initialize_from_anchor_model(saved_source['model'])
    model.requires_grad_(True)
    original_frozen, source_frozen = state_hash(frozen), state_hash(source)
    initial_model = state_hash(model)
    params = list(model.parameters())
    total = c['smoke_steps'] if smoke else c['steps']
    optimizer = torch.optim.AdamW(params, lr=c['learning_rate'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda n: lr_factor(n, total, min(c['warmup_steps'], max(1, total//10))))
    contract = dict(k_c=8, training_intervals=[4, 8], steps=total, arm=arm,
        source_checkpoint=c['selected_checkpoint'], action_mode='naive3', teacher_steps=10,
        data=data.contract(), heldout=heldout.contract(), batch_size=2,
        recovery='One clean self-unroll and one same-sequence frozen predecessor-state substitution per step; age1 is clean.',
        loss=dict(first_five_action_l1=1.0, normalized_condition_mse=c['condition_weight'],
            independent_measurement_mse=c['measurement_weight'] if arm=='observed_recurrent' else 0.0),
        optimizer='AdamW 1e-4 wd0 clip1; 150-step warmup then cosine to 0.1x at step3000',
        inference_state='Recurrently carries corrected condition; exact refresh resets all state',
        initialization='Shared existing b and updater weights; observed arm copies updater into an independent measurement head',
        supervision='Full-condition original10 same-noise teacher; fixed data split; no new collection')
    write_json(out/'training_contract.json', contract)
    latest = out/'latest.pt'
    start, elapsed = 0, 0.0
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved['identity'] != run_id or saved['arm'] != arm or saved['contract'] != contract:
            raise RuntimeError('Incompatible resume')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start, elapsed = saved['step'], saved['seconds']
    tracker = None
    if not smoke and c.get('wandb_project'):
        try:
            import wandb
            tracker = wandb.init(project=c['wandb_project'], name='observation_correction_'+arm,
                id=run_id[:12]+'_'+arm, resume='allow', config=contract, dir=str(out),
                settings=wandb.Settings(init_timeout=20))
        except Exception as exc:
            print(f'WANDB_WARNING {exc}', flush=True)
    began = time.monotonic()
    def save(n):
        atomic_torch_save(dict(format='simvla_observation_correction_v1', identity=run_id,
            arm=arm, step=n, contract=contract, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), seconds=elapsed+time.monotonic()-began), latest)
        print(f'CHECKPOINT step={n} path={latest}', flush=True)
    try:
        progress = trange(start+1, total+1, desc=arm, mininterval=2)
        for n in progress:
            rng = random.Random(c['seed']*100000+n)
            # Alternate horizons, cycle every age within each horizon.
            interval = 4 if n % 2 else 8
            age = ((n-1)//2) % (interval-1) + 1
            optimizer.zero_grad(set_to_none=True)
            metrics = dict(step=n, interval=interval, age=age, action_l1=0., condition_mse=0., measurement_mse=0.)
            for recovery in (False, True):
                index = rng.randrange(len(data))
                sequence = move_batch(collate_exact_teacher_sequences([data[index]]), device)
                s = sample_from_sequence(sequence, age)
                if n <= 14:
                    with torch.no_grad():
                        teacher = action.decode_action_from_condition(s['target_condition'], s['proprio'],
                            steps=10, initial_noise=s['noise'])
                    diff = float((teacher-s['target_action']).abs().max())
                    if diff > 2e-4:
                        raise RuntimeError(f'Teacher cache/runtime differs: {diff}')
                condition, d = unroll(model, sequence, age, interval, source if recovery else None)
                pred = action_prediction(action, condition, s)
                act_loss = action_loss(pred, action.action_space.normalize_action(s['target_action']))
                cond_loss = scaled_mse(condition, s['target_condition'], s['anchor'], s['valid'])
                measurement_loss = (scaled_mse(d['measured'], s['target_condition'], s['anchor'], s['valid'])
                    if 'measured' in d else cond_loss.new_zeros(()))
                loss = act_loss + c['condition_weight']*cond_loss + c['measurement_weight']*measurement_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite objective')
                (loss/2).backward()
                for key, value in (('action_l1',act_loss), ('condition_mse',cond_loss), ('measurement_mse',measurement_loss)):
                    metrics[key] += float(value.detach())/2
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if not norm > 0:
                raise RuntimeError('Missing gradients')
            assert_frozen(frozen, source)
            optimizer.step(); scheduler.step()
            metrics.update(lr=optimizer.param_groups[0]['lr'], seconds=elapsed+time.monotonic()-began)
            progress.set_postfix(act=f"{metrics['action_l1']:.4f}", cond=f"{metrics['condition_mse']:.4f}")
            if n%50 == 0 or n in (1,total):
                append_jsonl(out/'metrics.jsonl', metrics)
                if tracker:
                    try: tracker.log(metrics, step=n)
                    except Exception as exc: print(f'WANDB_WARNING {exc}', flush=True)
            if n%500 == 0 or n==total:
                save(n)
    finally:
        if tracker:
            try: tracker.finish()
            except Exception: pass
    if state_hash(frozen)!=original_frozen or state_hash(source)!=source_frozen:
        raise RuntimeError('Frozen teacher/source changed')
    if state_hash(model)==initial_model and not start:
        raise RuntimeError('Training did not change weights')
    validate(c, model, source, action, heldout, out, smoke)
    saved = torch.load(latest, map_location='cpu', weights_only=False)
    write_json(out/'summary.json', dict(identity=run_id, verdict='SMOKE_PASS' if smoke else 'TRAIN_AND_OFFLINE_COMPLETE',
        steps=total, checkpoint=str(latest), checkpoint_sha256=sha(latest),
        training_seconds=saved['seconds'], parameters=sum(p.numel() for p in params),
        frozen_teacher_unchanged=True, architecture=arm))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config', required=True)
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--smoke', action='store_true')
    a=p.parse_args(); train(read_json(a.config), a.arm, a.smoke)
