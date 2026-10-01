"""Fine-tune small modules through the actual frozen action computation."""
import argparse
import hashlib
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.latentloop.modules.action_aligned_joint import (
    ARMS, trainable_groups, query_schedule, action_loss, condition_loss, differentiable_rollout,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop
from tools.simvla.action_aligned_campaign import CONFIG
from tools.simvla.compile_runtime import ActionStep
from tools.simvla.error_compensation_common import configure, identity, read_json, snapshots, write_json
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
from architectures.simvla.adapters.latentloop.native_v0_runtime import append_jsonl, move_batch
from .shared_refinement_train import load_runtime, lr_factor, assert_frozen
from .recursive_condition_inputs import query_inputs
from .generation_checkpoint import load_generation_checkpoint
from .generation_train import RankDisjointStepSampler
from .exact_teacher_cache import collate_exact_teacher_sequences
from .condition_mechanism import action_metrics, _balanced_indices


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def state_hash(module):
    return hashlib.sha256(''.join(n+tensor_hash(t) for n, t in module.state_dict().items()).encode()).hexdigest()


def inputs(adapter, action, sequence, age, fresh, *, train_condition=False):
    raw = sequence['proprio_sequence'][:, age]
    if fresh:
        condition = sequence['teacher_conditions'][:, age-1].detach()
    else:
        context, _, _, _ = query_inputs(adapter, action, sequence, age, track_grad=train_condition)
        condition = context.condition
    return (condition, action.normalize_proprio(raw), sequence['explicit_noises'][:, age-1],
        action.action_space.normalize_action(sequence['teacher_actions'][:, age-1]))


@torch.no_grad()
def offline(c, runtime, adapter, loop, step_model, output):
    device, _, _, action, _, heldout = runtime
    rows = []
    indices = _balanced_indices(heldout.identities, limit=c['heldout_windows'], seed=c['seed'])
    for index in tqdm(indices, desc='Heldout executed actions', mininterval=2):
        sequence = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
        for age in (1, 2, 3):
            condition, proprio, noise, target = inputs(adapter, action, sequence, age, False)
            prediction = differentiable_rollout(loop, step_model, condition, proprio, noise)
            decoded = action.action_space.postprocess(prediction)
            rows.append(dict(index=index, age=age, normalized_first5_l1=float(action_loss(prediction, target)),
                condition_normalized_mse=float(condition_loss(condition,
                    sequence['teacher_conditions'][:, age-1], sequence['valid_mask'].bool())),
                **action_metrics(decoded, sequence['teacher_actions'][:, age-1])))
    keys = [key for key in rows[0] if key not in ('index', 'age')]
    means = lambda rs: {key: sum(r[key] for r in rs)/len(rs) for key in keys}
    write_json(output/'offline_queries.json', rows)
    write_json(output/'offline_summary.json', dict(verdict='OFFLINE_COMPLETE_NO_SR_GATE',
        queries=len(rows), metrics=means(rows),
        by_condition_age={str(age): means([r for r in rows if r['age']==age]) for age in (1,2,3)}))


def run(c, arm, *, smoke=False):
    configure(c)
    run_id = identity(c)
    output = Path(c['output']) / ('smoke' if smoke else 'train') / arm
    output.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(c, snapshots(c))
    device, adapter, frozen, action, train, heldout = runtime
    updater, _ = load_generation_checkpoint(c['generation_checkpoint'], device=device)
    initial_hashes = dict(condition=state_hash(adapter), generation=state_hash(updater))
    train_c, train_g = trainable_groups(arm)
    adapter.requires_grad_(train_c).eval()
    updater.requires_grad_(train_g).eval()
    loop = SimVLAGenerationLoop(updater, frozen.transformer.action_decoder).eval()
    step_model = ActionStep(frozen.transformer).eval()
    named = [('condition.'+n, p) for n,p in adapter.named_parameters() if p.requires_grad]
    named += [('generation.'+n,p) for n,p in updater.named_parameters() if p.requires_grad]
    params = [p for _,p in named]
    optimizer = torch.optim.AdamW(params, lr=c['learning_rate'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: lr_factor(step, c['steps'], c['warmup_steps']))
    contract = dict(identity=run_id, arm=arm, initial_sha256=initial_hashes,
        parameters=sum(p.numel() for p in params), trainable_names=[n for n,p in named],
        train_condition=train_c, train_generation=train_g,
        loss='normalized condition MSE' if arm=='condition_latent' else 'normalized executed-prefix (first 5) action L1',
        gradient='full chain through frozen transformer and decoder; no detached student anchors',
        full_indices=[0,4,8], integration_steps=10, action_horizon=10, execution_horizon=5,
        query_cycle='predicted ages 1,2,3, then fresh original condition; fresh ages cycle 1,2,3',
        fresh_fraction=0.25, fixed_condition_step_fraction=0.75,
        planned_condition_updates=3750 if train_c else 0, planned_generation_updates=5000 if train_g else 0,
        inference='one Condition and one Generation; zero direct code; mask None as original parent',
        total_stored_parameters=sum(p.numel() for p in adapter.parameters())+sum(p.numel() for p in updater.parameters()),
        recorded_observations='teacher trajectories, not on-policy rollouts',
        train=train.contract(), heldout=heldout.contract(), scheduler_steps=c['steps'])
    write_json(output/'training_contract.json', contract)
    start, elapsed = 0, 0.0
    latest = output/'latest.pt'
    if latest.exists() and not smoke:
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved['identity']!=run_id or saved['arm']!=arm:
            raise RuntimeError('Resume identity mismatch')
        adapter.load_state_dict(saved['condition_state'], strict=True)
        updater.load_state_dict(saved['generation_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, elapsed = saved['step'], saved['seconds']
    total = c['smoke_steps'] if smoke else c['steps']
    if start > total:
        raise RuntimeError('Resume step exceeds training budget')
    if start < total:
        sampler = RankDisjointStepSampler(len(train), seed=c['seed'], rank=0, world_size=1,
            local_batch_size=c['batch_size'], start_step=start, stop_step=total)
        train.store._loaded.clear()
        heldout.store._loaded.clear()
        loader = DataLoader(train, batch_size=c['batch_size'], sampler=sampler,
            collate_fn=collate_exact_teacher_sequences, num_workers=0 if smoke else c['num_workers'],
            pin_memory=True, **({'multiprocessing_context':'spawn', 'persistent_workers':True}
                if not smoke and c['num_workers'] else {}))
        tracker = None
        if not smoke and c.get('wandb_project'):
            try:
                import wandb
                tracker = wandb.init(project=c['wandb_project'], name='action_aligned_'+arm,
                    id=run_id[:12]+'_'+arm, resume='allow', config={**c, **contract},
                    dir=str(output), settings=wandb.Settings(init_timeout=20))
                write_json(output/'wandb.json', dict(mode=tracker.settings.mode, url=tracker.url))
            except Exception as error:
                write_json(output/'wandb.json', dict(local_logging_only=True, error=str(error)))
        begun = time.monotonic()
        progress = tqdm(loader, total=total, initial=start, desc=arm, mininterval=2)
        try:
            for step, host in enumerate(progress, start=start+1):
                sequence = move_batch(host, device)
                age, fresh = query_schedule(step)
                condition, proprio, noise, target = inputs(adapter, action, sequence, age, fresh, train_condition=train_c)
                teacher_c = sequence['teacher_conditions'][:, age-1]
                if step <= 4:
                    with torch.no_grad():
                        exact = action.decode_action_from_condition(teacher_c, sequence['proprio_sequence'][:,age],
                            steps=10, initial_noise=noise, return_debug=True).action
                        check = differentiable_rollout(loop, step_model, condition, proprio, noise)
                    diff = float((exact-sequence['teacher_actions'][:,age-1]).abs().max())
                    write_json(output/f'batch_step{step}.json', dict(age=age, fresh=fresh,
                        task_id=host['task_id'].tolist(), episode_id=host['episode_id'],
                        anchor_query_index=host['anchor_query_index'].tolist(), max_teacher_action_diff=diff,
                        initial_action_sha256=tensor_hash(check)))
                    if diff > 2e-4:
                        raise RuntimeError(f'Teacher/cache action mismatch: {diff}')
                optimizer.zero_grad(set_to_none=True)
                if arm=='condition_latent':
                    loss = condition_loss(condition, teacher_c, sequence['valid_mask'].bool())
                else:
                    prediction = differentiable_rollout(loop, step_model, condition, proprio, noise)
                    loss = action_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                if loss.requires_grad:
                    loss.backward()
                gradients = {}
                for name, module, expected in [('condition', adapter, train_c and not fresh),
                                              ('generation', updater, train_g)]:
                    norms = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
                    norm = float(torch.stack(norms).norm()) if norms else 0.0
                    if expected != (norm > 0) or not torch.isfinite(torch.tensor(norm)):
                        raise RuntimeError(f'Unexpected {name} gradient: {norm}, expected active={expected}')
                    gradients[name+'_grad_norm'] = norm
                assert_frozen(frozen)
                if not train_c: assert_frozen(adapter)
                if not train_g: assert_frozen(updater)
                if loss.requires_grad:
                    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                scheduler.step()
                if step <= 4:
                    write_json(output/f'gradient_step{step}.json', dict(age=age, fresh=fresh, **gradients))
                if step==1 or step%c['log_interval']==0 or step==total:
                    metrics = dict(step=step, age=age, fresh=fresh, loss=float(loss),
                        lr=optimizer.param_groups[0]['lr'], seconds=elapsed+time.monotonic()-begun,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(), **gradients)
                    append_jsonl(output/'metrics.jsonl', metrics)
                    if tracker: tracker.log(metrics, step=step)
                    progress.set_postfix(loss=f'{float(loss):.4g}')
                if step%c['save_interval']==0 or step==total:
                    atomic_torch_save(dict(format='simvla_action_aligned_joint_v1', identity=run_id,
                        arm=arm, condition_state=adapter.state_dict(), generation_state=updater.state_dict(),
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step,
                        seconds=elapsed+time.monotonic()-begun, training_contract=contract), latest)
                    print(f'SAVED {arm} {step}/{total}', flush=True)
        finally:
            if tracker: tracker.finish()
    # Check fixed small-module states exactly, not only their gradient flags.
    final_hashes = dict(condition=state_hash(adapter), generation=state_hash(updater))
    for name, trainable in [('condition',train_c), ('generation',train_g)]:
        if (initial_hashes[name] != final_hashes[name]) != trainable:
            raise RuntimeError(f'{name} state mutation does not match training contract')
    write_json(output/'state_audit.json', dict(initial=initial_hashes, final=final_hashes,
        original_simvla_frozen=True, active_state_changed=True, inactive_state_unchanged=True))
    if not smoke:
        adapter.requires_grad_(False).eval()
        updater.requires_grad_(False).eval()
        if not (output/'offline_summary.json').exists():
            offline(c, runtime, adapter, loop, step_model, output)
    write_json(output/'summary.json', dict(verdict='SMOKE_PASS' if smoke else 'TRAIN_AND_OFFLINE_COMPLETE',
        identity=run_id, steps=total, checkpoint=str(latest), online_evaluation_required=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config', default=str(CONFIG))
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--smoke', action='store_true')
    a=p.parse_args()
    run(read_json(a.config), a.arm, smoke=a.smoke)
