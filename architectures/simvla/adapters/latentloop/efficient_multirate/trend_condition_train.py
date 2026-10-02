"""Bounded trend/residual pilot using cached teachers and the frozen action head."""
import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.latentloop.modules.trend_condition import (
    MODEL_ARMS as ARMS, TrendCondition, decomposition_loss, scaled_mse, teacher_decomposition,
)
from methods.latentloop.modules.action_aligned_joint import action_loss, differentiable_rollout
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop
from tools.simvla.error_compensation_common import configure, identity, read_json, snapshots, write_json
from tools.simvla.compile_runtime import ActionStep
from tools.simvla.compile_benchmark import sha
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
from architectures.simvla.adapters.latentloop.native_v0_runtime import append_jsonl, move_batch
from .shared_refinement_train import load_runtime, lr_factor, assert_frozen
from .action_aligned_train import state_hash
from .generation_checkpoint import load_generation_checkpoint
from .generation_train import RankDisjointStepSampler
from .exact_teacher_cache import collate_exact_teacher_sequences
from .condition_mechanism import action_metrics, _balanced_indices
from methods.latentloop.modules.observed_progress import ARMS as PROGRESS_ARMS, build_model, projection_coefficient

ARMS = (*ARMS, *PROGRESS_ARMS)


@torch.no_grad()
def offline(c, model, runtime, loop, step_model, output):
    device, _, _, action, _, heldout = runtime
    rows = []
    indices = _balanced_indices(heldout.identities, limit=c['heldout_windows'], seed=c['seed'])
    for index in tqdm(indices, desc='Heldout trend/action fidelity', mininterval=2):
        s = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
        ctx = model.prepare(s['anchor_condition'], s['image_sequence'][:, 0],
            s['proprio_sequence'][:, 0], s['valid_mask'].bool(), s['group_ids'])
        for age in range(1, model.max_age + 1):
            condition, residual = model.predict(ctx, age, s['image_sequence'][:, age], s['proprio_sequence'][:, age])
            predicted = differentiable_rollout(loop, step_model, condition,
                action.normalize_proprio(s['proprio_sequence'][:, age]), s['explicit_noises'][:, age-1])
            target_b, target_r = teacher_decomposition(ctx.anchor, s['teacher_conditions'], age)
            base = ctx.anchor + age * ctx.trend
            if model.arm in PROGRESS_ARMS:
                target_alpha = projection_coefficient(s['teacher_conditions'][:, age-1]-ctx.anchor,
                    ctx.trend, ctx.anchor, ctx.valid)
                target_r = s['teacher_conditions'][:, age-1]-ctx.anchor-target_alpha*ctx.trend
            elif model.arm == 'frozen_trend_residual':
                target_r = s['teacher_conditions'][:, age-1] - base
            record = dict(index=index, age=age, task_id=int(s['task_id'][0]),
                condition_mse=float(scaled_mse(condition, s['teacher_conditions'][:, age-1], ctx.anchor, ctx.valid)),
                condition_cosine=float(torch.nn.functional.cosine_similarity(condition[ctx.valid], s['teacher_conditions'][:, age-1][ctx.valid], dim=-1).mean()),
                base_condition_mse=float(scaled_mse(base, s['teacher_conditions'][:, age-1], ctx.anchor, ctx.valid)),
                trend_endpoint_mse=float(scaled_mse(model.max_age*ctx.trend, model.max_age*target_b, ctx.anchor, ctx.valid)),
                residual_mse=float(scaled_mse(residual, target_r, ctx.anchor, ctx.valid)),
                residual_target_energy=float(target_r[ctx.valid].square().mean()),
                full_displacement_energy=float((s['teacher_conditions'][:,age-1]-ctx.anchor)[ctx.valid].square().mean()),
                **action_metrics(action.action_space.postprocess(predicted), s['teacher_actions'][:,age-1]))
            rows.append(record)
            if model.arm in PROGRESS_ARMS:
                record.update(progress=float(model.last_alpha.flatten()[0]),
                    oracle_progress=float(target_alpha.flatten()[0]))
    keys = [k for k in rows[0] if k not in ('index', 'age', 'task_id')]
    mean = lambda rs: {k: sum(r[k] for r in rs)/len(rs) for k in keys}
    write_json(output/'offline_queries.json', rows)
    write_json(output/'offline_summary.json', dict(verdict='OFFLINE_COMPLETE_NO_SR_GATE', queries=len(rows),
        metrics=mean(rows), by_condition_age={str(a): mean([r for r in rows if r['age']==a]) for a in range(1, model.max_age+1)}))


def run(c, arm, smoke=False):
    configure(c)
    run_id = identity(c)
    output = Path(c['output'])/('smoke' if smoke else 'train')/arm
    output.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(c, snapshots(c))
    device, parent, frozen, action, train, heldout = runtime
    torch.manual_seed(c['seed'])
    max_age = c.get('training_k_c', 4) - 1
    model = build_model(parent, arm, max_age=max_age).to(device).requires_grad_(True).eval()
    initial_trend = None
    if arm == 'frozen_trend_residual' or arm in PROGRESS_ARMS:
        source = c['frozen_trend_checkpoint']
        if sha(source['path']) != source['sha256']:
            raise RuntimeError('Frozen trend checkpoint checksum mismatch')
        payload = torch.load(source['path'], map_location=device, weights_only=False)
        if payload['arm'] != 'trend_only' or payload['step'] != 3000 or payload['contract']['k_c'] != max_age+1:
            raise RuntimeError('Frozen trend training horizon/arm mismatch')
        model.initialize_frozen_trend({k.removeprefix('trend_head.'):v for k,v in payload['model'].items() if k.startswith('trend_head.')})
        initial_trend = state_hash(model.trend_head)
    generation, _ = load_generation_checkpoint(c['generation_checkpoint'], device=device)
    generation.requires_grad_(False).eval()
    initial_generation = state_hash(generation)
    loop = SimVLAGenerationLoop(generation, frozen.transformer.action_decoder).eval()
    step_model = ActionStep(frozen.transformer).eval()
    initial_model = state_hash(model)
    params = [p for p in model.parameters() if p.requires_grad]
    total = c['smoke_steps'] if smoke else c['steps']
    stage1 = total//2 if smoke else c['decomposition_steps']
    if not 0 < stage1 < total:
        raise ValueError('Both training stages must be nonempty')
    contract = dict(identity=run_id, arm=arm, steps=total, decomposition_steps=stage1,
        action_steps=total-stage1, parameters=sum(p.numel() for p in params),
        trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        mean_target=f'(teacher_C{max_age}-original_C0)/{max_age}; training labels only',
        inference='original_C0 + age*predicted_trend + absolute_residual; no previous prediction input',
        loss_stage1='anchor-variance-scaled teacher-condition MSE with frozen b' if initial_trend else 'anchor-variance-scaled MSE; mean of trend endpoint and residual terms when both exist',
        loss_stage2='normalized executed first-five action L1; only E_j and its observation encoder train' if initial_trend else 'normalized executed first-five action L1; all active Condition branches train',
        interpretation='Previously trained b is frozen; E_j corrects teacher_Cj-(C0+j*b)' if initial_trend else 'Trend is mean-supervised during initialization, action-adapted in stage2; no exact mean guarantee after training',
        optimizer='AdamW lr1e-4 wd0 clip1; separate optimizer and cosine schedule for each stage',
        generation_frozen=True, original_simvla_frozen=True, k_c=max_age+1, n_g=3, integration_steps=10, H=10, R=5,
        trend_frozen=initial_trend is not None, frozen_trend_checkpoint=c.get('frozen_trend_checkpoint'),
        total_parameters=sum(p.numel() for p in model.parameters()),
        residual_target='teacher_Cj - (C0 + j*frozen_b)' if initial_trend else None,
        train=train.contract(), heldout=heldout.contract(),
        forecast='all three residuals batched at refresh without later observations' if arm=='trend_forecast' else None)
    if arm in PROGRESS_ARMS:
        contract.update(inference='C0 + alpha(current observations)*frozen_b + E_perpendicular',
            interpretation='Progress and residual are separated in the same weighted metric as condition MSE; no SR guarantee',
            loss_stage1='anchor-variance-scaled condition MSE; equivalent orthogonal progress/residual fitting',
            loss_stage2='normalized executed first-five action L1; observation, progress and optional residual train',
            residual_target='teacher_Cj-C0-oracle_alpha*b; oracle is used for offline analysis only')
    write_json(output/'training_contract.json', contract)
    latest = output/'latest.pt'
    saved = torch.load(latest, map_location=device, weights_only=False) if latest.exists() and not smoke else None
    start, elapsed = 0, 0.0
    if saved:
        if saved['identity']!=run_id or saved['arm']!=arm or saved['format']!='simvla_trend_condition_v1':
            raise RuntimeError('Resume identity mismatch')
        model.load_state_dict(saved['model'], strict=True)
        start, elapsed = saved['step'], saved['seconds']
    if not 0 <= start <= total:
        raise RuntimeError('Invalid resume step')
    initial = dict(generation=initial_generation, condition=initial_model)
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
                tracker=wandb.init(project=c['wandb_project'], name='trend_'+arm, id=run_id[:12]+'_'+arm,
                    resume='allow', config={**c,**contract}, dir=str(output), settings=wandb.Settings(init_timeout=20))
                write_json(output/'wandb.json', dict(mode=tracker.settings.mode,url=tracker.url))
            except Exception as error:
                write_json(output/'wandb.json', dict(local_logging_only=True,error=str(error)))
        begun = time.monotonic()
        current_stage = None
        progress=tqdm(loader, total=total, initial=start, desc='trend/'+arm, mininterval=2)
        try:
            for step, host in enumerate(progress, start=start+1):
                stage = 'decomposition' if step<=stage1 else 'action'
                stage_step = step if stage=='decomposition' else step-stage1
                stage_total = stage1 if stage=='decomposition' else total-stage1
                if stage!=current_stage:
                    optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=0)
                    warmup=max(1,min(c['warmup_steps'],stage_total//10))
                    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda s, st=stage_total, w=warmup:lr_factor(s,st,w))
                    if saved and saved['stage']==stage:
                        optimizer.load_state_dict(saved['optimizer'])
                        scheduler.load_state_dict(saved['scheduler'])
                    current_stage=stage
                s=move_batch(host,device)
                age=(step-1)%max_age+1
                condition,trend,residual=model.sequence(s,age)
                if step<=max_age:
                    with torch.no_grad():
                        exact=action.decode_action_from_condition(s['teacher_conditions'][:,age-1],
                            s['proprio_sequence'][:,age],steps=10,initial_noise=s['explicit_noises'][:,age-1])
                    diff=float((exact-s['teacher_actions'][:,age-1]).abs().max())
                    write_json(output/f'batch_step{step}.json',dict(age=age,max_teacher_action_diff=diff))
                    if diff>2e-4: raise RuntimeError(f'Teacher/cache mismatch {diff}')
                optimizer.zero_grad(set_to_none=True)
                if stage=='decomposition':
                    loss=(scaled_mse(condition,s['teacher_conditions'][:,age-1],s['anchor_condition'],s['valid_mask'].bool())
                        if arm in PROGRESS_ARMS else decomposition_loss(model,s,age,condition,trend,residual))
                else:
                    predicted=differentiable_rollout(loop,step_model,condition,
                        action.normalize_proprio(s['proprio_sequence'][:,age]),s['explicit_noises'][:,age-1])
                    loss=action_loss(predicted,action.action_space.normalize_action(s['teacher_actions'][:,age-1]))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
                if not norm>0: raise RuntimeError('No active gradient')
                assert_frozen(frozen,generation)
                if initial_trend is not None:
                    assert_frozen(model.trend_head)
                optimizer.step()
                scheduler.step()
                elapsed_now=elapsed+time.monotonic()-begun
                if step==1 or step%c['log_interval']==0 or step in (stage1,stage1+1,total):
                    metric=dict(step=step,stage=stage,age=age,loss=float(loss),gradient_norm=float(norm),
                        seconds=elapsed_now,lr=optimizer.param_groups[0]['lr'],
                        remaining_seconds_estimate=(total-step)*(time.monotonic()-begun)/(step-start),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    append_jsonl(output/'metrics.jsonl',metric)
                    if tracker:
                        try: tracker.log(metric,step=step)
                        except Exception as error: print(f'WANDB_LOG_WARNING {error}',flush=True)
                    progress.set_postfix(stage=stage,loss=f'{float(loss):.4g}')
                if step%c['save_interval']==0 or step in (stage1,total):
                    atomic_torch_save(dict(format='simvla_trend_condition_v1',identity=run_id,arm=arm,
                        model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                        step=step,stage=stage,seconds=elapsed_now,contract=contract),latest)
                    print(f'SAVED {arm} {step}/{total}',flush=True)
                if not smoke and step in c.get('validation_at_steps', []):
                    offline(c,model,runtime,loop,step_model,output/'validation'/f'step_{step}')
        finally:
            if tracker:
                try: tracker.finish()
                except Exception as error: print(f'WANDB_FINISH_WARNING {error}',flush=True)
            progress.close()
            del loader
    if state_hash(generation)!=initial_generation: raise RuntimeError('Frozen Generation mutated')
    if initial_trend is not None and state_hash(model.trend_head) != initial_trend:
        raise RuntimeError('Frozen trend mutated')
    if state_hash(model)==initial_model: raise RuntimeError('Condition weights unchanged')
    write_json(output/'state_audit.json',dict(initial=initial,final_condition=state_hash(model),
        generation_unchanged=True,original_simvla_frozen=True,
        frozen_trend_sha256=initial_trend, trend_unchanged=True if initial_trend else None))
    if not smoke and not (output/'offline_summary.json').exists():
        model.requires_grad_(False).eval()
        offline(c,model,runtime,loop,step_model,output)
    final=torch.load(latest,map_location='cpu',weights_only=False)
    write_json(output/'summary.json',dict(verdict='SMOKE_PASS' if smoke else 'TRAIN_AND_OFFLINE_COMPLETE',
        identity=run_id,steps=total,checkpoint=str(latest),training_seconds=final['seconds'],
        parameters=contract['parameters'],gpu=torch.cuda.get_device_name(0)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run(read_json(a.config),a.arm,a.smoke)
