"""Bounded trend/residual pilot using cached teachers and the frozen action head."""
import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.latentloop.modules.trend_condition import (
    ARMS, TrendCondition, decomposition_loss, scaled_mse, teacher_decomposition,
)
from methods.latentloop.modules.action_aligned_joint import action_loss, differentiable_rollout
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop
from tools.simvla.error_compensation_common import configure, identity, read_json, snapshots, write_json
from tools.simvla.compile_runtime import ActionStep
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
from architectures.simvla.adapters.latentloop.native_v0_runtime import append_jsonl, move_batch
from .shared_refinement_train import load_runtime, lr_factor, assert_frozen
from .action_aligned_train import state_hash
from .generation_checkpoint import load_generation_checkpoint
from .generation_train import RankDisjointStepSampler
from .exact_teacher_cache import collate_exact_teacher_sequences
from .condition_mechanism import action_metrics, _balanced_indices


@torch.no_grad()
def offline(c, model, runtime, loop, step_model, output):
    device, _, _, action, _, heldout = runtime
    rows = []
    indices = _balanced_indices(heldout.identities, limit=c['heldout_windows'], seed=c['seed'])
    for index in tqdm(indices, desc='Heldout trend/action fidelity', mininterval=2):
        s = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
        ctx = model.prepare(s['anchor_condition'], s['image_sequence'][:, 0],
            s['proprio_sequence'][:, 0], s['valid_mask'].bool(), s['group_ids'])
        for age in (1, 2, 3):
            condition, residual = model.predict(ctx, age, s['image_sequence'][:, age], s['proprio_sequence'][:, age])
            predicted = differentiable_rollout(loop, step_model, condition,
                action.normalize_proprio(s['proprio_sequence'][:, age]), s['explicit_noises'][:, age-1])
            target_b, target_r = teacher_decomposition(ctx.anchor, s['teacher_conditions'], age)
            record = dict(index=index, age=age, task_id=int(s['task_id'][0]),
                condition_mse=float(scaled_mse(condition, s['teacher_conditions'][:, age-1], ctx.anchor, ctx.valid)),
                trend_endpoint_mse=float(scaled_mse(3*ctx.trend, 3*target_b, ctx.anchor, ctx.valid)),
                residual_mse=float(scaled_mse(residual, target_r, ctx.anchor, ctx.valid)),
                residual_target_energy=float(target_r[ctx.valid].square().mean()),
                full_displacement_energy=float((s['teacher_conditions'][:,age-1]-ctx.anchor)[ctx.valid].square().mean()),
                **action_metrics(action.action_space.postprocess(predicted), s['teacher_actions'][:,age-1]))
            rows.append(record)
    keys = [k for k in rows[0] if k not in ('index', 'age', 'task_id')]
    mean = lambda rs: {k: sum(r[k] for r in rs)/len(rs) for k in keys}
    write_json(output/'offline_queries.json', rows)
    write_json(output/'offline_summary.json', dict(verdict='OFFLINE_COMPLETE_NO_SR_GATE', queries=len(rows),
        metrics=mean(rows), by_condition_age={str(a): mean([r for r in rows if r['age']==a]) for a in (1,2,3)}))


def run(c, arm, smoke=False):
    configure(c)
    run_id = identity(c)
    output = Path(c['output'])/('smoke' if smoke else 'train')/arm
    output.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(c, snapshots(c))
    device, parent, frozen, action, train, heldout = runtime
    torch.manual_seed(c['seed'])
    model = TrendCondition(parent, arm).to(device).requires_grad_(True).eval()
    generation, _ = load_generation_checkpoint(c['generation_checkpoint'], device=device)
    generation.requires_grad_(False).eval()
    initial_generation = state_hash(generation)
    loop = SimVLAGenerationLoop(generation, frozen.transformer.action_decoder).eval()
    step_model = ActionStep(frozen.transformer).eval()
    initial_model = state_hash(model)
    params = list(model.parameters())
    total = c['smoke_steps'] if smoke else c['steps']
    stage1 = total//2 if smoke else c['decomposition_steps']
    if not 0 < stage1 < total:
        raise ValueError('Both training stages must be nonempty')
    contract = dict(identity=run_id, arm=arm, steps=total, decomposition_steps=stage1,
        action_steps=total-stage1, parameters=sum(p.numel() for p in params),
        trainable_names=[n for n,p in model.named_parameters()],
        mean_target='(teacher_C3-original_C0)/3; training labels only',
        inference='original_C0 + age*predicted_trend + absolute_residual; no previous prediction input',
        loss_stage1='anchor-variance-scaled MSE; mean of trend endpoint and residual terms when both exist',
        loss_stage2='normalized executed first-five action L1 only; all active Condition branches adapted',
        interpretation='Trend is mean-supervised during initialization, action-adapted in stage2; no exact mean guarantee after training',
        optimizer='AdamW lr1e-4 wd0 clip1; separate optimizer and cosine schedule for each stage',
        generation_frozen=True, original_simvla_frozen=True, k_c=4, n_g=3, integration_steps=10, H=10, R=5,
        train=train.contract(), heldout=heldout.contract(),
        forecast='all three residuals batched at refresh without later observations' if arm=='trend_forecast' else None)
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
                age=(step-1)%3+1
                condition,trend,residual=model.sequence(s,age)
                if step<=3:
                    with torch.no_grad():
                        exact=action.decode_action_from_condition(s['teacher_conditions'][:,age-1],
                            s['proprio_sequence'][:,age],steps=10,initial_noise=s['explicit_noises'][:,age-1])
                    diff=float((exact-s['teacher_actions'][:,age-1]).abs().max())
                    write_json(output/f'batch_step{step}.json',dict(age=age,max_teacher_action_diff=diff))
                    if diff>2e-4: raise RuntimeError(f'Teacher/cache mismatch {diff}')
                optimizer.zero_grad(set_to_none=True)
                if stage=='decomposition':
                    loss=decomposition_loss(model,s,age,condition,trend,residual)
                else:
                    predicted=differentiable_rollout(loop,step_model,condition,
                        action.normalize_proprio(s['proprio_sequence'][:,age]),s['explicit_noises'][:,age-1])
                    loss=action_loss(predicted,action.action_space.normalize_action(s['teacher_actions'][:,age-1]))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
                if not norm>0: raise RuntimeError('No active gradient')
                assert_frozen(frozen,generation)
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
        finally:
            if tracker:
                try: tracker.finish()
                except Exception as error: print(f'WANDB_FINISH_WARNING {error}',flush=True)
            progress.close()
            del loader
    if state_hash(generation)!=initial_generation: raise RuntimeError('Frozen Generation mutated')
    if state_hash(model)==initial_model: raise RuntimeError('Condition weights unchanged')
    write_json(output/'state_audit.json',dict(initial=initial,final_condition=state_hash(model),
        generation_unchanged=True,original_simvla_frozen=True))
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
