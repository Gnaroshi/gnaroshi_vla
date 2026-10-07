"""Paired 3K training with explicit condition/action gradient routing."""
import argparse
from pathlib import Path
import random
import time

import torch
from tqdm import trange

from methods.latentloop.modules.condition_output_split import ARMS, ConditionOutputSplit, geometry
from methods.latentloop.modules.trend_condition import scaled_mse
from methods.latentloop.modules.action_aligned_joint import action_loss
from tools.simvla.rollout_state_repair import sample_from_sequence
from tools.simvla.observation_correction_train import action_prediction
from tools.simvla.error_compensation_common import configure, identity, snapshots, read_json, write_json, sha
from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, append_jsonl
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime, lr_factor, assert_frozen
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash


def unroll(model, sequence, age, interval):
    ctx = model.prepare(sequence['anchor_condition'], sequence['image_sequence'][:,0],
        sequence['proprio_sequence'][:,0], sequence['valid_mask'], sequence['group_ids'], interval)
    bases = []
    for j in range(1, age+1):
        output, diagnostics = model.predict(ctx,j,sequence['image_sequence'][:,j],sequence['proprio_sequence'][:,j])
        bases.append(diagnostics['base'])
    return output, bases, diagnostics


@torch.no_grad()
def validate(c, model, action, heldout, out, smoke):
    indices = _balanced_indices(heldout.identities,limit=2 if smoke else 30,seed=c['seed'])
    records = []
    for index in indices:
        sequence = move_batch(collate_exact_teacher_sequences([heldout[index]]),'cuda')
        for interval in (4,8):
            ctx = model.prepare(sequence['anchor_condition'],sequence['image_sequence'][:,0],
                sequence['proprio_sequence'][:,0],sequence['valid_mask'],sequence['group_ids'],interval)
            for age in range(1,interval):
                s = sample_from_sequence(sequence,age)
                torch.cuda.synchronize(); began=time.perf_counter()
                condition,d = model.predict(ctx,age,s['images'],s['proprio'])
                torch.cuda.synchronize(); condition_ms=1000*(time.perf_counter()-began)
                torch.cuda.synchronize(); began=time.perf_counter()
                predicted=action_prediction(action,condition,s,grad=False)
                torch.cuda.synchronize(); action_ms=1000*(time.perf_counter()-began)
                target=action.action_space.normalize_action(s['target_action'])
                record=dict(window=index,interval=interval,age=age,
                    action_l1=float(action_loss(predicted,target)),condition_ms=condition_ms,action_ms=action_ms,
                    addition_rms=float(d['addition'][s['valid']].square().mean().sqrt()),
                    base=geometry(d['base'],s['target_condition'],s['valid']),
                    output=geometry(condition,s['target_condition'],s['valid']))
                if age==1:
                    record['hold']=geometry(s['anchor'],s['target_condition'],s['valid'])
                records.append(record)
    write_json(out/'validation.json',dict(records=records,indices=indices,
        identities=[heldout.identities[i] for i in indices],
        reference='Original backbone condition and cached original NFE10 normalized continuous actions',
        timing='sd1 eager device-synchronized component time; separate from rb2 full-policy latency'))


def train(c, arm, smoke=False):
    configure(c)
    out=Path(c['output'])/('smoke' if smoke else 'train')/arm
    out.mkdir(parents=True,exist_ok=True)
    run_id=identity(c)
    device,parent,frozen,action,data,heldout=load_runtime(c,snapshots(c))
    model=ConditionOutputSplit(parent,arm).to(device).eval().requires_grad_(True)
    frozen_hash,initial_hash=state_hash(frozen),state_hash(model)
    total=c['smoke_steps'] if smoke else c['steps']
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda n:lr_factor(n,total,min(c['warmup_steps'],max(1,total//10))))
    contract=dict(arm=arm,steps=total,training_intervals=[4,8],action_mode='naive3',teacher_steps=10,
        source_checkpoint_sha256=sha(c['condition_checkpoint']),data=data.contract(),heldout=heldout.contract(),
        initial_weights_sha256=initial_hash,batch_size=2,seed=c['seed'],
        condition_loss='Mean anchor-variance-scaled raw MSE at every unrolled age',
        action_loss='First-five normalized continuous action L1 at sampled final age, original10 same-noise target',
        condition_weight=c['condition_weight'],
        current_action_gradient='Only action_condition_updater; base and observation feature detached at its input',
        future_condition_gradient='carry_output can reach earlier extra heads; carry_base reaches base updater and encoder',
        optimizer=dict(name='AdamW',lr=c['learning_rate'],weight_decay=0,clip=1,
            schedule='warmup then cosine to 0.1x',warmup=min(c['warmup_steps'],max(1,total//10))))
    write_json(out/'training_contract.json',contract)
    latest=out/'latest.pt'; start=0; elapsed=0.
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False)
        if saved['identity']!=run_id or saved['contract']!=contract:
            raise RuntimeError('Incompatible resume')
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start,elapsed=saved['step'],saved['seconds']
    began=time.monotonic(); tracker=None
    if not smoke and c.get('wandb_project'):
        try:
            import wandb
            tracker=wandb.init(project=c['wandb_project'],name='condition_output_split_'+arm,
                id=run_id[:12]+'_'+arm,resume='allow',config=contract,dir=str(out),
                settings=wandb.Settings(init_timeout=20))
        except Exception as exc:
            print(f'WANDB_WARNING {exc}',flush=True)
    def save(n):
        atomic_torch_save(dict(format='simvla_condition_output_split_v1',identity=run_id,arm=arm,
            step=n,contract=contract,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),seconds=elapsed+time.monotonic()-began),latest)
        print(f'CHECKPOINT step={n} path={latest}',flush=True)
    try:
        progress=trange(start+1,total+1,desc=arm,mininterval=2)
        for n in progress:
            rng=random.Random(c['seed']*100000+n)
            interval=4 if n%2 else 8
            age=((n-1)//2)%(interval-1)+1
            optimizer.zero_grad(set_to_none=True)
            metrics=dict(step=n,interval=interval,age=age,action_l1=0.,condition_mse=0.)
            for _ in range(2):
                seq=move_batch(collate_exact_teacher_sequences([data[rng.randrange(len(data))]]),device)
                s=sample_from_sequence(seq,age)
                if n<=14:
                    with torch.no_grad():
                        teacher=action.decode_action_from_condition(s['target_condition'],s['proprio'],steps=10,initial_noise=s['noise'])
                    difference=float((teacher-s['target_action']).abs().max())
                    if difference>2e-4: raise RuntimeError(f'Cache/runtime mismatch {difference}')
                condition,bases,_=unroll(model,seq,age,interval)
                loss_c=sum(scaled_mse(base,seq['teacher_conditions'][:,j],s['anchor'],s['valid'])
                    for j,base in enumerate(bases))/len(bases)
                pred=action_prediction(action,condition,s)
                loss_a=action_loss(pred,action.action_space.normalize_action(s['target_action']))
                loss=loss_a+c['condition_weight']*loss_c
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                (loss/2).backward()
                metrics['action_l1']+=float(loss_a.detach())/2
                metrics['condition_mse']+=float(loss_c.detach())/2
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if not norm>0: raise RuntimeError('Missing gradients')
            assert_frozen(frozen)
            optimizer.step(); scheduler.step()
            metrics.update(lr=optimizer.param_groups[0]['lr'],seconds=elapsed+time.monotonic()-began,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
            progress.set_postfix(act=f"{metrics['action_l1']:.4f}",cond=f"{metrics['condition_mse']:.4f}")
            if n%50==0 or n in (1,total):
                append_jsonl(out/'metrics.jsonl',metrics)
                if tracker:
                    try: tracker.log(metrics,step=n)
                    except Exception as exc: print(f'WANDB_WARNING {exc}',flush=True)
            if n%500==0 or n==total: save(n)
    finally:
        if tracker:
            try: tracker.finish()
            except Exception: pass
    if state_hash(frozen)!=frozen_hash: raise RuntimeError('Frozen model changed')
    if start==0 and state_hash(model)==initial_hash: raise RuntimeError('Unchanged trainable model')
    validate(c,model,action,heldout,out,smoke)
    saved=torch.load(latest,map_location='cpu',weights_only=False)
    write_json(out/'summary.json',dict(identity=run_id,verdict='SMOKE_PASS' if smoke else 'TRAIN_AND_OFFLINE_COMPLETE',
        steps=total,checkpoint=str(latest),checkpoint_sha256=sha(latest),training_seconds=saved['seconds'],
        parameters=sum(p.numel() for p in model.parameters()),initial_weights_sha256=initial_hash,
        architecture=arm,frozen_teacher_unchanged=True))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); train(read_json(a.config),a.arm,a.smoke)
