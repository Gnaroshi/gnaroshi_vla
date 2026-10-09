"""Paired 3K training with explicit condition/action gradient routing."""
import argparse
import hashlib
from pathlib import Path
import random
import time

import torch
from tqdm import trange

from methods.latentloop.modules.condition_output_split import ARMS, GRADIENT_CONTRACTS, ConditionOutputSplit, geometry
from methods.latentloop.modules.trend_condition import scaled_mse
from methods.latentloop.modules.action_aligned_joint import action_loss
from tools.simvla.rollout_state_repair import sample_from_sequence
from tools.simvla.error_compensation_common import configure, identity, snapshots, read_json, write_json, sha
from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, append_jsonl
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime, lr_factor, assert_frozen
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash


def student_steps(c):
    n = c.get('student_steps', 3)
    if type(n) is not int or n not in (1, 2, 3):
        raise ValueError('Student action steps must be 1, 2 or 3')
    return n


def action_prediction(action, condition, s, *, steps=3, grad=True):
    return action.decode_action_from_condition(condition, s['proprio'], steps=steps,
        initial_noise=s['noise'], requires_grad=grad, return_debug=True).final_action_latent


def noise_supervision(c, action, sample, step, sample_index, *, heldout=False):
    """Keep paired teacher/student noise; local RNG never changes window sampling."""
    count = c.get('heldout_action_noise_samples' if heldout else 'action_noise_samples', 1)
    if type(count) is not int or count not in (1, 2, 3):
        raise ValueError('Unsupported noise sample count')
    pairs = [(sample, action.action_space.normalize_action(sample['target_action']).detach())]
    scope = 'heldout' if heldout else 'train'
    for index in range(1, count):
        key = f"condition_noise:{scope}:{c['seed']}:{step}:{sample_index}:{index}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63 - 1)
        generator = torch.Generator(device=sample['noise'].device).manual_seed(seed)
        noise = torch.randn(sample['noise'].shape, generator=generator,
            device=sample['noise'].device, dtype=sample['noise'].dtype)
        with torch.no_grad():
            target = action.decode_action_from_condition(sample['target_condition'],
                sample['proprio'], steps=10, initial_noise=noise)
            target = action.action_space.normalize_action(target).detach()
        pairs.append(({**sample, 'noise': noise}, target))
    return pairs


def check_gradient_control(current, control, *, smoke=False):
    keys = ('arm','training_intervals','action_mode','teacher_steps','source_checkpoint_sha256',
        'data','heldout','initial_weights_sha256','batch_size','seed','condition_loss','action_loss',
        'condition_weight','initialization','sample_step_offset','continuation','optimizer_initialization',
        'solver_transition')
    if not smoke:
        keys += ('steps','total_training_steps','optimizer')
    if current.get('action_gradient_mode') != 'joint' or control.get('action_gradient_mode','detached') != 'detached':
        raise RuntimeError('Gradient comparison modes do not match')
    for key in keys:
        if current[key] != control[key]:
            raise RuntimeError('Matched gradient control differs: '+key)


def check_continuation_contract(previous, current, c):
    gradient_change = previous.get('action_gradient_mode','detached') != current.get('action_gradient_mode','detached')
    if gradient_change:
        if not (c.get('gradient_transition') == 'detached_to_joint'
                and previous.get('action_gradient_mode','detached') == 'detached'
                and current.get('action_gradient_mode') == 'joint'
                and tuple(previous[k] for k in ('current_action_gradient','future_condition_gradient')) == GRADIENT_CONTRACTS['detached']
                and tuple(current[k] for k in ('current_action_gradient','future_condition_gradient')) == GRADIENT_CONTRACTS['joint']):
            raise RuntimeError('Unapproved continuation gradient change')
    for key in ('data','heldout','batch_size','seed','teacher_steps',
            'source_checkpoint_sha256','condition_weight','current_action_gradient','future_condition_gradient'):
        if gradient_change and key in ('current_action_gradient','future_condition_gradient'):
            continue
        if previous[key] != current[key]:
            raise RuntimeError('Continuation data/objective mismatch: '+key)
    if previous['action_mode'] != current['action_mode']:
        if not (previous['action_mode'] == 'naive3' and current['action_mode'] in ('naive1','naive2')
                and c.get('solver_transition') == 'naive3_to_'+current['action_mode']):
            raise RuntimeError('Unapproved continuation action solver change')


def build_initial_model(parent, c, arm):
    mode=c.get('initialization','pretrained')
    if mode not in ('pretrained','fresh','continuation'):
        raise ValueError('Unknown initialization: '+mode)
    if mode=='fresh':
        from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
        from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
        ref=next(parent.parameters())
        torch.manual_seed(c['seed'])
        parent=NativeSimVLAV0(num_views=parent.num_views,proprio_dim=parent.proprio_dim,
            condition_dim=parent.condition_dim,delta_dim=parent.delta_dim,rank_dim=parent.rank_dim,
            max_tokens=parent.condition_updater.max_tokens,
            num_token_groups=parent.condition_updater.num_token_groups).to(ref)
        install_exact_uint8_delta_path(parent)
    return ConditionOutputSplit(parent,arm,gradient_mode=c.get('action_gradient_mode','detached'))


def load_continuation(model,c,arm):
    if c.get('initialization')!='continuation':
        return None
    spec=c['initial_models'][arm]
    summary=read_json(spec['summary'])
    prior_steps = c.get('continuation_source_steps', 3000)
    if (summary['identity']!=spec['identity'] or summary['steps']!=prior_steps
            or summary['verdict']!='TRAIN_AND_OFFLINE_COMPLETE'):
        raise RuntimeError('Unfinished or mismatched continuation source')
    checkpoint=summary['checkpoint']
    if sha(checkpoint)!=summary['checkpoint_sha256']:
        raise RuntimeError('Continuation checkpoint hash changed')
    from tools.simvla.condition_output_split_eval import load_payload
    saved=load_payload(checkpoint,arm,spec['identity'],steps=prior_steps,
        action_mode=c.get('continuation_source_action_mode', 'naive3'))
    model.load_state_dict(saved['model'],strict=True)
    return dict(checkpoint_sha256=summary['checkpoint_sha256'],identity=spec['identity'],
        prior_steps=prior_steps,prior_training_seconds=summary['training_seconds'],contract=saved['contract'])


def sampling_step(c, local_step):
    n=c.get('sample_step_offset',0)+local_step
    if n<1: raise ValueError('Sample step must be positive')
    interval=4 if n%2 else 8
    return random.Random(c['seed']*100000+n),interval,((n-1)//2)%(interval-1)+1


def unroll(model, sequence, age, interval, *, codes=None):
    ctx = model.prepare(sequence['anchor_condition'], sequence['image_sequence'][:,0],
        sequence['proprio_sequence'][:,0], sequence['valid_mask'], sequence['group_ids'], interval)
    bases = []
    handle = model.delta_encoder.register_forward_hook(lambda _m, _inputs, output: codes.append(output)) if codes is not None else None
    try:
        for j in range(1, age+1):
            output, diagnostics = model.predict(ctx,j,sequence['image_sequence'][:,j],sequence['proprio_sequence'][:,j])
            bases.append(diagnostics['base'])
    finally:
        if handle is not None: handle.remove()
    return output, bases, diagnostics


@torch.no_grad()
def validate_features(c, model, auxiliary, heldout, out, smoke):
    indices = _balanced_indices(heldout.identities, limit=2 if smoke else 30, seed=c['seed'])
    records = []
    previous_codes = {}
    for index in indices:
        sequence = move_batch(collate_exact_teacher_sequences([heldout[index]]), 'cuda')
        for interval in (4, 8):
            codes = []
            unroll(model, sequence, interval-1, interval, codes=codes)
            _, values = auxiliary.objective(codes, sequence, c['feature_alignment']['mode'])
            _, zero = auxiliary.objective([torch.zeros_like(x) for x in codes], sequence,
                c['feature_alignment']['mode'])
            record = dict(window=index, interval=interval, **{k:float(v) for k,v in values.items()},
                zero_code={k:float(v) for k,v in zero.items()})
            if interval in previous_codes:
                _, other = auxiliary.objective(previous_codes[interval], sequence,c['feature_alignment']['mode'])
                record['other_window_code'] = {k:float(v) for k,v in other.items()}
            previous_codes[interval] = [x.detach() for x in codes]
            records.append(record)
    write_json(out/'feature_validation.json', dict(records=records,
        target='Image-token LayerNorm(original condition), or consecutive difference; cached teacher, heldout episodes'))


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
                predicted=action_prediction(action,condition,s,steps=student_steps(c),grad=False)
                torch.cuda.synchronize(); action_ms=1000*(time.perf_counter()-began)
                target=action.action_space.normalize_action(s['target_action'])
                record=dict(window=index,interval=interval,age=age,
                    action_l1=float(action_loss(predicted,target)),condition_ms=condition_ms,action_ms=action_ms,
                    addition_rms=float(d['addition'][s['valid']].square().mean().sqrt()),
                    base=geometry(d['base'],s['target_condition'],s['valid']),
                    output=geometry(condition,s['target_condition'],s['valid']))
                if age==1:
                    record['hold']=geometry(s['anchor'],s['target_condition'],s['valid'])
                if c.get('heldout_action_noise_samples', 1) > 1:
                    pairs = noise_supervision(c, action, s, index, age, heldout=True)
                    record['unseen_noise_action_l1'] = [float(action_loss(
                        action_prediction(action, condition, extra, steps=student_steps(c), grad=False), target))
                        for extra, target in pairs[1:]]
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
    model=build_initial_model(parent,c,arm).to(device).eval().requires_grad_(True)
    continuation=load_continuation(model,c,arm)
    frozen_hash,initial_hash=state_hash(frozen),state_hash(model)
    total=c['smoke_steps'] if smoke else c['steps']
    auxiliary = None
    if c.get('feature_alignment', {}).get('mode', 'none') != 'none':
        from methods.latentloop.modules.condition_feature_supervision import ConditionFeatureSupervision
        auxiliary = ConditionFeatureSupervision(model.delta_dim, model.condition_dim,
            model.condition_updater.max_tokens, c['seed']).to(device)
    trainable = list(model.parameters()) + (list(auxiliary.parameters()) if auxiliary is not None else [])
    optimizer=torch.optim.AdamW(trainable,lr=c['learning_rate'],weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda n:lr_factor(n,total,min(c['warmup_steps'],max(1,total//10))))
    contract=dict(arm=arm,steps=total,training_intervals=[4,8],action_mode=f'naive{student_steps(c)}',teacher_steps=10,
        source_checkpoint_sha256=sha(c['condition_checkpoint']),data=data.contract(),heldout=heldout.contract(),
        initial_weights_sha256=initial_hash,batch_size=2,seed=c['seed'],
        condition_loss='Mean anchor-variance-scaled raw MSE at every unrolled age',
        action_loss='First-five normalized continuous action L1 at sampled final age, original10 same-noise target',
        condition_weight=c['condition_weight'],
        current_action_gradient=GRADIENT_CONTRACTS[model.gradient_mode][0],
        future_condition_gradient=GRADIENT_CONTRACTS[model.gradient_mode][1],
        optimizer=dict(name='AdamW',lr=c['learning_rate'],weight_decay=0,clip=1,
            schedule='warmup then cosine to 0.1x',warmup=min(c['warmup_steps'],max(1,total//10))))
    if 'initialization' in c:
        contract.update(initialization=c['initialization'],sample_step_offset=c.get('sample_step_offset',0),
            continuation=continuation,optimizer_initialization='fresh at phase start',
            total_training_steps=c.get('sample_step_offset',0)+total)
    if 'action_gradient_mode' in c:
        contract.update(action_gradient_mode=model.gradient_mode,gradient_transition=c.get('gradient_transition'))
    if 'action_noise_samples' in c:
        contract.update(action_noise_samples=c['action_noise_samples'],
            action_loss='Mean first-five normalized continuous action L1 over paired noise targets at sampled final age',
            noise_contract='Cached original10 target plus independently seeded Gaussian original10 targets; same noise per teacher/student pair; updater inputs independent of noise',
            heldout_action_noise_samples=c.get('heldout_action_noise_samples', 1))
    if 'feature_alignment' in c:
        contract['feature_alignment'] = c['feature_alignment']
    if continuation:
        previous=continuation['contract']
        check_continuation_contract(previous,contract,c)
    if 'solver_transition' in c:
        contract['solver_transition']=c['solver_transition']
    if 'matched_controls' in c:
        check_gradient_control(contract,c['matched_controls'][arm]['contract'],smoke=smoke)
    write_json(out/'training_contract.json',contract)
    latest=out/'latest.pt'; start=0; elapsed=0.
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False)
        if saved['identity']!=run_id or saved['contract']!=contract:
            raise RuntimeError('Incompatible resume')
        model.load_state_dict(saved['model'],strict=True)
        if auxiliary is not None:
            auxiliary.load_state_dict(saved['feature_reader'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start,elapsed=saved['step'],saved['seconds']
    began=time.monotonic(); tracker=None
    if not smoke and c.get('wandb_project'):
        try:
            import wandb
            tracker=wandb.init(project=c['wandb_project'],name=c.get('run_label','condition_output_split')+'_'+arm,
                id=run_id[:12]+'_'+arm,resume='allow',config=contract,dir=str(out),
                settings=wandb.Settings(init_timeout=20))
        except Exception as exc:
            print(f'WANDB_WARNING {exc}',flush=True)
    def save(n):
        atomic_torch_save(dict(format='simvla_condition_output_split_v1',identity=run_id,arm=arm,
            step=n,contract=contract,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),seconds=elapsed+time.monotonic()-began,
            **({'feature_reader':auxiliary.state_dict()} if auxiliary is not None else {})),latest)
        print(f'CHECKPOINT step={n} path={latest}',flush=True)
    try:
        progress=trange(start+1,total+1,desc=arm,mininterval=2)
        for n in progress:
            rng,interval,age=sampling_step(c,n)
            optimizer.zero_grad(set_to_none=True)
            metrics=dict(step=n,interval=interval,age=age,action_l1=0.,condition_mse=0.,feature_loss=0.)
            for sample_index in range(2):
                seq=move_batch(collate_exact_teacher_sequences([data[rng.randrange(len(data))]]),device)
                s=sample_from_sequence(seq,age)
                if n<=14:
                    with torch.no_grad():
                        teacher=action.decode_action_from_condition(s['target_condition'],s['proprio'],steps=10,initial_noise=s['noise'])
                    difference=float((teacher-s['target_action']).abs().max())
                    if difference>2e-4: raise RuntimeError(f'Cache/runtime mismatch {difference}')
                codes = [] if auxiliary is not None else None
                condition,bases,_=unroll(model,seq,age,interval,codes=codes)
                loss_c=sum(scaled_mse(base,seq['teacher_conditions'][:,j],s['anchor'],s['valid'])
                    for j,base in enumerate(bases))/len(bases)
                pairs = noise_supervision(c, action, s, c.get('sample_step_offset', 0)+n, sample_index)
                loss_a = sum(action_loss(action_prediction(action, condition, item, steps=student_steps(c)), target)
                    for item, target in pairs) / len(pairs)
                # Zero-initialized residuals learn their output projection before encoder gradients open.
                gradient_check_step = 2 if c.get('initialization') == 'fresh' else 1
                if model.gradient_mode == 'joint' and n == gradient_check_step and sample_index == 0:
                    groups = {'observation_encoder':list(model.delta_encoder.parameters()),
                        'condition_updater':list(model.condition_updater.parameters()),
                        'action_condition_updater':list(model.action_condition_updater.parameters())}
                    parameters = [p for ps in groups.values() for p in ps]
                    gradients = torch.autograd.grad(loss_a,parameters,retain_graph=True,allow_unused=True)
                    norms,offset = {},0
                    for name,ps in groups.items():
                        gs=gradients[offset:offset+len(ps)]; offset+=len(ps)
                        norm=sum(float(g.detach().float().square().sum()) for g in gs if g is not None)**.5
                        if not 0 < norm < float('inf'):
                            raise RuntimeError('Missing/nonfinite action gradient: '+name)
                        norms[name]=norm
                    write_json(out/'action_gradient_check.json',dict(verdict='ACTION_GRADIENT_PASS',
                        identity=run_id,arm=arm,student_steps=student_steps(c),norms=norms))
                loss=loss_a+c['condition_weight']*loss_c
                if auxiliary is not None:
                    feature_loss, feature_values = auxiliary.objective(codes,seq,c['feature_alignment']['mode'])
                    if n == 1 and sample_index == 0:
                        gs = torch.autograd.grad(feature_loss, tuple(model.delta_encoder.parameters()),
                            retain_graph=True, allow_unused=True)
                        encoder_norm = sum(float(g.detach().square().sum()) for g in gs if g is not None)**.5
                        if not 0 < encoder_norm < float('inf'):
                            raise RuntimeError('Missing direct feature-supervision encoder gradient')
                        write_json(out/'feature_gradient_check.json',dict(verdict='FEATURE_GRADIENT_PASS',
                            identity=run_id,encoder_gradient_norm=encoder_norm,teacher_target_requires_grad=False))
                    loss = loss + c['feature_alignment']['weight']*feature_loss
                    metrics['feature_loss'] += float(feature_loss.detach())/2
                    for key,value in feature_values.items():
                        metric='feature_'+key+'_mse'
                        metrics[metric]=metrics.get(metric,0.)+float(value.detach())/2
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                (loss/2).backward()
                metrics['action_l1']+=float(loss_a.detach())/2
                metrics['condition_mse']+=float(loss_c.detach())/2
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if auxiliary is not None:
                torch.nn.utils.clip_grad_norm_(auxiliary.parameters(),1.,error_if_nonfinite=True)
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
    if auxiliary is not None:
        validate_features(c,model,auxiliary,heldout,out,smoke)
    saved=torch.load(latest,map_location='cpu',weights_only=False)
    write_json(out/'summary.json',dict(identity=run_id,verdict='SMOKE_PASS' if smoke else 'TRAIN_AND_OFFLINE_COMPLETE',
        steps=total,checkpoint=str(latest),checkpoint_sha256=sha(latest),training_seconds=saved['seconds'],
        parameters=sum(p.numel() for p in model.parameters()),initial_weights_sha256=initial_hash,
        training_only_parameters=sum(p.numel() for p in auxiliary.parameters()) if auxiliary is not None else 0,
        architecture=arm,frozen_teacher_unchanged=True,
        initialization=c.get('initialization','pretrained'),
        total_training_steps=c.get('sample_step_offset',0)+total,
        prior_training_seconds=continuation['prior_training_seconds'] if continuation else 0))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); train(read_json(a.config),a.arm,a.smoke)
