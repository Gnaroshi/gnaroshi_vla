"""Collect disjoint student states and run a matched training-data intervention."""
import argparse
import hashlib
from pathlib import Path
import random
import shutil
import time
from types import MethodType

import numpy as np
import torch
from tqdm import trange

from tools.simvla.error_compensation_common import configure, identity, read_json, write_json, sha, snapshots
from tools.simvla.trend_condition_eval import make_policy as trend_policy, check_counts
from tools.simvla.error_compensation_eval import run as evaluate
from methods.latentloop.modules.observed_progress import build_model
from methods.latentloop.modules.trend_condition import scaled_mse
from methods.latentloop.modules.action_aligned_joint import action_loss, differentiable_rollout
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, configure_strict_torch_determinism, append_jsonl
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime, lr_factor, assert_frozen
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import load_generation_checkpoint
from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
from tools.simvla.compile_runtime import ActionStep

VARIANTS = ('offline_control','fresh_original_control','rollout_repair')
DRIVERS = ('original','student')
DATA_DRIVER = {'fresh_original_control':'original','rollout_repair':'student'}


def state_hash_array(value):
    return hashlib.sha256(np.asarray(value,dtype=np.float64).tobytes()).hexdigest()


def disjoint_indices(raw, official, count, seed):
    forbidden={state_hash_array(x) for x in official}
    candidates=[]
    seen=set(forbidden)
    for i,x in enumerate(raw):
        h=state_hash_array(x)
        if h not in seen and np.isfinite(np.asarray(x)).all():
            candidates.append(i)
            seen.add(h)
    random.Random(seed).shuffle(candidates)
    if len(candidates)<count:
        raise RuntimeError('Insufficient distinct non-evaluation initial states')
    return candidates[:count]


def source_rgb(value):
    # Live RGB is divided by 255 on CPU before transfer to CUDA. Reconstruct
    # on that same backend: CUDA division may differ by one float32 ULP.
    value=value.detach().cpu()
    if value.dtype==torch.uint8: return value
    if value.dtype!=torch.float32: raise ValueError('Expected live float32 RGB')
    encoded=(value*255).round().to(torch.uint8)
    if not torch.equal(encoded.float()/255,value.float()):
        raise RuntimeError('Live RGB is not losslessly representable as source uint8')
    return encoded


def reservoir_slot(seen,capacity,rng):
    slot=seen-1 if seen<=capacity else rng.randrange(seen)
    return slot if slot<capacity else None


def load_selected(c):
    spec=c['selected_checkpoint']
    if sha(spec['path'])!=spec['sha256']: raise RuntimeError('Selected checkpoint changed')
    p=torch.load(spec['path'],map_location='cpu',weights_only=False)
    if p['arm']!=spec['arm'] or p['step']!=spec.get('step',6000) or p['contract']['k_c']!=8:
        raise RuntimeError('Selected architecture/step/horizon mismatch')
    return p


def policy(c, variant='selected', smoke=False, k_c=8):
    def load(_c,_row):
        if variant=='selected': return load_selected(c)
        path=Path(c['output'])/('smoke' if smoke else 'train')/variant/'latest.pt'
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['identity']!=identity(c) or p['repair_variant']!=variant or p['step']!=(c['smoke_steps'] if smoke else c['steps']):
            raise RuntimeError('Repair checkpoint provenance mismatch')
        return p
    return trend_policy(c,c['selected_checkpoint']['arm'],smoke=smoke,k_c=k_c,checkpoint_loader=load)


def sample_from_sequence(s, age):
    return dict(anchor=s['anchor_condition'],anchor_images=s['image_sequence'][:,0],
        images=s['image_sequence'][:,age],anchor_proprio=s['proprio_sequence'][:,0],
        proprio=s['proprio_sequence'][:,age],valid=s['valid_mask'].bool(),groups=s['group_ids'],
        target_condition=s['teacher_conditions'][:,age-1],noise=s['explicit_noises'][:,age-1],
        target_action=s['teacher_actions'][:,age-1],age=age)


def prediction(model,s):
    ctx=model.prepare(s['anchor'],s['anchor_images'],s['anchor_proprio'],s['valid'],s['groups'])
    return model.predict(ctx,s['age'],s['images'],s['proprio'])[0]


@torch.no_grad()
def collect(c, shard, smoke=False, driver='student'):
    if driver not in DRIVERS: raise ValueError(driver)
    configure(c)
    configure_strict_torch_determinism(c['collection_seed'])
    torch.set_num_threads(1)
    from libero.libero import benchmark
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs,get_libero_env
    out=Path(c['output'])/('collection_smoke' if smoke else 'collection')/driver
    out.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out).free < c['collection_min_free_gib']*1024**3:
        raise RuntimeError('Insufficient space for bounded student-state collection')
    p=policy(c)
    p.action_noise_seed_base=c['collection_noise_seed']
    suite=benchmark.get_benchmark_dict()['libero_10']()
    selected_records=[]
    counts,slots={},{}
    original,original_full=p._v0_update,p._full_refresh
    current={}
    def full(self,batch,*,policy_query_index):
        condition,action,seed=original_full(batch,policy_query_index=policy_query_index)
        if driver=='original':
            noise,seed=self._paired_initial_noise(condition,batch['proprio'],policy_query_index)
            action=self.action_adapter.decode_action_from_condition(condition,batch['proprio'],steps=10,initial_noise=noise)
            self.cached_action_chunk=action.detach()
        return condition,action,seed
    def update(self,batch,*,age,policy_query_index):
        result=original(batch,age=age,policy_query_index=policy_query_index)
        counts[age]=counts.get(age,0)+1
        rng=random.Random(c['collection_seed']+current['task_id']*1000000+current['state_index']*1000+policy_query_index)
        slot=reservoir_slot(counts[age],c['collection_samples_per_age'],rng)
        if slot is None and driver=='student': return result
        ctx=self._trend_context
        # The driver determines executed actions; sample selection never does.
        teacher=self.condition_adapter.encode_condition(input_ids=batch['input_ids'],
            image_input=batch['image_input'],image_mask=batch['image_mask'])
        if teacher.shape!=ctx.anchor.shape: raise RuntimeError('Online token shape changed')
        noise,seed=self._paired_initial_noise(teacher,batch['proprio'],policy_query_index)
        target=self.action_adapter.decode_action_from_condition(teacher,batch['proprio'],steps=10,initial_noise=noise)
        executed=result
        if driver=='original':
            executed=(teacher,target,seed)
            self.cached_condition,self.cached_action_chunk=teacher.detach(),target.detach()
        if slot is None: return executed
        teacher_generation,_=self._decode(teacher,batch['proprio'],policy_query_index=policy_query_index)
        norm=self.action_adapter.action_space.normalize_action
        record=dict(anchor=ctx.anchor,anchor_images=ctx.images,images=batch['raw_rgb'],
            anchor_proprio=ctx.proprio,proprio=batch['proprio'],valid=ctx.valid,groups=ctx.groups,
            target_condition=teacher,noise=noise,target_action=target,age=age)
        for key in ('anchor_images','images'): record[key]=source_rgb(record[key])
        record={k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for k,v in record.items()}
        record['metadata']=dict(**current,query=policy_query_index,noise_seed=seed,
            condition_mse=float(scaled_mse(result[0],teacher,ctx.anchor,ctx.valid)),
            student_action_l1=float(action_loss(norm(result[1]),norm(target))),
            original_condition_generation_l1=float(action_loss(norm(teacher_generation),norm(target))))
        key=(age,slot)
        if key in slots: selected_records[slots[key]]=record
        else:
            slots[key]=len(selected_records); selected_records.append(record)
        return executed
    p._full_refresh=MethodType(full,p)
    p._v0_update=MethodType(update,p)
    summaries=[]
    tasks=[9] if smoke else [t for t in range(9,-1,-1) if t%4==shard]
    for task_id in tasks:
        task=suite.get_task(task_id)
        pool=c['collection_states'][str(task_id)]
        raw=torch.load(pool['path'],map_location='cpu',weights_only=False)
        if sha(pool['path'])!=pool['sha256']: raise RuntimeError('Collection states changed')
        env,prompt=get_libero_env(task,256,7)
        try:
            for number,index in enumerate(pool['indices'][:1] if smoke else pool['indices']):
                path=out/f'task{task_id}_state{index}.pt'
                marker=path.with_suffix('.json')
                if marker.exists():
                    info=read_json(marker)
                    if info['identity']!=identity(c) or info['driver']!=driver or sha(path)!=info['sha256']: raise RuntimeError('Collection resume mismatch')
                    summaries.append(info)
                    continue
                current.clear()
                current.update(driver=driver,task_id=task_id,state_index=index,state_hash=state_hash_array(raw[index]),
                    split='heldout' if number==4 else 'train')
                configure_strict_torch_determinism(c['collection_seed']+task_id*100+index)
                env.seed(7)
                env.reset()
                obs=env.set_init_state(raw[index])
                for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
                p.reset()
                p.task_id,p.trial_id=task_id,index
                selected_records.clear()
                counts.clear(); slots.clear()
                begun=last_progress=time.monotonic()
                for step in range(41 if smoke else 900):
                    action=p.act(*build_env_obs(obs),prompt).action
                    if not np.isfinite(action).all(): raise RuntimeError(driver+' driver produced a nonfinite action')
                    obs,_,success,_=env.step(action.tolist())
                    if time.monotonic()-last_progress>30:
                        status=dict(task=task_id,state=index,actions=step+1,samples=len(selected_records),completed_episodes=len(summaries))
                        write_json(out/f'shard{shard}_progress.json',status)
                        print(f'COLLECT_PROGRESS {status}',flush=True)
                        last_progress=time.monotonic()
                    if success: break
                if smoke and {r['age'] for r in selected_records}!=set(range(1,8)):
                    raise RuntimeError('Collection smoke missed a condition age')
                atomic_torch_save(selected_records,path)
                info=dict(identity=identity(c),**current,records=len(selected_records),
                    sha256=sha(path),file=path.name,success=bool(success),actions=step+1,seconds=time.monotonic()-begun,
                    use='training collection, not official success evaluation',teacher_action_applied=driver=='original')
                write_json(marker,info)
                summaries.append(info)
                print(f'COLLECT driver={driver} shard={shard} task={task_id} state={index} records={len(selected_records)}',flush=True)
        finally: env.close()
    write_json(out/f'shard{shard}_summary.json',dict(identity=identity(c),verdict='COLLECTION_COMPLETE',
        episodes=summaries,records=sum(x['records'] for x in summaries)))


def load_records(c,smoke=False):
    directory=Path(c['output'])/('collection_smoke' if smoke else 'collection')
    records=[]
    sources=[dict(path=str(directory),identity=identity(c),label='current')]
    sources+=c.get('record_sources_smoke' if smoke else 'record_sources',[])
    for source in sources:
        for marker in sorted(Path(source['path']).glob('*/task*_state*.json')):
            info=read_json(marker)
            path=marker.parent/info['file']
            if info['identity']!=source['identity'] or sha(path)!=info['sha256']: raise RuntimeError('Invalid collected sample')
            for r in torch.load(path,map_location='cpu',mmap=True,weights_only=False):
                r['metadata']={**r['metadata'],'collection_source':source['label']}
                records.append(r)
    if not records: raise RuntimeError('No collected states')
    return records


@torch.no_grad()
def validation(c,model,action,loop,step,heldout,records,out):
    groups={}
    labels=['cached_original_states','fresh_original_states','student_states']
    if c.get('variant_settings'): labels+=['previous_student_states','current_student_states']
    for label in labels:
        values=[]
        if label=='cached_original_states':
            ids=_balanced_indices(heldout.identities,limit=30,seed=c['seed'])
            samples=(sample_from_sequence(move_batch(collate_exact_teacher_sequences([heldout[i]]),'cuda'),a)
                for i in ids for a in range(1,8))
        else:
            driver='original' if label=='fresh_original_states' else 'student'
            source=label.split('_')[0] if label in ('previous_student_states','current_student_states') else None
            samples=(move_batch(r,'cuda') for r in records if r['metadata']['split']=='heldout' and r['metadata']['driver']==driver
                and (source is None or r['metadata']['collection_source']==source))
        for s in samples:
            condition=prediction(model,s)
            pred=differentiable_rollout(loop,step,condition,action.normalize_proprio(s['proprio']),s['noise'])
            values.append(dict(age=s['age'],condition_mse=float(scaled_mse(condition,s['target_condition'],s['anchor'],s['valid'])),
                action_l1=float(action_loss(pred,action.action_space.normalize_action(s['target_action'])))))
        if not values: raise RuntimeError('Empty heldout data group')
        groups[label]=dict(queries=len(values),condition_mse=np.mean([v['condition_mse'] for v in values]),
            action_l1=np.mean([v['action_l1'] for v in values]),by_age={str(a):dict(
                action_l1=np.mean([v['action_l1'] for v in values if v['age']==a]),
                condition_mse=np.mean([v['condition_mse'] for v in values if v['age']==a])) for a in range(1,8)})
    write_json(out/'validation.json',groups)


def training_settings(c,variant):
    settings=c.get('variant_settings',{}).get(variant)
    if settings is not None: return settings
    if variant not in VARIANTS: raise ValueError(variant)
    return dict(train_trend=False,driver=DATA_DRIVER.get(variant),sources=['current'])


def sample_pool(records,settings,age):
    return [r for r in records if r['age']==age and r['metadata']['split']=='train'
        and r['metadata']['driver']==settings['driver']
        and r['metadata']['collection_source'] in settings['sources']]


def train(c,variant,smoke=False):
    configure(c)
    torch.set_num_threads(1)
    run_id=identity(c)
    out=Path(c['output'])/('smoke' if smoke else 'train')/variant
    out.mkdir(parents=True,exist_ok=True)
    device,parent,frozen,action,data,heldout=load_runtime(c,snapshots(c))
    source=load_selected(c)
    model=build_model(parent,source['arm'],max_age=7).to(device).eval()
    model.load_state_dict(source['model'],strict=True)
    model.requires_grad_(True)
    settings=training_settings(c,variant)
    model.trend_head.requires_grad_(settings['train_trend'])
    generation,_=load_generation_checkpoint(c['generation_checkpoint'],device=device)
    generation.eval().requires_grad_(False)
    loop=SimVLAGenerationLoop(generation,frozen.transformer.action_decoder).eval()
    step_model=ActionStep(frozen.transformer).eval()
    frozen_b,frozen_g=state_hash(model.trend_head),state_hash(generation)
    original_model=state_hash(model)
    records=load_records(c,smoke)
    by_age={a:sample_pool(records,settings,a) for a in range(1,8)} if settings['driver'] else {}
    if any(not r for r in by_age.values()): raise RuntimeError('Training collection missing an age/driver')
    if not smoke and variant=='offline_control' and not (out/'before/validation.json').exists():
        validation(c,model,action,loop,step_model,heldout,records,out/'before')
    params=[p for p in model.parameters() if p.requires_grad]
    total=c['smoke_steps'] if smoke else c['steps']
    optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:lr_factor(n,total,min(150,max(1,total//10))))
    latest=out/'latest.pt'
    start=0; elapsed=0.
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=False)
        if saved['identity']!=run_id or saved['repair_variant']!=variant: raise RuntimeError('Resume mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start=saved['step']; elapsed=saved['seconds']
    # Fixed per-step draws give exact data replay after a technical interruption.
    contract={**source['contract'], 'steps':total,'repair_variant':variant,'source_checkpoint':c['selected_checkpoint'],
        'loss':'normalized first-five action L1 only','optimizer':'AdamW 1e-4 wd0 clip1; cosine to 0.1x, batch2 as two microbatches',
        'training_data':settings,'trend_frozen':not settings['train_trend'],
        'generation_frozen':True,'no_new_inference_operations':True}
    write_json(out/'training_contract.json',contract)
    tracker=None
    if not smoke and c.get('wandb_project'):
        try:
            import wandb
            tracker=wandb.init(project=c['wandb_project'],name='rollout_state_'+variant,id=run_id[:12]+'_'+variant,
                resume='allow',config=contract,dir=str(out),settings=wandb.Settings(init_timeout=20))
        except Exception as exc: print(f'WANDB_WARNING {exc}',flush=True)
    began=time.monotonic()
    def save(n):
        atomic_torch_save(dict(format='simvla_trend_condition_v1',identity=run_id,arm=source['arm'],repair_variant=variant,
            model=model.state_dict(),step=n,contract=contract,seconds=elapsed+time.monotonic()-began,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),latest)
    try:
        for n in trange(start+1,total+1,desc=variant,mininterval=2):
            age=(n-1)%7+1
            rng=random.Random(c['seed']*100000+n)
            ids=[rng.randrange(len(data)) for _ in range(2)]
            samples=[sample_from_sequence(move_batch(collate_exact_teacher_sequences([data[i]]),device),age) for i in ids]
            if settings['driver']:
                pool=by_age[age]
                # Equal draw probability per collection round, regardless of
                # successful episode length or reservoir size.
                source_label=rng.choice(settings['sources'])
                pool=[r for r in pool if r['metadata']['collection_source']==source_label]
                if not pool: raise RuntimeError('Collection round missing an age')
                samples[1]=move_batch(pool[rng.randrange(len(pool))],device)
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for s in samples:
                if n<=7:
                    with torch.no_grad(): exact=action.decode_action_from_condition(s['target_condition'],s['proprio'],steps=10,initial_noise=s['noise'])
                    diff=float((exact-s['target_action']).abs().max())
                    if diff>2e-4: raise RuntimeError(f'Collected/original teacher action mismatch {diff}')
                pred=differentiable_rollout(loop,step_model,prediction(model,s),action.normalize_proprio(s['proprio']),s['noise'])
                loss=action_loss(pred,action.action_space.normalize_action(s['target_action']))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                (loss/2).backward(); losses.append(float(loss))
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if not norm>0: raise RuntimeError('No gradient')
            assert_frozen(frozen,generation,*([] if settings['train_trend'] else [model.trend_head]))
            optimizer.step(); scheduler.step()
            if n%50==0 or n in (1,total):
                metrics=dict(step=n,age=age,loss=sum(losses)/2,lr=optimizer.param_groups[0]['lr'],seconds=elapsed+time.monotonic()-began)
                append_jsonl(out/'metrics.jsonl',metrics)
                if tracker:
                    try: tracker.log(metrics,step=n)
                    except Exception as exc: print(f'WANDB_WARNING {exc}',flush=True)
            if n%500==0 or n==total: save(n)
    finally:
        if tracker:
            try: tracker.finish()
            except Exception: pass
    if (not settings['train_trend'] and state_hash(model.trend_head)!=frozen_b) or state_hash(generation)!=frozen_g:
        raise RuntimeError('Frozen weights changed')
    if settings['train_trend'] and state_hash(model.trend_head)==frozen_b: raise RuntimeError('Trend was not trained')
    if state_hash(model)==original_model: raise RuntimeError('No model change')
    if not smoke: validation(c,model,action,loop,step_model,heldout,records,out)
    saved=torch.load(latest,map_location='cpu',weights_only=False)
    write_json(out/'summary.json',dict(identity=run_id,verdict='SMOKE_PASS' if smoke else 'TRAIN_AND_OFFLINE_COMPLETE',steps=total,
        checkpoint=str(latest),checkpoint_sha256=sha(latest),training_seconds=saved['seconds'],parameters=sum(p.numel() for p in params),
        original_and_generation_frozen=True,trend_frozen=not settings['train_trend'],inference_architecture=source['arm']))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('collect','train','eval'))
    p.add_argument('--config',required=True)
    p.add_argument('--shard',type=int,choices=range(4),default=0)
    p.add_argument('--driver',choices=DRIVERS,default='student')
    p.add_argument('--variant',default=VARIANTS[0])
    p.add_argument('--k-c',type=int,choices=(4,8),default=8)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); c=read_json(a.config)
    if a.command=='collect': collect(c,a.shard,a.smoke,a.driver)
    elif a.command=='train': train(c,a.variant,a.smoke)
    else:
        def factory(c,row,*,smoke,k_c): return policy(c,row,smoke,k_c)
        def counts(pol,row,calls,k): return check_counts(pol,c['selected_checkpoint']['arm'],calls,k)
        evaluate(c,a.variant,smoke=a.smoke,k_c=a.k_c,policy_factory=factory,counter_checker=counts)


if __name__=='__main__': main()
