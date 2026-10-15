"""Common demonstrations, frozen policy, official bridge objective and recurrent Ours."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from tqdm import trange

from dual_loop_runtime import atomic_json,load_components,read_json,sha256
from condition_bridge_common import Windows,make_module,bridge_update,bridge_objective,learning_rate,load_module
from train_pi05_condition import action_postprocessor
from train_pi05_generation import save
from architectures.openpi.adapters.latentloop.aligned_condition import pack_kv
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook,PrefixEmbeddingState
from architectures.openpi.adapters.latentloop.policy_io import explicit_policy_noise


def embedding(prefix):
    return PrefixEmbeddingState(prefix.embeddings,prefix.pad_mask,prefix.attention_pattern,prefix.position_ids)


def objective(model,module,window,config,index,postprocess,method,recursive=False):
    observations,raw_states,raw_actions,_=window
    hook=PrefixKVHook(model)
    previous=hook.extract(observations[0]).state.detach()
    losses=[];metrics=[]
    for age in range(1,len(observations)):
        target=hook.extract(observations[age]).state.detach()
        if method=='bridge':
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction,raw=bridge_update(module,previous,embedding(target),raw_states[age-1],raw_actions[age-1])
            loss=bridge_objective(raw,target)
            losses.append(loss);metrics.append(dict(age=age,loss=float(loss.detach())))
            previous=prediction.detach() if recursive else target
        else:
            prediction,_=module(previous,observations[age-1],observations[age],age=age)
            z,zt=pack_kv(prediction).float(),pack_kv(target).float()
            mask=target.pad_mask[:,None,None,None,:,None].expand_as(zt)
            # Relative raw KV error preserves scale, important for attention logits.
            scale=zt.square().mean(-1,keepdim=True).clamp_min(1e-6)
            latent=((z-zt).square()/scale).masked_select(mask).mean()
            noise=explicit_policy_noise((1,10,model.config.action_dim),seed=config['train_seed']*1000000+index*8+age,device='cuda')
            with torch.no_grad():teacher=postprocess(hook.sample_actions_from_state(target,observations[age].state,noise,num_steps=1)[0])
            student=postprocess(hook.sample_actions_from_state(prediction,observations[age].state,noise,num_steps=1)[0])
            action=F.l1_loss(student[:,:5],teacher[:,:5])
            loss=action+0.05*latent
            losses.append(loss);metrics.append(dict(age=age,loss=float(loss.detach()),action_l1=float(action.detach()),kv_relative_mse=float(latent.detach())))
            previous=prediction
    return torch.stack(losses).mean(),metrics


def run(args):
    c=read_json(args.config);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    base,model,_=load_components(c);data=Windows(c,base);postprocess=action_postprocessor(c)
    length=8 if args.extension else 4
    first=data.window(0,length=length);prefix=PrefixKVHook(model).extract(first[0][0]).state
    module,descriptor=make_module(args.method,prefix,first[0][0]);module=module.cuda().train()
    if args.parent:
        parent,payload=load_module(args.parent,c);module.load_state_dict(parent.state_dict(),strict=True);del parent
    if args.method=='bridge':
        from torch.utils.checkpoint import checkpoint
        for block in module.blocks:
            original=block.forward
            def forward(*inputs,_forward=original):
                return checkpoint(_forward,*inputs,use_reentrant=False) if torch.is_grad_enabled() else _forward(*inputs)
            block.forward=forward
    refinement=args.extension or bool(args.parent)
    optimizer=torch.optim.AdamW(module.parameters(),lr=learning_rate(0,args.steps,args.method,refinement),weight_decay=0. if args.method=='ours' else 1e-4)
    online=None
    if args.online_data:
        if args.method!='bridge' or not args.parent:raise ValueError('DAgger requires a frozen collecting bridge')
        from bridge_dagger import OnlineBridgePairs
        online=OnlineBridgePairs(args.online_data,base,model,args.parent,c,args.collection_interval)
    contract=dict(config_id=c['config_id'],method=args.method,steps=args.steps,extension=args.extension,parent_sha256=sha256(args.parent) if args.parent else None,
        length=length,online_data=args.online_data,online_mix_fraction=0.5 if online else 0.,
        collection_interval=args.collection_interval,teacher_steps=1,student_steps=1,train_episodes=len(data.roles['train']),heldout_episodes=len(data.roles['checkpoint_validation']),
        parameters=sum(p.numel() for p in module.parameters()),descriptor=descriptor,
        bridge_training='official module and delta-MSE+cosine; stage R0 demonstration pairs; refinement mixes demonstration and actual collected on-policy pairs equally')
    atomic_json(out/'contract.json',contract)
    latest=out/'latest.pt';start=0;elapsed=0.
    if latest.exists():
        p=torch.load(latest,map_location='cpu',weights_only=False)
        if p['contract']!=contract:raise ValueError('Resume contract differs')
        module.load_state_dict(p['module']);optimizer.load_state_dict(p['optimizer']);start,elapsed=p['step'],p['seconds']
    # Exact K1 action/cache reconstruction before learning, using the common baseline.
    hook=PrefixKVHook(model)
    with torch.no_grad():
        noise=explicit_policy_noise((1,10,model.config.action_dim),seed=7,device='cuda')
        sampler=getattr(model.sample_actions,'_torchdynamo_orig_callable',model.sample_actions)
        native=sampler('cuda',first[0][0],noise=noise.clone(),num_steps=1)
        rebuilt=hook.sample_actions_from_state(prefix,first[0][0].state,noise.clone(),num_steps=1)[0]
        diff=float((native-rebuilt).abs().max())
    atomic_json(out/'parity.json',dict(k1_maxdiff=diff,cache=hook.cache_allclose(prefix,hook.extract(first[0][0]).source_cache)))
    if diff!=0:raise RuntimeError('K1 parity failed')
    del first,prefix,native,rebuilt
    began=time.monotonic()
    for step in trange(start,args.steps,desc=args.method+('_k8' if args.extension else '_k4'),mininterval=2):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=learning_rate(step,args.steps,args.method,refinement)
        window=None
        if online and step%2:
            loss=online.loss(module,step);metrics=[dict(source='on_policy',loss=float(loss.detach()))]
        else:
            window=data.window(step,length=length)
            loss,metrics=objective(model,module,window,c,step,postprocess,args.method,args.extension)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(module.parameters(),1.,error_if_nonfinite=True)
        if not norm>0:raise RuntimeError('Missing module gradient')
        if any(p.grad is not None for p in model.parameters()):raise RuntimeError('Frozen baseline gradient')
        optimizer.step();seconds=elapsed+time.monotonic()-began;n=step+1
        if n%50==0 or n in (1,args.steps):
            row=dict(step=n,seconds=seconds,loss=float(loss.detach()),ages=metrics,gradient_norm=float(norm),lr=optimizer.param_groups[0]['lr'])
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            atomic_json(out/'progress.json',row)
        if n%500==0 or n==args.steps:
            payload=dict(config_id=c['config_id'],contract=contract,method=args.method,descriptor=descriptor,module=module.state_dict(),optimizer=optimizer.state_dict(),step=n,total_steps=args.steps,seconds=seconds)
            save(latest,payload)
            if n==args.steps:save(out/'final.pt',payload)
        del loss,window
    module.eval();rows=[]
    with torch.no_grad():
        for i in range(1 if args.smoke else data.validation_count):
            window=data.window(i,'checkpoint_validation',length)
            loss,metrics=objective(model,module,window,c,100000+i,postprocess,args.method,args.extension)
            rows.append(dict(**window[3],loss=float(loss),ages=metrics))
    atomic_json(out/'validation.json',dict(windows=rows,selection='fixed final step'))
    atomic_json(out/'summary.json',dict(complete=True,smoke=args.smoke,method=args.method,config_id=c['config_id'],steps=args.steps,
        elapsed_seconds=seconds,seconds_per_step=seconds/args.steps,parameters=contract['parameters'],
        checkpoint=str(out/'final.pt'),checkpoint_sha256=sha256(out/'final.pt'),peak_vram_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['ours','bridge'],required=True);p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--extension',action='store_true');p.add_argument('--parent');p.add_argument('--smoke',action='store_true')
    p.add_argument('--online-data');p.add_argument('--collection-interval',type=int,default=4)
    run(p.parse_args())
