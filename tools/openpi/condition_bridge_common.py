"""Shared pi0.5 interface and explicit official-code bridge adaptation."""
import importlib.util
import math
import os
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from dual_loop_runtime import TrainingPairs,read_json
from architectures.openpi.adapters.latentloop.aligned_condition import AlignedConditionUpdater,ConditionConfig
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVState


def official_class():
    path=Path(os.environ['LATENT_BRIDGE_UPSTREAM_ROOT'])/'scripts/pi0/pi0_bridge_kv.py'
    spec=importlib.util.spec_from_file_location('official_pi0_bridge_kv',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Pi0BridgeKV


def kv_tensor(prefix):
    if prefix.values[0].shape[1]!=1:raise ValueError('Official bridge expects one KV head')
    return torch.stack([torch.cat((k[:,0],v[:,0]),-1) for k,v in zip(prefix.pre_rope_keys,prefix.values)],1)


def kv_state(template,embedding,values):
    dim=template.values[0].shape[-1]
    return PrefixKVState(**embedding.__dict__,
        pre_rope_keys=tuple(values[:,l,:,:dim][:,None].to(template.values[l].dtype) for l in range(template.num_layers)),
        values=tuple(values[:,l,:,dim:][:,None].to(template.values[l].dtype) for l in range(template.num_layers)))


def bridge_update(module,previous,embedding,previous_raw_state,previous_first_action):
    old=kv_tensor(previous).float();b,l,s,d=old.shape
    dtype=next(module.parameters()).dtype
    delta=module((embedding.embeddings-previous.embeddings).to(dtype),embedding.embeddings.to(dtype),
        old.permute(0,2,1,3).reshape(b,s,l*d).to(dtype),previous_raw_state.to(dtype),previous_first_action.to(dtype))
    updated=old+torch.stack(delta,1).float()
    return kv_state(previous,embedding,updated),updated


def bridge_objective(predicted,target):
    # Official per-layer delta-MSE + 1-cos(next-KV); residual addition cancels in MSE.
    target=kv_tensor(target).float()
    return F.mse_loss(predicted,target)+(1-F.cosine_similarity(predicted,target,dim=-1).mean())


def make_module(method,prefix,observation,descriptor=None):
    if method=='ours':
        config=ConditionConfig(**descriptor) if descriptor else replace(ConditionConfig.from_prefix(prefix,observation),max_age=7)
        module=AlignedConditionUpdater(config)
        return module, module.descriptor()
    config=descriptor or dict(kv_dim=prefix.values[0].shape[-1],num_layers=prefix.num_layers,
        seq_len=prefix.num_tokens,emb_dim=prefix.embeddings.shape[-1],hidden_dim=768,num_heads=12,
        num_blocks=10,state_dim=8,action_dim=7)
    return official_class()(**config),config


class Windows(TrainingPairs):
    def window(self,index,role='train',length=4):
        import numpy as np
        from architectures.openpi.adapters.latentloop.streaming_teacher import _raw_policy_observation
        from architectures.openpi.adapters.latentloop.policy_io import prepare_policy_observation
        from architectures.openpi.adapters.latentloop.cache_contract_v2 import resolve_task_identity
        rows=[e for e in self.roles[role] if e.query_count>=length]
        if len(rows)!=len(self.roles[role]):raise ValueError('Window length excludes an episode')
        if role=='checkpoint_validation':
            e=rows[index//3];start=round((index%3)*(e.query_count-length)/2)
        else:
            rng=np.random.default_rng(np.random.SeedSequence([self.config['train_seed'],index]))
            e=rows[int(rng.integers(len(rows)))];start=int(rng.integers(e.query_count-length+1))
        observations=[];states=[];actions=[]
        for q in range(start,start+length):
            sample=self.dataset[e.dataset_frame_start+q*5]
            task=resolve_task_identity(int(sample['task_index']),str(sample['prompt']),self.final_manifest)
            if (task['suite'],int(task['benchmark_task_index']))!=(e.suite,e.benchmark_task_index):raise ValueError('Task mismatch')
            if int(sample['frame_index'])!=q*5 or bool(torch.as_tensor(sample['actions_is_pad'][:5]).any()):raise ValueError('Episode boundary')
            raw=_raw_policy_observation(sample)
            observations.append(prepare_policy_observation(self.policy,raw)[0])
            states.append(torch.as_tensor(raw['observation/state'],device='cuda',dtype=torch.float32)[None,:8])
            actions.append(torch.as_tensor(sample['actions'][0,:7],device='cuda',dtype=torch.float32)[None])
        return observations,states,actions,dict(episode_id=e.episode_id,task_id=e.benchmark_task_index,query_index=start,role=role)


def learning_rate(step,total,method,extension=False):
    peak=(1e-4 if method=='ours' else 3e-4)*(0.1 if extension else 1.)
    warm=max(1,total//20)
    if step<warm:return peak*(step+1)/warm
    fraction=(step-warm)/max(1,total-warm)
    return peak*(0.1+0.9*0.5*(1+math.cos(math.pi*fraction)))


def load_module(path,config,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['config_id']!=config['config_id'] or saved['step']!=saved['total_steps']:raise ValueError('Unfinished/mixed checkpoint')
    module,_=make_module(saved['method'],None,None,saved['descriptor'])
    module.load_state_dict(saved['module'],strict=True)
    return module.to(device).eval(),saved
