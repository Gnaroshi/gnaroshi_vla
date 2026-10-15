"""Common compiled NFE1 runtime for original, recurrent Condition and official KV bridge."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from dual_loop_runtime import atomic_json,load_components,read_json
from condition_bridge_common import load_module,bridge_update
from architectures.openpi.adapters.latentloop import prefix_kv_hook as hooks
from architectures.openpi.adapters.latentloop.policy_io import explicit_policy_noise,policy_noise_seed,prepare_policy_observation,postprocess_policy_actions


class ServingPolicy:
    def __init__(self,c,row,checkpoint=None):
        self.config,self.row=c,row
        self.base,self.model,_=load_components(c)
        self.method='baseline' if row=='baseline_nfe1' else row.split('_')[0]
        self.interval=1 if self.method=='baseline' else int(row.split('_k')[1])
        self.module=None
        if checkpoint:
            self.module,payload=load_module(checkpoint,c)
            if payload['method']!=self.method:raise RuntimeError('Wrong method checkpoint')
            self.module.requires_grad_(False)
        self.hook=hooks.PrefixKVHook(self.model)
        # One synchronization at the query boundary; event timings measure components.
        hooks._sync=lambda device:None
        self.graphs={}
        def compile_function(name,fn):
            from torch._dynamo.backends.registry import lookup_backend
            backend=lookup_backend('inductor')
            def observed(gm,inputs):
                self.graphs[name]=self.graphs.get(name,0)+1
                return backend(gm,inputs)
            return torch.compile(fn,backend=observed,dynamic=False,fullgraph=False)
        self.model.denoise_step=compile_function('action_expert',self.model.denoise_step)
        self.model.embed_prefix=compile_function('vision_embedding',self.model.embed_prefix)
        if self.method=='bridge':self.module.forward=compile_function('bridge',self.module.forward)
        elif self.method=='ours':
            self.module.delta_encoder.forward=compile_function('delta_encoder',self.module.delta_encoder.forward)
            self.module.condition_updater.forward=compile_function('condition_updater',self.module.condition_updater.forward)
        self.metadata=dict(config_id=c['config_id'],row=row,checkpoint=str(checkpoint),
            H=10,R=5,NFE=1,interval=self.interval,renderer='egl',compiled_components=['vision_embedding','action_expert',self.method],
            comparison='same LoRA baseline, demonstrations and heldout manifest; official Pi0BridgeKV architecture/objective with documented interface/data adaptation')
        self.warmed=False;self.reset()

    def reset(self):
        self.prefix=self.observation=self.previous_state=self.previous_action=None
        self.query_index=0;self.episode=None

    def query(self,obs,noise,raw_state,*,events=True):
        metrics=dict(query_index=self.query_index,full_prefix_calls=0,condition_updater_calls=0,
            action_expert_calls=1,generation_updater_calls=0,flow_iterations=1)
        recorded=[]
        def timed(name,fn):
            if not events:return fn()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();out=fn();b.record();recorded.append((name,a,b));return out
        if self.query_index%self.interval==0:
            emb,state,_=timed('prefix_embedding_ms',lambda:self.hook.embed(obs))
            self.prefix=timed('prefix_transformer_ms',lambda:self.hook.extract_from_embedding(emb,state)).state
            metrics['full_prefix_calls']=1
        elif self.method=='ours':
            self.prefix,_=timed('condition_updater_ms',lambda:self.module(self.prefix,self.observation,obs,age=self.query_index%self.interval))
            metrics['condition_updater_calls']=1
        else:
            emb,_,_=timed('prefix_embedding_ms',lambda:self.hook.embed(obs))
            def update():
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    return bridge_update(self.module,self.prefix,emb,self.previous_state,self.previous_action)[0]
            self.prefix=timed('condition_updater_ms',update);metrics['condition_updater_calls']=1
        cache=timed('cache_rebuild_ms',lambda:self.hook.rebuild_cache(self.prefix))
        output=timed('action_expert_ms',lambda:noise-self.model.denoise_step(obs.state,self.prefix.pad_mask,cache,noise,
            torch.ones(noise.shape[0],device=noise.device,dtype=torch.float32)))
        self.prefix=self.prefix.detach();self.observation=obs;self.previous_state=raw_state
        self.query_index+=1
        return output,metrics,recorded

    @torch.no_grad()
    def infer(self,raw):
        raw=dict(raw);request=raw.pop('latentloop')
        if request['policy_path']!=self.row:raise ValueError('Wrong row')
        identity=(int(request['task_id']),int(request['episode_id']))
        seed=policy_noise_seed(self.config['noise_seed'],'libero_10',*identity,int(request['query_index']))
        if not self.warmed:
            observation,transformed=prepare_policy_observation(self.base,raw)
            raw_state=torch.as_tensor(raw['observation/state'],device='cuda',dtype=torch.float32)[None,:8]
            noise=explicit_policy_noise((1,10,self.model.config.action_dim),seed=seed,device='cuda')
            # Exercise all ages before timing the first actual environment query.
            for _ in range(2*self.interval+2):
                action,_,_=self.query(observation,noise,raw_state,events=False)
                self.previous_action=torch.as_tensor(postprocess_policy_actions(self.base,transformed['state'],action)['actions'][0],device='cuda')[None]
            required={'vision_embedding','action_expert'}|({'bridge'} if self.method=='bridge' else {'delta_encoder','condition_updater'} if self.method=='ours' else set())
            if not required.issubset(self.graphs):raise RuntimeError('Missing compile graphs: '+str(required-set(self.graphs)))
            self.reset()
            rebuilt,_,_=self.query(observation,noise,raw_state,events=False)
            sampler=getattr(self.model.sample_actions,'_torchdynamo_orig_callable',self.model.sample_actions)
            native=sampler('cuda',observation,noise=noise.clone(),num_steps=1)
            difference=float((native-rebuilt).abs().max())
            if difference>1e-5:raise RuntimeError('Compiled K1 prefix/action reconstruction differs: '+str(difference))
            self.metadata.update(k1_max_action_diff=difference,compile_graphs=dict(self.graphs))
            if hasattr(self,'runtime_path'):atomic_json(self.runtime_path,self.metadata)
            torch.cuda.synchronize();self.reset();self.warmed=True
        if identity!=self.episode or request.get('reset'):
            if int(request['query_index'])!=0:raise ValueError('Episode must start at query0')
            self.reset();self.episode=identity
        if int(request['query_index'])!=self.query_index:raise ValueError('Noncontiguous query')
        torch.cuda.synchronize();started=time.perf_counter()
        observation,transformed=prepare_policy_observation(self.base,raw)
        raw_state=torch.as_tensor(raw['observation/state'],device='cuda',dtype=torch.float32)[None,:8]
        noise=explicit_policy_noise((1,10,self.model.config.action_dim),seed=seed,device='cuda')
        prep_ms=(time.perf_counter()-started)*1000
        action,metrics,events=self.query(observation,noise,raw_state)
        output=postprocess_policy_actions(self.base,transformed['state'],action)
        self.previous_action=torch.as_tensor(output['actions'][0],device='cuda',dtype=torch.float32)[None]
        torch.cuda.synchronize()
        metrics.update(infer_ms=(time.perf_counter()-started)*1000,preprocessing_ms=prep_ms,noise_seed=seed,
            task_id=identity[0],episode_id=identity[1],peak_vram_bytes=torch.cuda.max_memory_allocated())
        metrics.update({name:begin.elapsed_time(end) for name,begin,end in events})
        metrics['compile_graphs']=dict(self.graphs)
        if not np.isfinite(output['actions']).all():raise RuntimeError('Nonfinite actions')
        return dict(actions=output['actions'],latentloop_metrics=metrics)


if __name__=='__main__':
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--row',required=True)
    p.add_argument('--checkpoint');p.add_argument('--port',type=int,required=True);p.add_argument('--ready',required=True)
    a=p.parse_args();policy=ServingPolicy(read_json(a.config),a.row,a.checkpoint)
    policy.runtime_path=Path(a.ready).parent/'runtime.json'
    atomic_json(a.ready,policy.metadata)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=policy.metadata).serve_forever()
