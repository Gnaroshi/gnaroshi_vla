"""Reconstruct frozen collecting-policy KV on saved on-policy observations."""
from pathlib import Path
import numpy as np
import torch

from condition_bridge_common import load_module,bridge_update,bridge_objective
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook
from architectures.openpi.adapters.latentloop.policy_io import prepare_policy_observation


class OnlineBridgePairs:
    def __init__(self,path,base,model,checkpoint,config,interval):
        self.paths=sorted((Path(path)/'episodes').glob('task*_trial*.npz'))
        if len(self.paths)!=300:raise ValueError('Need all300 disjoint on-policy episodes')
        self.base,self.hook,self.interval,self.seed=base,PrefixKVHook(model),interval,config['train_seed']
        self.collector,_=load_module(checkpoint,config)
        self.collector.requires_grad_(False).eval()

    def sample(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,index,973]))
        for attempt in range(len(self.paths)):
            path=self.paths[int(rng.integers(len(self.paths)))]
            with np.load(path) as saved:
                raw_states=saved['state']
                available=[i for i in range(1,len(raw_states)) if i%self.interval]
                if not available:continue
                t=int(rng.choice(available));anchor=t-t%self.interval
                front,wrist,first_action,prompt=saved['image'],saved['wrist'],saved['first_action'],str(saved['prompt'])
                observations=[]
                for i in range(anchor,t+1):
                    raw={'observation/image':front[i],'observation/wrist_image':wrist[i],
                        'observation/state':raw_states[i],'prompt':prompt}
                    observations.append(prepare_policy_observation(self.base,raw)[0])
                states=torch.as_tensor(raw_states[anchor:t],device='cuda',dtype=torch.float32)
                actions=torch.as_tensor(first_action[anchor:t],device='cuda',dtype=torch.float32)
            with torch.no_grad():
                previous=self.hook.extract(observations[0]).state
                for i in range(1,len(observations)-1):
                    emb,_,_=self.hook.embed(observations[i])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        previous,_=bridge_update(self.collector,previous,emb,states[i-1:i,:8],actions[i-1:i,:7])
                target=self.hook.extract(observations[-1]).state
            return previous.detach(),target.detach(),states[-1:,:8],actions[-1:,:7]
        raise RuntimeError('No non-refresh queries in collection')

    def loss(self,module,index):
        from train_condition_bridge import embedding
        previous,target,state,action=self.sample(index)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,raw=bridge_update(module,previous,embedding(target),state,action)
        return bridge_objective(raw,target)
