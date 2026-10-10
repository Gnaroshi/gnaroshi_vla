"""Identical two-head work in both arms, including total policy latency."""
import argparse
from collections import Counter
from pathlib import Path
from types import MethodType

import torch

from methods.latentloop.modules.condition_output_split import ARMS, ConditionOutputSplit
from tools.simvla.error_compensation_common import identity, read_json
from tools.simvla.error_compensation_eval import make_policy as original_policy, run
from tools.simvla.observation_correction_eval import tensor_age_call
from tools.simvla.trend_condition_eval import ComponentTimers


def load_payload(path, arm, expected_identity=None, steps=3000, action_mode='naive3'):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if (p['format']!='simvla_condition_output_split_v1' or p['arm']!=arm or p['step']!=steps
            or action_mode not in ('naive1','naive3')
            or p['contract']['action_mode']!=action_mode
            or p['contract']['training_intervals'] not in ([4,8],[2],[3],[4])):
        raise RuntimeError('Condition output split checkpoint mismatch')
    if expected_identity is not None and p['identity']!=expected_identity:
        raise RuntimeError('Source identity mismatch')
    return p


def predictor_only_update(self, previous, code, *, valid_mask, group_ids, age):
    base = self.condition_updater(previous, code, valid_mask=valid_mask, group_ids=group_ids, age=age).condition
    return base, base, base


def attach(policy, parent, payload, arm, interval, compiler=None, use_action_condition_updater=True):
    nfe=int(payload['contract']['action_mode'].removeprefix('naive'))
    if nfe not in (1,3) or policy.nfe!=nfe:
        raise RuntimeError('Training and deployed action solver differ')
    model=ConditionOutputSplit(parent,arm).to('cuda').eval().requires_grad_(False)
    model.load_state_dict(payload['model'],strict=True)
    if not use_action_condition_updater:
        if arm != 'carry_base':
            raise ValueError('Predictor-only control requires carry_base')
        model.update=MethodType(predictor_only_update,model)
    policy._uses_action_condition_updater=use_action_condition_updater
    timers=ComponentTimers() if compiler is None else None
    counts=Counter()
    for attr,name in [('delta_encoder','observation_encoder'),('condition_updater','condition_updater'),
            ('action_condition_updater','action_condition_updater')]:
        module=getattr(model,attr)
        if compiler is not None and attr=='action_condition_updater':
            module.forward=compiler.wrap(name,module.forward)
        elif timers is not None:
            module.forward=timers.wrap(module.forward,name)
        forward=module.forward
        def measured(*args,_fn=forward,_name=name,**kwargs):
            counts[_name]+=1
            if compiler is not None and _name!='observation_encoder':
                return tensor_age_call(_fn,*args,**kwargs)
            return _fn(*args,**kwargs)
        module.forward=measured
    if timers is not None:
        policy.condition_adapter.encode_condition=timers.wrap(policy.condition_adapter.encode_condition,'backbone')
        policy._decode=timers.wrap(policy._decode,'action_generation')
        model.predict=timers.wrap(model.predict,'condition_query')
    original_full,original_reset=policy._full_refresh,policy.reset
    policy.native_v0=model; policy.row_name=arm
    policy.k_c=policy.refresh_every=interval
    policy._condition_component_calls=counts
    def reset(self):
        original_reset(); self._split_context=None; counts.clear()
        if timers is not None: timers.events.clear()
    def full(self,batch,*,policy_query_index):
        condition,action,seed=original_full(batch,policy_query_index=policy_query_index)
        self._split_context=model.prepare(condition,batch['raw_rgb'],batch['proprio'],
            self.condition_layout.valid_mask,self.condition_layout.group_ids,interval)
        return condition,action,seed
    def update(self,batch,*,age,policy_query_index):
        if self._split_context is None: raise RuntimeError('Missing refreshed condition')
        condition,_=model.predict(self._split_context,age,batch['raw_rgb'],batch['proprio'])
        self.metrics.counters['num_condition_updater_calls']+=1
        self.metrics.counters['num_observation_encoder_calls']+=1
        self.metrics.counters['num_action_condition_updater_calls']+=int(use_action_condition_updater)
        action,seed=self._decode(condition,batch['proprio'],policy_query_index=policy_query_index)
        self.cached_condition,self.cached_action_chunk=condition.detach(),action.detach()
        return condition,action,seed
    policy.reset=MethodType(reset,policy); policy._full_refresh=MethodType(full,policy)
    policy._v0_update=MethodType(update,policy)
    if timers is not None:
        policy.extra_episode_metrics=lambda:dict(component_timing=timers.report(),
            condition_parameters=sum(p.numel() for p in model.parameters()),
            active_condition_parameters=sum(p.numel() for name,p in model.named_parameters()
                if use_action_condition_updater or not name.startswith('action_condition_updater.')),
            uses_action_condition_updater=use_action_condition_updater)
    policy.reset(); return policy


def check_policy(policy):
    c=policy.metrics.counters; queries=int(c['num_policy_queries']); k=policy.k_c
    full=(queries+k-1)//k; light=queries-full
    extra=light if getattr(policy,'_uses_action_condition_updater',True) else 0
    expected=dict(observation_encoder=light,condition_updater=light,action_condition_updater=extra)
    if (c['num_full_vlm_calls']!=full or c.get('num_condition_updater_calls',0)!=light
            or c.get('num_action_condition_updater_calls',0)!=extra
            or policy.nfe not in (1,3) or c['num_action_transformer_calls']!=policy.nfe*queries
            or c.get('num_generation_decoder_only_steps',0)!=0
            or queries!=(policy.step_index+4)//5
            or any(policy._condition_component_calls.get(key,0)!=value for key,value in expected.items())
            or set(policy._condition_component_calls)-set(expected)):
        raise RuntimeError('Condition/action invocation contract mismatch')
    return dict(queries=queries,full_vlm=full,transformer=policy.nfe*queries,**expected)


def make_policy(c,arm,*,smoke=False,k_c=8):
    from tools.simvla.condition_output_split_train import student_steps
    mode=f'naive{student_steps(c)}'
    policy=original_policy(c,'condition_'+mode,k_c=min(k_c,4))
    payload=load_payload(Path(c['output'])/('smoke' if smoke else 'train')/arm/'latest.pt',arm,
        identity(c),c['smoke_steps'] if smoke else c['steps'],action_mode=mode)
    return attach(policy,policy.native_v0,payload,arm,k_c)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--k-c',type=int,choices=(2,3,4,8),required=True)
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    def checker(policy,row,calls,k):
        result=check_policy(policy)
        if calls.get('condition',0)!=result['condition_updater'] or calls.get('transformer',0)!=result['transformer']:
            raise RuntimeError('Independent hooks disagree')
        return result
    run(read_json(a.config),a.arm,smoke=a.smoke,k_c=a.k_c,policy_factory=make_policy,counter_checker=checker)
