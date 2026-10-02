"""Trend policies preserve the paired H=10/R=5 environment contract."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
from types import MethodType

import torch

from methods.latentloop.modules.trend_condition import ARMS, TrendCondition
from methods.latentloop.modules.observed_progress import build_model
from tools.simvla.error_compensation_common import identity, read_json
from tools.simvla.error_compensation_eval import make_policy as parent_policy, run


class ComponentTimers:
    def __init__(self):
        self.events = defaultdict(list)

    def wrap(self, fn, name):
        def measured(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            self.events[name].append((start, end))
            return result
        return measured

    def report(self):
        torch.cuda.synchronize()
        return {name: dict(calls=len(events), cuda_ms_total=sum(a.elapsed_time(b) for a,b in events))
            for name, events in self.events.items()}


def expected_counts(arm, queries, k_c):
    full = (queries+k_c-1)//k_c
    light = queries-full
    return dict(transformer=3*queries, generation=7*queries,
        condition=light if arm in ('direct_anchor','trend_residual','frozen_trend_residual','progress_residual','progress_spatial') else full if arm=='trend_forecast' else 0,
        trend=0 if arm=='direct_anchor' else full,
        observation=light if arm in ('direct_anchor','trend_residual','frozen_trend_residual','progress_only','progress_residual','progress_spatial') else 0,
        full_vlm=full, queries=queries, lightweight_conditions=light)


def check_counts(policy, row, calls, k_c):
    queries = int(policy.metrics.counters['num_policy_queries'])
    expected = expected_counts(row, queries, k_c)
    actual = {key:calls.get(key,0) for key in ('transformer','generation','condition')}
    actual.update(trend=policy._trend_counts['trend'],observation=policy._trend_counts['observation'],
        full_vlm=policy.metrics.counters['num_full_vlm_calls'],queries=queries,
        lightweight_conditions=policy.metrics.counters['num_condition_updater_calls'])
    if actual!=expected or queries!=(policy.step_index+4)//5:
        raise RuntimeError(f'Invocation mismatch: {actual} != {expected}')
    return actual


def make_policy(c, row, *, smoke=False, k_c=4, checkpoint_loader=None, generation_mode='learned'):
    if generation_mode not in ('learned','naive3'):
        raise ValueError(generation_mode)
    policy=parent_policy(c,'parent' if generation_mode=='learned' else 'condition_naive3',k_c=k_c)
    payload=(checkpoint_loader(c,row) if checkpoint_loader else
        torch.load(Path(c['output'])/('smoke' if smoke else 'train')/row/'latest.pt',
        map_location='cuda',weights_only=False))
    if checkpoint_loader is None and (payload['format']!='simvla_trend_condition_v1' or payload['identity']!=identity(c)
            or payload['arm']!=row or payload['step']!=(c['smoke_steps'] if smoke else c['steps'])):
        raise RuntimeError('Incompatible or incomplete trend checkpoint')
    model=build_model(policy.native_v0,row,max_age=payload['contract']['k_c']-1).to('cuda').requires_grad_(False).eval()
    model.load_state_dict(payload['model'],strict=True)
    policy.native_v0=model
    policy.row_name=row
    timers=ComponentTimers()
    if model.trend_head is not None:
        model.trend_head.forward=timers.wrap(model.trend_head.forward,'trend_head')
    if model.delta_encoder is not None:
        model.delta_encoder.forward=timers.wrap(model.delta_encoder.forward,'observation_encoder')
    if row == 'progress_spatial':
        model.spatial_encode=timers.wrap(model.spatial_encode,'observation_encoder_spatial')
    if hasattr(model,'progress_head'):
        model.progress_head.forward=timers.wrap(model.progress_head.forward,'progress_head')
    if model.condition_updater is not None:
        model.condition_updater.forward=timers.wrap(model.condition_updater.forward,'residual_head')
    policy.condition_adapter.encode_condition=timers.wrap(policy.condition_adapter.encode_condition,'backbone')
    policy._decode=timers.wrap(policy._decode,'generation_total')
    original_full, original_reset = policy._full_refresh, policy.reset

    def reset(self):
        original_reset()
        self._trend_context=None
        self._trend_counts=Counter()
        timers.events.clear()

    def full(self,batch,*,policy_query_index):
        condition,action,seed=original_full(batch,policy_query_index=policy_query_index)
        self._trend_context=model.prepare(condition,batch['raw_rgb'],batch['proprio'],
            self.condition_layout.valid_mask,self.condition_layout.group_ids)
        self._trend_counts['trend']+=int(model.trend_head is not None)
        return condition,action,seed

    def update(self,batch,*,age,policy_query_index):
        if self._trend_context is None: raise RuntimeError('No trend anchor')
        tick=time.perf_counter()
        condition,_=model.predict(self._trend_context,age,batch['raw_rgb'],batch['proprio'])
        self.metrics.latencies.setdefault('condition_updater_ms',[]).append((time.perf_counter()-tick)*1000)
        observed=int(model.delta_encoder is not None)
        self.metrics.counters['num_condition_updater_calls']+=1
        self.metrics.counters['num_observation_encoder_calls']+=observed
        self._trend_counts['observation']+=observed
        action,seed=self._decode(condition,batch['proprio'],policy_query_index=policy_query_index)
        self.cached_condition=condition.detach()
        self.cached_action_chunk=action.detach()
        return condition,action,seed

    # condition_query is a nested timer including observation/residual, excluding generation.
    model.predict=timers.wrap(model.predict,'condition_query')
    model.prepare=timers.wrap(model.prepare,'refresh_preparation')
    policy.reset=MethodType(reset,policy)
    policy._full_refresh=MethodType(full,policy)
    policy._v0_update=MethodType(update,policy)
    policy.extra_episode_metrics=lambda: dict(component_timing=timers.report(),
        condition_method=row,condition_parameters=sum(p.numel() for p in model.parameters()))
    policy.reset()
    return policy


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--row',choices=ARMS,required=True)
    p.add_argument('--k-c',type=int,choices=(4,),default=4)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run(read_json(a.config),a.row,smoke=a.smoke,k_c=a.k_c,policy_factory=make_policy,counter_checker=check_counts)
