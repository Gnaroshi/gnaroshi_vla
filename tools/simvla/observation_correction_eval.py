"""Common stateful adapter for eager sd1 and compiled rb2 evaluation."""
import argparse
from collections import Counter
from pathlib import Path
from types import MethodType

import torch

from methods.latentloop.modules.observation_correction import ARMS, ObservationCorrection, expected_condition_counts
from tools.simvla.error_compensation_common import identity, read_json
from tools.simvla.error_compensation_eval import make_policy as original_policy, run
from tools.simvla.trend_condition_eval import ComponentTimers


def load_payload(path, arm, *, expected_identity=None, steps=3000):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload['format']!='simvla_observation_correction_v1' or payload['arm']!=arm
            or payload['step']!=steps or payload['contract']['action_mode']!='naive3'
            or payload['contract']['training_intervals']!=[4,8]):
        raise RuntimeError('Condition checkpoint contract mismatch')
    if expected_identity is not None and payload['identity']!=expected_identity:
        raise RuntimeError('Condition checkpoint source mismatch')
    return payload


def attach(policy, parent, payload, arm, interval, compiler=None):
    model = ObservationCorrection(parent, arm).to('cuda').eval().requires_grad_(False)
    model.load_state_dict(payload['model'], strict=True)
    timers = ComponentTimers() if compiler is None else None
    components = [('delta_encoder','observation_encoder'), ('condition_updater','condition_updater'),
        ('trend_head','trend_head'), ('measurement','measurement_head'), ('correction_gain','correction_gain')]
    counts = Counter()
    for attr, name in components:
        module = getattr(model, attr)
        if module is None: continue
        # Replay already compiled these two original modules before constructing
        # the extended stateful policy. Keep those wrappers and their graph audit.
        if compiler is not None and attr not in ('delta_encoder','condition_updater'):
            module.forward = compiler.wrap(name, module.forward)
        elif timers is not None:
            module.forward = timers.wrap(module.forward, name)
        forward = module.forward
        def measured(*args, _fn=forward, _name=name, **kwargs):
            counts[_name] += 1
            return _fn(*args, **kwargs)
        module.forward = measured
    if timers is not None:
        policy.condition_adapter.encode_condition = timers.wrap(policy.condition_adapter.encode_condition, 'backbone')
        policy._decode = timers.wrap(policy._decode, 'action_generation')
        model.predict = timers.wrap(model.predict, 'condition_query')
        model.prepare = timers.wrap(model.prepare, 'refresh_preparation')
    original_full, original_reset = policy._full_refresh, policy.reset
    policy.native_v0 = model
    policy.row_name = arm
    policy.k_c = policy.refresh_every = interval
    policy._condition_component_calls = counts
    policy._condition_arm = arm
    def reset(self):
        original_reset()
        self._trend_context = None
        counts.clear()
        if timers is not None: timers.events.clear()
    def full(self, batch, *, policy_query_index):
        condition, action, seed = original_full(batch, policy_query_index=policy_query_index)
        self._trend_context = model.prepare(condition, batch['raw_rgb'], batch['proprio'],
            self.condition_layout.valid_mask, self.condition_layout.group_ids, interval)
        return condition, action, seed
    def update(self, batch, *, age, policy_query_index):
        if self._trend_context is None: raise RuntimeError('Missing refresh state')
        condition, _ = model.predict(self._trend_context, age, batch['raw_rgb'], batch['proprio'])
        self.metrics.counters['num_condition_updater_calls'] += 1
        self.metrics.counters['num_observation_encoder_calls'] += 2 if arm=='observed_recurrent' else 1
        action, seed = self._decode(condition, batch['proprio'], policy_query_index=policy_query_index)
        self.cached_condition, self.cached_action_chunk = condition.detach(), action.detach()
        return condition, action, seed
    policy.reset = MethodType(reset, policy)
    policy._full_refresh = MethodType(full, policy)
    policy._v0_update = MethodType(update, policy)
    if timers is not None:
        policy.extra_episode_metrics = lambda: dict(component_timing=timers.report(),
            condition_parameters=sum(p.numel() for p in model.parameters()))
    policy.reset()
    return policy


def check_policy(policy):
    q = int(policy.metrics.counters['num_policy_queries'])
    arm, interval = policy._condition_arm, policy.k_c
    expected = expected_condition_counts(arm, q, interval)
    counters = policy.metrics.counters
    if (counters['num_full_vlm_calls']!=expected['full']
            or counters['num_condition_updater_calls']!=expected['light']
            or counters['num_action_transformer_calls']!=3*q
            or counters.get('num_generation_decoder_only_steps',0)!=0
            or q!=(policy.step_index+4)//5):
        raise RuntimeError('VLM/action/query execution contract mismatch')
    actual = policy._condition_component_calls
    target = dict(condition_updater=expected['light'], observation_encoder=expected['observation'],
        trend_head=expected['trend'], measurement_head=expected['measurement'], correction_gain=expected['measurement'])
    if any(actual.get(key,0)!=value for key,value in target.items()):
        raise RuntimeError(f'Condition invocation mismatch: {actual} != {target}')
    return dict(queries=q, full_vlm=expected['full'], transformer=3*q, **target)


def make_policy(c, arm, *, smoke=False, k_c=8):
    policy = original_policy(c, 'condition_naive3', k_c=min(k_c,4))
    payload = load_payload(Path(c['output'])/('smoke' if smoke else 'train')/arm/'latest.pt', arm,
        expected_identity=identity(c), steps=c['smoke_steps'] if smoke else c['steps'])
    return attach(policy, policy.native_v0, payload, arm, k_c)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--k-c',type=int,choices=(4,8),required=True)
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    def checker(policy,row,calls,k):
        result=check_policy(policy)
        if calls.get('condition',0)!=result['condition_updater'] or calls.get('transformer',0)!=result['transformer']:
            raise RuntimeError('Independent module hooks disagree')
        return result
    run(read_json(a.config), a.arm, smoke=a.smoke, k_c=a.k_c, policy_factory=make_policy, counter_checker=checker)
