"""Frozen trend controls and explicitly labelled K6/K8 extrapolation."""
import argparse
from types import MethodType

import torch

from tools.simvla.error_compensation_common import read_json, sha
from tools.simvla.error_compensation_eval import run
from tools.simvla.trend_condition_eval import make_policy as trend_policy


ROWS = {
    'trend_naive3': dict(k_c=4, generation='naive3', hold=False),
    'hold_learned3': dict(k_c=4, generation='learned', hold=True),
    'trend_k6': dict(k_c=6, generation='learned', hold=False),
    'trend_k8': dict(k_c=8, generation='learned', hold=False),
    'hold_k8': dict(k_c=8, generation='learned', hold=True),
}


def load_checkpoint(c, arm):
    spec = c['trend_checkpoint']
    if arm != 'trend_only' or sha(spec['path']) != spec['sha256']:
        raise RuntimeError('Frozen trend checkpoint hash or arm mismatch')
    payload = torch.load(spec['path'], map_location='cpu', weights_only=False)
    if (payload['format'] != 'simvla_trend_condition_v1'
            or payload['identity'] != spec['identity'] or payload['arm'] != arm
            or payload['step'] != 3000):
        raise RuntimeError('Frozen trend checkpoint provenance mismatch')
    return payload


def condition_at_age(context, age, k_c, hold):
    if k_c not in (4, 6, 8) or not 1 <= age < k_c:
        raise ValueError('Age outside this evaluation window')
    # Use the same per-query slope at all K. Ages 4..7 are extrapolation,
    # not rescaled interpolation or evidence of long-window training.
    condition = context.anchor if hold else context.anchor + age * context.trend
    return condition, torch.zeros_like(context.anchor)


def expected_counts(row, queries):
    spec = ROWS[row]
    full = (queries + spec['k_c'] - 1) // spec['k_c']
    return dict(transformer=3 * queries,
        generation=(7 * queries if spec['generation']=='learned' else 0),
        condition=0, trend=(0 if spec['hold'] else full), observation=0,
        full_vlm=full, queries=queries, lightweight_conditions=queries-full)


def check_counts(policy, row, calls, k_c):
    if ROWS[row]['k_c'] != k_c:
        raise RuntimeError('Unexpected refresh interval')
    q = int(policy.metrics.counters['num_policy_queries'])
    observed = {key:calls.get(key,0) for key in ('transformer','generation','condition')}
    observed.update(trend=policy._trend_counts['trend'], observation=policy._trend_counts['observation'],
        full_vlm=policy.metrics.counters['num_full_vlm_calls'], queries=q,
        lightweight_conditions=policy.metrics.counters['num_condition_updater_calls'])
    expected = expected_counts(row,q)
    if observed != expected or q != (policy.step_index+4)//5:
        raise RuntimeError(f'Invocation mismatch: {observed} != {expected}')
    return observed


def make_policy(c, row, *, smoke=False, k_c=4):
    spec = ROWS[row]
    if spec['k_c'] != k_c:
        raise ValueError('Row and refresh interval disagree')
    # Construct the validated K4 parent first, then install the dedicated
    # linear/hold predictor. The recursive updater's supported ages stay intact.
    policy = trend_policy(c,'trend_only',k_c=4,checkpoint_loader=load_checkpoint,
        generation_mode=spec['generation'])
    model = policy.native_v0
    if spec['hold']:
        model.trend_head = None
    def predict(self, context, age, images=None, proprio=None):
        return condition_at_age(context,age,k_c,spec['hold'])

    if k_c > 4 or spec['hold']:
        # Replace the bounded predictor and retain the same CUDA-event scope.
        from tools.simvla.trend_condition_eval import ComponentTimers
        extra_timer = ComponentTimers()
        model.predict = extra_timer.wrap(MethodType(predict,model),'condition_query')
        old_reset = policy.reset
        def reset(self):
            old_reset()
            extra_timer.events.clear()
        policy.reset = MethodType(reset,policy)
        old_metrics = policy.extra_episode_metrics
        def metrics():
            result = old_metrics()
            result['component_timing'].update(extra_timer.report())
            return result
        policy.extra_episode_metrics = metrics
    policy.k_c = policy.refresh_every = k_c
    policy.row_name = policy.mode = row
    base_metrics = policy.extra_episode_metrics
    def metrics():
        result = base_metrics()
        result.update(condition_method='hold' if spec['hold'] else 'trend_only',
            generation_method=spec['generation'], condition_training_max_age=3,
            evaluation_max_age=k_c-1, extrapolation=(k_c>4 and not spec['hold']),
            trend_checkpoint_sha256=c['trend_checkpoint']['sha256'])
        return result
    policy.extra_episode_metrics = metrics
    policy.reset()
    return policy


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--row',choices=ROWS,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run(read_json(a.config),a.row,smoke=a.smoke,k_c=ROWS[a.row]['k_c'],
        policy_factory=make_policy,counter_checker=check_counts)
