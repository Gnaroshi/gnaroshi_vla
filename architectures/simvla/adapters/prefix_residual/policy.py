from pathlib import Path
from types import MethodType

import torch

from methods.prefix_residual.model import PrefixResidual
from .prefix import FrozenPrefix
from tools.simvla.error_compensation_common import identity, read_json, sha


def attach(policy, c, row, k, payload=None):
    prefix = FrozenPrefix(policy.model, c['prefix_depth'])
    updater = PrefixResidual(**c['model']).to(policy.device).eval().requires_grad_(False)
    if payload is not None:
        updater.load_state_dict(payload['model'], strict=True)
    updater.condition_updater = None
    from tools.simvla.trend_condition_eval import ComponentTimers
    timers = ComponentTimers()
    prefix.encode = timers.wrap(prefix.encode, 'current_prefix_including_vision')
    updater.forward = timers.wrap(updater.forward, 'suffix_change_update')
    policy.condition_adapter.encode_condition = timers.wrap(policy.condition_adapter.encode_condition, 'full_backbone')
    policy._decode = timers.wrap(policy._decode, 'action_head_nfe3')
    policy.native_v0 = updater
    policy.prefix_runtime = prefix
    policy.k_c = policy.refresh_every = k
    policy.row_name = policy.mode = row
    full, reset = policy._full_refresh, policy.reset
    policy.prefix_layer_calls = [0] * prefix.total_layers
    policy._prefix_count_handles = []
    for i, layer in enumerate(policy.model.vlm.model.text_model.layers):
        def count(_m, _a, index=i):
            policy.prefix_layer_calls[index] += 1
        policy._prefix_count_handles.append(layer.register_forward_pre_hook(count))

    def reset_state(self):
        reset()
        self._prefix_anchor = self._condition_anchor = None
        self.prefix_layer_calls[:] = [0] * prefix.total_layers
        timers.events.clear()

    def refresh(self, batch, *, policy_query_index):
        prefix.start_capture()
        try:
            condition, actions, seed = full(batch, policy_query_index=policy_query_index)
            self._prefix_anchor = prefix.finish_capture().detach()
        finally:
            prefix.capture_enabled = False
        self._condition_anchor = condition.detach()
        return condition, actions, seed

    def update(self, batch, *, age, policy_query_index):
        if self._prefix_anchor is None or age != policy_query_index % k:
            raise RuntimeError('Missing refresh state or invalid age')
        current = prefix.encode(batch)
        condition = updater(self._condition_anchor, self._prefix_anchor, current,
                            self.condition_layout.valid_mask, learned=row == 'learned')
        actions, seed = self._decode(condition, batch['proprio'], policy_query_index=policy_query_index)
        self.metrics.counters['num_condition_updater_calls'] += 1
        self.metrics.counters['num_prefix_only_calls'] += 1
        self.cached_condition, self.cached_action_chunk = condition.detach(), actions.detach()
        return condition, actions, seed

    policy.reset = MethodType(reset_state, policy)
    policy._full_refresh = MethodType(refresh, policy)
    policy._v0_update = MethodType(update, policy)
    policy.extra_episode_metrics = lambda: dict(component_timing=timers.report(),
        trained_parameters=sum(p.numel() for p in updater.parameters()) if row == 'learned' else 0,
        prefix_depth=prefix.depth, total_text_layers=prefix.total_layers,
        timing_note='Disjoint backbone/prefix/update/action intervals; total wall timing also includes preprocessing and queue.')
    policy.reset()
    return policy


def make_policy(c, row, *, smoke=False, k_c=4):
    from tools.simvla.error_compensation_eval import make_policy as parent
    payload = None
    if row == 'learned':
        directory = Path(c['output']) / ('smoke_train' if smoke else 'train') / f'k{k_c}'
        report = read_json(directory / 'summary.json')
        if sha(directory / 'latest.pt') != report['checkpoint_sha256']:
            raise RuntimeError('Checkpoint checksum changed')
        payload = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
        if payload['identity'] != identity(c) or payload['k'] != k_c or payload['steps'] != (2 if smoke else c['steps']):
            raise RuntimeError('Wrong checkpoint identity/age/steps')
    return attach(parent(c, 'condition_naive3', k_c=min(k_c, 4)), c, row, k_c, payload)


def check_counts(policy, row, actual, k_c=4):
    q = int(policy.metrics.counters['num_policy_queries'])
    full = (q + k_c - 1) // k_c
    if actual.get('transformer', 0) != 3*q or policy.metrics.counters['num_full_vlm_calls'] != full:
        raise RuntimeError('Action NFE or full VLM count changed')
    expected = [q if i < policy.prefix_runtime.depth else full for i in range(policy.prefix_runtime.total_layers)]
    if policy.prefix_layer_calls != expected:
        raise RuntimeError(f'Text-layer calls {policy.prefix_layer_calls} != {expected}')
    if q != (policy.step_index + 4)//5 or policy.metrics.counters['num_prefix_only_calls'] != q-full:
        raise RuntimeError('Query cadence or prefix-only count changed')
    return dict(queries=q, full_vlm=full, prefix_queries=q-full, text_layer_calls=expected,
                action_transformer=3*q, generation_updates=0)
