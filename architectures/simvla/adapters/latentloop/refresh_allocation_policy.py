"""Keep the trained Condition model and NFE1; change only backbone-call timing."""
from types import MethodType

from methods.latentloop.modules.refresh_allocation import RefreshAllocation


def attach(policy, strategy):
    import torch
    original_reset = policy.reset
    policy._allocation_strategy = strategy

    def reset(self):
        original_reset()
        seed = 710000 + int(self.task_id)*1000 + int(self.trial_id)
        self._allocation = RefreshAllocation(self._allocation_strategy, seed)
        self._allocation_previous_chunk = None
        self._allocation_probes = 0
        self._allocation_discarded = 0

    def refill(self, batch):
        t = int(self.query_index)
        self.metrics.counters['num_policy_queries'] += 1
        known = self._allocation.known_decision(t)
        candidate = seed = score = None
        if known is not True:
            age = t-self._allocation.last_full
            _, candidate, seed = self._v0_update(batch, age=age, policy_query_index=t)
            if self._allocation_strategy == 'overlap':
                previous = self._allocation_previous_chunk
                if previous is None:
                    raise RuntimeError('Missing previous action horizon')
                normalize = self.action_adapter.action_space.normalize_action
                # Both horizons refer to the same next five environment actions.
                score = float((normalize(candidate)[:, :5]-normalize(previous)[:, 5:10]).abs().mean())
            self._allocation_probes += int(known is None)
        full = self._allocation.decide(t, known, score)
        if full:
            self._allocation_discarded += int(candidate is not None)
            _, chunk, seed = self._full_refresh(batch, policy_query_index=t)
            source = 'full_refresh'
        else:
            chunk, source = candidate, 'native_v0'
        if tuple(chunk.shape[1:]) != (10, 7) or self.nfe != 1:
            raise RuntimeError('H10/NFE1 contract changed')
        self._allocation_previous_chunk = chunk.detach()
        self.action_queue.clear()
        self.action_queue.extend((a.detach(), source) for a in chunk[0, :5])
        record = self._allocation.records[-1]
        self.query_trace.append(dict(policy_query_index=t, age=record['age'], source=source,
            full_vlm_called=full, condition_updater_called=candidate is not None,
            action_noise_seed=seed, action_horizon=10, execution_horizon=5, k_c=8))
        self.query_index += 1
        return dict(refreshed=full, age=record['age'], queue_mode=source, action_noise_seed=seed)

    policy.reset = MethodType(reset, policy)
    policy._refill_action_queue = MethodType(refill, policy)
    policy.reset()
    return policy


def check(policy):
    values = policy._allocation.audit()
    q, full = values['queries'], values['backbone_calls']
    extra = policy._allocation_discarded
    c = policy.metrics.counters
    expected = dict(num_policy_queries=q, num_full_vlm_calls=full,
        num_condition_updater_calls=q-full+extra, num_action_transformer_calls=q+extra,
        num_generation_decoder_only_steps=0)
    if any(c.get(k, 0) != v for k, v in expected.items()):
        raise RuntimeError(f'Allocation counters differ: {dict(c)} versus {expected}')
    if q != (policy.step_index+4)//5:
        raise RuntimeError('Execution horizon changed')
    return dict(**values, discarded_action_candidates=extra,
        score_decisions=policy._allocation_probes, action_transformer_calls=q+extra)
