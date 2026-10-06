from collections import Counter, deque
from types import SimpleNamespace

import pytest
import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import SynchronizedConditionK_CPolicy
from architectures.simvla.adapters.refresh_calibration.policy import attach, check_counts, check_reset
from methods.refresh_calibration.model import RefreshCalibratedCondition


class FakePolicy:
    device = torch.device('cpu')
    replan_steps = 5
    log_action_chunks = False
    _refill_action_queue = SynchronizedConditionK_CPolicy._refill_action_queue

    def reset(self):
        self.metrics = SimpleNamespace(counters=Counter())
        self.query_index = self.step_index = 0
        self.action_queue = deque()
        self.query_trace = []
        self.cached_condition = self.cached_raw_rgb = self.cached_proprio = None
        self.cached_action_chunk = self.condition_layout = None

    def preprocess(self, image0, image1, proprio, prompt):
        return dict(raw_rgb=torch.stack((image0, image1))[None], proprio=proprio[None])

    def _decode(self, condition, proprio, *, policy_query_index):
        self.metrics.counters['num_action_transformer_calls'] += 3
        return condition.mean().expand(1, 10, 7), policy_query_index

    def _full_refresh(self, batch, *, policy_query_index):
        condition = torch.ones(1, 12, 32) * (1 + policy_query_index)
        self.condition_layout = SimpleNamespace(valid_mask=torch.ones(1, 12, dtype=torch.bool),
                                                group_ids=torch.zeros(1, 12, dtype=torch.long))
        self.metrics.counters['num_full_vlm_calls'] += 1
        return condition, *self._decode(condition, batch['proprio'], policy_query_index=policy_query_index)


@pytest.mark.parametrize('k', (4, 8))
@pytest.mark.parametrize('variant', ('fixed', 'anchor_input', 'ridge'))
def test_live_queue_and_reset(k, variant):
    torch.manual_seed(7)
    model = RefreshCalibratedCondition(variant, dim=32, width=8)
    payload = dict(contract=dict(model=dict(dim=32, width=8)), model=model.state_dict(),
                   language_bank={'pick cup': torch.randn(1, 32)})
    policy = attach(FakePolicy(), payload, variant, k)
    anchor_state = None
    for q in range(9):
        batch = policy.preprocess(torch.rand(32, 32, 3), torch.rand(32, 32, 3),
                                  torch.rand(8), 'pick_cup')
        policy._refill_action_queue(batch)
        assert len(policy.action_queue) == 5
        assert policy.query_trace[-1]['age'] == q % k
        if q % k == 0:
            anchor_state = policy._refresh_state
        else:
            assert policy._refresh_state is anchor_state
        policy.step_index += 5
    check_counts(policy, variant, {'transformer': 27}, k)
    assert len(policy.query_trace) == 9
    check_reset(policy)
    assert policy._refresh_state is None
    with pytest.raises(RuntimeError, match='Unregistered instruction'):
        policy.preprocess(torch.rand(32, 32, 3), torch.rand(32, 32, 3), torch.rand(8), 'unknown task')
