import math
import random

import pytest

from methods.latentloop.modules.refresh_allocation import RefreshAllocation, STRATEGIES


@pytest.mark.parametrize('strategy', STRATEGIES)
@pytest.mark.parametrize('length', [1, 7, 8, 9, 16, 180])
def test_exact_block_budget_and_age(strategy, length):
    model = RefreshAllocation(strategy)
    rng = random.Random(42)
    for t in range(length):
        known = model.known_decision(t)
        model.decide(t, known, rng.random() if known is not True else None)
    result = model.audit()
    assert result['backbone_calls'] in {2*(length//8)+(length%8 > 0), 2*(length//8)+min(2, length%8)}
    if length % 8 == 0:
        assert result['backbone_calls'] == length//4


def test_known_schedules():
    for strategy, slots in [('even', [0,4,8,12]), ('burst', [0,1,8,9])]:
        m = RefreshAllocation(strategy)
        actual = [t for t in range(16) if m.decide(t, m.known_decision(t))]
        assert actual == slots


def test_future_score_cannot_change_prefix():
    a, b = RefreshAllocation('overlap'), RefreshAllocation('overlap')
    for t in range(40):
        a.decide(t, a.known_decision(t), float(t % 7))
        b.decide(t, b.known_decision(t), float(t % 7) if t < 24 else 10000.)
    assert a.records[:24] == b.records[:24]


def test_invalid_inputs():
    m = RefreshAllocation('overlap')
    with pytest.raises(ValueError):
        m.known_decision(1)
    m.decide(0, m.known_decision(0))
    with pytest.raises(ValueError):
        m.decide(1, m.known_decision(1), math.nan)


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_policy_routing_counts_and_context_reset(strategy):
    from collections import Counter, deque
    from types import SimpleNamespace
    import torch
    from architectures.simvla.adapters.latentloop.refresh_allocation_policy import attach, check

    class Policy:
        task_id, trial_id, nfe = 0, 1, 1
        action_adapter = SimpleNamespace(action_space=SimpleNamespace(normalize_action=lambda x:x))

        def reset(self):
            self.query_index = self.step_index = self.context_age = 0
            self.metrics = SimpleNamespace(counters=Counter())
            self.query_trace = []
            self.action_queue = deque()

        def _full_refresh(self, batch, *, policy_query_index):
            self.context_age = 0
            self.metrics.counters['num_full_vlm_calls'] += 1
            self.metrics.counters['num_action_transformer_calls'] += 1
            return None,torch.full((1,10,7),float(policy_query_index)),policy_query_index

        def _v0_update(self, batch, *, age, policy_query_index):
            assert age == self.context_age + 1
            assert 1 <= age <= 7
            self.context_age = age
            self.metrics.counters['num_condition_updater_calls'] += 1
            self.metrics.counters['num_action_transformer_calls'] += 1
            return None,torch.full((1,10,7),float(policy_query_index%5)),policy_query_index

    model = attach(Policy(),strategy)
    for repeat in range(2):
        model.reset()
        for _ in range(180):
            model._refill_action_queue({})
            assert len(model.action_queue) == 5
            model.step_index += 5
            check(model)
        assert model._allocation.last_query == 179
