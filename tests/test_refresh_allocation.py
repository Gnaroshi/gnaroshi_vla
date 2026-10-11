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
