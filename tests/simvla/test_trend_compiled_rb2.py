from types import SimpleNamespace

import pytest
import torch

from tools.simvla.trend_compiled_rb2 import condition_at_age, expected_counts, check_policy, check_compiler, reset_policy


@pytest.mark.parametrize('row,k', [('trend_k4', 4), ('trend_k8', 8)])
def test_counts_across_refresh_and_queue(row, k):
    for actions in (1, 5, 6, 20, 21, 35, 36, 40, 41, 900):
        q = (actions+4)//5
        counters = dict(expected_counts(row, q), num_policy_queries=q)
        assert counters['num_trend_head_calls'] == (q+k-1)//k
        p = SimpleNamespace(metrics=SimpleNamespace(counters=counters), step_index=actions)
        check_policy(p, row)
        counters['num_observation_encoder_calls'] += 1
        with pytest.raises(RuntimeError):
            check_policy(p, row)


def test_k8_uses_same_slope_and_keeps_anchor():
    context = SimpleNamespace(anchor=torch.randn(1, 12, 960), trend=torch.randn(1, 12, 960))
    original = context.anchor.clone()
    for age in range(1, 8):
        result = condition_at_age(context, age, 8)
        assert torch.equal(result, original+age*context.trend)
        if age < 4:
            assert torch.equal(result, condition_at_age(context, age, 4))
    assert torch.equal(original, context.anchor)
    with pytest.raises(ValueError):
        condition_at_age(context, 8, 8)


def test_compilation_must_include_actual_trend_entrypoint():
    records = {n: dict(graphs=1) for n in ('vlm','action_transformer','action_decoder','generation_updater','trend_head')}
    c = SimpleNamespace(records=records)
    check_compiler(c, 'trend_k8')
    records['trend_head']['graphs'] = 0
    with pytest.raises(RuntimeError, match='trend_head'):
        check_compiler(c, 'trend_k8')


def test_reset_rejects_old_trend_context():
    p = SimpleNamespace(reset=lambda: None, query_index=0, step_index=0, action_queue=[], _trend_context=1)
    with pytest.raises(RuntimeError, match='Trend context'):
        reset_policy(p)
