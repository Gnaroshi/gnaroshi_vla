from collections import Counter, defaultdict
from types import SimpleNamespace

import pytest

from tools.simvla import condition_nfe_sweep as sweep


@pytest.mark.parametrize('row', sweep.ROWS)
@pytest.mark.parametrize('actions', [1, 5, 6, 40, 41, 900])
def test_counters_and_no_generation(row, actions):
    group, base, k, nfe = sweep.specification(row)
    q = (actions + 4)//5
    counts = sweep.expected_counts(row, q)
    assert counts['num_action_transformer_calls'] == nfe*q
    assert counts['num_generation_decoder_only_steps'] == 0
    assert counts['num_full_vlm_calls'] == sum(i % k == 0 for i in range(q))
    policy = SimpleNamespace(flow_steps=10, step_index=actions,
        metrics=SimpleNamespace(counters={**counts, 'num_policy_queries': q}))
    sweep.check_policy(policy, row)
    policy.metrics.counters['num_generation_decoder_only_steps'] = 1
    with pytest.raises(RuntimeError):
        sweep.check_policy(policy, row)


@pytest.mark.parametrize('row', sweep.ROWS)
def test_factory_changes_only_solver_and_bridge_interval(monkeypatch, row):
    noise = object()
    calls = []
    metrics = SimpleNamespace(counters=Counter(), latencies=defaultdict(list))
    def decode(condition, proprio, *, steps, initial_noise, return_debug):
        calls.append((steps, initial_noise))
        return SimpleNamespace(action='action', debug={'iterations': steps})
    policy = SimpleNamespace(flow_steps=10, refresh_every=2, metrics=metrics,
        _sync=lambda: None, _paired_initial_noise=lambda *a: (noise, 99),
        action_adapter=SimpleNamespace(decode_action_from_condition=decode))
    monkeypatch.setattr(sweep, 'attach_policy', lambda *a: policy)
    actual = sweep.make_policy(SimpleNamespace(loop=None), {}, row, {})
    assert actual.flow_steps == 10
    assert actual.NFE == sweep.specification(row)[3]
    assert actual._decode(None, None, policy_query_index=0) == ('action', 99)
    assert calls == [(actual.NFE, noise)]
    assert metrics.counters['num_action_transformer_calls'] == actual.NFE
    with pytest.raises(RuntimeError):
        sweep.make_policy(SimpleNamespace(loop=object()), {}, row, {})


@pytest.mark.parametrize('row', sweep.ROWS)
def test_compile_required_paths(row):
    names = sweep.required_components(sweep.specification(row)[1])
    compiler = SimpleNamespace(records={name: {'graphs': 1} for name in names})
    sweep.check_compiler(compiler, row)
    compiler.records['generation_updater'] = {'graphs': 1}
    with pytest.raises(RuntimeError):
        sweep.check_compiler(compiler, row)
    compiler.records = {}
    with pytest.raises(RuntimeError):
        sweep.check_compiler(compiler, row)


def test_only_identical_nfe3_references():
    assert len(sweep.ROWS) == 12
    assert len(sweep.REFERENCES) == 4
    assert all(sweep.specification(row)[3] == 3 for row in sweep.REFERENCES)
    for row in ('condition_k4_nfe1', 'condition_k2_nfe10'):
        with pytest.raises(ValueError):
            sweep.specification(row)


def test_reference_tally_and_duplicate_rejection(tmp_path):
    import csv
    contract = {'provenance': 'test'}
    row = 'naive_nfe3'
    result = dict(identity=sweep.campaign.digest(dict(campaign=sweep.campaign.digest(contract),
        suite='libero_10', seed='seed01', row=row, smoke=False)), episodes=500,
        verdict='EPISODES_COMPLETE', timing_valid_episodes=500, successes=500,
        success_rate=1.0, executed_actions=5000, pooled_policy_ms_per_action=2.0)
    sweep.write_json(tmp_path/'summary.json', result)
    records = [[t, i, 1, 10, 20, True] for t in range(10) for i in range(50)]
    def write():
        with (tmp_path/'outcomes.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['task_id', 'trial_id', 'success', 'episode_length', 'policy_ms_total', 'timing_valid'])
            writer.writerows(records)
    write()
    assert sweep.validate_reference(tmp_path, contract, row) == result
    records[-1] = records[0]
    write()
    with pytest.raises(RuntimeError):
        sweep.validate_reference(tmp_path, contract, row)
