import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.simvla import matched_reference_seeds as study


def fixture_values():
    manifest = dict(action_noise_seed_base=100, determinism_seed=7, environment_seed=7,
                    episodes=[dict(task_id=t, trial_id=i) for t in range(10) for i in range(50)])
    records = [dict(**row, identity='same', model_sha256='weights', success=True,
                    timing_valid=True, episode_length=5, policy_ms_total=20,
                    action_noise_seed_base=101) for row in manifest['episodes']]
    summary = dict(identity='same', episodes=500, successes=500, success_rate=1.,
                   timing_valid_episodes=500, policy_ms_per_action=4.)
    return manifest, records, summary


def test_only_noise_changes():
    manifest, _, _ = fixture_values()
    expected = copy.deepcopy(manifest);expected['action_noise_seed_base'] = 101
    assert study.second_manifest(manifest) == expected
    assert manifest['action_noise_seed_base'] == 100


def test_complete_records_and_latency():
    manifest, records, summary = fixture_values()
    assert study.validate_records(summary, records, manifest, 'same', 'weights', 101) == summary


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'seed', 'weight', 'latency', 'success'])
def test_reject_mixed_or_incomplete(change):
    manifest, records, summary = fixture_values()
    if change == 'missing':records.pop()
    if change == 'duplicate':records[-1] = dict(records[0])
    if change == 'seed':records[0]['action_noise_seed_base'] = 100
    if change == 'weight':records[0]['model_sha256'] = 'wrong'
    if change == 'latency':summary['policy_ms_per_action'] = 5.
    if change == 'success':records[0]['success'] = False
    with pytest.raises(RuntimeError):
        study.validate_records(summary, records, manifest, 'same', 'weights', 101)


def test_queue_independent_retries_and_full_length():
    plan = study.jobs(dict(identity='same'))
    assert len(plan) == 6
    ids = [j['id'] for j in plan]
    assert len(set(ids)) == 6
    for index, row in enumerate(study.ROWS):
        smoke, full = plan[2*index:2*index+2]
        assert smoke['deps'] == []
        assert full['deps'] == [smoke['id']]
        assert full['completion']['episodes'] == 500
        assert 'codex_outputs' not in ' '.join(full['cmd'])
        assert Path(full['summary']).parent.name == row


def test_future_predecessor_becomes_priority_when_it_starts(tmp_path):
    specs = study.predecessor_specs(tmp_path)
    assert not study.queue.predecessor_pending(specs)
    lock = study.queue.acquire_lock(tmp_path / 'evaluation/queue.lock')
    try:
        assert study.queue.predecessor_pending(specs)
        assert study.predecessor_specs(tmp_path) == specs
        assert study.queue.predecessor_pending(specs)
    finally:
        lock.close()
    assert not study.queue.predecessor_pending(specs)


@pytest.mark.parametrize('row', study.ROWS)
def test_scoped_seed_and_restoration(row, monkeypatch):
    from tools.simvla import age_specialization_eval as recurrent
    from tools.simvla import visual_memory_followup_eval as bridge
    from tools.simvla import condition_interval_compare as interval
    eval_module = recurrent if row == study.ROWS[0] else bridge
    old_study, old_writer = eval_module.study, eval_module.write_json
    def original(*args):return SimpleNamespace(action_noise_seed_base=100), None
    def bridge_factory(*args):return SimpleNamespace(action_noise_seed_base=args[-1]['action_noise_seed_base'])
    monkeypatch.setattr(study.prior, 'build_policy', original)
    monkeypatch.setattr(interval, 'policy_factory', bridge_factory)
    c = dict(manifest=dict(action_noise_seed_base=101), model=dict(config={}))
    with study.runtime_scope(row, c) as active:
        assert active is eval_module
        assert study.prior.build_policy('mode', {})[0].action_noise_seed_base == 101
        assert interval.policy_factory(None, {}, row, c['manifest']).action_noise_seed_base == 101
        assert active.study.checked()['manifest']['action_noise_seed_base'] == 101
    assert eval_module.study is old_study and eval_module.write_json is old_writer
    assert study.prior.build_policy is original and interval.policy_factory is bridge_factory
