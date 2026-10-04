import fcntl
from types import SimpleNamespace

import pytest

from tools.simvla import interval_followup_rb2 as followup


def test_five_predeclared_rows_no_duplicate_k8():
    assert list(followup.ROWS.values()) == [
        (4, 'ours_kc2_ng3'), (4, 'condition_naive3'), (4, 'condition_nfe10'),
        (5, 'ours_kc2_ng3'), (6, 'ours_kc2_ng3')]


@pytest.mark.parametrize('actions', [1, 5, 6, 20, 21, 25, 26, 30, 31, 40, 41, 900])
def test_query_refresh_and_action_head_counts(actions):
    queries = (actions + 4) // 5
    for row, (k, mode) in followup.ROWS.items():
        counters = followup.expected_counts(row, queries)
        full = len(range(0, queries, k))
        assert counters['num_full_vlm_calls'] == counters['num_trend_head_calls'] == full
        assert counters['num_condition_updater_calls'] == counters['num_observation_encoder_calls'] == queries-full
        assert counters['num_action_transformer_calls'] == queries*(10 if mode == 'condition_nfe10' else 3)
        assert counters['num_generation_decoder_only_steps'] == (7*queries if mode == 'ours_kc2_ng3' else 0)
        counters['num_policy_queries'] = queries
        policy = SimpleNamespace(metrics=SimpleNamespace(counters=counters), step_index=actions)
        followup.check_policy(policy, row)
        counters['num_action_transformer_calls'] += 1
        with pytest.raises(RuntimeError, match='num_action_transformer_calls'):
            followup.check_policy(policy, row)


def test_compile_checks_follow_actual_path():
    common = {'vlm', 'action_transformer', 'trend_head', 'observation_encoder', 'condition_updater'}
    for row, (_, mode) in followup.ROWS.items():
        required = common | ({'action_decoder', 'generation_updater'} if mode == 'ours_kc2_ng3' else set())
        compiler = SimpleNamespace(records={key: {'graphs': 1} for key in required})
        followup.check_compiler(compiler, row)
        for key in required:
            compiler.records[key]['graphs'] = 0
            with pytest.raises(RuntimeError, match='Compile bypass'):
                followup.check_compiler(compiler, row)
            compiler.records[key]['graphs'] = 1


def test_waits_for_entire_bridge_queue_not_gpu_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, 'PREDECESSOR', tmp_path)
    with pytest.raises(FileNotFoundError):
        followup.predecessor_busy()
    with (tmp_path / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert followup.predecessor_busy()
        fcntl.flock(lock, fcntl.LOCK_UN)
        assert not followup.predecessor_busy()


def test_pins_selected_checkpoint(monkeypatch):
    value = dict(selected_arm='frozen_trend_residual', checkpoints=dict(previous_joint=dict(
        file='previous_joint.pt', sha256=followup.CHECKPOINT_SHA256)))
    monkeypatch.setattr(followup, 'bundle', lambda _: value)
    spec = followup.selected_checkpoint()
    assert spec['step'] == 3000
    assert spec['path'].endswith('/previous_joint.pt')
    value['checkpoints']['previous_joint']['sha256'] = 'different'
    with pytest.raises(RuntimeError, match='identity changed'):
        followup.selected_checkpoint()


def test_replay_rejects_mismatched_mode(monkeypatch):
    monkeypatch.setattr(followup, 'Replay', lambda c, mode, compiler, samples: mode)
    for row, (k, mode) in followup.ROWS.items():
        c = dict(action_mode=mode, condition_interval=k)
        assert followup.replay_factory(c, row, None, None) == mode
        with pytest.raises(RuntimeError, match='declared row differ'):
            followup.replay_factory({**c, 'condition_interval': k+1}, row, None, None)


def test_row_failure_does_not_cancel_remaining_rows(tmp_path, monkeypatch):
    predecessor = tmp_path / 'bridge'
    predecessor.mkdir()
    followup.write_json(predecessor / 'pipeline_status.json', {'phase': 'complete'})
    monkeypatch.setattr(followup, 'OUTPUT', tmp_path / 'followup')
    monkeypatch.setattr(followup, 'PREDECESSOR', predecessor)
    monkeypatch.setattr(followup, 'preflight', lambda c: {})
    waiting = iter([True, False])
    monkeypatch.setattr(followup, 'predecessor_busy', lambda: next(waiting))
    monkeypatch.setattr(followup.time, 'sleep', lambda _: None)
    monkeypatch.setattr(followup, 'wait_idle', lambda: None)
    calls = []

    def evaluate(c, row, spec, *, output):
        calls.append(row)
        assert c['condition_interval'] == followup.ROWS[row][0]
        if len(calls) == 1:
            raise RuntimeError('technical error')
        return dict(successes=200, episodes=500, success_rate=0.4,
                    pooled_policy_ms_per_action=6.0, timing_valid_episodes=500)

    monkeypatch.setattr(followup, 'evaluate', evaluate)
    assert followup.run_all({'scope': 'test', 'selection': {}}) == 1
    assert calls == list(followup.ROWS)
    summary = followup.read_json(followup.OUTPUT / 'combined_summary.json')
    assert len(summary['results']) == 4 and len(summary['failures']) == 1
    assert followup.read_json(followup.OUTPUT / 'pipeline_status.json')['phase'] == 'finished_with_failures'


def test_duplicate_launcher_does_not_replace_status(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, 'OUTPUT', tmp_path)
    with (tmp_path / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            followup.run_all({})
