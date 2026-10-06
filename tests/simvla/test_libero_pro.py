import pickle
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from tools.simvla import libero_pro_zero_shot as pro
from tools.simvla.libero_pro_assets import load_numeric_states
from tools.simvla.gpu_followup_queue import acquire_lock, predecessor_pending
from tests.simvla.test_compiled_campaign import manifest


def archive(tmp_path, value):
    path = tmp_path/'states.pruned_init'
    with zipfile.ZipFile(path, 'w') as stream:
        stream.writestr('archive/data.pkl', pickle.dumps(value, protocol=4))
    return path


def test_numpy_states_only(tmp_path):
    value = np.arange(50*47, dtype=np.float64).reshape(50, 47)
    assert np.array_equal(load_numeric_states(archive(tmp_path, value)), value)
    for invalid in (np.zeros((49, 47)), np.ones((50, 47), dtype=np.float32),
                    np.full((50, 47), np.nan), np.zeros((50, 47), dtype=object)):
        with pytest.raises(ValueError):
            load_numeric_states(archive(tmp_path, invalid))


def test_no_arbitrary_pickle_globals(tmp_path):
    class Forbidden:
        def __reduce__(self):
            return eval, ('1+1',)
    with pytest.raises(pickle.UnpicklingError, match='Forbidden'):
        load_numeric_states(archive(tmp_path, Forbidden()))


def assets():
    return dict(tasks={s: [dict(task_id=i, name=f'task{i}', language=f'goal{i}')
        for i in range(10)] for s in pro.SUITES}, code_commit='code', data_revision='data')


@pytest.mark.parametrize('suite', pro.SUITES)
def test_pro_manifest_keeps_500_long_protocol(monkeypatch, suite):
    a = assets()
    monkeypatch.setattr(pro, 'verify', lambda: a)
    m = pro.make_manifest(manifest(), suite, a)
    pro.validate_manifest(m, suite, 'seed01')
    for field in ('execution_horizon', 'flow_steps', 'environment_seed', 'max_policy_actions'):
        wrong = {**m, field: m[field]+1}
        wrong['manifest_sha256'] = pro.campaign.digest({k: v for k, v in wrong.items() if k != 'manifest_sha256'})
        with pytest.raises(RuntimeError):
            pro.validate_manifest(wrong, suite, 'seed01')
    m['pro_tasks'] = []
    m['manifest_sha256'] = pro.campaign.digest({k: v for k, v in m.items() if k != 'manifest_sha256'})
    with pytest.raises(RuntimeError, match='prompt/state'):
        pro.validate_manifest(m, suite, 'seed01')


def test_pro_grid():
    c = dict(benchmark_suites=pro.SUITES, long_rows=pro.ROWS, seeds=['seed01'])
    jobs = pro.campaign.jobs(c)
    assert len(jobs) == len(set(jobs)) == 14
    assert len(jobs)*500 == 7000


@pytest.mark.parametrize('row', pro.ROWS)
def test_exact_paths(row):
    from tools.simvla.compiled_policy import expected_counts
    q = 9
    counts = pro.bridge.expected_counts(row, q) if row.startswith('bridge_') else expected_counts(row, q)
    p = SimpleNamespace(flow_steps=10, step_index=41,
        metrics=SimpleNamespace(counters={**counts, 'num_policy_queries': q}))
    pro.policy_checker(p, row)
    assert counts['num_action_transformer_calls'] == q*(10 if row == 'baseline' else 3)
    assert counts['num_generation_decoder_only_steps'] == (7*q if row == 'ours_kc2_ng3' else 0)


def test_waits_for_entire_nfe_queue(tmp_path):
    owner = acquire_lock(tmp_path/'queue.lock')
    try:
        assert predecessor_pending(tmp_path, 'queue.lock')
    finally:
        owner.close()
    assert not predecessor_pending(tmp_path, 'queue.lock')
    with pytest.raises(ValueError):
        predecessor_pending(tmp_path, '../queue.lock')


def test_actual_bddl_prompt_replaces_stale_filename():
    from collections import namedtuple
    Task = namedtuple('Task', 'name language')
    suite = object.__new__(pro.NumericSuite)
    suite.inner = SimpleNamespace(get_task=lambda i: Task('task', 'old filename instruction'))
    suite.tasks = [dict(name='task', language='changed BDDL goal')]
    assert suite.get_task(0).language == 'changed BDDL goal'
