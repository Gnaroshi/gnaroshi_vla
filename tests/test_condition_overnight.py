from types import SimpleNamespace

import pytest

from tools.simvla import condition_overnight as m


def test_complete_factorial_no_duplicate_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'OUTPUT', tmp_path)
    monkeypatch.setattr(m, 'identity', lambda c: 'identity')
    rows = m.row_specs()
    assert len(rows) == 8
    assert {(d['mode'], d['nfe'], d['arm']) for d in rows.values()} == {
        (mode, nfe, arm) for mode in ('joint', 'detached') for nfe in (1, 2) for arm in m.ARMS}
    jobs = m.jobs(dict(python='python'))
    assert len(jobs) == len({j['id'] for j in jobs}) == 24
    smokes = [j for j in jobs if j['id'].startswith('smoke_')]
    evals = [j for j in jobs if not j['id'].startswith('smoke_')]
    assert len(smokes) == 8 and len(evals) == 16
    assert all(j['completion']['episodes'] == 500 for j in evals)
    assert all(j['deps'][0] in {s['id'] for s in smokes} for j in evals)


@pytest.mark.parametrize('mode,nfe,arm', [('joint', 1, 'carry_base'), ('detached', 2, 'carry_output')])
def test_checkpoint_mode_solver_and_hash_are_enforced(monkeypatch, mode, nfe, arm):
    spec = dict(mode=mode, nfe=nfe, arm=arm, path='p', sha256='hash', source_identity='source')
    monkeypatch.setattr(m, 'sha', lambda p: 'hash')
    seen = []
    monkeypatch.setattr(m, 'load_payload', lambda *a, **kw: (
        seen.append(kw) or dict(contract=dict(action_gradient_mode=mode))))
    monkeypatch.setattr(m, 'original_policy', lambda c, row, **kw: SimpleNamespace(native_v0='parent', row=row))
    monkeypatch.setattr(m, 'attach', lambda p, parent, payload, arm, k: p)
    p = m.policy_factory(dict(models=dict(test=spec)), 'test', k_c=8)
    assert p.row == f'condition_naive{nfe}' and seen[0]['action_mode'] == f'naive{nfe}'
    monkeypatch.setattr(m, 'sha', lambda p: 'changed')
    with pytest.raises(RuntimeError, match='Checkpoint changed'):
        m.policy_factory(dict(models=dict(test=spec)), 'test')


def test_eta_accounts_for_four_parallel_gpus_and_finished_rows():
    pending = [f'kc4_{i}' for i in range(8)] + [f'kc8_{i}' for i in range(8)]
    d = m.estimate_remaining({}, {}, pending, 0)
    assert 12 < d['remaining_hours'] < 13
    rows = dict(kc4_a=dict(episodes=100, wall_seconds=1000))
    d = m.estimate_remaining(rows, {'4':dict(job='kc4_a')}, [], 0)
    assert d['remaining_hours'] == 4000/3600
    assert m.estimate_remaining({}, {}, [], 0)['remaining_hours'] == 0
