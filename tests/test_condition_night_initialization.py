import pytest

from tools.simvla import condition_overnight_initialization as f


def test_initialization_followup_reuses_pretrained_controls(monkeypatch, tmp_path):
    monkeypatch.setattr(f, 'OUTPUT', tmp_path)
    monkeypatch.setattr(f, 'identity', lambda c: 'identity')
    plan = f.jobs(dict(python='python'))
    assert len(plan) == len({j['id'] for j in plan}) == 8
    assert len([j for j in plan if j['completion']['episodes'] == 500]) == 4
    assert all('fresh' in j['id'] for j in plan)


def test_continuation_checks_original_initialization():
    f.check_origin(dict(initialization='continuation', continuation=dict(contract=dict(initialization='fresh'))))
    with pytest.raises(RuntimeError):
        f.check_origin(dict(initialization='continuation', continuation=dict(contract=dict(initialization='pretrained'))))
