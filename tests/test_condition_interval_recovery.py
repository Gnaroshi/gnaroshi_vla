from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.simvla import condition_interval_recovery as recovery
from tools.simvla import condition_interval_recovery_rb2 as rb2


def previous(k):
    return dict(output='old',training_k_c=k,training_intervals=[k],extra_source_files=[],
        initialization='continuation',steps=5000,warmup_steps=250,sample_step_offset=10000,
        interval_transition=dict(source=[4,8],target=[k]),initial_models={'carry_base':{}},
        continuation_source_steps=7000,continuation_source_action_mode='naive1')


@pytest.mark.parametrize('mode',recovery.MODES)
def test_cache_window_is_independent_from_refresh_interval(mode):
    init,k=recovery.MODES[mode]
    c=recovery.make_config(previous(k),mode)
    assert c['training_k_c']==8 and c['training_intervals']==[k]
    assert c['evaluation_condition_intervals']==[k]
    if init=='continue':
        assert c['steps']==5000 and c['sample_step_offset']==10000 and c['warmup_steps']==250
        assert c['initialization']=='continuation'
    else:
        assert c['steps']==10000 and c['sample_step_offset']==0 and c['warmup_steps']==500
        assert c['initialization']=='fresh' and 'initial_models' not in c and 'interval_transition' not in c


def test_dataset_preflight_calls_real_constructor_and_rejects_mismatch(monkeypatch):
    from architectures.simvla.adapters.latentloop.efficient_multirate import condition_mechanism as m
    calls=[]
    def make(c,p):
        calls.append((c,p));return SimpleNamespace(contract=lambda:{'window':8}),SimpleNamespace(contract=lambda:{'window':8})
    monkeypatch.setattr(m,'make_datasets',make)
    expected=dict(data={'window':8},heldout={'window':8})
    assert recovery.verify_dataset(dict(training_k_c=8),{},expected)==expected
    assert len(calls)==1
    with pytest.raises(RuntimeError):recovery.verify_dataset(dict(training_k_c=2),{},expected)
    with pytest.raises(RuntimeError):recovery.verify_dataset(dict(training_k_c=8),{},dict(data={},heldout={}))


def test_sd1_jobs_no_retraining_or_repeating_fresh_mixed_k4(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'identity',lambda c:'test')
    configs={m:dict(output=str(tmp_path/m),python='python',steps=5000 if m.startswith('continue') else 10000) for m in recovery.MODES}
    configs['fresh_mixed_control']=dict(output=str(tmp_path/'control'),python='python')
    jobs=recovery.jobs(configs);by_id={j['id']:j for j in jobs}
    assert len(jobs)==len(by_id)==32
    assert all(set(j.get('deps',())).issubset(by_id) for j in jobs)
    assert not any('fresh_mixed' in j['id'] and '_train' in j['id'] for j in jobs)
    assert 'fresh_mixed_k4' not in by_id
    assert sum(j['completion'].get('episodes')==500 for j in jobs)==8


def test_rb2_has_nine_full_rows_and_independent_dependencies():
    jobs=rb2.jobs()
    assert len(jobs)==len({j['id'] for j in jobs})==9
    assert all(j['completion']['episodes']==500 and not j.get('deps') for j in jobs)
    for j in jobs:
        if j['id'].startswith('fresh_mixed'):
            assert 'simvla_condition_noise/detached_noise1' in j['ready_file']
            assert 'upstream_status_file' not in j
        else:
            assert 'simvla_interval_recovery' in j['ready_file']
