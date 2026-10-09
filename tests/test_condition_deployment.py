from copy import deepcopy
import random
from types import SimpleNamespace

import pytest
import torch

from tools.simvla.condition_output_split_train import sampling_step, training_intervals, check_continuation_contract
from tools.simvla.condition_output_split_eval import load_payload
from tools.simvla import condition_deployment_pipeline as pipeline
from tools.simvla import condition_deployment_rb2 as rb2


def test_default_sampling_is_unchanged_and_window_draws_are_matched():
    for n in range(1,501):
        c=dict(seed=7,sample_step_offset=10000)
        rng,k,age=sampling_step(c,n)
        total=10000+n
        assert k==(4 if total%2 else 8)
        assert age==((total-1)//2)%(k-1)+1
        old=random.Random(700000+total)
        expected=[old.randrange(50000) for _ in range(2)]
        assert [rng.randrange(50000) for _ in range(2)]==expected
        for intervals in ([2],[3],[4]):
            rng,k,age=sampling_step({**c,'training_intervals':intervals},n)
            assert k==intervals[0] and 1<=age<k
            assert [rng.randrange(50000) for _ in range(2)]==expected


@pytest.mark.parametrize('k',[2,3,4])
def test_specialist_age_cycle_balanced(k):
    ages=[sampling_step(dict(seed=7,training_intervals=[k]),n)[2] for n in range(1,1+12*(k-1))]
    assert all(ages.count(j)==12 for j in range(1,k))


@pytest.mark.parametrize('intervals',[[8],[2,4],[],[True],[2.0],['2']])
def test_interval_config_rejects_invalid_values(intervals):
    with pytest.raises(ValueError): training_intervals(dict(training_intervals=intervals))


def test_interval_transition_requires_explicit_source_and_target():
    keys=('data','heldout','batch_size','seed','teacher_steps','source_checkpoint_sha256',
          'condition_weight','current_action_gradient','future_condition_gradient')
    prior={key:'same' for key in keys}
    prior.update(action_mode='naive1',training_intervals=[4,8])
    current={**prior,'training_intervals':[2]}
    with pytest.raises(RuntimeError): check_continuation_contract(prior,current,{})
    check_continuation_contract(prior,current,dict(interval_transition=dict(source=[4,8],target=[2])))
    with pytest.raises(RuntimeError):
        check_continuation_contract(prior,current,dict(interval_transition=dict(source=[4],target=[2])))


@pytest.mark.parametrize('intervals',[[4,8],[2],[3],[4]])
def test_loader_reads_specialist_but_still_rejects_wrong_solver(tmp_path,intervals):
    path=tmp_path/'model.pt'
    torch.save(dict(format='simvla_condition_output_split_v1',arm='carry_base',step=5000,
        identity='test',contract=dict(action_mode='naive1',training_intervals=intervals)),path)
    load_payload(path,'carry_base','test',steps=5000,action_mode='naive1')
    with pytest.raises(RuntimeError): load_payload(path,'carry_base','test',steps=5000)


def test_sd1_plan_four_independent_trainings_six_evaluations(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'identity',lambda c:'test')
    configs={m:dict(output=str(tmp_path/m),python='python') for m in pipeline.MODES}
    plan=pipeline.jobs(configs)
    assert len(plan)==len({j['id'] for j in plan})==22
    by_id={j['id']:j for j in plan}
    assert all(set(j['deps']).issubset(by_id) for j in plan)
    assert len([j for j in plan if j['id'].endswith('_train') and '_smoke_' not in j['id']])==4
    for row,(mode,k) in pipeline.ROWS.items():
        assert by_id[row]['deps']==[mode+'_train']
        assert by_id[row]['completion']['episodes']==500
        assert by_id[row]['cmd'][-1]==str(k)


def test_rb2_priority_and_independent_failure_isolation():
    plan=rb2.jobs()
    assert len(plan)==len({j['id'] for j in plan})==9
    assert [j['id'] for j in plan[:3]]==['reference_'+r for r in rb2.PRIORITY]
    assert all(not j.get('deps') for j in plan)
    assert all(j['completion']['episodes']==500 for j in plan)
    assert all('simvla_condition_deployment/' in j['ready_file'] for j in plan[3:])


def test_reference_uses_original_tree_and_inherits_gpu_lease(monkeypatch):
    monkeypatch.setattr(rb2,'verify_comparison_source',lambda:'6982a5a')
    monkeypatch.setenv('GNAROSHI_GPU_LEASE_FD','17')
    captured=[]
    monkeypatch.setattr(rb2.subprocess,'run',lambda *args,**kwargs:captured.append((args,kwargs)))
    rb2.run_comparison('ours_k2')
    args,kwargs=captured[0]
    assert kwargs['cwd']==rb2.COMPARISON_ROOT
    assert kwargs['env']['PYTHONPATH']==str(rb2.COMPARISON_ROOT)
    assert kwargs['pass_fds']==(17,)
    assert args[0][-3:]==['cell','--row','ours_k2']


def test_bundle_contract_parent_and_gradient_checks():
    previous={key:'same' for key in ('data','heldout','batch_size','seed','condition_weight',
        'current_action_gradient','future_condition_gradient','source_checkpoint_sha256')}
    contract={**previous,'training_intervals':[2],'total_training_steps':15000,'sample_step_offset':10000,
        'action_mode':'naive1','teacher_steps':10,'initialization':'continuation',
        'continuation':dict(checkpoint_sha256=pipeline.PARENT_SHA,contract=previous),
        'interval_transition':dict(source=[4,8],target=[2])}
    rb2.validate_training_contract(contract,'k2')
    for key in ('data','current_action_gradient','source_checkpoint_sha256'):
        c=deepcopy(contract);c[key]='changed'
        with pytest.raises(RuntimeError): rb2.validate_training_contract(c,'k2')
    with pytest.raises(RuntimeError): rb2.validate_training_contract(contract,'k3')


def test_continuation_loads_10k_parent_and_counts_both_prior_phases(tmp_path,monkeypatch):
    from tools.simvla import condition_output_split_train as trainer
    from tools.simvla import condition_output_split_eval as loader
    summary=dict(identity='test',steps=7000,total_training_steps=10000,verdict='TRAIN_AND_OFFLINE_COMPLETE',
        checkpoint='parent.pt',checkpoint_sha256='hash',training_seconds=1500,prior_training_seconds=1100)
    monkeypatch.setattr(trainer,'read_json',lambda p:summary)
    monkeypatch.setattr(trainer,'sha',lambda p:'hash')
    calls=[]
    def load(*args,**kwargs):
        assert kwargs==dict(steps=7000,action_mode='naive1')
        return dict(model={'weights':'test'},contract={})
    monkeypatch.setattr(loader,'load_payload',load)
    model=SimpleNamespace(load_state_dict=lambda weights,strict:calls.append((weights,strict)))
    c=dict(initialization='continuation',initial_models={'carry_base':dict(summary='s',identity='test')},
        continuation_source_steps=7000,continuation_source_action_mode='naive1')
    result=trainer.load_continuation(model,c,'carry_base')
    assert result['prior_steps']==10000 and result['prior_training_seconds']==2600
    assert calls==[({'weights':'test'},True)]
