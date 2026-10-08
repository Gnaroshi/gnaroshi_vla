from types import SimpleNamespace

import pytest
import torch

from tools.simvla.condition_output_split_train import student_steps, action_prediction, check_continuation_contract
from tools.simvla.condition_output_split_eval import load_payload


@pytest.mark.parametrize('steps',[1,2,3])
def test_action_prediction_passes_solver_noise_and_gradient(steps):
    class Action:
        def decode_action_from_condition(self,z,q,**kwargs):
            assert kwargs['steps']==steps
            assert kwargs['initial_noise'] is noise
            assert kwargs['requires_grad'] is True
            return SimpleNamespace(final_action_latent=z*steps)
    z=torch.ones(1,10,7,requires_grad=True); noise=torch.randn_like(z)
    output=action_prediction(Action(),z,dict(proprio=torch.zeros(1,8),noise=noise),steps=steps)
    output.sum().backward()
    assert torch.all(z.grad==steps)
    assert student_steps(dict(student_steps=steps))==steps


def test_student_steps_rejects_unsupported_or_inexact_values():
    assert student_steps({})==3
    for value in (0,4,10,1.5,True,'1'):
        with pytest.raises(ValueError): student_steps(dict(student_steps=value))


@pytest.mark.parametrize('nfe',[1,2])
def test_continuation_only_allows_explicit_solver_transition(nfe):
    keys=('data','heldout','batch_size','seed','teacher_steps','source_checkpoint_sha256',
          'condition_weight','current_action_gradient','future_condition_gradient')
    previous={key:'identical' for key in keys}; previous['action_mode']='naive3'
    current={**previous,'action_mode':f'naive{nfe}'}
    with pytest.raises(RuntimeError): check_continuation_contract(previous,current,{})
    allowed=dict(solver_transition=f'naive3_to_naive{nfe}')
    check_continuation_contract(previous,current,allowed)
    for key in keys:
        with pytest.raises(RuntimeError):
            check_continuation_contract(previous,{**current,key:'changed'},allowed)
    with pytest.raises(RuntimeError): check_continuation_contract(current,previous,allowed)


@pytest.mark.parametrize('nfe',[1,2])
def test_checkpoint_loader_requires_matching_solver(tmp_path,nfe):
    path=tmp_path/'model.pt'
    torch.save(dict(format='simvla_condition_output_split_v1',arm='carry_base',step=7000,identity='same',
        contract=dict(action_mode=f'naive{nfe}',training_intervals=[4,8])),path)
    load_payload(path,'carry_base','same',steps=7000,action_mode=f'naive{nfe}')
    with pytest.raises(RuntimeError): load_payload(path,'carry_base','same',steps=7000)
    with pytest.raises(RuntimeError): load_payload(path,'carry_base','wrong',steps=7000,action_mode=f'naive{nfe}')


def test_jobs_reuse_completed_three_k_and_transfer_each_model(tmp_path,monkeypatch):
    from tools.simvla import condition_initialization_pipeline as shared
    from tools.simvla.condition_solver_pipeline import PHASES
    monkeypatch.setattr(shared,'identity',lambda c:'test')
    configs={p:dict(output=str(tmp_path/p),python='python',steps=7000) for p in PHASES}
    jobs=shared.jobs(configs,export_module='tools.simvla.condition_solver_pipeline')
    assert len(jobs)==16
    by_id={j['id']:j for j in jobs}
    assert all(set(j['deps']).issubset(by_id) for j in jobs)
    for j in jobs:
        if 'smoke_train' in j['id']: assert j['deps']==[]
        if '_export_' in j['id']: assert 'tools.simvla.condition_solver_pipeline' in j['cmd']


@pytest.mark.parametrize('nfe',[1,2])
def test_rb2_queue_has_eight_distinct_new_rows_with_complete_episode_budget(nfe):
    from tools.simvla.condition_solver_rb2 import jobs,OUTPUT,PRIOR
    plan=jobs(nfe)
    assert len(plan)==len({j['id'] for j in plan})==8
    assert OUTPUT!=PRIOR
    for j in plan:
        assert 'tools.simvla.condition_solver_rb2' in j['cmd']
        assert j['completion']['episodes']==500
        assert f'incoming/simvla_condition_solver{"_nfe2" if nfe==2 else ""}/' in j['ready_file']
        assert j['cmd'][-2:]==['--student-steps',str(nfe)]
        assert f'nfe{nfe}_compiled' in j['summary']


@pytest.mark.parametrize('module_name',['condition_solver_pipeline','condition_solver_rb2'])
def test_followup_output_and_locks_are_disjoint_and_ordered(module_name):
    import importlib
    module=importlib.import_module('tools.simvla.'+module_name)
    one, incoming1=module.solver_paths(1)
    two, incoming2=module.solver_paths(2)
    assert one!=two and incoming1!=incoming2
    assert module.predecessor(2)==dict(path=str(one),lock='queue.lock')
    assert module.predecessor(1)==dict(path=str(module.PRIOR),lock='queue.lock')
    with pytest.raises(ValueError): module.solver_paths(3)


def test_nfe2_policy_factory_uses_the_training_solver(monkeypatch):
    from tools.simvla import condition_solver_rb2 as remote
    monkeypatch.setattr(remote,'sha',lambda p:'hash')
    loaded=[]
    def load(*args,**kwargs):
        loaded.append(kwargs['action_mode'])
        return {'contract':{'action_mode':kwargs['action_mode']}}
    monkeypatch.setattr(remote,'load_payload',load)
    monkeypatch.setattr(remote,'attach_policy',lambda *args:SimpleNamespace(nfe=3))
    monkeypatch.setattr(remote,'attach',lambda policy,*args:policy)
    config=dict(student_steps=2,model_checkpoint=dict(path='p',sha256='hash',source_identity='i',student_steps=2))
    replay=SimpleNamespace(native=None,compiler=None)
    policy=remote.policy_factory(replay,config,'pretrained_carry_base_k8',{})
    assert policy.nfe==2 and loaded==['naive2']
    with pytest.raises(RuntimeError):
        remote.policy_factory(replay,{**config,'student_steps':1},'pretrained_carry_base_k8',{})
