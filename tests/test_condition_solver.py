from types import SimpleNamespace

import pytest
import torch

from tools.simvla.condition_output_split_train import student_steps, action_prediction, check_continuation_contract
from tools.simvla.condition_output_split_eval import load_payload


@pytest.mark.parametrize('steps',[1,3])
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
    for value in (0,2,10,1.5,True,'1'):
        with pytest.raises(ValueError): student_steps(dict(student_steps=value))


def test_continuation_only_allows_explicit_solver_transition():
    keys=('data','heldout','batch_size','seed','teacher_steps','source_checkpoint_sha256',
          'condition_weight','current_action_gradient','future_condition_gradient')
    previous={key:'identical' for key in keys}; previous['action_mode']='naive3'
    previous['training_intervals']=[4,8]
    current={**previous,'action_mode':'naive1'}
    with pytest.raises(RuntimeError): check_continuation_contract(previous,current,{})
    allowed=dict(solver_transition='naive3_to_naive1')
    check_continuation_contract(previous,current,allowed)
    for key in keys:
        with pytest.raises(RuntimeError):
            check_continuation_contract(previous,{**current,key:'changed'},allowed)
    with pytest.raises(RuntimeError): check_continuation_contract(current,previous,allowed)


def test_checkpoint_loader_requires_matching_solver(tmp_path):
    path=tmp_path/'model.pt'
    torch.save(dict(format='simvla_condition_output_split_v1',arm='carry_base',step=7000,identity='same',
        contract=dict(action_mode='naive1',training_intervals=[4,8])),path)
    load_payload(path,'carry_base','same',steps=7000,action_mode='naive1')
    with pytest.raises(RuntimeError): load_payload(path,'carry_base','same',steps=7000)
    with pytest.raises(RuntimeError): load_payload(path,'carry_base','wrong',steps=7000,action_mode='naive1')


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


def test_rb2_queue_has_eight_distinct_new_rows_with_complete_episode_budget():
    from tools.simvla.condition_solver_rb2 import jobs,OUTPUT,PRIOR
    plan=jobs()
    assert len(plan)==len({j['id'] for j in plan})==8
    assert OUTPUT!=PRIOR
    for j in plan:
        assert 'tools.simvla.condition_solver_rb2' in j['cmd']
        assert j['completion']['episodes']==500
        assert 'incoming/simvla_condition_solver/' in j['ready_file']
