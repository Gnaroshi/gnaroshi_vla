import copy

import pytest
import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.condition_output_split import ARMS, GRADIENT_CONTRACTS, ConditionOutputSplit
from tools.simvla.condition_output_split_train import check_continuation_contract, check_gradient_control


def model_pair(arm):
    torch.set_num_threads(1); torch.manual_seed(42)
    parent=NativeSimVLAV0(condition_dim=16,max_tokens=8)
    torch.nn.init.normal_(parent.condition_updater.up.weight,std=.1)
    detached=ConditionOutputSplit(parent,arm)
    torch.nn.init.normal_(detached.action_condition_updater.up.weight,std=.1)
    joint=copy.deepcopy(detached); joint.gradient_mode='joint'
    return detached,joint


@pytest.mark.parametrize('arm',ARMS)
def test_joint_forward_identical_across_full_interval_and_action_gradients(arm):
    detached,joint=model_pair(arm)
    previous=torch.randn(1,4,16)
    kwargs=dict(valid_mask=torch.ones(1,4,dtype=torch.bool),group_ids=torch.zeros(1,4,dtype=torch.long))
    histories=[]; codes=[]
    for model in (detached,joint):
        carried=previous.clone().requires_grad_(True)
        outputs=[]; inputs=[]
        torch.manual_seed(8)
        for age in range(1,8):
            code=torch.randn(1,128,requires_grad=True); inputs.append(code)
            output,base,carried=model.update(carried,code,age=age,**kwargs)
            output.retain_grad(); base.retain_grad()
            outputs.append((output,base,carried))
        histories.append(outputs); codes.append(inputs)
        output.square().mean().backward()
    for d,j in zip(*histories):
        for x,y in zip(d,j): torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert all(code.grad is None for code in codes[0])
    assert all(code.grad is not None and code.grad.abs().sum()>0 for code in codes[1])
    assert all(p.grad is None for p in detached.condition_updater.parameters())
    assert joint.condition_updater.up.weight.grad.abs().sum()>0
    assert joint.action_condition_updater.up.weight.grad.abs().sum()>0
    previous_output=histories[1][0][0]
    assert (previous_output.grad is not None and previous_output.grad.abs().sum()>0) if arm=='carry_output' else previous_output.grad is None
    assert histories[1][0][1].grad.abs().sum()>0


@pytest.mark.parametrize('arm',ARMS)
def test_action_loss_reaches_observation_encoder(arm):
    _,model=model_pair(arm)
    z=torch.randn(1,4,16); images=torch.rand(1,2,3,64,64); q=torch.zeros(1,8)
    ctx=model.prepare(z,images,q,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,4,dtype=torch.long),4)
    output,_=model.predict(ctx,1,images+.1,q+.1)
    output.square().mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in model.delta_encoder.parameters() if p.grad is not None)>0


@pytest.mark.parametrize('nfe',[1,2])
def test_both_transitions_require_explicit_authorization(nfe):
    keys=('data','heldout','batch_size','seed','teacher_steps','source_checkpoint_sha256','condition_weight')
    old={k:'same' for k in keys}
    old.update(action_mode='naive3',current_action_gradient=GRADIENT_CONTRACTS['detached'][0],
        future_condition_gradient=GRADIENT_CONTRACTS['detached'][1])
    new={**old,'action_mode':f'naive{nfe}','action_gradient_mode':'joint',
        'current_action_gradient':GRADIENT_CONTRACTS['joint'][0],'future_condition_gradient':GRADIENT_CONTRACTS['joint'][1]}
    c=dict(solver_transition=f'naive3_to_naive{nfe}',gradient_transition='detached_to_joint')
    check_continuation_contract(old,new,c)
    for key in c:
        with pytest.raises(RuntimeError): check_continuation_contract(old,new,{k:v for k,v in c.items() if k!=key})
    for key in keys:
        with pytest.raises(RuntimeError): check_continuation_contract(old,{**new,key:'different'},c)
    with pytest.raises(RuntimeError):
        check_continuation_contract(old,{**new,'current_action_gradient':'wrong'},c)


def test_gradient_comparison_checks_budget_and_targets():
    keys=('arm','training_intervals','action_mode','teacher_steps','source_checkpoint_sha256',
        'data','heldout','initial_weights_sha256','batch_size','seed','condition_loss','action_loss',
        'condition_weight','initialization','sample_step_offset','continuation','optimizer_initialization',
        'solver_transition','steps','total_training_steps','optimizer')
    control={k:'same' for k in keys}
    current={**control,'action_gradient_mode':'joint'}
    check_gradient_control(current,control)
    for key in keys:
        with pytest.raises(RuntimeError): check_gradient_control({**current,key:'changed'},control)


def test_four_trainings_eight_online_rows_no_duplicate_controls(monkeypatch,tmp_path):
    from tools.simvla import condition_initialization_pipeline as pipeline
    from tools.simvla.condition_gradient_rb2 import ROWS,jobs
    monkeypatch.setattr(pipeline,'identity',lambda c:'test')
    configs={f'nfe{n}':dict(output=str(tmp_path/f'nfe{n}'),python='python',steps=7000) for n in (1,2)}
    training=pipeline.jobs(configs,export_module='tools.simvla.condition_gradient_pipeline')
    assert len(training)==16
    assert sum(j['id'].startswith(('nfe1_train_','nfe2_train_')) for j in training)==4
    by_id={j['id']:j for j in training}
    assert all(set(j['deps']).issubset(by_id) for j in training)
    assert set(ROWS.values())=={(n,a,k) for n in (1,2) for a in ARMS for k in (4,8)}
    plan=jobs()
    assert len(plan)==len({j['id'] for j in plan})==8
    for j in plan:
        assert j['completion']['episodes']==500
        assert 'incoming/simvla_condition_gradient/nfe' in j['ready_file']


@pytest.mark.parametrize('row,nfe',[('nfe1_carry_base_k8',1),('nfe2_carry_output_k4',2)])
def test_online_model_matches_solver_and_gradient(monkeypatch,row,nfe):
    from types import SimpleNamespace
    from tools.simvla import condition_gradient_rb2 as remote
    monkeypatch.setattr(remote,'sha',lambda p:'hash')
    calls=[]
    def load(*args,**kw):
        calls.append(kw['action_mode'])
        return dict(contract=dict(action_gradient_mode='joint'))
    monkeypatch.setattr(remote,'load_payload',load)
    monkeypatch.setattr(remote,'attach_policy',lambda *args:SimpleNamespace(nfe=3))
    monkeypatch.setattr(remote,'attach',lambda p,*args:p)
    c=dict(student_steps=nfe,model_checkpoint=dict(path='p',sha256='hash',source_identity='i',student_steps=nfe))
    policy=remote.policy_factory(SimpleNamespace(native=None,compiler=None),c,row,{})
    assert policy.nfe==nfe and calls==[f'naive{nfe}']
