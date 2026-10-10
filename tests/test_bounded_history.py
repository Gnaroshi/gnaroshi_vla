from copy import deepcopy
import signal

import pytest
import torch

from methods.latentloop.modules.bounded_history import BoundedHistoryCondition
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from tools.simvla.condition_output_split_train import build_initial_model, training_intervals, sampling_step
from tools.simvla import priority_handoff


@pytest.fixture
def inputs():
    torch.manual_seed(11)
    return (torch.randn(1,4,12),torch.randn(1,4,12),torch.randn(1,8),
            dict(valid_mask=torch.tensor([[True,True,True,False]]),group_ids=torch.zeros(1,4,dtype=torch.long),age=2))


def model(variant):
    parent=NativeSimVLAV0(condition_dim=12,delta_dim=8,max_tokens=8,num_token_groups=2)
    m=BoundedHistoryCondition(parent,variant).double()
    with torch.no_grad():
        m.condition_updater.up.weight.normal_(std=.1)
        m.condition_updater.gate_head.weight.normal_(std=.1)
    return m


def test_exact_conditional_history_difference_and_jacobian(inputs):
    anchor,previous,code,kw=inputs
    anchor,previous,code=anchor.double(),previous.double(),code.double()
    m=model('bounded')
    other=previous+torch.randn_like(previous)
    a,da=m.update(previous,anchor,code,**kw)
    b,db=m.update(other,anchor,code,**kw)
    torch.testing.assert_close(da['gate'],db['gate'],rtol=0,atol=0)
    torch.testing.assert_close(da['candidate'],db['candidate'],rtol=0,atol=0)
    torch.testing.assert_close(a-b,(1-da['gate'])*(previous-other),rtol=1e-12,atol=1e-12)
    assert (a-b).norm() <= (previous-other).norm()+1e-12
    previous.requires_grad_(True)
    output,d=m.update(previous,anchor,code,**kw)
    gradient=torch.autograd.grad(output.sum(),previous)[0]
    torch.testing.assert_close(gradient,(1-d['gate']).expand_as(previous),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(output[:,3],previous[:,3],rtol=0,atol=0)


def test_recurrent_matches_existing_updater(inputs):
    anchor,previous,code,kw=inputs
    anchor,previous,code=anchor.double(),previous.double(),code.double()
    m=model('recurrent')
    output,_=m.update(previous,anchor,code,**kw)
    expected=m.condition_updater(previous,code,**kw).condition
    torch.testing.assert_close(output,expected,rtol=1e-12,atol=1e-12)


def test_candidates_can_depend_on_current_observation(inputs):
    anchor,previous,code,kw=inputs
    m=model('bounded')
    a,da=m.update(previous.double(),anchor.double(),code.double(),**kw)
    b,db=m.update(previous.double(),anchor.double(),code.double()+1,**kw)
    assert not torch.equal(a,b)
    assert not torch.equal(da['candidate'],db['candidate'])


def test_matched_initialization_and_parameter_count():
    parent=NativeSimVLAV0(condition_dim=12,delta_dim=8,max_tokens=8,num_token_groups=2)
    a=build_initial_model(parent,dict(initialization='fresh',seed=7,bounded_history_variant='bounded'),'carry_base')
    b=build_initial_model(parent,dict(initialization='fresh',seed=7,bounded_history_variant='recurrent'),'carry_base')
    assert set(a.state_dict())==set(b.state_dict())
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    assert not hasattr(a,'action_condition_updater')
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())


@pytest.mark.parametrize('variant',['bounded','recurrent'])
def test_recursive_context_gradients_and_reset(variant,inputs):
    anchor,_,_,kw=inputs
    m=model(variant).float()
    images=torch.rand(1,2,3,16,16);q=torch.randn(1,8)
    ctx=m.prepare(anchor,images,q,kw['valid_mask'],kw['group_ids'],8)
    first=None
    for age in range(1,8):
        out,d=m.predict(ctx,age,images+age*.01,q+age*.01)
        if age==1:first=out;first.retain_grad()
    out.square().mean().backward()
    assert first.grad is not None and first.grad.abs().sum()>0
    for part in (m.condition_updater,m.delta_encoder):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in part.parameters())
    assert ctx.previous is out
    reset=m.prepare(anchor,images,q,kw['valid_mask'],kw['group_ids'],8)
    assert reset.previous is anchor and reset.age==0
    with pytest.raises(ValueError):m.predict(reset,2,images,q)


def test_k8_sampling_and_compiled_tensor_age(inputs):
    anchor,previous,code,kw=inputs
    c=dict(training_intervals=[8],seed=7)
    assert training_intervals(c)==[8]
    assert [sampling_step(c,n)[2] for n in range(1,8)]==list(range(1,8))
    m=model('bounded').float()
    expected=m.update(previous,anchor,code,**kw)[0]
    compiled=torch.compile(m.update,backend='eager')
    actual=compiled(previous,anchor,code,**{**kw,'age':torch.tensor([2])})[0]
    torch.testing.assert_close(actual,expected)


def test_handoff_finishes_only_after_workers_exit(tmp_path,monkeypatch):
    record=tmp_path/'handoff.json';queue=tmp_path/'queue';queue.mkdir()
    priority_handoff.write_json(queue/'queue_status.json',dict(active={},pending=['old']))
    owner=dict(pid=10,started='1');worker=dict(pid=11,started='2')
    priority_handoff.write_json(record,dict(phase='dispatch_deferred_workers_preserved',owner=owner,workers=[worker],queue=str(queue)))
    states=iter([True,True,False,False])
    sent=[]
    def alive(saved):
        if saved==worker:return next(states,False)
        return not sent
    monkeypatch.setattr(priority_handoff,'alive',alive)
    monkeypatch.setattr(priority_handoff.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(priority_handoff.os,'kill',lambda pid,sig:sent.append((pid,sig)))
    priority_handoff.finish(record)
    assert sent==[(10,signal.SIGTERM),(10,signal.SIGCONT)]
    assert priority_handoff.read_json(record)['phase']=='HANDOFF_COMPLETE'
    priority_handoff.finish(record)
    assert len(sent)==2


def test_handoff_pid_reuse_and_zombies(monkeypatch):
    saved=dict(pid=10,started='1')
    monkeypatch.setattr(priority_handoff,'process',lambda pid:dict(pid=pid,started='2',state='R'))
    assert not priority_handoff.alive(saved)
    monkeypatch.setattr(priority_handoff,'process',lambda pid:dict(pid=pid,started='1',state='Z'))
    assert not priority_handoff.alive(saved)
