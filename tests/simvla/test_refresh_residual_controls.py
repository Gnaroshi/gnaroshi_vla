import fcntl
import json

import pytest
import torch

from methods.refresh_calibration.model import RefreshCalibratedCondition
from tools.simvla.gpu_followup_queue import predecessor_pending
from architectures.simvla.adapters.refresh_calibration.policy import attach, check_counts
from test_refresh_calibration_policy import FakePolicy


@pytest.mark.parametrize('variant',('fixed','ridge','anchor_input'))
def test_repeated_observation_exactly_preserves_anchor(variant):
    torch.manual_seed(7)
    model=RefreshCalibratedCondition(variant,dim=32,width=8,anchor_exact=True)
    anchor=torch.randn(1,12,32)
    image=torch.rand(1,2,32,32,3); proprio=torch.rand(1,8)
    mask=torch.ones(1,12,dtype=torch.bool); mask[:,-1]=False
    state=model.prepare(anchor,image,proprio,torch.rand(1,32),mask,torch.zeros(1,12,dtype=torch.long))
    prediction=model.predict(state,image,proprio)
    assert torch.equal(prediction,anchor)
    altered=model.predict(state,torch.rand_like(image),proprio+.1)
    assert not torch.equal(altered[:,:-1],anchor[:,:-1])
    assert torch.equal(altered[:,-1],anchor[:,-1])
    altered.square().mean().backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients)>0


def test_wrong_scene_write_reanchors_its_reference():
    model=RefreshCalibratedCondition('ridge',dim=32,width=8,anchor_exact=True)
    anchor=torch.randn(1,12,32); image=torch.rand(1,2,32,32,3); proprio=torch.rand(1,8)
    state=model.prepare(anchor,image,proprio,torch.rand(1,32),torch.ones(1,12,dtype=torch.bool),torch.zeros(1,12,dtype=torch.long))
    state.correction=torch.randn_like(state.correction)
    model.update_reference(state)
    assert torch.equal(model.predict(state,image,proprio),anchor)


@pytest.mark.parametrize('k',(4,8))
@pytest.mark.parametrize('variant',('fixed','ridge'))
def test_policy_counts_reference_encoder(k,variant):
    model=RefreshCalibratedCondition(variant,dim=32,width=8,anchor_exact=True)
    payload=dict(contract=dict(model=dict(dim=32,width=8,anchor_exact=True)),model=model.state_dict(),
                 language_bank={'pick cup':torch.rand(1,32)})
    policy=attach(FakePolicy(),payload,variant,k)
    for q in range(9):
        b=policy.preprocess(torch.rand(32,32,3),torch.rand(32,32,3),torch.rand(8),'pick cup')
        policy._refill_action_queue(b); policy.step_index+=5
    result=check_counts(policy,variant,{'transformer':27},k)
    assert result['num_observation_encoder_calls']==9


def test_idle_gpus_can_follow_fully_assigned_predecessor(tmp_path):
    path=tmp_path/'queue.lock'
    spec=dict(path=str(tmp_path),lock='queue.lock',allow_when_all_assigned=True)
    with path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert predecessor_pending(spec)
        status=dict(pending=[],completed=['a'],failed={},active={'7':{'job':'b'}},total_jobs=2)
        (tmp_path/'queue_status.json').write_text(json.dumps(status))
        assert not predecessor_pending(spec)
        assert predecessor_pending(dict(path=str(tmp_path),lock='queue.lock'))
        status.update(pending=['c'],total_jobs=3)
        (tmp_path/'queue_status.json').write_text(json.dumps(status))
        assert predecessor_pending(spec)


@pytest.mark.parametrize('k',(4,8))
def test_hold_control_keeps_fresh_action_solver(monkeypatch,k):
    from tools.simvla import error_compensation_eval,refresh_residual_controls as followup
    parent=FakePolicy(); parent.reset()
    monkeypatch.setattr(error_compensation_eval,'make_policy',lambda *a,**kw:parent)
    policy=followup.hold_policy({},'hold',k_c=k)
    original_full=policy._full_refresh
    def full(batch,*,policy_query_index):
        cond,act,seed=original_full(batch,policy_query_index=policy_query_index)
        policy.cached_condition=cond
        return cond,act,seed
    policy._full_refresh=full
    for q in range(9):
        batch=policy.preprocess(torch.rand(32,32,3),torch.rand(32,32,3),torch.rand(8),'pick cup')
        policy._refill_action_queue(batch)
        assert len(policy.action_queue)==5
        policy.step_index+=5
    result=followup.check_hold(policy,'hold',{'transformer':27},k)
    assert result['condition']==0 and result['transformer']==27


def test_followup_jobs_have_no_bootstrap_rerun(monkeypatch):
    from tools.simvla import refresh_residual_controls as f
    monkeypatch.setattr(f,'identity',lambda c:'test')
    monkeypatch.setattr(f.pipeline,'identity',lambda c:'test')
    monkeypatch.setattr(f.pipeline,'MODULE',f.MODULE)
    monkeypatch.setattr(f.pipeline,'CELLS',tuple((v,k) for k in (4,8) for v in ('fixed','ridge')))
    jobs=f.sd_jobs({'python':'python','steps':3000})
    ids={j['id'] for j in jobs}
    assert len(ids)==len(jobs)==13
    assert not any('bootstrap' in j['id'] for j in jobs)
    for job in jobs:
        assert set(job.get('deps',())).issubset(ids)
