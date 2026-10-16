import pytest
from tools.simvla import action_coverage_campaign as c


def test_coverage_and_warm_audit():
    assert [c.sampled_age('fresh_joint',n) for n in range(1,10)]==[5,7,1,2,3,4,5,6,7]
    assert c.action_ages('fresh_joint',3)==[3]
    assert c.action_ages('fresh_all_query',7)==list(range(1,8))
    assert c.action_ages('protected_short',7)==[4,5,6,7]
    with pytest.raises(ValueError):c.sampled_age('fresh_joint',3,True)


def test_plan_and_scope_restore(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path)
    original=c.base.OUTPUT
    plan=c.jobs(dict(identity='fixed',manifest=dict(action_noise_seed_base=9)))
    assert len(plan)==39
    assert c.base.OUTPUT==original
    assert len({j['summary'] for j in plan})==39
    assert not any('protected_short_k4' in j['id'] for j in plan)
    assert next(j for j in plan if j['id']=='fresh_all_query_k8_seed1_smoke')['deps']==['fresh_all_query_k4_seed1_merge']
    assert all(j['cmd'][3]==c.MODULE for j in plan)


@pytest.mark.parametrize('arm,expected',[
    ('fresh_joint',[7]),('fresh_all_query',list(range(1,8))),('protected_short',[4,5,6,7])])
def test_current_query_targets_and_gradients(monkeypatch,arm,expected):
    import torch
    from types import SimpleNamespace
    from tools.simvla import condition_output_split_train as t
    from tools.simvla import rollout_state_repair as r
    from methods.latentloop.modules import trend_condition as g
    from methods.latentloop.modules import action_aligned_joint as a
    predictions=[torch.tensor(float(i+1),requires_grad=True) for i in range(7)]
    calls=[]
    monkeypatch.setattr(t,'unroll',lambda *args:(predictions[-1],predictions,{}))
    monkeypatch.setattr(r,'sample_from_sequence',lambda seq,age:dict(age=age,target_action=torch.tensor(float(age+10)),anchor=None,valid=None))
    def decode(action,z,sample,steps):
        calls.append(sample['age']);assert z is predictions[sample['age']-1];assert steps==1
        return z
    monkeypatch.setattr(t,'action_prediction',decode)
    monkeypatch.setattr(g,'scaled_mse',lambda *args:torch.tensor(0.))
    monkeypatch.setattr(a,'action_loss',lambda x,y:(x-y).abs())
    action=SimpleNamespace(action_space=SimpleNamespace(normalize_action=lambda x:x))
    model=SimpleNamespace(supervision_variant=arm)
    features=SimpleNamespace(bind=lambda seq:None)
    loss,la,lc=c.loss_for(dict(condition_weight=.05),model,action,
        dict(teacher_conditions=torch.zeros(1,7)),7,features)
    loss.backward()
    assert calls==expected
    assert float(la)==10
    assert [i+1 for i,z in enumerate(predictions) if z.grad is not None]==expected
