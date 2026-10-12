import copy

import pytest
import torch

from methods.latentloop.modules.bounded_history import BoundedHistoryCondition
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.visual_midpoint import VisualMidpointCondition, VisualReference


@pytest.mark.parametrize('arm,expected',[('midpoint',[4]),('late_every',[4,5,6,7])])
def test_initial_parity_causal_feature_schedule_and_gradient(arm,expected):
    torch.manual_seed(11)
    native=NativeSimVLAV0(condition_dim=32,delta_dim=128)
    parent=BoundedHistoryCondition(native,'recurrent')
    torch.nn.init.normal_(parent.condition_updater.up.weight,std=.01)
    model=VisualMidpointCondition(parent,arm)
    calls=[]
    def provide(age,images):
        calls.append(age)
        return torch.randn(1,8,32)
    model.feature_provider=provide
    images=torch.randn(1,2,3,16,16)
    args=(torch.randn(1,6,32),images,torch.randn(1,8),torch.ones(1,6,dtype=torch.bool),
          torch.zeros(1,6,dtype=torch.long),8)
    a,b=parent.prepare(*args),model.prepare(*args)
    for age in range(1,8):
        image,q=torch.randn_like(images),torch.randn(1,8)
        x,_=parent.predict(a,age,image,q);y,_=model.predict(b,age,image,q)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert calls==expected
    y.square().mean().backward()
    assert all(p.grad is None for p in model.early.parameters())
    assert model.visual.output.weight.grad.abs().sum()>0


def test_visual_features_affect_output_and_all_parameters_learn():
    model=VisualReference(32,16)
    torch.nn.init.eye_(model.output.weight)
    query,vision=torch.randn(1,6,16),torch.randn(1,8,32)
    valid=torch.tensor([[True,True,True,False,False,False]])
    first=model(query,vision,valid)
    second=model(query,vision.flip(1),valid)
    assert not torch.equal(first,second)
    assert torch.count_nonzero(first[:,3:])==0
    first.square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


def test_arms_have_identical_initial_parameters():
    parent=BoundedHistoryCondition(NativeSimVLAV0(),'recurrent')
    torch.manual_seed(7);a=VisualMidpointCondition(parent,'midpoint')
    torch.manual_seed(7);b=VisualMidpointCondition(parent,'late_every')
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])


def test_cached_features_select_current_query_only(tmp_path):
    from tools.simvla.visual_midpoint_features import CachedFeatures,feature_path
    p=feature_path(tmp_path,'query4');p.parent.mkdir(parents=True)
    torch.save(dict(identity='abc',query_id='query4',features=torch.ones(1,2,3)),p)
    provider=CachedFeatures(tmp_path,'abc');provider.bind(dict(query_ids=[[f'query{i}' for i in range(8)]]))
    assert torch.equal(provider(4,torch.zeros(1)),torch.ones(1,2,3))
    with pytest.raises(FileNotFoundError):provider(5,torch.zeros(1))


def test_no_vision_calls_during_preserved_k4_prefix():
    parent=BoundedHistoryCondition(NativeSimVLAV0(condition_dim=32),'recurrent')
    model=VisualMidpointCondition(parent,'late_every')
    images=torch.randn(1,2,3,16,16);q=torch.randn(1,8)
    ctx=model.prepare(torch.randn(1,6,32),images,q,torch.ones(1,6,dtype=torch.bool),torch.zeros(1,6,dtype=torch.long),4)
    for age in (1,2,3):model.predict(ctx,age,images,q)
    assert model.feature_provider is None


def test_stale_intervention_changes_only_vision_input_and_clears_at_reset():
    from types import SimpleNamespace
    from tools.simvla.visual_midpoint_followup import attach_stale
    calls=[]
    policy=SimpleNamespace(_full_refresh=lambda b,**kw:None,
        _v0_update=lambda b,**kw:calls.append(b),reset=lambda:None)
    policy=attach_stale(lambda *a:a[0],policy,None,8,None)
    first=dict(image_input='old',raw_rgb='old_rgb',proprio='old_q')
    current=dict(image_input='current',raw_rgb='current_rgb',proprio='current_q')
    policy._full_refresh(first,policy_query_index=0)
    policy._v0_update(current,age=4,policy_query_index=4)
    assert calls==[dict(image_input='old',raw_rgb='current_rgb',proprio='current_q')]
    assert current['image_input']=='current'
    policy.reset()
    with pytest.raises(RuntimeError):policy._v0_update(current,age=4,policy_query_index=4)


@pytest.mark.parametrize('arm',['midpoint','late_every'])
def test_stale_job_ready_after_smoke_without_waiting_for_full_eval(monkeypatch,arm):
    from tools.simvla import visual_midpoint_followup as followup
    from tools.simvla.gpu_followup_queue import ready
    monkeypatch.setattr(followup,'contract',lambda:dict(identity='control'))
    job=followup.stale_job(dict(identity='parent'),arm)
    assert job['ready_file'].endswith(f'/eval_smoke/{arm}/summary.json')
    assert job['ready_fields']==dict(verdict='EVAL_SMOKE_PASS',identity='parent',arm=arm)
    assert job['deps']==[]
    assert job['cmd'][-2:]==['--arm',arm]
    assert job['completion']['identity']=='control'
    assert followup.attribution_job(arm)['deps']==['stale_'+arm]


def test_attribution_uses_selected_vision_without_changing_image_input():
    from tools.simvla.visual_midpoint_attribution import InterventionFeatures
    calls=[]
    def current(age,images):
        calls.append(age)
        return images+age
    images=torch.ones(2)
    refresh=torch.zeros(2)
    assert torch.equal(InterventionFeatures(current,refresh,'current')(4,images),torch.full((2,),5.))
    assert torch.equal(InterventionFeatures(current,refresh,'refresh')(7,images),refresh)
    assert calls==[4]
    assert torch.equal(images,torch.ones(2))
