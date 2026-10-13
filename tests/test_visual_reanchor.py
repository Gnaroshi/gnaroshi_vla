import copy

import pytest
import torch

from methods.latentloop.modules.bounded_history import BoundedHistoryCondition
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.visual_reanchor import VisualReanchorCondition


def setup(schedule='midpoint',contrast=False):
    torch.manual_seed(7)
    parent=BoundedHistoryCondition(NativeSimVLAV0(condition_dim=32),'recurrent')
    torch.nn.init.normal_(parent.condition_updater.up.weight,std=.01)
    model=VisualReanchorCondition(parent,schedule,contrast)
    torch.nn.init.normal_(model.visual.output.weight,std=.01)
    images=torch.randn(1,2,3,16,16);proprio=torch.randn(1,8)
    args=(torch.randn(1,6,32),images,proprio,torch.ones(1,6,dtype=torch.bool),torch.zeros(1,6,dtype=torch.long),8)
    current,anchor=torch.randn(1,8,32),torch.randn(1,8,32)
    calls=[]
    def features(age,images):
        calls.append(age)
        return anchor if age==0 else current
    model.feature_provider=features
    return parent,model,args,calls


@pytest.mark.parametrize('schedule',['midpoint','late_every'])
@pytest.mark.parametrize('contrast',[False,True])
def test_preserved_short_prefix_and_feature_schedule(schedule,contrast):
    parent,model,args,calls=setup(schedule,contrast)
    left,right=parent.prepare(*args),model.prepare(*args)
    for age in range(1,8):
        images,q=torch.randn_like(args[1]),torch.randn_like(args[2])
        y,_=model.predict(right,age,images,q)
        if age<4:
            x,_=parent.predict(left,age,images,q)
            torch.testing.assert_close(x,y,rtol=0,atol=0)
    expected=[4] if schedule=='midpoint' else [4,5,6,7]
    assert calls==([v for age in expected for v in (age,0)] if contrast else expected)
    y.square().mean().backward()
    assert all(p.grad is None for p in model.early.parameters())
    assert model.visual.output.weight.grad.abs().sum()>0
    assert all(p.grad is not None for p in model.late.condition_updater.parameters())


@pytest.mark.parametrize('contrast',[False,True])
def test_vision_reestimate_independent_of_previous_prediction(contrast):
    _,model,args,_=setup(contrast=contrast)
    left=model.prepare(*args)
    with torch.no_grad():
        for age in (1,2,3):model.predict(left,age,args[1],args[2])
        right=copy.deepcopy(left)
        right.previous=torch.randn_like(right.previous)*10
        right.previous_images=torch.randn_like(right.previous_images)
        right.previous_proprio=torch.randn_like(right.previous_proprio)
        a,_=model.predict(left,4,args[1],args[2])
        b,_=model.predict(right,4,args[1],args[2])
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        # After the single vision query the recurrent update again uses its
        # preceding prediction; this path remains intentionally trainable.
        right.previous=torch.randn_like(right.previous)*10
        a,_=model.predict(left,5,args[1],args[2])
        b,_=model.predict(right,5,args[1],args[2])
        assert not torch.equal(a,b)


def test_four_arms_share_initial_parameters():
    states=[]
    for schedule in ('midpoint','late_every'):
        for contrast in (False,True):
            _,model,_,_=setup(schedule,contrast)
            states.append(model.state_dict())
    for state in states[1:]:
        for name,value in states[0].items():assert torch.equal(value,state[name])
