import copy

import pytest
import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.observation_correction import ARMS, ObservationCorrection, expected_condition_counts

torch.set_num_threads(1)


def fixture(arm):
    torch.manual_seed(3)
    parent=NativeSimVLAV0(condition_dim=32,max_tokens=8)
    model=ObservationCorrection(parent,arm).eval()
    z=torch.randn(1,6,32); image=torch.rand(1,2,3,32,32); proprio=torch.randn(1,8)
    valid=torch.tensor([[True,True,True,True,True,False]])
    ctx=model.prepare(z,image,proprio,valid,torch.zeros(1,6,dtype=torch.long),8)
    return model,ctx,image,proprio


@pytest.mark.parametrize('arm',ARMS)
def test_causal_state_and_padding(arm):
    model,ctx,image,proprio=fixture(arm)
    original=ctx.anchor.clone()
    for age in range(1,8):
        predicted,_=model.predict(ctx,age,image,proprio)
        assert ctx.age==age
        assert ctx.previous is predicted
        torch.testing.assert_close(predicted[:,5],original[:,5])
        torch.testing.assert_close(ctx.anchor,original)
    with pytest.raises(ValueError): model.predict(ctx,7,image,proprio)
    with pytest.raises(ValueError): model.predict(ctx,8,image,proprio)


def test_observation_estimate_independent_of_predicted_error():
    model,ctx,image,proprio=fixture('observed_recurrent')
    a,_=model.observation_estimate(ctx,1,image,proprio)
    ctx.previous=ctx.previous+100
    b,_=model.observation_estimate(ctx,1,image,proprio)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_midpoint_is_one_extra_trend_call():
    model,ctx,image,proprio=fixture('midpoint_recurrent')
    counts=[]
    hook=model.trend_head.register_forward_hook(lambda *args: counts.append(1))
    for age in range(1,8): model.predict(ctx,age,image,proprio)
    hook.remove()
    assert len(counts)==1
    assert expected_condition_counts('midpoint_recurrent',16,8)['trend']==4


def test_recurrent_correction_is_carried():
    model,ctx,image,proprio=fixture('recurrent')
    p,_=model.predict(ctx,1,image,proprio)
    ctx.previous=p+0.5*ctx.valid.unsqueeze(-1)
    q,_=model.predict(ctx,2,image,proprio)
    torch.testing.assert_close(q[:,:5],ctx.anchor[:,:5]+0.5)


def test_observed_gain_receives_gradient():
    model,ctx,image,proprio=fixture('observed_recurrent')
    ctx.previous=ctx.previous+0.3*ctx.valid.unsqueeze(-1)
    pred,d=model.predict(ctx,1,image,proprio)
    pred.square().mean().backward()
    assert model.correction_gain.readout.bias.grad.abs().max()>0
    assert 'measured' in d


def test_refresh_does_not_share_previous_context():
    model,ctx,image,proprio=fixture('observed_recurrent')
    model.predict(ctx,1,image,proprio)
    fresh=model.prepare(ctx.anchor,image,proprio,ctx.valid,ctx.groups,4)
    assert fresh.age==0
    assert fresh.previous is fresh.anchor
    assert fresh is not ctx
