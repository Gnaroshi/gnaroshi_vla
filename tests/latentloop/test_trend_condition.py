import copy

import pytest
import torch
from torch import nn

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.trend_condition import (
    ARMS, TrendCondition, teacher_decomposition, decomposition_loss,
)
from tools.simvla.trend_condition_eval import expected_counts, check_counts
from tools.simvla.trend_condition_campaign import jobs


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(4,128)

    def forward(self,pair):
        axes=tuple(range(1,pair.previous_images.ndim))
        x=torch.stack([pair.previous_images.float().mean(axes),pair.current_images.float().mean(axes),
            pair.previous_proprio.mean(1),pair.current_proprio.mean(1)],dim=1)
        return self.projection(x)


def fixture():
    torch.manual_seed(19)
    parent=NativeSimVLAV0(condition_dim=12,max_tokens=8)
    parent.delta_encoder=TinyEncoder()
    nn.init.normal_(parent.condition_updater.up.weight,std=.05)
    parent.condition_updater.gate_head.bias.data.fill_(1)
    s=dict(anchor_condition=torch.randn(2,5,12),teacher_conditions=torch.randn(2,3,5,12),
        image_sequence=torch.randn(2,4,2,8,8,3),proprio_sequence=torch.randn(2,4,8),
        valid_mask=torch.tensor([[True,True,True,False,False]]*2),group_ids=torch.zeros(2,5,dtype=torch.long))
    return parent,s


def test_teacher_decomposition_and_variance_identity():
    _,s=fixture()
    c=torch.cat([s['anchor_condition'][:,None],s['teacher_conditions']],1)
    d=c[:,1:]-c[:,:-1]
    b=d.mean(1)
    torch.testing.assert_close(((d-b[:,None])**2).sum(),(d**2).sum()-3*(b**2).sum())
    for age in (1,2,3):
        target_b,r=teacher_decomposition(c[:,0],c[:,1:],age)
        torch.testing.assert_close(c[:,0]+age*target_b+r,c[:,age])
    torch.testing.assert_close(r,torch.zeros_like(r),atol=1e-6,rtol=0)


@pytest.mark.parametrize('arm',ARMS)
def test_no_future_teacher_leak_and_immutable_anchor(arm):
    parent,s=fixture()
    model=TrendCondition(parent,arm)
    anchor=s['anchor_condition'].clone()
    first=model.sequence(s,1)[0]
    altered={k:v.clone() for k,v in s.items()}
    altered['teacher_conditions'].add_(200)
    altered['image_sequence'][:,2:].add_(80)
    altered['proprio_sequence'][:,2:].sub_(80)
    torch.testing.assert_close(first,model.sequence(altered,1)[0],rtol=0,atol=0)
    torch.testing.assert_close(s['anchor_condition'],anchor,rtol=0,atol=0)
    torch.testing.assert_close(first[~s['valid_mask']],anchor[~s['valid_mask']],rtol=0,atol=0)


@pytest.mark.parametrize('arm',ARMS)
def test_latest_observation_dependence(arm):
    parent,s=fixture()
    model=TrendCondition(parent,arm)
    before=model.sequence(s,2)[0]
    altered={k:v.clone() for k,v in s.items()}
    altered['image_sequence'][:,2].add_(10)
    after=model.sequence(altered,2)[0]
    assert torch.equal(before,after)==(arm in ('trend_only','trend_forecast'))


def test_forecast_is_one_batched_head_call_and_matches_individual_ages():
    parent,s=fixture()
    model=TrendCondition(parent,'trend_forecast')
    calls=[]
    handle=model.condition_updater.register_forward_hook(lambda *args:calls.append(1))
    ctx=model.prepare(s['anchor_condition'],s['image_sequence'][:,0],s['proprio_sequence'][:,0],
        s['valid_mask'],s['group_ids'])
    for age in (1,2,3): model.predict(ctx,age)
    assert len(calls)==1
    handle.remove()
    for age in (1,2,3):
        expected=model.condition_updater(ctx.anchor,ctx.anchor.new_zeros(2,128),
            valid_mask=ctx.valid,group_ids=ctx.groups,age=age).condition-ctx.anchor
        torch.testing.assert_close(ctx.forecast[:,age-1],expected,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('arm',ARMS)
def test_both_training_objectives_have_finite_gradients(arm):
    parent,s=fixture()
    model=TrendCondition(parent,arm)
    for objective in ('decomposition','action_surrogate'):
        for age in (1,2,3):
            model.zero_grad(set_to_none=True)
            c,b,r=model.sequence(s,age)
            loss=decomposition_loss(model,s,age,c,b,r) if objective=='decomposition' else c.square().mean()
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(g).all() for g in gradients)
            assert sum(float(g.square().sum()) for g in gradients)>0


@pytest.mark.parametrize('arm',ARMS)
def test_checkpoint_roundtrip_and_counter_contract(arm):
    parent,s=fixture()
    model=TrendCondition(copy.deepcopy(parent),arm)
    clone=TrendCondition(copy.deepcopy(parent),arm)
    clone.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(model.sequence(s,3)[0],clone.sequence(s,3)[0],rtol=0,atol=0)
    for q in (1,2,4,5,8,9):
        c=expected_counts(arm,q,4)
        assert c['full_vlm']==(q+3)//4
        assert c['lightweight_conditions']==q-c['full_vlm']
        assert c['condition']==(q-c['full_vlm'] if arm in ('direct_anchor','trend_residual')
            else c['full_vlm'] if arm=='trend_forecast' else 0)
        assert c['generation']==7*q and c['transformer']==3*q


def test_queue_dependencies_and_future_age_rejection():
    c=dict(python='/python',output='/result')
    plan=jobs(c,'/config',False)
    assert len(plan)==8
    for j in plan[4:]:
        assert len(j['deps'])==1 and j['deps'][0] in {k['id'] for k in plan[:4]}
    parent,s=fixture()
    model=TrendCondition(parent,'trend_only')
    ctx=model.prepare(s['anchor_condition'],s['image_sequence'][:,0],s['proprio_sequence'][:,0],
        s['valid_mask'],s['group_ids'])
    with pytest.raises(ValueError): model.predict(ctx,4)
