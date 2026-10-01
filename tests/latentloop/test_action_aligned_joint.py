from types import SimpleNamespace

import pytest
import torch
from torch import nn

from methods.latentloop.modules.action_aligned_joint import (
    ARMS, trainable_groups, query_schedule, action_loss, differentiable_rollout,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationHiddenUpdater, SimVLAGenerationLoop
from tools.simvla.action_aligned_campaign import jobs
from architectures.simvla.adapters.latentloop.efficient_multirate.recursive_condition_inputs import query_inputs


def test_matched_fresh_and_recursive_schedule():
    schedule = [query_schedule(step) for step in range(1,5001)]
    assert sum(fresh for age,fresh in schedule)==1250
    for age in (1,2,3):
        assert sum(a==age and not fresh for a,fresh in schedule)==1250
    assert schedule[:4]==[(1,False),(2,False),(3,False),(1,True)]
    with pytest.raises(ValueError): query_schedule(0)


@pytest.mark.parametrize('arm,expected', list(zip(ARMS, [(True,False),(True,False),(False,True),(True,True)])))
def test_factorized_trainability(arm, expected):
    assert trainable_groups(arm)==expected


def test_queue_does_not_repeat_baseline_or_skip_any_candidate():
    c=dict(python='python', output='/tmp/out', evaluation_condition_intervals=[4,2])
    plan=jobs(c,'config.json',False)
    assert len(plan)==12
    assert [j['id'] for j in plan[:4]]==['train_'+a for a in ARMS]
    for j in plan[4:]:
        assert len(j['deps'])==1
        assert '--k-c' in j['cmd']
        assert 'baseline' not in j['id']


def test_loss_uses_only_executed_prefix_and_continuous_gripper():
    pred=torch.zeros(1,10,7,requires_grad=True)
    target=torch.ones_like(pred)
    action_loss(pred,target).backward()
    assert torch.count_nonzero(pred.grad[:,:5])==35
    assert torch.count_nonzero(pred.grad[:,5:])==0
    assert pred.grad[0,0,6]!=0


class Step(nn.Module):
    def __init__(self):
        super().__init__()
        self.action=nn.Linear(7,8)
        self.decoder=nn.Linear(8,7)
    def forward(self,c,x,p,t):
        h=self.action(x)+c.mean(1)[:,None,:]+t[:,None,None]
        return h,self.decoder(h)


def test_frozen_transformer_passes_input_derivatives_and_checkpoint_is_equivalent():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    step=Step().requires_grad_(False)
    updater=SimVLAGenerationHiddenUpdater(hidden_dim=8,condition_dim=8,rank_dim=4).requires_grad_(False)
    loop=SimVLAGenerationLoop(updater,step.decoder)
    c=torch.randn(1,4,8,requires_grad=True)
    noise=torch.randn(1,10,7)
    p=torch.randn(1,8)
    target=torch.zeros_like(noise)
    y=differentiable_rollout(loop,step,c,p,noise,recompute=True)
    grad=torch.autograd.grad(action_loss(y,target),c)[0]
    assert grad.norm()>0
    plain=differentiable_rollout(loop,step,c,p,noise,recompute=False)
    plain_grad=torch.autograd.grad(action_loss(plain,target),c)[0]
    torch.testing.assert_close(y,plain,atol=0,rtol=0)
    torch.testing.assert_close(grad,plain_grad,atol=1e-7,rtol=1e-6)
    eps=1e-3
    c_plus,c_minus=c.detach().clone(),c.detach().clone()
    c_plus[0,0,0]+=eps
    c_minus[0,0,0]-=eps
    numerical=(action_loss(differentiable_rollout(loop,step,c_plus,p,noise),target)
        -action_loss(differentiable_rollout(loop,step,c_minus,p,noise),target))/(2*eps)
    torch.testing.assert_close(numerical,grad[0,0,0],atol=2e-4,rtol=2e-3)
    assert all(t.grad is None for t in step.parameters())
    assert all(t.grad is None for t in updater.parameters())


def test_condition_gradients_cross_all_recursive_updates():
    weight=nn.Parameter(torch.tensor(1.))
    adapter=SimpleNamespace(delta_encoder=lambda pair: pair.current_proprio,
        condition_updater=lambda c,code,age,**kw: SimpleNamespace(condition=c+weight*age))
    action=SimpleNamespace(normalize_proprio=lambda x:x,
        action_space=SimpleNamespace(normalize_action=lambda x:x))
    seq=dict(anchor_condition=torch.zeros(1,2,1), image_sequence=torch.zeros(1,4,1),
        proprio_sequence=torch.zeros(1,4,1), valid_mask=torch.ones(1,2,dtype=torch.bool),
        group_ids=torch.zeros(1,2), explicit_noises=torch.zeros(1,3,1), teacher_actions=torch.zeros(1,3,1))
    context,*_=query_inputs(adapter,action,seq,3,track_grad=True)
    context.condition.sum().backward()
    assert weight.grad.item()==12
    frozen,*_=query_inputs(adapter,action,seq,3)
    assert not frozen.condition.requires_grad
