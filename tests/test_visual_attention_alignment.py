import pytest
import torch
from methods.latentloop.modules.visual_attention_alignment import AdaptedFreshCondition,AttentionStudent
from test_visual_token_transport import warm
from test_visual_readout_distillation import parent,inputs,unroll,cpu_threads


@pytest.mark.parametrize('joint',[False,True])
def test_teacher_targets_and_deployment_schedule(warm,joint):
    teacher=AdaptedFreshCondition(warm)
    model=AttentionStudent(teacher,joint)
    data=inputs();calls=[]
    def provider(age,images):calls.append(age);return data[3][age]
    model.feature_provider=provider
    with torch.no_grad():unroll(model,data,7)
    assert calls==[4]
    assert model.attention_losses==[]
    calls.clear();model.distill_enabled=True
    _,outputs=unroll(model,data,7)
    assert calls==[4,5,6,7]
    assert len(model.attention_losses)==3
    loss=torch.stack(model.attention_losses).mean()
    if joint:loss=loss+outputs[-1][0].square().mean()
    loss.backward()
    assert model.visual.predictor.appearance[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.teacher_visual.parameters())
    assert all(p.grad is None for p in model.early.parameters())
    assert any(p.grad is not None for p in model.late.parameters())==joint
    assert all(p.grad is None for p in model.visual.key.parameters())


def test_fresh_training_changes_only_late_and_visual(warm):
    model=AdaptedFreshCondition(warm)
    assert all(p.requires_grad==n.startswith(('late.','visual.')) for n,p in model.named_parameters())
    data=inputs();calls=[]
    model.feature_provider=lambda age,images:(calls.append(age) or data[3][age])
    _,actual=unroll(model,data,7)
    _,early=unroll(warm.early,data,3)
    for (a,_),(b,_) in zip(actual[:3],early):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert calls==[4,5,6,7]


def test_stage_dependencies():
    from tools.simvla import visual_attention_campaign as c
    c.base.OUTPUT=c.training.OUTPUT
    jobs=c.jobs({'identity':'test'})
    byid={j['id']:j for j in jobs}
    assert byid['smoke_attention_distill']['deps']==['train_adapt_fresh']
    assert byid['smoke_attention_joint']['deps']==['train_attention_distill']
    assert len(jobs)==24
