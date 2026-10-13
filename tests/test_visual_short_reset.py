import pytest
import torch

from methods.latentloop.modules.bounded_history import BoundedHistoryCondition
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.visual_short_reset import VisualShortReset
from tools.simvla.visual_short_reset import target_age


def setup(anchor=False):
    torch.manual_seed(9)
    parent=BoundedHistoryCondition(NativeSimVLAV0(condition_dim=32),'recurrent')
    torch.nn.init.normal_(parent.condition_updater.up.weight,std=.01)
    model=VisualShortReset(parent,anchor)
    torch.nn.init.normal_(model.visual.output.weight,std=.01)
    images=[torch.randn(1,2,3,16,16) for _ in range(8)]
    proprio=[torch.randn(1,8) for _ in range(8)]
    args=(torch.randn(1,6,32),images[0],proprio[0],torch.ones(1,6,dtype=torch.bool),torch.zeros(1,6,dtype=torch.long),8)
    calls=[];features=torch.randn(1,8,32)
    def provide(age,images):
        calls.append(age)
        return features
    model.feature_provider=provide
    return parent,model,args,images,proprio,calls


@pytest.mark.parametrize('anchor',[False,True])
def test_short_updater_reused_with_local_age_and_future_gradient(anchor):
    parent,model,args,images,q,calls=setup(anchor)
    context=model.prepare(*args);reference=parent.prepare(*args)
    for age in range(1,8):
        z,_=model.predict(context,age,images[age],q[age])
        if age<4:
            expected,_=parent.predict(reference,age,images[age],q[age])
            torch.testing.assert_close(z,expected,rtol=0,atol=0)
        elif age==4:
            midpoint=z
            reference=model.early.prepare(z,images[age],q[age],args[3],args[4],4)
            assert midpoint.requires_grad
            midpoint.retain_grad()
        else:
            expected,_=model.early.predict(reference,age-4,images[age],q[age])
            torch.testing.assert_close(z,expected,rtol=0,atol=0)
    assert calls==[4]
    z.square().mean().backward()
    assert midpoint.grad.abs().sum()>0
    assert model.visual.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.early.parameters())
    assert context.age==7 and context.short_context.age==3
    fresh=model.prepare(*args)
    assert not hasattr(fresh,'short_context')


def test_future_observations_still_affect_outputs():
    _,model,args,images,q,_=setup()
    def run(change):
        ctx=model.prepare(*args)
        for age in range(1,8):
            z,_=model.predict(ctx,age,images[age]+(10 if change and age==7 else 0),q[age])
        return z
    assert not torch.equal(run(False),run(True))


def test_relative_age_mismatch_rejected():
    _,model,args,images,q,_=setup()
    with pytest.raises(ValueError):model.predict(model.prepare(*args),4,images[4],q[4])


def test_matched_targets_and_queue_followups(monkeypatch):
    from tools.simvla import visual_short_reset as study
    assert [target_age('rollout',i) for i in range(1,9)]==[4,5,6,7]*2
    assert [target_age('instant',i) for i in range(1,9)]==[4]*8
    monkeypatch.setattr(study.prior,'IS_RB2',False)
    jobs={j['id']:j for j in study.jobs(dict(identity='test'))}
    assert len(jobs)==14
    assert jobs['train_rollout']['deps']==['train_smoke_rollout']
    assert jobs['eval_rollout']['deps']==['eval_smoke_rollout']
    assert jobs['export_rollout']['deps']==['train_rollout']
    assert 'train_zero_previous' not in jobs
    monkeypatch.setattr(study.prior,'IS_RB2',True)
    jobs={j['id']:j for j in study.jobs(dict(identity='test'))}
    assert len(jobs)==8
    assert jobs['eval_smoke_rollout']['ready_file'].endswith('rollout/READY.json')
    assert all('train' not in j['cmd'][5:6] for j in jobs.values())


def test_payload_shape_matches_existing_models():
    from methods.latentloop.modules.visual_midpoint import VisualMidpointCondition
    from methods.latentloop.modules.visual_reanchor import VisualReanchorCondition
    parent,model,*_=setup()
    model.load_state_dict(VisualMidpointCondition(parent,'midpoint').state_dict(),strict=True)
    model.load_state_dict(VisualReanchorCondition(parent,'midpoint',False).state_dict(),strict=True)
