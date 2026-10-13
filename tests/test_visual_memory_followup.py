import pytest
import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.bounded_history import BoundedHistoryCondition
from methods.latentloop.modules.visual_memory import VisualMemoryCondition
from methods.latentloop.modules.visual_memory_controls import ReferenceMemoryCondition, ControlledReference


def fixture(spatial=False, reference_age=0, control='none'):
    torch.manual_seed(11)
    parent=BoundedHistoryCondition(NativeSimVLAV0(condition_dim=32),'recurrent')
    torch.nn.init.normal_(parent.condition_updater.up.weight,std=.01)
    model=ReferenceMemoryCondition(parent,spatial,reference_age,control)
    torch.nn.init.normal_(model.visual.output.weight,std=.01)
    images=[torch.rand(1,2,16,16,3) for _ in range(8)]
    qs=[torch.randn(1,8) for _ in range(8)]
    args=(torch.randn(1,6,32),images[0],qs[0],torch.ones(1,6,dtype=torch.bool),torch.zeros(1,6,dtype=torch.long),8)
    calls=[];vision=torch.randn(1,72,32)
    def provider(age,images):
        calls.append(age);return vision
    model.feature_provider=provider
    return parent,model,args,images,qs,calls,vision


@pytest.mark.parametrize('spatial',[False,True])
def test_refresh_capture_once_early_parity_and_frozen_weights(spatial):
    parent,model,args,images,qs,calls,vision=fixture(spatial)
    ctx=model.prepare(*args);early=parent.prepare(*args)
    for age in range(1,8):
        z,_=model.predict(ctx,age,images[age],qs[age])
        if age<4:
            expected,_=parent.predict(early,age,images[age],qs[age])
            torch.testing.assert_close(z,expected,rtol=0,atol=0)
        assert ctx.visual_memory is vision
    assert calls==[0] and not any(model.uses_vision(i) for i in range(8))
    z.square().mean().backward()
    assert model.visual.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.early.parameters())


@pytest.mark.parametrize('spatial',[False,True])
def test_midpoint_compatibility_with_completed_model(spatial):
    parent,model,args,images,qs,_,vision=fixture(spatial,4)
    reference=VisualMemoryCondition(parent,spatial)
    reference.load_state_dict(model.state_dict(),strict=True)
    reference.feature_provider=lambda age,images:vision
    a,b=model.prepare(*args),reference.prepare(*args)
    for age in range(1,8):
        x,_=model.predict(a,age,images[age],qs[age]);y,_=reference.predict(b,age,images[age],qs[age])
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_rgb_control_keeps_current_proprio_and_actual_history():
    _,model,args,images,qs,_,_=fixture(True,4,'frozen_rgb')
    ctx=model.prepare(*args)
    for age in range(1,5):model.predict(ctx,age,images[age],qs[age])
    received=[]
    hook=model.late.delta_encoder.register_forward_pre_hook(lambda module,args:received.append(args[0]))
    model.predict(ctx,5,images[5],qs[5]);hook.remove()
    pair=received[0]
    assert ctx.previous_images is images[5] and ctx.previous_proprio is qs[5]
    assert pair.previous_images is images[4] and pair.current_images is images[4]
    assert pair.current_proprio is qs[5]


def test_mean_vision_removes_position_within_each_camera():
    torch.manual_seed(4)
    module=ControlledReference(32,64,False,'mean_vision')
    torch.nn.init.normal_(module.output.weight)
    z=torch.randn(1,6,64);v=torch.randn(1,72,32);valid=torch.ones(1,6,dtype=torch.bool)
    images=torch.rand(1,2,16,16,3)
    shuffled=v.reshape(1,2,36,32).flip(2).reshape_as(v)
    torch.testing.assert_close(module(z,v,valid,images,images),module(z,shuffled,valid,images,images))


def test_grid_control_keeps_state_dict_and_changes_attention():
    torch.manual_seed(6)
    plain=ControlledReference(32,64,True,'none')
    for layer in (plain.output,plain.spatial_key,plain.spatial_value):torch.nn.init.normal_(layer.weight,std=.05)
    control=ControlledReference(32,64,True,'permuted_grid');control.load_state_dict(plain.state_dict(),strict=True)
    q=torch.randn(1,6,64);v=torch.randn(1,72,32);valid=torch.ones(1,6,dtype=torch.bool)
    a,b=torch.rand(1,2,16,16,3),torch.rand(1,2,16,16,3)
    assert not torch.equal(plain(q,v,valid,a,b),control(q,v,valid,a,b))


def test_queue_and_bridge_intervals(monkeypatch):
    from tools.simvla import visual_memory_followup as study
    monkeypatch.setattr(study.prior,'IS_RB2',False)
    plan=study.jobs(dict(identity='test'))
    assert len(plan)==30 and len({j['id'] for j in plan})==30
    assert sum(j['id'].startswith('eval_') and not j['id'].startswith('eval_smoke') for j in plan)==12
    monkeypatch.setattr(study.prior,'IS_RB2',True)
    plan=study.jobs(dict(identity='test'))
    assert len(plan)==14 and plan[0]['id']=='eval_smoke_bridge_f8'
    assert 'eval_bridge_f8' in plan[2]['deps']
    assert all('f4' not in j['id'] for j in plan)
    for f in (2,3,6,8):assert study.row_spec(f'bridge_f{f}')['interval']==f
    with pytest.raises(ValueError):study.row_spec('midpoint_spatial_unknown')
