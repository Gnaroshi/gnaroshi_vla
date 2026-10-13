import pytest
import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.bounded_history import BoundedHistoryCondition
from methods.latentloop.modules.visual_midpoint import VisualMidpointCondition
from methods.latentloop.modules.visual_memory import VisualMemoryCondition,SpatialPairEncoder,MemoryReference


def setup(spatial=False):
    torch.manual_seed(7)
    parent=BoundedHistoryCondition(NativeSimVLAV0(condition_dim=32),'recurrent')
    torch.nn.init.normal_(parent.condition_updater.up.weight,std=.01)
    model=VisualMemoryCondition(parent,spatial)
    torch.nn.init.normal_(model.visual.output.weight,std=.01)
    images=[torch.rand(1,2,16,16,3) for _ in range(8)]
    qs=[torch.randn(1,8) for _ in range(8)]
    args=(torch.randn(1,6,32),images[0],qs[0],torch.ones(1,6,dtype=torch.bool),torch.zeros(1,6,dtype=torch.long),8)
    calls=[];vision=torch.randn(1,72,32)
    def provider(age,images):
        calls.append(age)
        return vision
    model.feature_provider=provider
    return parent,model,args,images,qs,calls,vision


@pytest.mark.parametrize('spatial',[False,True])
def test_frozen_early_parity_memory_schedule_and_reset(spatial):
    parent,model,args,images,qs,calls,vision=setup(spatial)
    ctx=model.prepare(*args);reference=parent.prepare(*args)
    for age in range(1,8):
        z,_=model.predict(ctx,age,images[age],qs[age])
        if age<4:
            expected,_=parent.predict(reference,age,images[age],qs[age])
            torch.testing.assert_close(z,expected,rtol=0,atol=0)
        else:assert ctx.visual_memory is vision
    assert calls==[4]
    z.square().mean().backward()
    assert model.visual.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.early.parameters())
    assert not hasattr(model.prepare(*args),'visual_memory')
    assert [model.uses_vision(i) for i in range(8)]==[False]*4+[True]+[False]*3


def test_hold_matches_original_if_same_features_are_supplied():
    parent,model,args,images,qs,_,vision=setup()
    reference=VisualMidpointCondition(parent,'late_every')
    reference.load_state_dict(model.state_dict(),strict=True)
    reference.feature_provider=lambda age,images:vision
    a,b=model.prepare(*args),reference.prepare(*args)
    for age in range(1,8):
        x,_=model.predict(a,age,images[age],qs[age]);y,_=reference.predict(b,age,images[age],qs[age])
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_uint8_live_float_spatial_input_parity_and_view_order():
    module=SpatialPairEncoder()
    a=torch.randint(0,256,(1,2,17,19,3),dtype=torch.uint8)
    b=torch.randint(0,256,a.shape,dtype=torch.uint8)
    x=module(a,b);y=module(a.float()/255,b.float()/255)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert x.shape==(1,72,64)
    torch.testing.assert_close(module(a.flip(1),b.flip(1)),x.reshape(1,2,36,64).flip(1).reshape_as(x))


def test_fixed_pool_matches_24_to_6_adaptive_pool():
    x=torch.randn(2,64,24,24)
    torch.testing.assert_close(torch.nn.AvgPool2d(4,4)(x),torch.nn.functional.adaptive_avg_pool2d(x,(6,6)))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_strict_deterministic_spatial_backward():
    previous=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        encoder=SpatialPairEncoder().cuda()
        a=torch.rand(1,2,224,224,3,device='cuda')
        encoder(a,a.flip(2)).square().mean().backward()
        assert encoder.net[0].weight.grad.abs().sum()>0
    finally:
        torch.use_deterministic_algorithms(previous)


def test_spatial_branch_initial_parity_then_action_surrogate_gradient():
    torch.manual_seed(7);plain=MemoryReference(32,64,False)
    rich=MemoryReference(32,64,True);rich.load_state_dict(plain.state_dict(),strict=False)
    query=torch.randn(1,6,64);features=torch.randn(1,72,32);valid=torch.ones(1,6,dtype=torch.bool)
    a,b=torch.rand(1,2,16,16,3),torch.rand(1,2,16,16,3)
    assert torch.equal(plain(query,features,valid,a,b),rich(query,features,valid,a,b))
    optimizer=torch.optim.AdamW(rich.parameters(),lr=.01)
    target=torch.randn(1,6,64)
    for _ in range(4):
        optimizer.zero_grad();(rich(query,features,valid,a,b)-target).square().mean().backward();optimizer.step()
    assert rich.spatial.net[0].weight.grad.abs().sum()>0
    assert rich.spatial_key.weight.grad.abs().sum()>0
    assert rich.spatial_value.weight.grad.abs().sum()>0
    assert not torch.equal(rich(query,features,valid,a,b),rich(query,features,valid,a,b.flip(2)))


def test_shards_cover_500_once_and_ordered_queue(monkeypatch):
    from tools.simvla import visual_memory as study
    manifest=dict(episodes=[dict(task_id=t,trial_id=r) for t in range(10) for r in range(50)])
    for count in (1,2,4):
        ids=[(e['task_id'],e['trial_id']) for s in range(count) for e in study.selected_manifest(manifest,s,count)['episodes']]
        assert len(ids)==len(set(ids))==500
    monkeypatch.setattr(study.prior,'IS_RB2',False)
    plan={j['id']:j for j in study.jobs(dict(identity='test'))}
    assert plan['eval_smoke_memory']['deps']==['merge_hold','train_memory']
    assert plan['eval_smoke_spatial']['deps']==['merge_memory','train_spatial']
    assert len(plan['merge_hold']['deps'])==2 and len(plan['merge_memory']['deps'])==4
    assert 'merge_hold' not in plan['train_memory']['deps']
    monkeypatch.setattr(study.prior,'IS_RB2',True)
    plan={j['id']:j for j in study.jobs(dict(identity='test'))}
    assert plan['eval_smoke_memory']['ready_file'].endswith('memory/READY.json')
    assert len(plan['merge_memory']['deps'])==1


def test_merge_rejects_duplicate_or_incomplete_rows(tmp_path,monkeypatch):
    from tools.simvla import visual_memory as study
    monkeypatch.setattr(study,'OUTPUT',tmp_path)
    monkeypatch.setattr(study,'checked',lambda:dict(identity='test',manifest=dict(episodes=[])))
    with pytest.raises(FileNotFoundError):study.merge('hold')


def test_merge_complete_then_rejects_model_and_episode_mix(tmp_path,monkeypatch):
    from tools.simvla import visual_memory as study
    manifest=dict(episodes=[dict(task_id=t,trial_id=r) for t in range(10) for r in range(50)])
    monkeypatch.setattr(study,'OUTPUT',tmp_path)
    monkeypatch.setattr(study.prior,'IS_RB2',False)
    monkeypatch.setattr(study,'checked',lambda:dict(identity='test',manifest=manifest))
    for shard in range(2):
        out=tmp_path/'shards/hold'/f'part{shard}'/'rows/hold'
        subset=study.selected_manifest(manifest,shard,2)['episodes']
        for spec in subset:
            study.write_json(out/'episodes'/f"{spec['task_id']}_{spec['trial_id']}.json",dict(
                **spec,identity='test',model_sha256='model',success=spec['trial_id']<40,
                timing_valid=True,policy_ms_total=100.,episode_length=10))
        study.write_json(out/'summary.json',dict(verdict='EVAL_COMPLETE',identity='test',
            episodes=250,successes=200,hardware='test',elapsed_seconds=5,timing_scope='test'))
    study.merge('hold')
    summary=study.read_json(tmp_path/'rows/hold/summary.json')
    assert summary['episodes']==500 and summary['successes']==400
    assert summary['policy_ms_per_action']==10.
    path=tmp_path/'shards/hold/part0/rows/hold/episodes/0_0.json'
    record=study.read_json(path)
    study.write_json(path,dict(record,model_sha256='other'))
    with pytest.raises(RuntimeError,match='Incomplete Long500'):study.merge('hold')
    study.write_json(path,dict(record,trial_id=2))
    with pytest.raises(RuntimeError,match='Shard episode mismatch'):study.merge('hold')
