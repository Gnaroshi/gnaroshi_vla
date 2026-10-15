import copy
import pytest
import torch

from methods.latentloop.modules.visual_readout_distillation import ReadoutDistillationCondition
from methods.latentloop.modules.visual_token_transport import TokenTransport, TransportCondition, FreshVisionCondition
from test_visual_readout_distillation import parent, inputs, unroll, cpu_threads


@pytest.fixture
def warm(parent):
    model = ReadoutDistillationCondition(parent, 'warm_retain').eval()
    with torch.no_grad():
        for p in model.late.parameters(): p.add_(.003)
    return model


def test_camera_local_transport_is_convex_and_matches_permutation():
    m = TokenTransport()
    _, images, _, _ = inputs(batch=2)
    key, value = torch.randn(2,72,64), torch.randn(2,72,64)
    weights, _, _ = m.correspondence(key,value,images[4],images[7])
    assert weights.shape == (2,2,36,36)
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2,2,36))
    moved = m.apply_transport(weights, value).reshape(2,2,36,64)
    original = value.reshape(2,2,36,64)
    assert (moved >= original.amin(2,keepdim=True)-1e-6).all()
    assert (moved <= original.amax(2,keepdim=True)+1e-6).all()
    permutation = torch.eye(36).flip(0)[None,None].expand(2,2,-1,-1)
    torch.testing.assert_close(m.apply_transport(permutation,value).reshape(2,2,36,64),original.flip(2))
    same_camera = value.clone(); same_camera[:,36:] += 1000
    torch.testing.assert_close(m.apply_transport(weights,same_camera)[:,:36],m.apply_transport(weights,value)[:,:36])


def test_matched_initialization_and_new_content_only_when_enabled():
    torch.manual_seed(7); pure = TokenTransport(False)
    torch.manual_seed(7); extended = TokenTransport(True)
    for n,p in pure.state_dict().items(): torch.testing.assert_close(p,extended.state_dict()[n],atol=0,rtol=0)
    _,images,_,_ = inputs()
    key,value = torch.randn(1,72,64),torch.randn(1,72,64)
    args=(key,value,images[4],images[7],7)
    for a,b in zip(pure(*args),extended(*args)): torch.testing.assert_close(a,b,atol=0,rtol=0)
    with torch.no_grad(): extended.appearance[-1].bias.fill_(.1)
    for a,b in zip(pure(*args),extended(*args)): torch.testing.assert_close(a+.1,b)


@pytest.mark.parametrize('arm',['fresh_vision','transport','transport_residual'])
def test_parent_frozen_schedule_early_parity_and_no_online_teacher(warm,arm,monkeypatch):
    model = FreshVisionCondition(warm) if arm=='fresh_vision' else TransportCondition(warm,arm)
    for n,p in warm.state_dict().items(): torch.testing.assert_close(p,model.state_dict()[n],atol=0,rtol=0)
    assert all(p.requires_grad == (n.startswith('visual.predictor.') and arm!='fresh_vision') for n,p in model.named_parameters())
    data=inputs(); calls=[]
    def provide(age,images): calls.append(age); return data[3][age]
    model.feature_provider=provide
    if arm!='fresh_vision':
        monkeypatch.setattr(model.visual,'target_readout',lambda *a: (_ for _ in ()).throw(AssertionError('online teacher')))
    with torch.no_grad():
        _,early=unroll(warm.early,data,3)
        _,actual=unroll(model,data,7)
    for (a,_),(b,_) in zip(actual[:3],early): torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert calls==([4,5,6,7] if arm=='fresh_vision' else [4])
    assert [a for a in range(8) if model.uses_vision(a)]==calls
    fresh=model.prepare(*data[0]); assert not hasattr(fresh,'visual_memory')


@pytest.mark.parametrize('arm',['transport','transport_residual'])
def test_gradients_targets_reference_and_age4_exact(warm,arm):
    model=TransportCondition(warm,arm)
    args,images,_,features=inputs()
    query=torch.randn(1,6,64)
    actual=model.visual(query,features[4],args[3],images[4],images[4],4)
    expected=warm.visual(query,features[4],args[3],images[4],images[4])
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    optimizer=torch.optim.AdamW(model.visual.predictor.parameters(),lr=1e-3)
    before={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    for _ in range(2):
        optimizer.zero_grad()
        prediction=model.visual.predicted_readout(features[4],images[4],images[7],7)
        target=model.visual.target_readout(features[7])
        assert all(not p.requires_grad for p in target)
        loss=sum((p-t).square().mean() for p,t in zip(prediction,target));loss.backward()
        assert model.visual.predictor.encoder[0].weight.grad.abs().sum()>0
        assert model.visual.predictor.query.weight.grad.abs().sum()>0
        optimizer.step()
    for n,p in model.named_parameters():
        if n in before:
            torch.testing.assert_close(p,before[n],atol=0,rtol=0);assert p.grad is None
    result=model.visual.predicted_readout(features[4],images[4],images[7],7)
    model.visual.predicted_readout(features[4],images[4],images[6],6)
    again=model.visual.predicted_readout(features[4],images[4],images[7],7)
    for a,b in zip(result,again): torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_compilation_and_uint8_float_agree(warm):
    model=TransportCondition(warm,'transport_residual')
    args,images,_,features=inputs(uint8=True)
    q=torch.randn(1,6,64)
    original=model.visual.forward
    compiled=torch.compile(original,backend='eager',fullgraph=True)
    try:
        for age in (4,5,6,7):
            a=compiled(q,features[4],args[3],images[4],images[age],age)
            b=original(q,features[4],args[3],images[4].float()/255,images[age].float()/255,age)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    finally: torch._dynamo.reset()


def test_queue_order_and_gpu_policy(monkeypatch,tmp_path):
    from tools.simvla import visual_transport_campaign as campaign
    from tools.simvla import visual_readout_campaign as base
    from tools.simvla import visual_transport_train as training
    monkeypatch.setattr(base,'OUTPUT',tmp_path)
    monkeypatch.setattr(base,'ARMS',training.ARMS)
    monkeypatch.setattr(training,'OUTPUT',tmp_path)
    c=dict(identity='test',manifest=dict(episodes=[dict(task_id=t,trial_id=n) for t in range(10) for n in range(50)]))
    for remote in (False,True):
        monkeypatch.setattr(training.prior,'IS_RB2',remote)
        jobs=campaign.jobs(c); byid={j['id']:j for j in jobs}
        assert len(byid)==len(jobs)
        assert 'train_fresh_vision' not in byid
        assert ('train_transport' in byid)==(not remote)
        assert base.gpu_ids()==((0,) if remote else (4,5,6,7))
        for before,after in zip(training.ARMS,training.ARMS[1:]):
            assert 'merge_'+before in byid['eval_smoke_'+after]['deps']
        for arm in training.ARMS:
            total=sum(byid[f'eval_{arm}_{i}']['completion']['episodes'] for i in range(1 if remote else 4))
            assert total==500


def test_capacity_metric_and_known_spatial_permutation():
    from tools.simvla.visual_transport_evidence import camera_local_distances
    reference=(torch.randn(2,36,64),torch.randn(2,36,64))
    target=tuple(x.flip(1) for x in reference)
    distances=camera_local_distances(reference,target)
    error,indices=distances.min(-1)
    torch.testing.assert_close(error,torch.zeros(2,36),atol=0,rtol=0)
    assert torch.equal(indices,torch.arange(36).flip(0)[None].expand(2,-1))
    held=distances.diagonal(dim1=1,dim2=2)
    expected=sum(((r-t).square().mean(-1)/t.square().mean(-1).clamp_min(1e-6)) for r,t in zip(reference,target))/2
    torch.testing.assert_close(held,expected)
