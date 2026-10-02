import copy
import pytest
import torch
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.observed_progress import build_model, projection_coefficient, orthogonal_component, ARMS
from methods.latentloop.modules.trend_condition import scaled_mse
from tests.latentloop.test_frozen_trend_residual import fixture


def test_projection_metric_and_best_coefficient():
    torch.manual_seed(2)
    anchor,b,delta=[torch.randn(2,5,12) for _ in range(3)]
    valid=torch.tensor([[True,True,False,False,True]]*2)
    a=projection_coefficient(delta,b,anchor,valid)
    e=orthogonal_component(delta,b,anchor,valid)
    torch.testing.assert_close(projection_coefficient(e,b,anchor,valid),torch.zeros_like(a),atol=1e-6,rtol=0)
    for x in (-10,0,1,3,7,20):
        assert scaled_mse(a*b,delta,anchor,valid)<=scaled_mse(x*b,delta,anchor,valid)+1e-6
    torch.testing.assert_close(projection_coefficient(delta,b*0,anchor,valid),a*0)
    assert torch.isfinite(orthogonal_component(delta,b*0,anchor,valid)).all()


@pytest.mark.parametrize('arm',ARMS)
def test_initial_identity_causality_freeze_and_train(arm):
    source,_,s=fixture(8)
    parent=NativeSimVLAV0(condition_dim=12,max_tokens=8)
    model=build_model(parent,arm,max_age=7)
    model.initialize_frozen_trend(source.trend_head.state_dict())
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    before={k:v.clone() for k,v in model.trend_head.state_dict().items()}
    c,b,e=model.sequence(s,7)
    torch.testing.assert_close(c,source.sequence(s,7)[0],rtol=0,atol=0)
    for _ in range(3):
        opt.zero_grad()
        c,b,e=model.sequence(s,7)
        scaled_mse(c,s['teacher_conditions'][:,6],s['anchor_condition'],s['valid_mask']).backward()
        assert model.progress_head[-1].weight.grad.abs().max()>0
        opt.step()
    for k,v in before.items(): torch.testing.assert_close(model.trend_head.state_dict()[k],v,atol=0,rtol=0)
    assert all(p.grad is None for p in model.trend_head.parameters())
    c,b,e=model.sequence(s,1)
    torch.testing.assert_close(projection_coefficient(e,b,s['anchor_condition'],s['valid_mask']),torch.zeros(2,1,1),atol=1e-5,rtol=0)
    changed={k:v.clone() for k,v in s.items()}
    changed['teacher_conditions'].add_(100)
    changed['image_sequence'][:,2:].add_(100)
    torch.testing.assert_close(c,model.sequence(changed,1)[0],atol=0,rtol=0)
    changed['image_sequence'][:,1].add_(.5)
    assert not torch.equal(c,model.sequence(changed,1)[0])
    with pytest.raises(ValueError): model.sequence(s,8)


def test_spatial_path_preserves_global_features():
    parent=NativeSimVLAV0(condition_dim=12,max_tokens=8)
    source,_,s=fixture(8)
    model=build_model(parent,'progress_spatial',max_age=7)
    model.initialize_frozen_trend(source.trend_head.state_dict())
    from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair
    pair=NativeV0ObservationPair(s['image_sequence'][:,0],s['image_sequence'][:,1],s['proprio_sequence'][:,0],s['proprio_sequence'][:,1])
    code,local=model.spatial_encode(pair,s['anchor_condition'])
    torch.testing.assert_close(code,model.delta_encoder(pair),atol=0,rtol=0)
    assert local.shape==(2,5,64)


def test_queue_has_four_matched_trainings_and_eight_nonduplicate_evals():
    from tools.simvla.observed_progress_campaign import jobs
    c=dict(python='/python',output='/output',evaluation_condition_intervals=[8,4])
    plan=jobs(c,'config.json',False)
    assert len(plan)==len({r['id'] for r in plan})==13
    assert sum(r['id'].startswith('train_') for r in plan)==4
    assert sum(r['id'].startswith('eval_') for r in plan)==8
    assert all('eval_kc8_'+r['id'].removeprefix('eval_kc4_') in r['deps'] for r in plan if r['id'].startswith('eval_kc4_'))


@pytest.mark.parametrize('queries',[1,4,5,8,9,180])
def test_rb2_controls_counts(queries):
    from tools.simvla.trend_controls_rb2 import expected_counts
    naive=expected_counts('trend_k4_naive3',queries)
    hold=expected_counts('hold_k4_generation3',queries)
    assert naive['num_full_vlm_calls']==hold['num_full_vlm_calls']==(queries+3)//4
    assert naive['num_action_transformer_calls']==hold['num_action_transformer_calls']==3*queries
    assert naive['num_generation_decoder_only_steps']==0
    assert hold['num_generation_decoder_only_steps']==7*queries
    assert naive['num_trend_head_calls']==(queries+3)//4
    assert hold['num_trend_head_calls']==0
