import copy

import pytest
import torch

from methods.latentloop.modules.native_simvla_v0 import TokenSharedConditionUpdater, NativeSimVLAV0
from methods.latentloop.modules.short_visual_training import build_model
from methods.latentloop.modules.token_interaction import TokenInteraction, InteractingUpdater, install


@pytest.mark.parametrize('kind', ['attention','mlp'])
def test_capacity_zero_init_and_padding(kind):
    torch.manual_seed(3)
    block = TokenInteraction(kind)
    assert sum(p.numel() for p in block.parameters()) == 16384
    x = torch.randn(2,7,64)
    valid = torch.tensor([[True]*5+[False]*2,[False]*7])
    assert torch.equal(block(x,valid), torch.zeros_like(x))
    torch.nn.init.normal_(block.output.weight, std=.1)
    output = block(x,valid)
    changed = x.clone();changed[~valid]=100*torch.randn_like(changed[~valid])
    assert torch.equal(output[valid],block(changed,valid)[valid])
    assert output[~valid].count_nonzero()==0
    assert torch.isfinite(output).all()


@pytest.mark.parametrize('kind', ['attention','mlp'])
def test_two_steps_gradient_and_intertoken_dependence(kind):
    block = TokenInteraction(kind)
    optimizer = torch.optim.AdamW(block.parameters(),lr=.01)
    x,target = torch.randn(1,5,64),torch.randn(1,5,64)
    valid = torch.ones(1,5,dtype=torch.bool)
    for _ in range(2):
        optimizer.zero_grad();(block(x,valid)-target).square().mean().backward();optimizer.step()
    optimizer.zero_grad();(block(x,valid)-target).square().mean().backward()
    assert all(p.grad is not None and p.grad.count_nonzero() for p in block.parameters())
    other=x.clone();other[:,1]+=torch.randn(64)
    same=torch.equal(block(x,valid)[:,0],block(other,valid)[:,0])
    assert same == (kind=='mlp')


@pytest.mark.parametrize('kind', ['attention','mlp'])
def test_updater_exact_zero_parity(kind):
    torch.manual_seed(7)
    parent=TokenSharedConditionUpdater(max_age=7).eval()
    torch.nn.init.normal_(parent.up.weight)
    candidate=InteractingUpdater(parent,kind).eval()
    z=torch.randn(2,9,960);code=torch.randn(2,128)
    valid=torch.ones(2,9,dtype=torch.bool);valid[:,-1]=False
    kwargs=dict(valid_mask=valid,group_ids=torch.zeros(2,9,dtype=torch.long),
                age=torch.tensor([4,7]),token_feature=torch.randn(2,9,64))
    left,right=parent(z,code,**kwargs),candidate(z,code,**kwargs)
    for key in ('condition','residual','gate'):
        assert torch.equal(getattr(left,key),getattr(right,key))
    assert torch.equal(right.condition[:,-1],z[:,-1])


@pytest.mark.parametrize('warm', [True,False])
@pytest.mark.parametrize('kind', ['attention','mlp'])
def test_install_trainable_contract(warm,kind):
    model=build_model(NativeSimVLAV0(),'fresh_joint')
    early={k:v.clone() for k,v in model.early.state_dict().items()}
    install(model,kind,warm=warm)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==(16384 if warm else 819330)
    assert all(torch.equal(v,model.early.state_dict()[k]) for k,v in early.items())
    assert not any(p.requires_grad for p in model.early.parameters())
    with pytest.raises(ValueError,match='already'):
        install(model,kind,warm=warm)


def test_layout_validation():
    block=InteractingUpdater(TokenSharedConditionUpdater(max_age=7),'attention')
    with pytest.raises(ValueError,match='age'):
        block(torch.zeros(1,5,960),torch.zeros(1,128),valid_mask=torch.ones(1,5,dtype=torch.bool),
              group_ids=torch.zeros(1,5,dtype=torch.long),age=8)


def test_schedules_and_matched_targets():
    from tools.simvla.token_interaction_campaign import sampling, arm_contract
    c=dict(seed=7,step_counts=dict(warm_attention=5000,warm_mlp=5000,fresh_attention=10000,fresh_mlp=10000))
    for stage in ('warm','fresh'):
        left,right=stage+'_attention',stage+'_mlp'
        for step in (1,2,3,4,5,6,100,5000):
            r1,a1=sampling(c,left,step);r2,a2=sampling(c,right,step)
            assert (r1.random(),a1)==(r2.random(),a2)
        local=arm_contract(c,left)
        assert local['warmup']==local['steps']//20


def test_early_unchanged_and_recursive_gradient():
    parent=build_model(NativeSimVLAV0(),'fresh_joint')
    # Test the actual expert-routing interface without loading original assets.
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
    for _,expert in parent.experts():
        install_exact_uint8_delta_path(expert)
        torch.nn.init.normal_(expert.condition_updater.up.weight,std=.02)
    model=install(copy.deepcopy(parent),'attention',warm=True).eval()
    z=torch.randn(1,7,960);rgb=torch.randint(0,256,(1,2,32,32,3),dtype=torch.uint8);q=torch.randn(1,8)
    valid=torch.ones(1,7,dtype=torch.bool);groups=torch.zeros(1,7,dtype=torch.long)
    contexts=[m.prepare(z,rgb,q,valid,groups,8) for m in (parent,model)]
    for age in range(1,4):
        left,_=parent.predict(contexts[0],age,rgb,q)
        right,_=model.predict(contexts[1],age,rgb,q)
        assert torch.equal(left,right)
    model.feature_provider=lambda age,images:torch.ones(1,72,960)
    z4,_=model.predict(contexts[1],4,rgb,q)
    z4.retain_grad()
    z5,_=model.predict(contexts[1],5,rgb,q)
    z5.square().mean().backward()
    assert z4.grad is not None and z4.grad.count_nonzero()
    assert model.late.condition_updater.mixer.output.weight.grad.count_nonzero()
    assert all(p.grad is None for p in model.early.parameters())


def test_selection_uses_new_arm_names_without_training_scope():
    from tools.simvla.token_interaction_campaign import ARMS, row_name, select_candidate
    result=dict(rows={},training={})
    for arm in ARMS:
        result['rows'][row_name(arm)]=dict(episodes=500,successes=400,policy_ms_per_action=12.)
        result['training'][arm]=dict(total_training_steps=75000 if arm.startswith('warm') else 20000)
    assert select_candidate(result)[0]=='fresh_attention'
    result['rows'][row_name('warm_attention')]['successes']=401
    assert select_candidate(result)[0]=='warm_attention'
    result['rows'][row_name('warm_attention')]['policy_ms_per_action']=None
    with pytest.raises(RuntimeError,match='Incomplete'):
        select_candidate(result)


def test_remote_jobs_require_both_smokes_and_evaluations():
    from tools.simvla.token_interaction_rb2 import jobs
    from tools.simvla.short_visual_campaign import validate_jobs
    plan=jobs(dict(identity='test'))
    validate_jobs(plan)
    assert len(plan)==5
    assert set(plan[-1]['deps'])=={'selected_seed1_eval','selected_seed2_eval'}
    for job in plan:
        assert job['completion']['identity']=='test'


@pytest.mark.parametrize('warm', [True,False])
@pytest.mark.parametrize('kind', ['attention','mlp'])
def test_actual_parent_classes_support_original_only_bypass(warm,kind):
    from tools.simvla.geometry_comparison_rb2 import construct
    native=NativeSimVLAV0()
    parent=construct(native) if warm else build_model(native,'fresh_joint')
    model=install(parent,kind,warm=warm)
    assert model.prepare(None,None,None,None,None,1) is None
    z=torch.randn(1,7,960);rgb=torch.zeros(1,2,32,32,3,dtype=torch.uint8);q=torch.randn(1,8)
    valid=torch.ones(1,7,dtype=torch.bool);groups=torch.zeros(1,7,dtype=torch.long)
    for interval in (4,8):
        context=model.prepare(z,rgb,q,valid,groups,interval)
        assert context.interval==interval
        assert torch.equal(context.previous,z)
