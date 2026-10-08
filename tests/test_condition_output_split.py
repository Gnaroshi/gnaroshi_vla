import copy
import os

import pytest
import torch
from torch.nn import functional as F

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.condition_output_split import ARMS,ConditionOutputSplit,geometry


def test_rb2_environment_is_self_contained(monkeypatch):
    from tools.simvla.condition_output_split_rb2 import environment
    for key in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('MUJOCO_GL','osmesa')
    monkeypatch.setenv('GALLIUM_DRIVER','llvmpipe')
    monkeypatch.setenv('LIBGL_ALWAYS_SOFTWARE','1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4,5,6,7')
    env=environment(0)
    assert env['PATH']==os.environ['PATH']
    assert env['CUDA_VISIBLE_DEVICES']=='0'
    assert env['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert env['MUJOCO_GL']==env['PYOPENGL_PLATFORM']=='egl'
    assert env['OMP_NUM_THREADS']==env['MKL_NUM_THREADS']=='1'
    assert env['TORCHINDUCTOR_COMPILE_THREADS']=='2'
    assert env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']=='1'
    assert 'GALLIUM_DRIVER' not in env and 'LIBGL_ALWAYS_SOFTWARE' not in env
    with pytest.raises(ValueError): environment(4)


def test_fresh_initialization_ignores_parent_weights():
    from tools.simvla.condition_output_split_train import build_initial_model
    parent=NativeSimVLAV0(condition_dim=16,max_tokens=8)
    torch.nn.init.constant_(parent.condition_updater.up.weight,4.)
    first=build_initial_model(copy.deepcopy(parent),dict(initialization='fresh',seed=17),'carry_output')
    for p in parent.parameters():
        with torch.no_grad(): p.add_(5.)
    second=build_initial_model(parent,dict(initialization='fresh',seed=17),'carry_base')
    for key,value in first.state_dict().items():
        torch.testing.assert_close(value,second.state_dict()[key],rtol=0,atol=0)
    assert first.condition_updater.up.weight.count_nonzero()==0
    assert first.action_condition_updater.up.weight.count_nonzero()==0
    images=torch.randint(0,256,(1,2,64,64,3),dtype=torch.uint8)
    z=torch.randn(1,4,16); q=torch.zeros(1,8)
    ctx=first.prepare(z,images,q,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,4,dtype=torch.long),8)
    predicted,_=first.predict(ctx,1,images,q)
    torch.testing.assert_close(predicted,z,rtol=0,atol=0)


def test_pretrained_initialization_keeps_base_encoder_and_weights():
    from tools.simvla.condition_output_split_train import build_initial_model
    parent=NativeSimVLAV0(condition_dim=16,max_tokens=8)
    torch.nn.init.constant_(parent.condition_updater.up.weight,.4)
    model=build_initial_model(parent,{},'carry_base')
    assert model.delta_encoder is parent.delta_encoder
    assert torch.all(model.condition_updater.up.weight==.4)
    assert model.action_condition_updater.up.weight.count_nonzero()==0


def test_phase_two_sampling_continues_without_repeating_phase_one():
    from tools.simvla.condition_output_split_train import sampling_step
    for step in (1,14,100,7000):
        a,ka,aa=sampling_step(dict(seed=7,sample_step_offset=3000),step)
        b,kb,ab=sampling_step(dict(seed=7),3000+step)
        assert (ka,aa)==(kb,ab)
        assert [a.randrange(100000) for _ in range(2)]==[b.randrange(100000) for _ in range(2)]


def test_continuation_validates_and_restores_full_model(tmp_path):
    import json
    from tools.simvla.condition_output_split_train import load_continuation
    from tools.simvla.compile_benchmark import sha
    model=ConditionOutputSplit(NativeSimVLAV0(condition_dim=16,max_tokens=8),'carry_base')
    path=tmp_path/'model.pt'; summary=tmp_path/'summary.json'
    torch.save(dict(format='simvla_condition_output_split_v1',arm='carry_base',step=3000,identity='source',
        contract=dict(action_mode='naive3',training_intervals=[4,8]),model=model.state_dict()),path)
    summary.write_text(json.dumps(dict(identity='source',steps=3000,verdict='TRAIN_AND_OFFLINE_COMPLETE',
        checkpoint=str(path),checkpoint_sha256=sha(path),training_seconds=10.)))
    c=dict(initialization='continuation',initial_models=dict(carry_base=dict(summary=str(summary),identity='source')))
    other=ConditionOutputSplit(NativeSimVLAV0(condition_dim=16,max_tokens=8),'carry_base')
    restored=load_continuation(other,c,'carry_base')
    assert restored['prior_steps']==3000
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)
    c['initial_models']['carry_base']['identity']='wrong'
    with pytest.raises(RuntimeError): load_continuation(other,c,'carry_base')


def test_initialization_pipeline_dependencies(tmp_path,monkeypatch):
    from tools.simvla import condition_initialization_pipeline as pipeline
    monkeypatch.setattr(pipeline,'identity',lambda c:'test')
    configs={p:dict(output=str(tmp_path/p),python='python',steps=3000 if p=='fresh_3k' else 7000)
        for p in pipeline.PHASES}
    jobs=pipeline.jobs(configs); by_id={j['id']:j for j in jobs}
    assert len(jobs)==len(by_id)==22
    assert sum('_export_' in j['id'] for j in jobs)==4
    for arm in ARMS:
        assert by_id[f'fresh_10k_smoke_train_{arm}']['deps']==[f'fresh_3k_train_{arm}']
        assert by_id[f'pretrained_10k_smoke_train_{arm}']['deps']==[]
    for j in jobs:
        assert set(j['deps']).issubset(by_id)


def test_initialization_evaluation_does_not_repeat_three_k_rows():
    from tools.simvla.condition_initialization_rb2 import ROWS,jobs
    assert len(ROWS)==8
    assert set(ROWS.values())=={(init,arm,k) for init in ('pretrained','fresh') for arm in ARMS for k in (4,8)}
    for job in jobs():
        assert job['completion']['episodes']==500
        assert '_10k/' in job['ready_file']


@pytest.fixture
def pair():
    torch.set_num_threads(1); torch.manual_seed(42)
    parent=NativeSimVLAV0(condition_dim=16,max_tokens=8)
    torch.nn.init.normal_(parent.condition_updater.up.weight,std=.1)
    models=[ConditionOutputSplit(copy.deepcopy(parent),arm) for arm in ARMS]
    inputs=dict(previous=torch.randn(1,4,16),code=torch.randn(1,128),
        valid_mask=torch.tensor([[True,True,True,False]]),group_ids=torch.zeros(1,4,dtype=torch.long),age=1)
    return models,inputs


def test_matching_initial_weights_and_zero_addition(pair):
    models,inputs=pair
    assert sum(p.numel() for p in models[0].parameters())==sum(p.numel() for p in models[1].parameters())
    for key,value in models[0].state_dict().items():
        torch.testing.assert_close(value,models[1].state_dict()[key],rtol=0,atol=0)
    outputs=[model.update(**inputs) for model in models]
    for output,base,carried in outputs:
        torch.testing.assert_close(output,base,rtol=0,atol=0)
        torch.testing.assert_close(carried,base,rtol=0,atol=0)
        torch.testing.assert_close(output[:,3],inputs['previous'][:,3],rtol=0,atol=0)


def test_routing_only_changes_carried_value(pair):
    models,inputs=pair
    for model in models:
        torch.nn.init.constant_(model.action_condition_updater.up.bias,.2)
    output0,base0,carry0=models[0].update(**inputs)
    output1,base1,carry1=models[1].update(**inputs)
    torch.testing.assert_close(output0,output1,rtol=0,atol=0)
    torch.testing.assert_close(base0,base1,rtol=0,atol=0)
    assert not torch.equal(carry0,carry1)
    torch.testing.assert_close(carry0,output0,rtol=0,atol=0)
    torch.testing.assert_close(carry1,base1,rtol=0,atol=0)


@pytest.mark.parametrize('index',[0,1])
def test_current_action_gradient_only_extra_head(pair,index):
    models,inputs=pair; model=models[index]
    inputs['code'].requires_grad_(True)
    output,base,_=model.update(**inputs)
    output.square().mean().backward()
    assert inputs['code'].grad is None
    assert all(p.grad is None for p in model.condition_updater.parameters())
    assert model.action_condition_updater.up.weight.grad.abs().sum()>0


@pytest.mark.parametrize('index',[0,1])
def test_future_condition_gradient_matches_routing(pair,index):
    models,inputs=pair; model=models[index]
    _,_,carried=model.update(**inputs)
    next_inputs={**inputs,'previous':carried,'age':2}
    _,base,_=model.update(**next_inputs)
    base.square().mean().backward()
    grad=model.action_condition_updater.up.weight.grad
    assert (grad is not None and grad.abs().sum()>0) if index==0 else grad is None


def test_added_output_depends_on_input(pair):
    models,inputs=pair; model=models[1]
    torch.nn.init.normal_(model.action_condition_updater.up.weight,std=.1)
    output,base,_=model.update(**inputs)
    other,other_base,_=model.update(**{**inputs,'code':inputs['code']+2})
    assert not torch.allclose(output-base,other-other_base)


def test_geometry_identity_and_cosine_mse_disagreement():
    target=torch.tensor([[[1.,0.]]]); valid=torch.ones(1,1,dtype=torch.bool)
    close=geometry(torch.tensor([[[.9,.1]]]),target,valid)
    aligned=geometry(torch.tensor([[[2.,0.]]]),target,valid)
    assert close['raw_mse']<aligned['raw_mse']
    assert close['cosine']<aligned['cosine']
    for d in (close,aligned):
        assert abs(d['raw_mse']-d['norm_difference_mse']-d['weighted_direction_mse'])<1e-6
        assert abs(d['raw_mse']-d['token_mean_mse']-d['centered_mse'])<1e-6


def test_query_sequence_and_refresh(pair):
    models,inputs=pair; model=models[1]
    images=torch.rand(1,2,3,64,64); proprio=torch.zeros(1,8)
    ctx=model.prepare(inputs['previous'],images,proprio,inputs['valid_mask'],inputs['group_ids'],8)
    with pytest.raises(ValueError): model.predict(ctx,2,images,proprio)
    for age in range(1,8):
        out,d=model.predict(ctx,age,images,proprio)
        torch.testing.assert_close(ctx.previous,d['base'],rtol=0,atol=0)
    with pytest.raises(ValueError): model.predict(ctx,8,images,proprio)
    fresh=model.prepare(inputs['previous'],images,proprio,inputs['valid_mask'],inputs['group_ids'],4)
    assert fresh.age==0
    torch.testing.assert_close(fresh.previous,inputs['previous'],rtol=0,atol=0)


def test_extra_head_does_not_copy_patched_forward(pair):
    models,inputs=pair
    parent=NativeSimVLAV0(condition_dim=16,max_tokens=8)
    original=parent.condition_updater.forward
    parent.condition_updater.forward=lambda *args,**kwargs:original(*args,**kwargs)
    model=ConditionOutputSplit(parent,'carry_base')
    assert 'forward' not in model.action_condition_updater.__dict__


@pytest.mark.parametrize('actions',[1,5,6,39,40,41,900])
@pytest.mark.parametrize('interval',[4,8])
def test_full_and_light_call_counts(actions,interval):
    from types import SimpleNamespace
    from tools.simvla.condition_output_split_eval import check_policy
    queries=(actions+4)//5; full=(queries+interval-1)//interval; light=queries-full
    counts=dict(num_policy_queries=queries,num_full_vlm_calls=full,
        num_condition_updater_calls=light,num_action_condition_updater_calls=light,
        num_action_transformer_calls=3*queries)
    actual=dict(observation_encoder=light,condition_updater=light,action_condition_updater=light) if light else {}
    policy=SimpleNamespace(k_c=interval,step_index=actions,metrics=SimpleNamespace(counters=counts),
        _condition_component_calls=actual)
    check_policy(policy)
    counts['num_action_transformer_calls']+=1
    with pytest.raises(RuntimeError): check_policy(policy)
