from types import SimpleNamespace
import copy

import pytest
import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules import readout_intervention as m
from tools.simvla import readout_causality as c
from tools.simvla import geometry_comparison_rb2 as portable
from tools.simvla.short_visual_campaign import validate_jobs


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        yield
    torch.set_num_threads(old)


def inputs():
    return dict(z=torch.randn(1,6,32),images=torch.rand(1,8,2,16,16,3),
        proprio=torch.randn(1,8,8),valid=torch.ones(1,6,dtype=torch.bool),
        groups=torch.arange(6)[None],features=torch.randn(8,1,72,32),
        original=torch.randn(8,1,6,32))


def ready(mode):
    native=NativeSimVLAV0(condition_dim=32,max_tokens=16)
    model=m.ReadoutIntervention(native,mode).eval()
    with torch.no_grad():
        for p in (model.early.condition_updater.up.weight,
                  model.late.condition_updater.up.weight,model.visual.output.weight,
                  model.visual.predictor.appearance[-1].weight):
            p.normal_(std=.03)
    return native,model


def run(model,x):
    calls=[]
    def features(age,images):
        calls.append(age)
        return x['features'][age]
    model.feature_provider=features
    ctx=model.prepare(x['z'],x['images'][:,0],x['proprio'][:,0],x['valid'],x['groups'],8)
    outputs=[]
    with torch.no_grad():
        for age in range(1,8):
            model.original_previous=x['original'][age-1]
            model.original_previous_age=age-1
            out,_=model.predict(ctx,age,x['images'][:,age],x['proprio'][:,age])
            outputs.append(out)
    return outputs,calls


def test_control_exact_loaded_constructor_parity():
    native,model=ready('control');x=inputs()
    reference=portable.construct(native).eval()
    reference.load_state_dict(model.state_dict(),strict=True)
    left,calls=run(model,x);right,_=run(reference,x)
    assert calls==[4]
    for a,b in zip(left,right):assert torch.equal(a,b)


@pytest.mark.parametrize('mode',m.MODES)
def test_same_weights_early_parity_and_visual_call_routing(mode):
    _,control=ready('control');_,model=ready(mode)
    model.load_state_dict(control.state_dict(),strict=True)
    x=inputs();before={k:v.clone() for k,v in model.state_dict().items()}
    original,_=run(control,x);values,calls=run(model,x)
    assert calls==([4,5,6,7] if mode in ('current_content','both') else [4])
    assert all(torch.equal(a,b) for a,b in zip(original[:3],values[:3]))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert not any(p.requires_grad for p in model.parameters())
    assert [i for i in range(1,8) if model.uses_vision(i)]==calls
    if mode!='control':assert any(not torch.equal(a,b) for a,b in zip(original[4:],values[4:]))


def test_reference_used_only_in_attention_query(monkeypatch):
    _,model=ready('original_selector');x=inputs();seen=[];updater_inputs=[]
    original=m.make_query
    def capture(updater,condition,code,groups):
        seen.append(condition.clone())
        return original(updater,condition,code,groups)
    monkeypatch.setattr(m,'make_query',capture)
    hook=model.late.condition_updater.register_forward_pre_hook(
        lambda module,args:updater_inputs.append(args[0].clone()))
    outputs,_=run(model,x);hook.remove()
    for j in range(4):
        assert torch.equal(seen[j],x['original'][j+3])
        assert torch.equal(updater_inputs[j],outputs[j+2])
        assert not torch.equal(seen[j],updater_inputs[j])


def test_missing_reference_fails_before_late_update():
    _,model=ready('original_selector');x=inputs()
    model.feature_provider=lambda age,images:x['features'][age]
    ctx=model.prepare(x['z'],x['images'][:,0],x['proprio'][:,0],x['valid'],x['groups'],8)
    for age in range(1,4):model.predict(ctx,age,x['images'][:,age],x['proprio'][:,age])
    with pytest.raises(RuntimeError,match='preceding query'):
        model.predict(ctx,4,x['images'][:,4],x['proprio'][:,4])


@pytest.mark.parametrize('mode',m.MODES)
def test_reference_call_count_and_no_short_query_overhead(mode):
    ages=[i for i in range(1,8) if m.needs_previous_reference(mode,8,i)]
    assert ages==([3,4,5,6] if m.uses_original_selector(mode) else [])
    assert not any(m.needs_previous_reference(mode,4,i) for i in range(4))


def test_online_reference_comes_from_previous_query_and_reset(monkeypatch):
    _,model=ready('original_selector')
    monkeypatch.setattr(c,'ORIGINAL_ATTACH',lambda policy,*args:policy)
    observed=[]
    def update(batch,*,age,policy_query_index):
        if age>=4:observed.append((age,model.original_previous.item(),model.original_previous_age))
        return ('unchanged_chunk',age)
    policy=SimpleNamespace(k_c=8,reset=lambda:None,_v0_update=update,
        condition_adapter=SimpleNamespace(encode_condition=lambda **kw:kw['input_ids'].float()))
    policy=c.attach(policy,model,8,None)
    for age in range(1,8):
        result=policy._v0_update(dict(input_ids=torch.tensor(age),image_input=None,image_mask=None),
            age=age,policy_query_index=age)
        assert result==('unchanged_chunk',age)
    assert observed==[(4,3.,3),(5,4.,4),(6,5.,5),(7,6.,6)]
    assert policy._selector_backbone_calls==4
    policy.reset()
    assert model.original_previous is None and policy._selector_backbone_calls==0


@pytest.mark.parametrize('remote',[False,True])
def test_queue_dependencies_and_no_repeated_online_control(tmp_path,monkeypatch,remote):
    monkeypatch.setattr(c.prior,'IS_RB2',remote);monkeypatch.setattr(c,'OUTPUT',tmp_path)
    jobs=c.jobs(dict(identity='test'));validate_jobs(jobs)
    assert len(jobs)==(6 if remote else 8)
    if remote:
        assert [j['id'] for j in jobs if j['completion'].get('episodes')==500]==[
            a+'_eval' for a in c.ARMS]
        assert not any('control' in j['id'] for j in jobs)
    assert all(j['cmd'][3]==c.MODULE for j in jobs)


def test_goal_progress_excludes_initially_satisfied_goals():
    def record(states,final,success=False):
        return dict(success=success,query_goal_progress=[dict(satisfied=s) for s in states],
                    terminal_goal_progress=dict(satisfied=final))
    records=[record([[True,False],[True,False]],[False,False]),
             record([[False,False],[True,False]],[False,False]),
             record([[False,False],[True,False]],[True,False]),
             record([[False,False],[True,True]],[True,True],True)]
    assert c.failure_progress(records)==dict(failed=3,failed_after_new_goal=2,new_goal_lost_at_end=1)
