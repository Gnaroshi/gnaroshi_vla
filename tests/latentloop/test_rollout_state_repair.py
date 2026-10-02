import json
import random
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import numpy as np
import pytest
import torch

from tools.simvla.rollout_state_repair import disjoint_indices,source_rgb,reservoir_slot,sample_from_sequence,prediction
from tools.simvla.rollout_repair_pipeline import jobs,choose_candidate,dependency_ready,ARMS
from tools.simvla.rollout_repair_rb2 import expected_counts,ROWS
from methods.latentloop.modules.observed_progress import build_model
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from tests.latentloop.test_frozen_trend_residual import fixture


def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data))


def test_initial_state_split_uses_values_not_row_indices():
    raw=np.arange(30,dtype=float).reshape(10,3)
    official=raw[[8,2,1,5,0]]
    a=disjoint_indices(raw,official,5,7)
    assert set(a)=={3,4,6,7,9}
    assert a==disjoint_indices(raw,official,5,7)
    with pytest.raises(RuntimeError): disjoint_indices(raw,official,6,7)


def test_rgb_roundtrip_exact():
    raw=torch.arange(256,dtype=torch.uint8).reshape(1,1,16,16,1).expand(-1,-1,-1,-1,3)
    assert torch.equal(source_rgb(raw.float()/255),raw)
    with pytest.raises(RuntimeError): source_rgb(torch.full((1,2,2,2,3),.123456))


def test_reservoir_is_bounded_and_reaches_late_queries():
    rng=random.Random(7); slots={}
    for i in range(1,1001):
        slot=reservoir_slot(i,4,rng)
        if slot is not None: slots[slot]=i
    assert len(slots)==4 and max(slots.values())>100
    assert min(slots)>=0 and max(slots)<4


@pytest.mark.parametrize('driver',['original','student'])
@pytest.mark.parametrize('sampled',[False,True])
def test_collection_driver_controls_execution_not_teacher_sampling(tmp_path,monkeypatch,driver,sampled):
    from tools.simvla import rollout_state_repair as repair
    from tools.simvla.error_compensation_common import sha
    condition=torch.ones(1,2,3)
    batch=dict(input_ids=None,image_input=None,image_mask=None,
        raw_rgb=torch.zeros(1,2,4,4,3),proprio=torch.zeros(1,8))
    ctx=SimpleNamespace(anchor=condition,images=batch['raw_rgb'],proprio=batch['proprio'],
        valid=torch.ones(1,2,dtype=torch.bool),groups=torch.zeros(1,2,dtype=torch.long))
    class Policy:
        def __init__(self):
            self.condition_adapter=SimpleNamespace(encode_condition=lambda **kw:condition*2)
            self.action_adapter=SimpleNamespace(
                decode_action_from_condition=lambda cond,prop,**kw:torch.full((1,10,7),float(cond.mean())*10),
                action_space=SimpleNamespace(normalize_action=lambda a:a))
            self.reset()
        def reset(self): self.steps=0; self._trend_context=ctx
        def _paired_initial_noise(self,condition,proprio,index): return torch.zeros(1,10,7),index
        def _decode(self,condition,proprio,**kw): return torch.ones(1,10,7)*3,0
        def _full_refresh(self,batch,**kw): return condition*2,torch.ones(1,10,7)*3,0
        def _v0_update(self,batch,**kw): return condition,torch.ones(1,10,7)*3,0
        def act(self,*args):
            q=self.steps//5
            if self.steps%5==0:
                if q%8==0: result=self._full_refresh(batch,policy_query_index=q)
                else: result=self._v0_update(batch,age=q%8,policy_query_index=q)
                self.current_action=result[1][0,0].numpy()
            self.steps+=1
            return SimpleNamespace(action=self.current_action)
    class Env:
        def __init__(self): self.actions=[]
        def seed(self,value): pass
        def reset(self): pass
        def set_init_state(self,value): return None
        def step(self,action): self.actions.append(action); return None,0,False,{}
        def close(self): pass
    env=Env(); pol=Policy()
    fake_libero=ModuleType('libero.libero')
    fake_libero.benchmark=SimpleNamespace(get_benchmark_dict=lambda:{'libero_10':lambda:SimpleNamespace(get_task=lambda i:i)})
    monkeypatch.setitem(sys.modules,'libero.libero',fake_libero)
    fake_runner=ModuleType('architectures.simvla.wrappers.dcld_eval.rollout_runner')
    fake_runner.build_env_obs=lambda obs:()
    fake_runner.get_libero_env=lambda task,*args:(env if task==9 else Env(),'task')
    monkeypatch.setitem(sys.modules,fake_runner.__name__,fake_runner)
    monkeypatch.setattr(repair,'configure',lambda c:None)
    monkeypatch.setattr(repair,'configure_strict_torch_determinism',lambda seed:None)
    monkeypatch.setattr(repair,'identity',lambda c:'test')
    monkeypatch.setattr(repair,'policy',lambda c:pol)
    if not sampled: monkeypatch.setattr(repair,'reservoir_slot',lambda *args:None)
    states=tmp_path/'states.pt'; torch.save(torch.zeros(1,3),states)
    c=dict(output=str(tmp_path),collection_seed=7,collection_noise_seed=7,
        collection_min_free_gib=0,collection_samples_per_age=4,
        collection_states={str(i):dict(path=str(states),sha256=sha(states),indices=[0]) for i in (9,5,1)})
    # Full collection avoids the smoke assertion that each age was retained.
    repair.collect(c,1,False,driver)
    assert len(env.actions)==910
    np.testing.assert_array_equal(np.asarray(env.actions[10:]),20 if driver=='original' else 3)
    marker=json.loads((tmp_path/'collection'/driver/'task9_state0.json').read_text())
    assert marker['driver']==driver and marker['teacher_action_applied']==(driver=='original')
    assert marker['records']==(28 if sampled else 0)
    if sampled:
        records=torch.load(tmp_path/'collection'/driver/'task9_state0.pt',weights_only=False)
        assert {r['age'] for r in records}==set(range(1,8))
        assert all(r['images'].dtype==torch.uint8 for r in records)
        assert all(torch.equal(r['target_action'],torch.full((1,10,7),20.)) for r in records)


@pytest.mark.parametrize('arm',['frozen_trend_residual','progress_only','progress_residual','progress_spatial'])
def test_pair_adapter_is_same_as_sequence_and_freezes_b(arm):
    source,_,s=fixture(8)
    model=build_model(NativeSimVLAV0(condition_dim=12,max_tokens=8),arm,max_age=7)
    model.initialize_frozen_trend(source.trend_head.state_dict())
    s['explicit_noises']=torch.randn(2,7,10,7)
    s['teacher_actions']=torch.randn(2,7,10,7)
    for age in (1,7):
        pair=sample_from_sequence(s,age)
        value=prediction(model,pair)
        torch.testing.assert_close(value,model.sequence(s,age)[0],rtol=0,atol=0)
        value.square().mean().backward()
        assert all(p.grad is None for p in model.trend_head.parameters())


def test_queue_collection_dependencies_and_three_matched_trainings():
    c=dict(output='/output',python='/python')
    plan=jobs(c,Path('/config'),False)
    assert len(plan)==18 and len({p['id'] for p in plan})==18
    training=[p for p in plan if p['id'].startswith('train_')]
    assert len(training)==3
    assert all(set(p['deps'])=={f'collect_{d}_{i}' for d in ('original','student') for i in range(4)} for p in training)
    assert len([p for p in plan if p['id'].startswith('eval_')])==6
    export=next(p for p in plan if p['id']=='export_rb2')
    assert set(export['deps'])=={p['id'] for p in training}


def test_selection_has_no_arbitrary_sr_gate(tmp_path):
    for i,arm in enumerate(ARMS):
        for k in (4,8):
            put(tmp_path/'online'/f'kc{k}_{arm}'/'summary.json',dict(verdict='EVALUATION_COMPLETE',episodes=500,successes=i,policy_ms_per_action=10))
        f=tmp_path/'train'/arm/'latest.pt'; f.parent.mkdir(parents=True); f.write_bytes(arm.encode())
    put(tmp_path/'pipeline_status.json',dict(phase='train_and_evaluate'))
    assert not dependency_ready(tmp_path)
    put(tmp_path/'pipeline_status.json',dict(phase='complete'))
    put(tmp_path/'campaign_complete.json',dict(verdict='COMPLETE',failed={}))
    assert dependency_ready(tmp_path)
    chosen,report=choose_candidate(tmp_path)
    assert chosen['arm']==ARMS[-1] and report['development_seed_reused']
    put(tmp_path/'pipeline_status.json',dict(phase='technical_failure'))
    with pytest.raises(RuntimeError): dependency_ready(tmp_path)


@pytest.mark.parametrize('q',[1,8,9,180])
def test_rb2_control_and_repair_call_budget(q):
    for row in ROWS:
        c=expected_counts(row,q)
        assert c['num_full_vlm_calls']==(q+7)//8
        assert c['num_condition_updater_calls']==q-(q+7)//8
        assert c['num_action_transformer_calls']==q*(10 if row=='trend_k8_full10' else 3)
        assert c['num_generation_decoder_only_steps']==(0 if row=='trend_k8_full10' else 7*q)
