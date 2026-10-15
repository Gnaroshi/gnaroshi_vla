import importlib.util
import os
from pathlib import Path
import sys

import torch
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/openpi'))
from condition_bridge_common import kv_tensor,kv_state,bridge_update,bridge_objective,learning_rate
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixEmbeddingState,PrefixKVState


def example():
    keys=tuple(torch.randn(1,1,4,256) for _ in range(18))
    values=tuple(torch.randn_like(k) for k in keys)
    emb=PrefixEmbeddingState(torch.randn(1,4,8),torch.ones(1,4,dtype=torch.bool),torch.zeros(1,4,dtype=torch.bool),torch.arange(4)[None])
    return PrefixKVState(**emb.__dict__,pre_rope_keys=keys,values=values),emb


def test_kv_roundtrip_preserves_all_layers_and_tokens():
    prefix,emb=example();packed=kv_tensor(prefix);rebuilt=kv_state(prefix,emb,packed)
    assert packed.shape==(1,18,4,512)
    for a,b in zip(prefix.pre_rope_keys+prefix.values,rebuilt.pre_rope_keys+rebuilt.values):torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_zero_delta_and_official_loss_equivalence():
    from condition_bridge_common import official_class
    torch.set_num_threads(1)
    prefix,emb=example()
    model=official_class()(emb_dim=8,seq_len=4,hidden_dim=24,num_heads=12,num_blocks=1,state_dim=8)
    predicted,packed=bridge_update(model,prefix,emb,torch.randn(1,8),torch.randn(1,7))
    torch.testing.assert_close(packed,kv_tensor(prefix),atol=0,rtol=0)
    target,_=example()
    path=Path(os.environ['LATENT_BRIDGE_UPSTREAM_ROOT'])/'scripts/pi0/train_pi0_bridge_kv.py'
    spec=importlib.util.spec_from_file_location('official_bridge_training',path)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    batch=dict(prev_kv=kv_tensor(prefix),next_kv=kv_tensor(target),delta_kv=kv_tensor(target)-kv_tensor(prefix),
        prev_emb=emb.embeddings,curr_emb=emb.embeddings,state=torch.zeros(1,8),action=torch.zeros(1,7))
    for head in model.layer_heads:
        torch.nn.init.normal_(head[-1].weight,std=.01)
    _,packed=bridge_update(model,prefix,emb,batch['state'],batch['action'])
    expected,_=official.compute_loss(model,batch,'cpu')
    actual=bridge_objective(packed,target)
    torch.testing.assert_close(actual,expected)
    actual.backward();assert model.layer_heads[0][-1].weight.grad.abs().sum()>0


def test_lr_budgets_and_recursive_age_descriptor():
    assert learning_rate(499,10000,'ours')==1e-4
    assert learning_rate(9999,10000,'ours')<1.001e-5
    assert learning_rate(249,5000,'bridge',True)==pytest.approx(3e-5)
    from condition_bridge_common import make_module
    from types import SimpleNamespace
    prefix,_=example();module,desc=make_module('ours',prefix,SimpleNamespace(state=torch.randn(1,32)))
    assert desc['max_age']==7


def test_plan_has_all_requested_intervals():
    import run_condition_bridge as runner
    c=dict(config_id='test',condition_steps=10000,extension_steps=5000)
    jobs=runner.plan(c)
    rows=[j['row'] for j in jobs if j['kind']=='eval' and not j['smoke'] and not j.get('collect')]
    assert rows==['baseline_nfe1','ours_k2','ours_k3','ours_k4','bridge_k2','bridge_k3','bridge_k4','ours_k8','bridge_k8']
    assert len({j['id'] for j in jobs})==len(jobs)
    assert len([j for j in jobs if j.get('collect')])==2
