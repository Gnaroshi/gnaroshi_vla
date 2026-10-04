from types import SimpleNamespace

import pytest

from tools.simvla.bridge_interval_sweep import ROWS,interval,expected_counts,check_policy,check_compiler,make_policy


@pytest.mark.parametrize('k',range(1,9))
@pytest.mark.parametrize('actions',[1,5,6,40,41,900])
def test_refresh_bridge_and_action_budgets(k,actions):
    row=f'latent_bridge_f{k}'; q=(actions+4)//5
    counts=expected_counts(row,q)
    assert counts['num_full_vlm_calls']==sum(i%k==0 for i in range(q))
    assert counts['num_latent_bridge_calls']==sum(i%k!=0 for i in range(q))
    assert counts['num_action_transformer_calls']==10*q
    assert counts['num_generation_decoder_only_steps']==0
    policy=SimpleNamespace(step_index=actions,metrics=SimpleNamespace(counters={**counts,'num_policy_queries':q}))
    check_policy(policy,row)
    policy.metrics.counters['num_latent_bridge_calls']+=1
    with pytest.raises(RuntimeError): check_policy(policy,row)


def test_k1_must_never_execute_bridge():
    compiler=SimpleNamespace(records={k:dict(graphs=1) for k in ('vlm','action_transformer')})
    check_compiler(compiler,'latent_bridge_f1')
    with pytest.raises(RuntimeError): check_compiler(compiler,'latent_bridge_f8')
    compiler.records['bridge_predict_next']=dict(graphs=1)
    check_compiler(compiler,'latent_bridge_f8')
    with pytest.raises(RuntimeError): check_compiler(compiler,'latent_bridge_f1')


def test_factory_only_changes_refresh_interval(monkeypatch):
    from tools.simvla import bridge_interval_sweep as m
    for row in ROWS:
        sentinel=object()
        monkeypatch.setattr(m,'attach_policy',lambda *args:SimpleNamespace(refresh_every=2,bridge=sentinel,flow_steps=10))
        policy=make_policy(None,{},row,{})
        assert policy.refresh_every==interval(row) and policy.flow_steps==10 and policy.bridge is sentinel
    with pytest.raises(ValueError): interval('latent_bridge_f9')
