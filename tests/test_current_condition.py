import copy

import pytest
import torch

from methods.latentloop.modules.current_condition import install, frozen_hash, assert_frozen
from methods.latentloop.modules.attention_query_contract import AttentionQueryCondition
from methods.latentloop.modules.visual_readout_distillation import ReadoutDistillationCondition
from test_visual_readout_distillation import parent, inputs, unroll, cpu_threads


@pytest.fixture
def model(parent):
    source=ReadoutDistillationCondition(parent,'joint_readout').eval()
    return AttentionQueryCondition(source,parent,'original_query_target').eval()


def test_initial_outputs_and_carried_predictions_exact(model):
    reference=copy.deepcopy(model)
    install(model)
    data=inputs();calls=[]
    model.feature_provider=lambda age,images:(calls.append(age) or data[3][age])
    reference.feature_provider=lambda age,images:data[3][age]
    with torch.no_grad():
        context,actual=unroll(model,data)
        ref_context,expected=unroll(reference,data)
    assert calls==[4]
    for (z,d),(target,_) in zip(actual,expected):
        torch.testing.assert_close(z,target,atol=0,rtol=0)
        torch.testing.assert_close(d['carried'],target,atol=0,rtol=0)
    torch.testing.assert_close(context.previous,ref_context.previous,atol=0,rtol=0)


def test_changed_current_head_leaves_all_future_predictions_exact(model):
    reference=copy.deepcopy(model);install(model)
    with torch.no_grad():model.current_head['updater'].up.bias.add_(.5)
    data=inputs()
    model.feature_provider=reference.feature_provider=lambda age,images:data[3][age]
    with torch.no_grad():
        _,actual=unroll(model,data);_,expected=unroll(reference,data)
    for age,((z,d),(target,_)) in enumerate(zip(actual,expected),1):
        assert torch.equal(z,target)==(age!=4)
        assert torch.equal(d['carried'],target)


def test_only_current_head_receives_gradients(model):
    install(model);data=inputs();model.feature_provider=lambda age,images:data[3][age]
    frozen=frozen_hash(model)
    context,outputs=unroll(model,data,4)
    output=outputs[-1][0]
    output.square().mean().backward()
    assert not context.previous.requires_grad
    for name,module in model.current_head.items():
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()),name
    original=torch.nn.Linear(2,2).requires_grad_(False)
    assert_frozen(model,original)
    torch.optim.SGD(model.current_head.parameters(),lr=.01).step()
    assert frozen_hash(model)==frozen


def test_bypass_and_state_roundtrip(model):
    other=copy.deepcopy(model);install(model);install(other)
    other.load_state_dict(model.state_dict(),strict=True)
    data=inputs(interval=1)
    assert model.prepare(*data[0]) is None
    with pytest.raises(ValueError):install(model)


def test_three_trainings_have_independent_dependencies(monkeypatch,tmp_path):
    from tools.simvla import current_condition_campaign as c
    monkeypatch.setattr(c.training,'OUTPUT',tmp_path)
    jobs=c.jobs(dict(identity='test',steps=5000,manifest=dict(action_noise_seed_base=7)))
    c.campaign.validate_jobs(jobs)
    assert len(jobs)==27
    byid={j['id']:j for j in jobs}
    for arm in c.ARMS:
        assert byid[arm+'_smoke']['deps']==[]
        assert byid[arm+'_runtime']['deps']==[arm+'_smoke']
        assert byid[arm+'_train']['deps']==[arm+'_runtime']
        assert byid[arm+'_k8_seed1_smoke']['deps']==[arm+'_train']
        assert sum(byid[arm+'_k8_seed1_'+str(i)]['completion']['episodes'] for i in range(4))==500
    assert all('codex_outputs' not in s for j in jobs for s in j['cmd'])


def test_all_targets_share_same_samples():
    import random
    from methods.latentloop.modules.current_condition import ARMS
    values={arm:[random.Random(7*100000+step).randrange(1000) for step in range(1,15)] for arm in ARMS}
    assert len({tuple(v) for v in values.values()})==1
