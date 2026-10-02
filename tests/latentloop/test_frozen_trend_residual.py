import copy
from types import SimpleNamespace

import pytest
import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.trend_condition import TrendCondition, teacher_decomposition, decomposition_loss
from tests.latentloop.test_trend_condition import TinyEncoder
from tools.simvla.trend_residual_pipeline import expected_counts, check_compiler, check_policy


def fixture(k):
    torch.manual_seed(42)
    parent=NativeSimVLAV0(condition_dim=12,max_tokens=8)
    parent.delta_encoder=TinyEncoder()
    source=TrendCondition(copy.deepcopy(parent),'trend_only',max_age=k-1)
    torch.nn.init.normal_(source.trend_head.up.weight,std=.01)
    model=TrendCondition(parent,'frozen_trend_residual',max_age=k-1)
    model.initialize_frozen_trend(source.trend_head.state_dict())
    seq=dict(anchor_condition=torch.randn(2,5,12),teacher_conditions=torch.randn(2,k-1,5,12),
        image_sequence=torch.randn(2,k,2,8,8,3),proprio_sequence=torch.randn(2,k,8),
        valid_mask=torch.tensor([[True,True,True,False,False]]*2),group_ids=torch.zeros(2,5,dtype=torch.long))
    return source,model,seq


@pytest.mark.parametrize('k',[4,8])
def test_zero_initial_residual_exactly_preserves_b_only(k):
    source,model,s=fixture(k)
    for age in range(1,k):
        c,b,e=model.sequence(s,age)
        assert torch.count_nonzero(e)==0
        torch.testing.assert_close(c,source.sequence(s,age)[0],rtol=0,atol=0)
    assert model.condition_updater.age_embedding.num_embeddings==k
    assert not any(p.requires_grad for p in model.trend_head.parameters())
    with pytest.raises(ValueError): model.sequence(s,k)


@pytest.mark.parametrize('k',[4,8])
def test_training_updates_residual_but_never_frozen_trend(k):
    _,model,s=fixture(k)
    before={n:p.clone() for n,p in model.trend_head.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0)
    for age in range(1,k):
        optimizer.zero_grad(set_to_none=True)
        c,b,e=model.sequence(s,age)
        loss=decomposition_loss(model,s,age,c,b,e)
        loss.backward()
        assert model.condition_updater.up.weight.grad.abs().max()>0
        assert all(p.grad is None for p in model.trend_head.parameters())
        optimizer.step()
    assert torch.count_nonzero(model.sequence(s,k-1)[2])>0
    for n,p in model.trend_head.state_dict().items():
        torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    changed={n:t.clone() for n,t in s.items()}
    changed['image_sequence'][:,k-1].add_(20)
    assert not torch.equal(model.sequence(s,k-1)[0],model.sequence(changed,k-1)[0])
    changed={n:t.clone() for n,t in s.items()}
    changed['teacher_conditions'].add_(999)
    changed['image_sequence'][:,2:].add_(999)
    changed['proprio_sequence'][:,2:].add_(999)
    torch.testing.assert_close(model.sequence(s,1)[0],model.sequence(changed,1)[0],rtol=0,atol=0)


def test_k8_decomposition_reconstructs_all_seven_labels():
    _,_,s=fixture(8)
    for age in range(1,8):
        b,e=teacher_decomposition(s['anchor_condition'],s['teacher_conditions'],age)
        torch.testing.assert_close(s['anchor_condition']+age*b+e,s['teacher_conditions'][:,age-1])


@pytest.mark.parametrize('row',['residual_k4','trend_trained_k8','residual_k8'])
def test_compiled_residual_counters_and_required_components(row):
    for actions in (1,5,6,20,21,35,36,40,41,900):
        q=(actions+4)//5
        p=SimpleNamespace(step_index=actions,metrics=SimpleNamespace(counters={**expected_counts(row,q),'num_policy_queries':q}))
        check_policy(p,row)
        p.metrics.counters['num_residual_calls']+=1
        with pytest.raises(RuntimeError): check_policy(p,row)
    c=SimpleNamespace(records={n:dict(graphs=1) for n in
        ['vlm','action_transformer','action_decoder','generation_updater','trend_head','observation_encoder','condition_updater']})
    check_compiler(c,row)
    c.records['condition_updater']['graphs']=0
    if row.startswith('residual'):
        with pytest.raises(RuntimeError): check_compiler(c,row)
    else: check_compiler(c,row)


def test_k8_dataset_joins_consecutive_queries_without_crossing_episodes(monkeypatch):
    from architectures.simvla.adapters.latentloop.efficient_multirate import exact_teacher_cache as cache
    from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import query_identity
    queries={}
    windows=[]
    for ep,n in [('a',12),('b',4)]:
        for j in range(n):
            queries[query_identity(0,ep,j)]=dict(metadata=dict(task_id=0,episode_id=ep,query_index=j))
        windows.extend([[query_identity(0,ep,j+a) for a in range(4)] for j in range(0,n,4)])
    fake=SimpleNamespace(manifest=dict(windows=windows),locators=queries,query=lambda key:queries[key])
    monkeypatch.setattr(cache,'ExactTeacherStore',lambda path:fake)
    original=cache.ExactTeacherSequenceDataset('/unused',split='all')
    joined=cache.ExactTeacherSequenceDataset('/unused',split='all',window_queries=8)
    assert len(original)==4 and len(joined)==2
    assert joined.identities==((0,'a',0),(0,'a',4))
    assert joined.windows[1][-1]==query_identity(0,'a',11)
    del queries[query_identity(0,'a',7)]
    with pytest.raises(ValueError,match='no exact teacher windows'):
        cache.ExactTeacherSequenceDataset('/unused',split='all',window_queries=8)


def test_longer_windows_keep_original_episode_split_contract(monkeypatch):
    from architectures.simvla.adapters.latentloop.efficient_multirate import condition_mechanism as module
    def dataset(cache, *, split, window_queries=4, **kwargs):
        identities=((0,split,0),(0,split,4)) if window_queries==4 else ((0,split,0),)
        return SimpleNamespace(identities=identities,split_sha256=split if window_queries==4 else 'extended_'+split)
    monkeypatch.setattr(module,'ExactTeacherSequenceDataset',dataset)
    payload=dict(training_config=dict(heldout_fraction=.2,split_seed=1,
        dataset_splits=dict(train_split_sha256='train',heldout_split_sha256='heldout')))
    train,heldout=module.make_datasets(dict(cache='/unused',training_k_c=8),payload)
    assert train.split_sha256=='extended_train' and heldout.split_sha256=='extended_heldout'
    payload['training_config']['dataset_splits']['train_split_sha256']='wrong'
    with pytest.raises(RuntimeError,match='Cache split'):
        module.make_datasets(dict(cache='/unused',training_k_c=8),payload)
