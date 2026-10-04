import random
from pathlib import Path

import pytest
import torch

from tools.simvla.rollout_round2_pipeline import settings,jobs,VARIANTS
from tools.simvla.rollout_round2_rb2 import expected_counts,ROWS
from tools.simvla.rollout_state_repair import training_settings,sample_pool,prediction
from tests.latentloop.test_frozen_trend_residual import fixture


def test_factorial_only_changes_data_and_trend_training():
    s=settings()
    assert len(s)==4
    assert s['previous_fixed']['sources']==s['previous_joint']['sources']==['previous']
    assert s['aggregate_fixed']['sources']==s['aggregate_joint']['sources']==['previous','current']
    for v in VARIANTS:
        assert training_settings({'variant_settings':s},v)==s[v]
        assert s[v]['train_trend']==v.endswith('joint')


def test_old_defaults_are_preserved():
    assert training_settings({},'rollout_repair')==dict(driver='student',sources=['current'],train_trend=False)
    assert training_settings({},'offline_control')['driver'] is None
    with pytest.raises(ValueError): training_settings({},'unknown')


def test_data_selection_excludes_heldout_and_original_driver():
    records=[dict(age=a,metadata=dict(split=split,driver=driver,collection_source=source))
        for a in (1,7) for split in ('train','heldout') for driver in ('student','original') for source in ('previous','current')]
    for name,s in settings().items():
        values=sample_pool(records,s,7)
        assert len(values)==len(s['sources'])
        assert all(r['metadata']['split']=='train' and r['metadata']['driver']=='student' and r['age']==7 for r in values)


def test_four_gpu_collection_and_four_matched_trainings():
    plan=jobs(dict(output='/tmp/test',python='/python'),Path('/config'),False)
    assert len(plan)==17
    assert len([j for j in plan if j['id'].startswith('collect')])==4
    train=[j for j in plan if j['id'].startswith('train')]
    assert len(train)==4 and len({tuple(j['deps']) for j in train})==1
    assert len([j for j in plan if j['id'].startswith('eval')])==8
    assert next(j for j in plan if j['id']=='export_rb2')['deps']==['train_'+v for v in VARIANTS]


@pytest.mark.parametrize('q',[1,4,8,9,180])
def test_all_rb2_budgets(q):
    for row,(k,mode) in ROWS.items():
        c=expected_counts(row,q)
        assert c['num_full_vlm_calls']==(q+k-1)//k
        assert c['num_trend_head_calls']==c['num_full_vlm_calls']
        assert c['num_action_transformer_calls']==q*(10 if mode=='condition_nfe10' else 3)
        assert c['num_generation_decoder_only_steps']==(7*q if mode=='ours_kc2_ng3' else 0)


@pytest.mark.parametrize('train_b',[False,True])
def test_joint_flag_changes_training_not_inference_equation(train_b):
    source,model,s=fixture(8)
    model.requires_grad_(True); model.trend_head.requires_grad_(train_b)
    before=model.sequence(s,7)[0].detach()
    loss=model.sequence(s,7)[0].square().mean(); loss.backward()
    grads=[p.grad for p in model.trend_head.parameters()]
    assert any(g is not None and bool(g.abs().sum()>0) for g in grads)==train_b
    torch.testing.assert_close(before,model.sequence(s,7)[0],rtol=0,atol=0)
