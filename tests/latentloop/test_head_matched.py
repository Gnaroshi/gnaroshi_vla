from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.simvla import head_matched_pipeline as pipeline
from tools.simvla import head_matched_rb2 as rb2
from tools.simvla.rollout_state_repair import training_ages,training_action,sample_pool
from tools.simvla.trend_condition_eval import expected_counts
from tools.simvla.interval_followup_rb2 import expected_counts as compiled_counts
from architectures.simvla.adapters.dcld.simvla_action_adapter import SimVLAActionAdapter


def test_matched_axes_and_training_budget():
    s=pipeline.settings()
    assert len(s)==4 and set(s)==set(rb2.ROWS)
    for v,(k,mode) in pipeline.VARIANTS.items():
        assert s[v]==dict(train_trend=True,driver='student',sources=['previous'],training_k_c=k,generation_mode=mode)
        assert training_ages(s[v])==tuple(range(1,k))
        compiled_mode='ours_kc2_ng3' if mode=='learned' else 'condition_naive3'
        assert rb2.ROWS[v]==(k,compiled_mode)
    assert training_ages({})==tuple(range(1,8))
    with pytest.raises(ValueError): training_ages({'training_k_c':2})


@pytest.mark.parametrize('smoke',[True,False])
def test_four_gpu_queue_no_new_collection(smoke):
    jobs=pipeline.jobs(dict(output='/tmp/test',python='/python'),Path('/config'),smoke)
    train=[j for j in jobs if j['id'].startswith('train_')]
    assert len(train)==4 and all(j['deps']==[] for j in train)
    assert len([j for j in jobs if j['id'].startswith('collect')])==0
    evaluations=[j for j in jobs if j['id'].startswith('eval_')]
    assert len(evaluations)==4
    for v,(k,_) in pipeline.VARIANTS.items():
        j=next(j for j in evaluations if j['id']==f'eval_kc{k}_{v}')
        assert j['deps']==['train_'+v]
        assert j['cmd'][j['cmd'].index('--k-c')+1]==str(k)
    if not smoke:
        assert next(j for j in jobs if j['id']=='export_rb2')['deps']==['train_'+v for v in pipeline.VARIANTS]


def test_no_heldout_or_new_round_data_in_training():
    records=[dict(age=age,metadata=dict(split=split,driver=driver,collection_source=source))
        for age in range(1,8) for split in ('train','heldout')
        for driver in ('original','student') for source in ('previous','current')]
    for settings in pipeline.settings().values():
        for age in training_ages(settings):
            chosen=sample_pool(records,settings,age)
            assert len(chosen)==1
            assert chosen[0]['metadata']==dict(split='train',driver='student',collection_source='previous')


class ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(.3),requires_grad=False)
        self.times=[]

    def forward(self,vlm_features,action_with_noise,proprio,t):
        self.times.append(float(t[0]))
        return action_with_noise*self.weight+vlm_features.mean()+proprio.mean()+t[:,None,None]


def test_naive_train_uses_official_three_step_latent_and_input_gradients():
    transformer=ToyTransformer()
    model=nn.Module(); model.transformer=transformer
    model.action_space=SimpleNamespace(postprocess=lambda x: x,dim_action=7)
    model.num_actions=10
    adapter=SimVLAActionAdapter(model)
    condition=torch.ones(1,4,8,requires_grad=True)
    sample=dict(proprio=torch.zeros(1,8),noise=torch.ones(1,10,7))
    actual=training_action({'generation_mode':'naive3'},adapter,None,None,condition,sample)
    assert transformer.times==pytest.approx([1.,2/3,1/3])
    expected=adapter.decode_action_from_condition(condition,sample['proprio'],steps=3,
        initial_noise=sample['noise'],return_debug=True).final_action_latent
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.square().mean().backward()
    assert condition.grad.abs().sum()>0 and transformer.weight.grad is None


@pytest.mark.parametrize('q',[1,4,5,8,9,180])
def test_sd1_and_rb2_count_same_operators(q):
    for row,(k,mode) in pipeline.VARIANTS.items():
        eager=expected_counts('frozen_trend_residual',q,k,generation_mode=mode)
        compiled=compiled_counts(row,q,rows=rb2.ROWS)
        assert eager['transformer']==compiled['num_action_transformer_calls']==3*q
        assert eager['generation']==compiled['num_generation_decoder_only_steps']
        assert eager['full_vlm']==compiled['num_full_vlm_calls']
        assert eager['observation']==compiled['num_observation_encoder_calls']


def test_transfer_payload_must_match_training_solver(monkeypatch):
    settings=pipeline.settings()
    value=dict(variant_settings=settings,selected_arm='frozen_trend_residual',source_identity='locked',
        checkpoints={row:dict(file=row+'.pt',sha256='hash',step=3000) for row in settings})
    def load(path,**kwargs):
        row=path.stem
        return dict(identity='locked',repair_variant=row,step=3000,contract=dict(training_data=settings[row]))
    monkeypatch.setattr(torch,'load',load)
    for row in settings:
        assert rb2.checkpoint_spec(value,row)['step']==3000
    value['variant_settings']['k4_naive3']['generation_mode']='learned'
    with pytest.raises(RuntimeError,match='mode/interval'):
        rb2.checkpoint_spec(value,'k4_naive3')
