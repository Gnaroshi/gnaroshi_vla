import copy
import pytest
import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.condition_output_split import ConditionOutputSplit
from tools.simvla.condition_output_split_train import configure_trainable_modules,condition_predictor_hash,check_continuation_contract
from tools.simvla.fixed_condition_pipeline import combine_models,MODES,HYBRIDS,jobs


def test_frozen_predictor_and_only_action_head_gradient():
    torch.set_num_threads(1);torch.manual_seed(7)
    m=ConditionOutputSplit(NativeSimVLAV0(condition_dim=16,max_tokens=8),'carry_base')
    trainable=configure_trainable_modules(m,dict(freeze_condition_predictor=True))
    assert {id(p) for p in trainable}=={id(p) for p in m.action_condition_updater.parameters()}
    h=condition_predictor_hash(m)
    opt=torch.optim.AdamW(trainable,lr=1e-3)
    previous=torch.randn(1,4,16);code=torch.randn(1,128)
    for age in (1,2,3):
        opt.zero_grad();out,base,previous=m.update(previous,code,valid_mask=torch.ones(1,4,dtype=torch.bool),group_ids=torch.zeros(1,4,dtype=torch.long),age=age)
        assert not base.requires_grad
        out.square().mean().backward();opt.step()
        assert all(p.grad is None for p in m.delta_encoder.parameters())
        assert all(p.grad is None for p in m.condition_updater.parameters())
    assert condition_predictor_hash(m)==h
    with pytest.raises(ValueError):configure_trainable_modules(ConditionOutputSplit(NativeSimVLAV0(condition_dim=16,max_tokens=8),'carry_output'),dict(freeze_condition_predictor=True))


def test_default_training_retains_all_parameters():
    m=ConditionOutputSplit(NativeSimVLAV0(condition_dim=16,max_tokens=8),'carry_base')
    assert {id(p) for p in configure_trainable_modules(m,{})}=={id(p) for p in m.parameters()}


def test_only_declared_frozen_transition_is_allowed():
    previous={k:1 for k in ('data','heldout','batch_size','seed','teacher_steps','source_checkpoint_sha256','condition_weight','current_action_gradient')}
    previous.update(future_condition_gradient='carry_output can reach earlier extra heads; carry_base reaches base updater and encoder',action_mode='naive1',training_intervals=[4,8])
    current={**previous,'future_condition_gradient':'Frozen delta_encoder and condition_updater at every age'}
    check_continuation_contract(previous,current,dict(freeze_condition_predictor=True))
    with pytest.raises(RuntimeError):check_continuation_contract(previous,current,{})
    with pytest.raises(RuntimeError):check_continuation_contract(previous,{**current,'data':2},dict(freeze_condition_predictor=True))


def test_hybrid_exact_state_selection_and_inputs_unmodified():
    a={'delta_encoder.x':torch.tensor([1.]),'condition_updater.x':torch.tensor([2.]),'action_condition_updater.x':torch.tensor([3.])}
    b={k:v+10 for k,v in a.items()};before=copy.deepcopy(a)
    c=combine_models(a,b)
    assert c['delta_encoder.x'].item()==1 and c['condition_updater.x'].item()==2
    assert c['action_condition_updater.x'].item()==13
    for k in a:torch.testing.assert_close(a[k],before[k])
    with pytest.raises(ValueError):combine_models(a,{**b,'extra':torch.tensor(0.)})


def test_complete_nonduplicate_queue(tmp_path,monkeypatch):
    from tools.simvla import fixed_condition_pipeline as sd1,fixed_condition_rb2 as rb2
    monkeypatch.setattr(sd1,'identity',lambda c:'test')
    cfgs={m:dict(output=str(tmp_path/m),python='python') for m in (*MODES,*HYBRIDS)}
    plan=jobs(cfgs);ids={j['id'] for j in plan}
    assert len(ids)==len(plan)==25
    assert sum(j['completion'].get('episodes')==500 for j in plan)==11
    assert sum(j['id'].endswith('_train') and 'smoke' not in j['id'] for j in plan)==3
    assert all(set(j.get('deps',())).issubset(ids) for j in plan)
    assert len(rb2.jobs())==9
    assert all(j['completion']['episodes']==500 for j in rb2.jobs())
