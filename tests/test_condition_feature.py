import copy
import json

import pytest
import torch
from torch.nn import functional as F

from methods.latentloop.modules.condition_feature_supervision import ConditionFeatureSupervision, MODES
from methods.latentloop.modules.condition_output_split import ConditionOutputSplit
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from tools.simvla.condition_output_split_train import unroll, load_continuation


def sample():
    return dict(anchor_condition=torch.randn(1,4,16,requires_grad=True),
        teacher_conditions=torch.randn(1,7,4,16,requires_grad=True),
        valid_mask=torch.tensor([[True,True,True,False]]),group_ids=torch.tensor([[1,2,4,0]]))


@pytest.mark.parametrize('mode',['absolute','delta','both'])
def test_only_image_targets_and_encoder_code_receive_gradients(mode):
    s=sample();reader=ConditionFeatureSupervision(8,16,4)
    codes=[torch.randn(1,8,requires_grad=True) for _ in range(3)]
    loss,values=reader.objective(codes,s,mode)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(x.grad is not None and x.grad.abs().sum()>0 for x in codes)
    assert s['anchor_condition'].grad is None and s['teacher_conditions'].grad is None
    changed={**s,'teacher_conditions':s['teacher_conditions'].detach().clone()}
    changed['teacher_conditions'][:,:,2:]+=1000*torch.randn_like(changed['teacher_conditions'][:,:,2:])
    other,_=reader.objective([x.detach() for x in codes],changed,mode)
    torch.testing.assert_close(loss.detach(),other,rtol=0,atol=0)


def test_consecutive_delta_target_and_combined_loss():
    s=sample();reader=ConditionFeatureSupervision(8,16,4)
    codes=[torch.randn(1,8) for _ in range(3)]
    loss,values=reader.objective(codes,s,'both')
    expected=[];previous=F.layer_norm(s['anchor_condition'].detach(),(16,))
    for j,code in enumerate(codes):
        current=F.layer_norm(s['teacher_conditions'][:,j].detach(),(16,))
        prediction=reader(code,4)['delta']
        expected.append((prediction[:,:2]-(current-previous)[:,:2]).square().mean())
        previous=current
    torch.testing.assert_close(values['delta'],torch.stack(expected).mean())
    torch.testing.assert_close(loss,(values['absolute']+values['delta'])/2)


def test_reader_does_not_advance_rng_or_enter_deployed_model():
    torch.set_num_threads(1);torch.manual_seed(7)
    parent=NativeSimVLAV0(condition_dim=16,max_tokens=4)
    model=ConditionOutputSplit(parent,'carry_base',gradient_mode='joint')
    before=copy.deepcopy(model.state_dict());rng=torch.random.get_rng_state().clone()
    reader=ConditionFeatureSupervision(128,16,4)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all('reader' not in k for k in model.state_dict())
    assert all(id(p) not in {id(x) for x in model.parameters()} for p in reader.parameters())


def test_code_capture_preserves_forward_and_is_removed_after_unroll():
    torch.set_num_threads(1)
    parent=NativeSimVLAV0(condition_dim=16,max_tokens=4)
    torch.nn.init.normal_(parent.condition_updater.up.weight,std=.01)
    m=ConditionOutputSplit(parent,'carry_base',gradient_mode='joint')
    s=sample();s['image_sequence']=torch.rand(1,8,2,3,64,64);s['proprio_sequence']=torch.randn(1,8,8)
    a=unroll(m,s,3,4);codes=[];b=unroll(m,s,3,4,codes=codes)
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    assert len(codes)==3 and all(x.shape==(1,128) for x in codes)
    assert not m.delta_encoder._forward_hooks
    reader=ConditionFeatureSupervision(128,16,4)
    loss,_=reader.objective(codes,s,'both');loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.delta_encoder.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in m.condition_updater.parameters())
    assert all(p.grad is None for p in m.action_condition_updater.parameters())


def test_generic_continuation_source_uses_declared_steps_and_solver(monkeypatch,tmp_path):
    from tools.simvla import condition_output_split_train as train
    from tools.simvla import condition_output_split_eval as ev
    summary=tmp_path/'summary.json'
    summary.write_text(json.dumps(dict(identity='source',steps=10000,verdict='TRAIN_AND_OFFLINE_COMPLETE',
        checkpoint='source.pt',checkpoint_sha256='hash',training_seconds=120)))
    monkeypatch.setattr(train,'sha',lambda path:'hash')
    captured={}
    def load(path,arm,identity,**kwargs):
        captured.update(kwargs);return dict(model={'weight':torch.ones(1,1)},contract={'initialization':'fresh'})
    monkeypatch.setattr(ev,'load_payload',load)
    c=dict(initialization='continuation',continuation_source_steps=10000,
        continuation_source_action_mode='naive1',initial_models={'carry_base':dict(summary=str(summary),identity='source')})
    m=torch.nn.Linear(1,1,bias=False);d=load_continuation(m,c,'carry_base')
    assert captured==dict(steps=10000,action_mode='naive1') and d['prior_steps']==10000
    assert m.weight.item()==1


def test_queue_covers_four_training_controls_and_eight_unique_rows(monkeypatch,tmp_path):
    from tools.simvla import condition_feature_pipeline as p
    from tools.simvla.condition_feature_rb2 import ROWS
    monkeypatch.setattr(p,'identity',lambda c:'id')
    configs={m:dict(output=str(tmp_path/m),python='python') for m in MODES}
    plan=p.jobs(configs);ids={j['id'] for j in plan}
    assert len(plan)==len(ids)==24
    assert all(set(j.get('deps',())).issubset(ids) for j in plan)
    assert sum(j['completion'].get('steps')==5000 for j in plan)==4
    assert sum(j['completion'].get('episodes')==500 for j in plan)==8
    assert set(ROWS.values())=={(m,k) for m in MODES for k in (4,8)}
