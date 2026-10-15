import copy
from types import SimpleNamespace

import pytest
import torch

from methods.latentloop.modules.attention_query_contract import ARMS, AttentionQueryCondition
from methods.latentloop.modules.visual_attention_alignment import AdaptedFreshCondition, AttentionStudent
from test_visual_token_transport import warm
from test_visual_readout_distillation import parent, inputs, unroll, cpu_threads


@pytest.fixture
def trained(warm):
    teacher=AdaptedFreshCondition(warm).eval()
    student=AttentionStudent(teacher,joint=True).eval()
    with torch.no_grad():
        for p in student.late.parameters():
            p.add_(.001)
        student.visual.predictor.appearance[-1].weight.add_(.002)
    return student,teacher


@pytest.mark.parametrize('arm', ARMS)
def test_matched_weights_early_parity_and_online_vision(trained,arm,monkeypatch):
    student,teacher=trained
    model=AttentionQueryCondition(student,teacher,arm)
    data=inputs();calls=[]
    model.feature_provider=lambda age,images:(calls.append(age) or data[3][age])
    student.feature_provider=lambda age,images:data[3][age]
    for name,value in student.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
    monkeypatch.setattr(model.teacher_visual,'forward',lambda *args: (_ for _ in ()).throw(AssertionError('online teacher')))
    with torch.no_grad():
        _,actual=unroll(model,data)
        _,expected=unroll(student,data)
    assert calls==[4]
    assert model.teacher_previous is None
    through=3 if arm=='refresh_query_reader' else 7
    for (a,_),(b,_) in zip(actual[:through],expected[:through]):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert [i for i in range(8) if model.uses_vision(i)]==[4]
    assert all(p.requires_grad==n.startswith(('late.','visual.predictor.')) for n,p in model.named_parameters())


@pytest.mark.parametrize('arm',ARMS)
def test_target_source_and_gradients(trained,arm):
    model=AttentionQueryCondition(*trained,arm)
    data=inputs();model.feature_provider=lambda age,images:data[3][age]
    model.distill_enabled=True
    context=model.prepare(*data[0])
    for age in range(1,5):model.predict(context,age,data[1][age],data[2][age])
    previous=context.previous.detach().clone()
    queries=[]
    hook=model.teacher_visual.register_forward_pre_hook(lambda m,a:queries.append(a[0].clone()))
    model.teacher_previous=data[0][0]+1.
    output,_=model.predict(context,5,data[1][5],data[2][5])
    hook.remove()
    assert len(queries)==3
    torch.testing.assert_close(queries[0],queries[1 if arm=='student_query_control' else 2],atol=0,rtol=0)
    assert not torch.equal(queries[1],queries[2])
    assert not any(q.requires_grad for q in queries)
    loss=output.square().mean()+torch.stack(model.attention_losses).mean()
    loss.backward()
    assert model.late.condition_updater.up.weight.grad.abs().sum()>0
    assert model.visual.predictor.appearance[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert not torch.equal(previous,output.detach())


@pytest.mark.parametrize('arm',ARMS)
def test_teacher_target_dependency_on_previous_prediction(trained,arm):
    model=AttentionQueryCondition(*trained,arm)
    data=inputs();model.feature_provider=lambda age,images:data[3][age]
    model.distill_enabled=True;model.teacher_previous=data[0][0]
    context=model.prepare(*data[0])
    with torch.no_grad():
        for age in range(1,5):model.predict(context,age,data[1][age],data[2][age])
        changed=copy.deepcopy(context)
        changed.previous=changed.previous+torch.randn_like(changed.previous)*.5
        targets=[];queries=[];updater_inputs=[]
        h=model.teacher_visual.register_forward_hook(lambda m,a,o:targets.append(o.clone()))
        q=model.visual.register_forward_pre_hook(lambda m,a:queries.append(a[0].clone()))
        u=model.late.condition_updater.register_forward_pre_hook(lambda m,a:updater_inputs.append(a[0].clone()))
        a,_=model.predict(context,5,data[1][5],data[2][5])
        b,_=model.predict(changed,5,data[1][5],data[2][5])
        h.remove();q.remove();u.remove()
    assert torch.equal(targets[0],targets[3]) == (arm!='student_query_control')
    assert torch.equal(queries[0],queries[1]) == (arm=='refresh_query_reader')
    assert not torch.equal(updater_inputs[0],updater_inputs[1])
    assert not torch.equal(a,b)


def test_training_original_previous_index():
    from tools.simvla.attention_query_train import unroll as train_unroll
    class Stub:
        def prepare(self,*args):return None
        def predict(self,c,j,images,proprio):
            seen.append(self.teacher_previous)
            return j,{'base':j}
    seq=dict(anchor_condition='refresh',teacher_conditions=torch.arange(7)[None],
        image_sequence=torch.zeros(1,8),proprio_sequence=torch.zeros(1,8),valid_mask=None,group_ids=None)
    seen=[];model=Stub();train_unroll(model,seq,7)
    assert seen[0]=='refresh'
    assert [int(x) for x in seen[1:]]==list(range(6))
    assert model.teacher_previous is None


def test_queue_three_independent_arms_and_all_500_ids(monkeypatch,tmp_path):
    from tools.simvla import attention_query_campaign as c
    monkeypatch.setattr(c.base,'OUTPUT',tmp_path)
    monkeypatch.setattr(c.base,'ARMS',ARMS)
    monkeypatch.setattr(c.training,'OUTPUT',tmp_path)
    monkeypatch.setattr(c.base.prior,'IS_RB2',False)
    jobs=c.jobs({'identity':'test'});byid={j['id']:j for j in jobs}
    assert len(byid)==len(jobs)==24
    assert c.base.gpu_ids()==(4,5,6,7)
    manifest={'episodes':[{'task_id':t,'trial_id':r} for t in range(10) for r in range(50)]}
    ids=[]
    for shard in range(4):
        ids.extend(c.base.selected_manifest(manifest,shard,4)['episodes'])
    assert len({(x['task_id'],x['trial_id']) for x in ids})==500
    for i,arm in enumerate(ARMS):
        assert sum(byid[f'eval_{arm}_{s}']['completion']['episodes'] for s in range(4))==500
        if i:assert byid['smoke_'+arm]['deps']==['train_'+ARMS[i-1]]
        assert all('codex_outputs' not in x for x in byid['train_'+arm]['cmd'])
