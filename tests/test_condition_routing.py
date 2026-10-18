from types import SimpleNamespace

import pytest
import torch

from methods.latentloop.modules.condition_routing import ARMS, install, route


@pytest.mark.parametrize('arm',ARMS)
def test_route_exact_tensors_and_no_mutation(arm):
    predicted=torch.full((1,3,4),2.)
    original=torch.full_like(predicted,7.)
    action,carry=route(predicted,original,arm)
    assert action is (original if arm in ('current_action','both') else predicted)
    assert carry is (original if arm in ('next_prediction','both') else predicted)
    assert torch.equal(predicted,torch.full_like(predicted,2.))


def test_route_invalid_inputs():
    with pytest.raises(ValueError):route(torch.zeros(1),torch.zeros(2),'both')
    with pytest.raises(ValueError):route(torch.zeros(1),torch.zeros(1),'unknown')


class Model:
    def prepare(self,*args):return args[-1]

    def predict(self,context,age,images,proprio):
        context.previous=context.previous+1
        context.age=age
        context.previous_images,context.previous_proprio=images,proprio
        return context.previous,dict(carried=context.previous,base=context.previous)


@pytest.mark.parametrize('arm',ARMS)
def test_recursive_input_and_current_decode_are_separate(arm):
    model=install(Model(),arm)
    context=SimpleNamespace(previous=torch.zeros(1),age=0,visual_memory='unchanged')
    for age in (1,2,3):
        value,_=model.predict(context,age,'images','proprio')
        assert value.item()==age
    model.routing_reference=torch.tensor([10.])
    value,_=model.predict(context,4,'current images','current proprio')
    assert value.item()==(10 if arm in ('current_action','both') else 4)
    assert context.previous.item()==(10 if arm in ('next_prediction','both') else 4)
    assert context.age==4 and context.visual_memory=='unchanged'
    assert context.previous_images=='current images'
    assert context.previous_proprio=='current proprio'
    assert model.routing_reference is None
    next_value,_=model.predict(context,5,'next images','next proprio')
    assert next_value.item()==(11 if arm in ('next_prediction','both') else 5)


def test_required_reference_bypass_and_duplicate_guard():
    model=install(Model(),'both')
    assert model.prepare(None,None,None,None,None,1) is None
    assert model.prepare(None,None,None,None,None,4)==4
    with pytest.raises(ValueError):install(model,'both')
    with pytest.raises(RuntimeError):model.predict(SimpleNamespace(previous=torch.zeros(1)),4,None,None)


def test_scope_restores_helpers():
    from tools.simvla import condition_routing_campaign as study
    before=(study.base.OUTPUT,study.base.checked,study.evaluation.attach_visual,study.evaluation.check)
    with study.scope():
        assert study.base.checked is study.checked
        assert study.evaluation.check is study.check
    assert before==(study.base.OUTPUT,study.base.checked,study.evaluation.attach_visual,study.evaluation.check)


def test_three_full_rows_use_disjoint_shards_and_output_paths():
    from tools.simvla import condition_routing_campaign as study
    plan=study.jobs(dict(identity='test',manifest=dict(action_noise_seed_base=123)))
    assert len(plan)==19
    assert len({j['summary'] for j in plan})==19
    for arm in ARMS:
        parts=[j for j in plan if j['id'].startswith(arm+'_') and j['completion'].get('episodes')==125]
        assert len(parts)==4
        assert sum(j['completion']['episodes'] for j in parts)==500
        assert all(j['deps']==[arm+'_k8_seed1_smoke'] for j in parts)
    assert not any('codex_outputs' in ' '.join(j['cmd']) for j in plan)


@pytest.mark.parametrize('row',('bridge_f2','bridge_f3'))
def test_bridge_only_campaign_never_uses_recurrent_evaluator(row,monkeypatch):
    from tools.simvla import matched_reference_seeds as study
    from tools.simvla import visual_memory_followup_eval as bridge
    monkeypatch.setattr(study,'ROWS',('bridge_f2','bridge_f3'))
    with study.runtime_scope(row,dict(model=dict(config={}),manifest=dict(action_noise_seed_base=1))) as active:
        assert active is bridge
        assert active.study.row_spec(row)==dict(kind='bridge',interval=int(row[-1]))
