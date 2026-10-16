import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from methods.latentloop.modules.geometry_target_followup import ARMS, MeanCapacityPool, configure_model
from methods.latentloop.modules.local_geometry_input import LocalGeometryPool
from methods.latentloop.modules.attention_block_isolation import build_model
from test_attention_block_isolation import source
from test_visual_token_transport import warm
from test_visual_readout_distillation import parent, inputs, unroll, cpu_threads


def test_capacity_budget_and_initialization():
    capacity, geometry = MeanCapacityPool(), LocalGeometryPool(64, 'subpatch_projection')
    assert sum(p.numel() for p in capacity.parameters()) == sum(p.numel() for p in geometry.parameters()) == 65536
    x = torch.randn(2,64,24,24)
    torch.testing.assert_close(capacity(x),F.avg_pool2d(x,4,4),rtol=0,atol=0)
    a, b = torch.zeros(1,64,4,4), torch.zeros(1,64,4,4)
    a[:,:,0,0], b[:,:,3,3] = 1,1
    with torch.no_grad():
        capacity.correction[-1].weight.normal_()
    torch.testing.assert_close(capacity(a),capacity(b),rtol=0,atol=0)


@pytest.mark.parametrize('arm',ARMS)
def test_online_parity_target_source_and_frozen_gradient(source,arm,monkeypatch):
    original = build_model(*source,'visual_predictor_only').eval()
    model = configure_model(copy.deepcopy(original),arm)
    data = inputs()
    model.feature_provider = original.feature_provider = lambda age,images:data[3][age]
    with torch.no_grad():
        _, expected = unroll(original,data)
        _, actual = unroll(model,data)
    for (a,_),(b,_) in zip(expected,actual):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    model.distill_enabled = True
    model.teacher_previous = data[0][0] + 1.
    queries = []
    hook = model.teacher_visual.register_forward_pre_hook(lambda m,a:queries.append(a[0].clone()))
    _, actual = unroll(model,data)
    hook.remove()
    for i in range(0,len(queries),3):
        torch.testing.assert_close(queries[i],queries[i+1 if arm=='capacity_only' else i+2],rtol=0,atol=0)
    loss = actual[-1][0].square().mean() + torch.stack(model.attention_losses).mean()
    loss.backward()
    assert all(p.requires_grad == n.startswith('visual.predictor.') for n,p in model.named_parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0
    if arm == 'capacity_only':
        assert model.visual.predictor.encoder[-1].correction[-1].weight.grad.abs().sum() > 0
    elif arm == 'subpatch_original_target':
        assert model.visual.predictor.encoder[-1].projection.weight.grad.abs().sum() > 0
    model.distill_enabled = False
    monkeypatch.setattr(model.teacher_visual,'forward',lambda *a:pytest.fail('Unexpected online teacher'))
    with torch.no_grad():
        unroll(model,data)


@pytest.fixture
def campaign(monkeypatch,tmp_path):
    from tools.simvla import geometry_target_campaign as c
    for name in ('training','OUTPUT','INCOMING','ARMS','MODULE','jobs','evaluate'):
        monkeypatch.setattr(c.base,name,getattr(c.base,name))
    for name in ('training','MODULE'):
        monkeypatch.setattr(c.template,name,getattr(c.template,name))
    monkeypatch.setattr(c.reuse,'training',c.reuse.training)
    monkeypatch.setattr(c.training,'OUTPUT',tmp_path)
    monkeypatch.setattr(c.training.prior,'IS_RB2',False)
    c.configure()
    return c


def test_plan_has_runtime_checks_before_training_and_three_full_rows(campaign):
    c = campaign
    plan = c.jobs(dict(identity='test'))
    assert len(plan) == len({j['id'] for j in plan}) == 27
    assert len({j['summary'] for j in plan}) == 27
    for _,local in c.phases(plan):
        ids = {j['id'] for j in local}
        assert all(set(j['deps']).issubset(ids) for j in local)
    for arm in ARMS:
        train = next(j for j in plan if j['id']=='train_'+arm)
        assert train['deps'] == ['runtime_'+arm]
        assert sum(j['completion'].get('episodes',0) for j in plan if j['id'].startswith('eval_'+arm+'_')) == 500
    assert c.base.gpu_ids() == (4,5,6,7)


def test_run_automatically_continues_after_three_experiments(campaign,monkeypatch,tmp_path):
    from tools.simvla import geometry_target_confirmation as f
    c = campaign
    monkeypatch.setattr(c.base,'prepare',lambda:dict(identity='test'))
    monkeypatch.setattr(c.base,'audit',lambda _:None)
    monkeypatch.setattr(c,'summarize',lambda:dict(rows={},complete=True))
    monkeypatch.setattr(f,'OUTPUT',tmp_path/'confirmation')
    monkeypatch.setattr(f,'prepare',lambda _:dict(identity='confirm'))
    monkeypatch.setattr(f,'summarize',lambda:None)
    calls=[]
    def queue(output,plan,**kwargs):
        assert kwargs['gpus'] == (4,5,6,7)
        calls.append(plan)
        c.base.write_json(output/'queue_status.json',dict(completed=[j['id'] for j in plan],failed={}))
    monkeypatch.setattr(c.base.queue,'run_queue',queue)
    assert c.run() == 0
    assert len(calls)==6
    state = c.base.read_json(tmp_path/'queue_status.json')
    assert state['total_jobs']==39 and len(state['completed'])==39
    assert len(calls[-1])==len(calls[-2])==6


def test_confirmation_selection_and_job_counts():
    from tools.simvla import geometry_target_confirmation as f
    rows=dict(mean_pool_control=dict(successes=500,policy_ms_per_action=1.),
        a=dict(successes=400,policy_ms_per_action=12.),b=dict(successes=401,policy_ms_per_action=15.),
        c=dict(successes=401,policy_ms_per_action=11.))
    assert f.select_candidate(rows)=='c'
    plan=f.jobs(dict(identity='confirm'))
    assert len(plan)==len({j['id'] for j in plan})==12
    for arm in f.ARMS:
        local=[j for j in plan if j['arm']==arm]
        assert all(set(j['deps']).issubset({j['id'] for j in local}) for j in local)
        assert sum(j['completion'].get('episodes',0) for j in local if j['id'].endswith(('_0','_1','_2','_3')))==500


def test_confirmation_updates_manifest_and_actual_policy_config(monkeypatch,tmp_path):
    from tools.simvla import geometry_target_confirmation as f
    from tools.simvla.compile_benchmark import write_json
    c=dict(confirmation=dict(action_noise_seed_base=12),manifest=dict(action_noise_seed_base=11),
           model=dict(config=dict(action_noise_seed_base=11)),identity='parent')
    monkeypatch.setattr(f.training,'checked',lambda:c)
    monkeypatch.setattr(f.geometry,'checked',lambda:dict(identity='old'))
    monkeypatch.setattr(f.geometry,'OUTPUT',tmp_path/'geometry')
    monkeypatch.setattr(f,'OUTPUT',tmp_path/'confirmation')
    write_json(tmp_path/'geometry/contract.json',dict(identity='old'))
    for arm in ('mean_pool_control','subpatch_projection'):
        write_json(tmp_path/'geometry/train'/arm/'summary.json',dict(identity='old',
            verdict='TRAIN_COMPLETE',checkpoint_sha256=arm))
    rows={arm:dict(successes=400+i,policy_ms_per_action=12.) for i,arm in enumerate(('mean_pool_control','subpatch_projection'))}
    result=f.prepare(dict(rows=rows,complete=False))
    assert result['manifest']['action_noise_seed_base']==result['model']['config']['action_noise_seed_base']==12
    assert c['manifest']['action_noise_seed_base']==11
    assert result['sources']['selected']['arm']=='subpatch_projection'


def test_confirmation_rejects_runtime_seed_mismatch(monkeypatch,tmp_path):
    from tools.simvla import geometry_target_confirmation as f
    from tools.simvla import attention_query_eval as evaluation
    manifest=dict(action_noise_seed_base=12,episodes=[dict(task_id=t,trial_id=n) for t in range(10) for n in range(50)])
    c=dict(manifest=manifest,sources=dict(control=dict(arm='mean_pool_control')))
    monkeypatch.setattr(f,'checked',lambda:c)
    monkeypatch.setattr(f.base,'OUTPUT',tmp_path)
    monkeypatch.setattr(evaluation.prior,'build_policy',lambda *a:(SimpleNamespace(action_noise_seed_base=11),None))
    monkeypatch.setattr(evaluation,'evaluate',lambda *a,**kw:evaluation.prior.build_policy('recurrent_k4',{}))
    with pytest.raises(RuntimeError,match='Runtime policy'):
        f.evaluate('control',0,True)


def test_confirmation_reuses_checkpoint_without_retraining(monkeypatch):
    from tools.simvla import geometry_target_confirmation as f
    c=dict(sources=dict(selected=dict(provider='followup',arm='mean_original_target',checkpoint_sha256='fixed')))
    monkeypatch.setattr(f.training,'checked',lambda:dict(identity='train'))
    monkeypatch.setattr(f.training,'model_payload',lambda contract,arm:({'arm':arm},'fixed'))
    value,checksum=f.model_payload(c,'selected')
    assert value['arm']=='mean_original_target' and checksum=='fixed'
