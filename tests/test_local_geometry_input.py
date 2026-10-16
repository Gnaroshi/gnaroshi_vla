import copy

import pytest
import torch
from torch.nn import functional as F

from methods.latentloop.modules.local_geometry_input import ARMS, LocalGeometryPool, configure_model
from methods.latentloop.modules.attention_block_isolation import build_model
from test_attention_block_isolation import source
from test_visual_token_transport import warm
from test_visual_readout_distillation import parent, inputs, unroll, cpu_threads


@pytest.mark.parametrize('mode', ARMS[1:])
def test_zero_initialization_exact_and_constant_invariant(mode):
    layer = LocalGeometryPool(3, mode)
    x = torch.randn(4,3,24,24)
    torch.testing.assert_close(layer(x), F.avg_pool2d(x,4,4), rtol=0, atol=0)
    with torch.no_grad():
        layer.projection.weight.normal_()
    x = torch.ones(4,3,24,24)
    torch.testing.assert_close(layer(x), F.avg_pool2d(x,4,4), rtol=0, atol=0)


@pytest.mark.parametrize('mode', ARMS[1:])
def test_preserves_subpatch_location_discarded_by_mean(mode):
    layer = LocalGeometryPool(1, mode)
    left, right = torch.zeros(1,1,4,4), torch.zeros(1,1,4,4)
    left[0,0,1,0] = right[0,0,1,3] = 1
    torch.testing.assert_close(F.avg_pool2d(left,4),F.avg_pool2d(right,4),rtol=0,atol=0)
    with torch.no_grad():
        layer.projection.weight[0,0 if mode == 'spatial_moments' else 4,0,0] = 1
    assert not torch.equal(layer(left),layer(right))


def test_full_subpatch_can_resolve_same_first_moments():
    moment, full = LocalGeometryPool(1,'spatial_moments'), LocalGeometryPool(1,'subpatch_projection')
    diagonal, anti = torch.zeros(1,1,4,4), torch.zeros(1,1,4,4)
    diagonal[0,0,0,0] = diagonal[0,0,3,3] = 1
    anti[0,0,0,3] = anti[0,0,3,0] = 1
    with torch.no_grad():
        moment.projection.weight.fill_(1)
        full.projection.weight[0,0,0,0] = 1
    torch.testing.assert_close(moment(diagonal),moment(anti),rtol=0,atol=0)
    assert not torch.equal(full(diagonal),full(anti))


@pytest.mark.parametrize('arm,extra', tuple(zip(ARMS,(0,8192,65536))))
def test_model_parity_parameters_targets_and_gradients(source,arm,extra,monkeypatch):
    baseline = build_model(*source,'visual_predictor_only').eval()
    model = configure_model(copy.deepcopy(baseline),arm)
    data = inputs()
    baseline.feature_provider = model.feature_provider = lambda age,images:data[3][age]
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()) == extra
    assert all(p.requires_grad == n.startswith('visual.predictor.') for n,p in model.named_parameters())
    assert model.arm == baseline.arm == 'student_query_control'
    with torch.no_grad():
        _, original = unroll(baseline,data)
        _, actual = unroll(model,data)
        for (a,_),(b,_) in zip(original,actual):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        baseline.distill_enabled = model.distill_enabled = True
        baseline.teacher_previous = model.teacher_previous = data[0][0]
        unroll(baseline,data)
        unroll(model,data)
        for a,b in zip(baseline.attention_losses,model.attention_losses):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    model.distill_enabled = False
    monkeypatch.setattr(model.teacher_visual,'forward',lambda *a:pytest.fail('Online teacher call'))
    before = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    _, actual = unroll(model,data)
    actual[-1][0].tanh().square().mean().backward()
    if extra:
        grad = model.visual.predictor.encoder[-1].projection.weight.grad
        assert grad is not None and grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    opt.step()
    for n,p in model.named_parameters():
        if n in before:
            torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    with torch.no_grad():
        _, actual = unroll(model,data,3)
    for (a,_),(b,_) in zip(original[:3],actual):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('mode',ARMS[1:])
def test_pool_compile_and_checkpoint_roundtrip(mode):
    model = LocalGeometryPool(3,mode)
    x = torch.randn(2,3,24,24)
    compiled = torch.compile(model,backend='eager',fullgraph=True)
    torch.testing.assert_close(compiled(x),model(x),rtol=0,atol=0)
    restored = LocalGeometryPool(3,mode)
    restored.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)


def test_training_scope_restored():
    from tools.simvla import local_geometry_train as training
    old = training.driver.OUTPUT
    with pytest.raises(ValueError):
        with training.training_scope():
            assert training.driver.OUTPUT == training.OUTPUT
            raise ValueError('test')
    assert training.driver.OUTPUT == old


@pytest.fixture
def campaign(monkeypatch,tmp_path):
    from tools.simvla import local_geometry_campaign as c
    for name in ('training','OUTPUT','INCOMING','ARMS','MODULE','jobs','evaluate'):
        monkeypatch.setattr(c.base,name,getattr(c.base,name))
    for name in ('training','MODULE'):
        monkeypatch.setattr(c.template,name,getattr(c.template,name))
    monkeypatch.setattr(c.reuse,'training',c.reuse.training)
    monkeypatch.setattr(c.training,'OUTPUT',tmp_path)
    monkeypatch.setattr(c.training.prior,'IS_RB2',False)
    c.configure()
    return c


def test_three_independent_training_then_ordered_1500_episodes(campaign):
    c = campaign
    plan = c.template.jobs({'identity':'test'})
    assert len(plan) == len({j['id'] for j in plan}) == 24
    phases = c.phases(plan)
    assert [len(jobs) for _,jobs in phases] == [6,8,8,8]
    for _,jobs in phases:
        ids = {j['id'] for j in jobs}
        assert all(set(j['deps']).issubset(ids) for j in jobs)
    for arm in ARMS:
        rows = [j for j in plan if j['id'].startswith('eval_'+arm+'_')]
        assert sum(j['completion']['episodes'] for j in rows) == 500
    assert c.base.gpu_ids() == (4,5,6,7)
    assert all('codex_outputs' not in ' '.join(j['cmd']) for j in plan)


def test_followups_survive_independent_job_failure(campaign,monkeypatch):
    c = campaign
    monkeypatch.setattr(c.base,'prepare',lambda:{'identity':'test'})
    monkeypatch.setattr(c.base,'audit',lambda _:None)
    monkeypatch.setattr(c.reuse,'summarize',lambda:None)
    calls = []
    def run(output,plan,**kwargs):
        calls.append(plan)
        c.base.write_json(output/'queue_status.json',dict(completed=[],failed={plan[0]['id']:'test'}))
        return 1
    monkeypatch.setattr(c.base.queue,'run_queue',run)
    assert c.run() == 1
    assert len(calls) == 4


def test_initially_satisfied_goal_is_not_new_progress():
    from tools.simvla.local_geometry_evidence import classify
    record = dict(query_goal_progress=[dict(satisfied=[False,False,True])],
                  terminal_goal_progress=dict(satisfied=[False,False,True]))
    labels = classify(record)
    assert labels['some_at_end']
    assert not labels['newly_achieved_after_start']


def test_losing_an_achieved_goal_without_simultaneous_success():
    from tools.simvla.local_geometry_evidence import classify
    record = dict(query_goal_progress=[dict(satisfied=s) for s in
                  [[False,False],[True,False],[False,True]]],
                  terminal_goal_progress=dict(satisfied=[False,True]))
    labels = classify(record)
    assert labels['newly_achieved_after_start'] and labels['lost_newly_achieved']
    assert labels['all_ever_not_simultaneous']


def test_runtime_smoke_loads_only_matching_two_step_checkpoint(monkeypatch,tmp_path):
    from tools.simvla import local_geometry_inference_smoke as runtime
    from tools.simvla.compile_benchmark import write_json,sha
    monkeypatch.setattr(runtime.training,'OUTPUT',tmp_path)
    folder = tmp_path/'train_smoke'/ARMS[0]
    folder.mkdir(parents=True)
    checkpoint = folder/'latest.pt'
    torch.save(dict(identity='test',arm=ARMS[0],step=2,model={}),checkpoint)
    summary = dict(verdict='TRAIN_SMOKE_PASS',identity='test',steps=2,
                   checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
    write_json(folder/'summary.json',summary)
    payload,_ = runtime.model_payload({'identity':'test'},ARMS[0])
    assert payload['step'] == 2
    with pytest.raises(RuntimeError):
        runtime.model_payload({'identity':'other'},ARMS[0])
    summary['steps'] = 10000
    write_json(folder/'summary.json',summary)
    with pytest.raises(RuntimeError):
        runtime.model_payload({'identity':'test'},ARMS[0])


def test_final_comparison_uses_equal_budget_control(campaign,monkeypatch,tmp_path):
    from tools.simvla import local_geometry_final_comparison as final
    c = campaign
    monkeypatch.setattr(c.base,'checked',lambda:dict(identity='test',matched_parent_sha256='parent'))
    monkeypatch.setattr(c.training.parent,'OUTPUT',tmp_path/'parent')
    c.base.write_json(tmp_path/'parent/rows/visual_predictor_only/summary.json',dict(success_rate=.794))
    summaries,records = {},{}
    for index,arm in enumerate(ARMS):
        summaries[arm] = dict(success_rate=(400+index)/500,policy_ms_per_action=12.+index)
        records[arm] = [dict(task_id=t,trial_id=n,success=(t*50+n)<400+index)
                        for t in range(10) for n in range(50)]
        c.base.write_json(tmp_path/'rows'/arm/'summary.json',summaries[arm])
    monkeypatch.setattr(c.base,'merged_result',lambda _,arm:(summaries[arm],records[arm]))
    result = final.summarize()
    assert result['complete'] and result['equal_budget_control']==ARMS[0]
    assert result['matched_comparisons'][ARMS[1]]['success_rate_difference_pp']==pytest.approx(.2)
    assert result['matched_comparisons'][ARMS[2]]['paired_outcomes']['net_success_change']==2
