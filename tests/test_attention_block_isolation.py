import copy

import pytest
import torch

from methods.latentloop.modules.attention_block_isolation import ARMS, build_model
from methods.latentloop.modules.visual_attention_alignment import AdaptedFreshCondition, AttentionStudent
from test_visual_token_transport import warm
from test_visual_readout_distillation import parent, inputs, unroll, cpu_threads


@pytest.fixture
def source(warm):
    teacher = AdaptedFreshCondition(warm).eval()
    student = AttentionStudent(teacher, joint=True).eval()
    with torch.no_grad():
        student.visual.predictor.appearance[-1].weight.add_(.002)
    return student, teacher


@pytest.mark.parametrize('arm', ARMS)
def test_exact_forward_targets_and_no_online_teacher(source, arm, monkeypatch):
    student, teacher = source
    model = build_model(student, teacher, arm)
    data = inputs()
    student.feature_provider = model.feature_provider = lambda age, images: data[3][age]
    for name, value in student.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    with torch.no_grad():
        _, reference = unroll(student, data)
        _, actual = unroll(model, data)
        for (a, _), (b, _) in zip(reference, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        student.distill_enabled = model.distill_enabled = True
        model.teacher_previous = data[0][0]
        unroll(student, data)
        unroll(model, data)
        assert len(model.attention_losses) == len(student.attention_losses) == 3
        for a, b in zip(model.attention_losses, student.attention_losses):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.distill_enabled = False
    monkeypatch.setattr(model.teacher_visual, 'forward', lambda *a: pytest.fail('Online teacher call'))
    with torch.no_grad():
        unroll(model, data)


@pytest.mark.parametrize('arm', ARMS)
def test_only_selected_block_changes_with_action_gradient(source, arm):
    model = build_model(*source, arm)
    data = inputs()
    model.feature_provider = lambda age, images: data[3][age]
    prefix = 'visual.predictor.' if arm == 'visual_predictor_only' else 'late.'
    assert all(p.requires_grad == n.startswith(prefix) for n, p in model.named_parameters())
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    # A fixed nonlinear action surrogate must pass gradients through the frozen
    # updater/visual block, rather than disabling its entire computation graph.
    _, results = unroll(model, data)
    prediction = results[-1][0]
    prediction.tanh().square().mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    for name, value in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    with torch.no_grad():
        _, actual = unroll(model, data, 3)
        _, expected = unroll(reference, data, 3)
    for (a, _), (b, _) in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_common_initial_weights(source):
    models = [build_model(*source, arm) for arm in ARMS]
    for name, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)


def test_training_scope_restores_driver_on_failure(monkeypatch):
    from tools.simvla import attention_block_train as training
    old = {name: getattr(training.driver, name) for name in ('OUTPUT', 'checked', 'initial_model', 'parent')}
    with pytest.raises(ValueError):
        with training.training_scope():
            assert training.driver.OUTPUT == training.OUTPUT
            assert training.driver.initial_model is training.initial_model
            raise ValueError('test')
    for name, value in old.items():
        assert getattr(training.driver, name) is value


@pytest.fixture
def campaign(monkeypatch, tmp_path):
    from tools.simvla import attention_block_campaign as c
    # Register original globals for teardown before configuring the shared runner.
    for name in ('training', 'OUTPUT', 'INCOMING', 'ARMS', 'MODULE', 'jobs', 'evaluate'):
        monkeypatch.setattr(c.base, name, getattr(c.base, name))
    for name in ('training', 'MODULE'):
        monkeypatch.setattr(c.template, name, getattr(c.template, name))
    monkeypatch.setattr(c.training, 'OUTPUT', tmp_path)
    monkeypatch.setattr(c.training.prior, 'IS_RB2', False)
    c.configure()
    return c


def test_queue_reuses_completed_joint_control(campaign):
    c = campaign
    jobs = c.template.jobs({'identity': 'test'})
    assert len(jobs) == len({j['id'] for j in jobs}) == 16
    assert c.base.gpu_ids() == (4, 5, 6, 7)
    for arm in ARMS:
        selected = [j for j in jobs if j['id'] == 'train_' + arm]
        assert len(selected) == 1 and selected[0]['completion']['steps'] == 10000
        rows = [j for j in jobs if j['id'].startswith('eval_' + arm + '_')]
        assert sum(j['completion']['episodes'] for j in rows) == 500
    assert all('attention_joint' not in j['id'] for j in jobs)
    assert all('codex_outputs' not in ' '.join(j['cmd']) for j in jobs)


def test_independent_arm_survives_first_failure(campaign, monkeypatch):
    c = campaign
    monkeypatch.setattr(c.base, 'prepare', lambda: {'identity': 'test'})
    monkeypatch.setattr(c.base, 'audit', lambda _: None)
    monkeypatch.setattr(c, 'wait_for_predecessor', lambda: [])
    monkeypatch.setattr(c, 'summarize', lambda: None)
    calls = []
    def run(output, plan, **kwargs):
        calls.append(plan)
        c.base.write_json(output / 'queue_status.json', dict(completed=[], failed={plan[0]['id']: 'test'}))
        return 1
    monkeypatch.setattr(c.base.queue, 'run_queue', run)
    assert c.run_ordered() == 1
    assert len(calls) == 2 and all(len(plan) == 8 for plan in calls)


@pytest.mark.parametrize('phase', ['running', 'waiting_for_predecessor', 'interrupted'])
def test_waits_for_all_predecessor_jobs(campaign, monkeypatch, tmp_path, phase):
    c = campaign
    monkeypatch.setattr(c.predecessor, 'OUTPUT', tmp_path)
    c.base.write_json(tmp_path / 'queue_status.json', dict(phase=phase))
    monkeypatch.setattr(c.base.queue, 'predecessor_pending', lambda _: False)
    assert not c.predecessor_complete()


@pytest.mark.parametrize('phase', ['complete', 'finished_with_failures'])
def test_completed_predecessor_must_release_lock(campaign, monkeypatch, tmp_path, phase):
    c = campaign
    monkeypatch.setattr(c.predecessor, 'OUTPUT', tmp_path)
    c.base.write_json(tmp_path / 'queue_status.json', dict(phase=phase))
    monkeypatch.setattr(c.base.queue, 'predecessor_pending', lambda _: True)
    assert not c.predecessor_complete()
    monkeypatch.setattr(c.base.queue, 'predecessor_pending', lambda _: False)
    assert c.predecessor_complete()


def test_pairing_rejects_missing_or_duplicate_episodes():
    from tools.simvla.attention_block_campaign import paired_changes
    old = [dict(task_id=t, trial_id=n, success=True) for t in range(10) for n in range(50)]
    new = copy.deepcopy(old)
    new[0]['success'] = False
    result = paired_changes(old, new)
    assert result['net_success_change'] == -1
    assert result['lost_successes'] == [[0, 0]]
    with pytest.raises(RuntimeError):
        paired_changes(old, new[:-1])
    with pytest.raises(RuntimeError):
        paired_changes(old, new[:-1] + [new[0]])
