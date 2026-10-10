from types import SimpleNamespace

import pytest
import torch

from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.condition_output_split import ConditionOutputSplit
from tools.simvla.condition_output_split_train import (
    action_target, teacher_steps, check_continuation_contract, sampling_step,
)
from tools.simvla.condition_output_split_eval import predictor_only_update, check_policy
from tools.simvla import action_target_alignment as pipeline


def test_original10_default_uses_cached_normalized_target():
    target = torch.randn(1, 10, 7)
    action = SimpleNamespace(action_space=SimpleNamespace(normalize_action=lambda t: t * 2))
    torch.testing.assert_close(action_target({}, action, dict(target_action=target)), target * 2)
    assert teacher_steps({}) == 10


def test_original1_target_uses_current_teacher_condition_same_noise_and_no_grad():
    condition = torch.randn(1, 4, 16, requires_grad=True)
    proprio = torch.randn(1, 8)
    noise = torch.randn(1, 10, 7)
    calls = []
    def decode(c, q, **kwargs):
        calls.append(kwargs)
        assert c is condition and q is proprio and kwargs['initial_noise'] is noise
        assert not torch.is_grad_enabled() and kwargs['requires_grad'] is False
        return SimpleNamespace(final_action_latent=noise + c.mean())
    s = dict(target_condition=condition, proprio=proprio, noise=noise)
    out = action_target(dict(teacher_steps=1), SimpleNamespace(decode_action_from_condition=decode), s)
    assert not out.requires_grad and calls[0]['steps'] == 1
    torch.testing.assert_close(out, noise + condition.detach().mean())


@pytest.mark.parametrize('n', [True, 1.0, 2, 3, 0, '1'])
def test_only_explicit_supported_teacher_solvers(n):
    with pytest.raises(ValueError):
        teacher_steps(dict(teacher_steps=n))


def test_teacher_transition_is_explicit_and_data_cannot_change():
    old = {k: 'same' for k in ('data', 'heldout', 'batch_size', 'seed', 'source_checkpoint_sha256',
                             'condition_weight', 'current_action_gradient', 'future_condition_gradient')}
    old.update(teacher_steps=10, action_mode='naive1', training_intervals=[4, 8])
    new = {**old, 'teacher_steps': 1}
    approved = dict(teacher_transition=dict(source=10, target=1))
    check_continuation_contract(old, new, approved)
    with pytest.raises(RuntimeError):
        check_continuation_contract(old, new, {})
    with pytest.raises(RuntimeError):
        check_continuation_contract(old, {**new, 'data': 'different'}, approved)
    with pytest.raises(RuntimeError):
        check_continuation_contract(new, old, approved)


def test_teacher_selection_does_not_change_window_or_age_sampling():
    for n in range(1, 30):
        base = dict(seed=7, sample_step_offset=10000, training_intervals=[4, 8])
        r1, k1, a1 = sampling_step({**base, 'teacher_steps': 1}, n)
        r10, k10, a10 = sampling_step({**base, 'teacher_steps': 10}, n)
        assert (k1, a1, [r1.randrange(1000) for _ in range(2)]) == (k10, a10, [r10.randrange(1000) for _ in range(2)])


def test_predictor_only_has_identical_recurrence_and_zero_second_head_calls():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = ConditionOutputSplit(NativeSimVLAV0(condition_dim=16, max_tokens=8), 'carry_base').eval()
    previous = torch.randn(1, 4, 16)
    valid = torch.ones(1, 4, dtype=torch.bool)
    groups = torch.zeros(1, 4, dtype=torch.long)
    count = []
    hook = model.action_condition_updater.register_forward_hook(lambda *_: count.append(1))
    for age in (1, 2, 3):
        code = torch.randn(1, 128)
        with torch.no_grad():
            _, reference, carried = model.update(previous, code, valid_mask=valid, group_ids=groups, age=age)
            before = len(count)
            output, base, next_previous = predictor_only_update(model, previous, code, valid_mask=valid, group_ids=groups, age=age)
        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(next_previous, carried)
        assert len(count) == before and output is base and base is next_previous
        previous = next_previous
    hook.remove()


def test_predictor_only_counter_rejects_an_extra_forward():
    counts = dict(num_policy_queries=9, num_full_vlm_calls=3, num_condition_updater_calls=6,
                  num_action_condition_updater_calls=0, num_action_transformer_calls=9)
    policy = SimpleNamespace(metrics=SimpleNamespace(counters=counts), k_c=4, nfe=1, step_index=45,
                             _uses_action_condition_updater=False,
                             _condition_component_calls=dict(observation_encoder=6, condition_updater=6))
    assert check_policy(policy)['action_condition_updater'] == 0
    policy._condition_component_calls['action_condition_updater'] = 1
    with pytest.raises(RuntimeError):
        check_policy(policy)


def test_queue_has_two_new_trainings_nine_evals_and_reuses_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'identity', lambda c: 'test')
    configs = {m: dict(output=str(tmp_path / m), python='python') for m in (*pipeline.MODES, 'predictor_only')}
    plan = pipeline.jobs(configs)
    ids = {j['id'] for j in plan}
    assert len(plan) == len(ids) == 18
    assert sum(j['completion'].get('episodes') == 500 for j in plan) == 9
    assert sum(j['id'].endswith('_train') and 'smoke' not in j['id'] for j in plan) == 2
    assert all(set(j.get('deps', ())).issubset(ids) for j in plan)
    assert all('fresh_fixed' not in j['id'] and 'fresh_joint' not in j['id'] for j in plan)
