from types import SimpleNamespace

import pytest
import torch

from tools.simvla import action_component_causality as study


@pytest.mark.parametrize('arm', study.ARMS)
def test_component_merge_preserves_inputs_and_requested_components(arm):
    predicted = torch.arange(70.).reshape(1,10,7)
    original = predicted + 100
    result = study.merge_action(predicted, original, arm)
    assert torch.equal(predicted, torch.arange(70.).reshape(1,10,7))
    assert torch.equal(original, predicted+100)
    assert torch.equal(result[..., :6], original[..., :6] if arm != 'gripper_midpoint' else predicted[..., :6])
    assert torch.equal(result[..., 6:], original[..., 6:] if arm != 'pose_midpoint' else predicted[..., 6:])


def test_invalid_inputs_and_temporal_coverage():
    with pytest.raises(ValueError):
        study.merge_action(torch.zeros(1,10,6), torch.zeros(1,10,6), 'pose_midpoint')
    with pytest.raises(ValueError):
        study.intervention_ages('invalid')
    assert study.intervention_ages('gripper_midpoint') == (4,)
    assert study.intervention_ages('pose_midpoint') == (4,)
    assert study.intervention_ages('late_actions') == (5,6,7)


class Policy:
    def __init__(self):
        self.condition_adapter = SimpleNamespace(encode_condition=lambda **kw:torch.tensor([9.]))
        self.action_adapter = SimpleNamespace(decode_action_from_condition=self.decode)
        self.reset()

    def reset(self):
        self._age_context = SimpleNamespace(previous=torch.tensor([0.]))
        self.cached_condition = self._age_context.previous

    def _v0_update(self, batch, *, age, policy_query_index):
        self._age_context.previous = self._age_context.previous+1
        self.cached_condition = self._age_context.previous
        chunk = torch.ones(1,10,7)*age
        self.cached_action_chunk = chunk
        return self.cached_condition,chunk,123

    def _paired_initial_noise(self, z, proprio, query):
        return torch.ones(1,10,7),123

    def decode(self, z, proprio, *, steps, initial_noise):
        assert steps == 1 and torch.equal(initial_noise,torch.ones(1,10,7))
        return torch.ones(1,10,7)*99


@pytest.mark.parametrize('arm', study.ARMS)
def test_same_input_recurrence_survives_action_replacement(arm):
    p = study.install_action_intervention(Policy(),SimpleNamespace(component_arm=arm))
    batch = dict(input_ids=None,image_input=None,image_mask=None,proprio=None)
    for age in range(1,8):
        z,chunk,seed = p._v0_update(batch,age=age,policy_query_index=age)
        assert z.item() == age and p._age_context.previous.item() == age
        assert p.cached_condition is z and torch.equal(p.cached_action_chunk,chunk)
        assert seed == 123
        if age in study.intervention_ages(arm):
            expected = study.merge_action(torch.ones_like(chunk)*age,torch.ones_like(chunk)*99,arm)
            assert torch.equal(chunk,expected)
        else:
            assert torch.equal(chunk,torch.ones_like(chunk)*age)
    assert p._component_calls == p._reference_action_calls == p._observed_backbone_calls == len(study.intervention_ages(arm))
    p.reset()
    assert p._component_calls == p._reference_action_calls == p._observed_backbone_calls == 0


def test_noise_mismatch_fails_before_actuation():
    p = study.install_action_intervention(Policy(),SimpleNamespace(component_arm='gripper_midpoint'))
    p._paired_initial_noise = lambda *args:(torch.ones(1,10,7),124)
    with pytest.raises(RuntimeError,match='noise seeds'):
        p._v0_update(dict(input_ids=None,image_input=None,image_mask=None,proprio=None),age=4,policy_query_index=4)


def test_scope_restore_and_three_complete_rows():
    before = (study.base.OUTPUT,study.base.checked,study.evaluation.attach_visual,study.evaluation.check)
    with study.scope():
        assert study.evaluation.check is study.check
    assert before == (study.base.OUTPUT,study.base.checked,study.evaluation.attach_visual,study.evaluation.check)
    plan = study.jobs(dict(identity='test',manifest=dict(action_noise_seed_base=123)))
    assert len(plan) == 18 and len({j['summary'] for j in plan}) == 18
    for arm in study.ARMS:
        shards = [j for j in plan if j['id'].startswith(arm+'_') and j['completion'].get('episodes') == 125]
        assert len(shards) == 4
        assert all(j['deps'] == [arm+'_k8_seed1_smoke'] for j in shards)
    assert not any('codex_outputs' in ' '.join(j['cmd']) for j in plan)
