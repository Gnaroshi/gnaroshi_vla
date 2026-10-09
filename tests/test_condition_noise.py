import copy
from types import SimpleNamespace

import pytest
import torch

from tools.simvla.condition_output_split_train import noise_supervision, sampling_step


class FakeAction:
    def __init__(self):
        self.calls = []
        self.action_space = SimpleNamespace(normalize_action=lambda x: x * 2)

    def decode_action_from_condition(self, condition, proprio, *, steps, initial_noise):
        self.calls.append((steps, torch.is_grad_enabled(), initial_noise.clone()))
        return initial_noise + condition


def sample():
    return dict(noise=torch.zeros(1,10,7), target_action=torch.ones(1,10,7),
        target_condition=torch.ones(1,10,7,requires_grad=True), proprio=torch.zeros(1,8))


def test_cached_control_has_no_extra_teacher_work():
    action=FakeAction(); s=sample()
    pairs=noise_supervision({'seed':7},action,s,5,0)
    assert len(pairs)==1 and pairs[0][0] is s and not action.calls
    torch.testing.assert_close(pairs[0][1],2*s['target_action'])
    assert not pairs[0][1].requires_grad


def test_noise_reproducible_paired_and_disjoint_from_validation():
    action=FakeAction(); s=sample()
    c=dict(seed=7,action_noise_samples=2,heldout_action_noise_samples=3)
    rng=torch.random.get_rng_state().clone()
    a=noise_supervision(c,action,s,10,1)
    b=noise_supervision(c,action,s,10,1)
    v=noise_supervision(c,action,s,10,1,heldout=True)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert torch.equal(a[1][0]['noise'],b[1][0]['noise'])
    assert not torch.equal(a[1][0]['noise'],v[1][0]['noise'])
    assert not torch.equal(v[1][0]['noise'],v[2][0]['noise'])
    for entries in (a,b,v):
        for item,target in entries[1:]:
            torch.testing.assert_close(target,2*(item['noise']+s['target_condition']))
            assert not target.requires_grad
    assert all(steps==10 and not grad for steps,grad,_ in action.calls)
    assert s['noise'].count_nonzero()==0
    assert all(a[1][0][k] is s[k] for k in ('proprio','target_condition'))


def test_noise_draws_do_not_change_window_and_age_schedule():
    c=dict(seed=7)
    a=sampling_step(c,123)
    noise_supervision({**c,'action_noise_samples':2},FakeAction(),sample(),123,0)
    b=sampling_step(c,123)
    assert a[1:]==b[1:] and a[0].getstate()==b[0].getstate()


def test_four_fresh_models_and_eight_complete_evaluations(monkeypatch,tmp_path):
    from tools.simvla import condition_noise_pipeline as pipeline
    from tools.simvla.condition_noise_rb2 import ROWS
    monkeypatch.setattr(pipeline,'OUTPUT',tmp_path)
    monkeypatch.setattr(pipeline,'source_config',lambda:dict(extra_source_files=[],python='python',seed=7))
    monkeypatch.setattr(pipeline,'prepare',lambda c:None)
    monkeypatch.setattr(pipeline,'identity',lambda c:'test')
    configs=pipeline.configurations()
    assert len(configs)==4 and len(ROWS)==8
    for c in configs.values():
        assert c['initialization']=='fresh' and c['student_steps']==1
        assert c['steps']==10000 and c['sample_step_offset']==0
        assert c['warmup_steps']==500 and c['heldout_action_noise_samples']==3
    plan=pipeline.jobs(configs); ids={j['id'] for j in plan}
    assert len(plan)==len(ids)==24
    assert all(set(j.get('deps',())).issubset(ids) for j in plan)
    assert sum(j['completion'].get('episodes')==500 for j in plan)==8
    assert sum(j['completion'].get('steps')==10000 for j in plan)==4


@pytest.mark.parametrize('mode',['detached','joint'])
def test_fresh_initialization_is_independent_of_old_checkpoint_values(mode):
    from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
    from tools.simvla.condition_output_split_train import build_initial_model
    parent=NativeSimVLAV0(condition_dim=16,max_tokens=8)
    c=dict(initialization='fresh',seed=7,action_gradient_mode=mode)
    a=build_initial_model(copy.deepcopy(parent),c,'carry_base')
    with torch.no_grad():
        for p in parent.parameters(): p.add_(10)
    b=build_initial_model(parent,c,'carry_base')
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
