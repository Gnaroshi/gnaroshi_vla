import copy
from types import SimpleNamespace

import pytest
import torch

from tools.simvla.condition_state_audit import MODES, intervention


class Model:
    def predict(self, context, age, images, proprio):
        self.input = context.previous.clone()
        prediction = context.previous + images
        measured = context.anchor + 2 * images
        gain = torch.full_like(prediction[..., :1], 0.2)
        condition = prediction + gain * (measured - prediction)
        condition = torch.where(context.valid.unsqueeze(-1), condition, context.anchor)
        context.previous, context.age = condition, age
        return condition, dict(predicted=prediction, measured=measured, gain=gain)


def context():
    return SimpleNamespace(anchor=torch.zeros(1, 2, 3), previous=torch.ones(1, 2, 3),
        valid=torch.tensor([[True, False]]), age=1)


@pytest.mark.parametrize('mode', MODES)
def test_independent_interventions_and_carried_state(mode):
    c, model = context(), Model()
    previous = torch.full_like(c.anchor, 5)
    current = torch.full_like(c.anchor, 9)
    value, d = intervention(model, c, 2, torch.ones_like(c.anchor), None, mode, previous, current)
    memory = 5 if mode in ('teacher_history', 'teacher_both') else 1
    measured = 9 if mode in ('teacher_measurement', 'teacher_both') else 2
    expected = memory + 1 + 0.2 * (measured - memory - 1)
    torch.testing.assert_close(value[0, 0], torch.full((3,), expected))
    assert c.previous is value
    assert c.age == 2
    torch.testing.assert_close(value[0, 1], c.anchor[0, 1])
    torch.testing.assert_close(d['gain'], torch.full((1, 2, 1), 0.2))
    torch.testing.assert_close(d['measured'][0, 0], torch.full((3,), 2.0))


def test_normal_is_identical_and_oracle_values_do_not_leak():
    a, b = context(), context()
    x = torch.ones_like(a.anchor)
    expected, _ = Model().predict(a, 2, x, None)
    actual, _ = intervention(Model(), b, 2, x, None, 'normal', x * 999, x * -999)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_measurement_replacement_keeps_history_and_gate():
    a, b = context(), context()
    m1, m2 = Model(), Model()
    x = torch.ones_like(a.anchor)
    _, d1 = intervention(m1, a, 2, x, None, 'normal', x * 5, x * 9)
    _, d2 = intervention(m2, b, 2, x, None, 'teacher_measurement', x * 5, x * 9)
    torch.testing.assert_close(m1.input, m2.input, rtol=0, atol=0)
    for key in ('gain', 'predicted', 'measured'):
        torch.testing.assert_close(d1[key], d2[key], rtol=0, atol=0)


def test_rejects_nonfinite_teacher():
    c = context()
    with pytest.raises(ValueError):
        intervention(Model(), c, 2, c.anchor, None, 'normal', c.anchor, c.anchor + float('nan'))


@torch.inference_mode()
def test_real_module_normal_parity_through_seven_recursive_queries():
    from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
    from methods.latentloop.modules.observation_correction import ObservationCorrection

    torch.manual_seed(3)
    model = ObservationCorrection(NativeSimVLAV0(condition_dim=32, max_tokens=8),
        'observed_recurrent').eval()
    anchor = torch.randn(1, 6, 32)
    image = torch.rand(1, 2, 3, 32, 32)
    proprio = torch.randn(1, 8)
    valid = torch.tensor([[True, True, True, True, True, False]])
    a = model.prepare(anchor, image, proprio, valid, torch.zeros(1, 6, dtype=torch.long), 8)
    b = copy.deepcopy(a)
    for age in range(1, 8):
        image, proprio = torch.rand_like(image), torch.randn_like(proprio)
        expected, _ = model.predict(a, age, image, proprio)
        actual, _ = intervention(model, b, age, image, proprio, 'normal',
            torch.randn_like(anchor), torch.randn_like(anchor))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(a.previous, b.previous, rtol=0, atol=0)
