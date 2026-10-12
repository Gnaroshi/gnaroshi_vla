import copy

import pytest
import torch

from methods.latentloop.modules.condition_token_routing import TokenReference, ReferencedUpdater, TokenRoutedCondition
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0, TokenSharedConditionUpdater
from methods.latentloop.modules.bounded_history import BoundedHistoryCondition


@pytest.mark.parametrize('arm', ['attention', 'tokenwise'])
def test_zero_initialization_preserves_original_exactly(arm):
    torch.manual_seed(12)
    base = TokenSharedConditionUpdater(condition_dim=32, delta_dim=12, max_age=7)
    torch.nn.init.normal_(base.up.weight)
    model = ReferencedUpdater(copy.deepcopy(base), arm)
    z, code = torch.randn(2, 7, 32), torch.randn(2, 12)
    kw = dict(valid_mask=torch.ones(2, 7, dtype=torch.bool), group_ids=torch.zeros(2, 7, dtype=torch.long), age=5)
    expected = base(z, code, **kw)
    actual = model(z, code, anchor=torch.randn_like(z), **kw)
    for field in ('condition', 'residual', 'gate'):
        assert torch.equal(getattr(expected, field), getattr(actual, field))


def ready_reference(arm):
    torch.manual_seed(7)
    m = TokenReference(64, arm)
    torch.nn.init.eye_(m.output.weight)
    return m


def test_cross_token_dependence_is_only_in_attention_arm():
    current, anchor = torch.randn(1, 5, 64), torch.randn(1, 5, 64)
    valid = torch.ones(1, 5, dtype=torch.bool)
    altered = anchor.clone(); altered[:, 3] = torch.randn(1, 64) * 3
    for arm in ('attention', 'tokenwise'):
        model = ready_reference(arm)
        a, b = model(current, anchor, valid), model(current, altered, valid)
        assert bool(torch.equal(a[:, 0], b[:, 0])) == (arm == 'tokenwise')


@pytest.mark.parametrize('arm', ['attention', 'tokenwise'])
def test_padding_cannot_affect_valid_outputs(arm):
    model = ready_reference(arm)
    current, anchor = torch.randn(2, 5, 64), torch.randn(2, 5, 64)
    valid = torch.tensor([[True, True, False, False, False], [False]*5])
    a = model(current, anchor, valid)
    altered = anchor.clone(); altered[~valid] = torch.randn_like(altered[~valid]) * 100
    b = model(current, altered, valid)
    torch.testing.assert_close(a[valid], b[valid], rtol=0, atol=0)
    assert torch.isfinite(b).all() and torch.count_nonzero(b[~valid]) == 0


def test_controls_have_identical_parameters_and_initial_weights():
    a, b = ready_reference('attention'), ready_reference('tokenwise')
    assert sum(p.numel() for p in a.parameters()) == 16640
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y)


@pytest.mark.parametrize('arm', ['attention', 'tokenwise'])
def test_reference_all_parameters_receive_gradients_after_output_opens(arm):
    model = ready_reference(arm)
    out = model(torch.randn(2, 5, 64), torch.randn(2, 5, 64), torch.ones(2, 5, dtype=torch.bool))
    out.square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize('arm', ['attention', 'tokenwise'])
def test_early_model_frozen_and_full_initial_trajectory_equal(arm):
    torch.manual_seed(11)
    native = NativeSimVLAV0(condition_dim=32, delta_dim=128)
    parent = BoundedHistoryCondition(native, 'recurrent')
    torch.nn.init.normal_(parent.condition_updater.up.weight, std=.01)
    m = TokenRoutedCondition(parent, arm)
    images = torch.randn(1, 2, 3, 16, 16)
    args = (torch.randn(1, 6, 32), images, torch.randn(1, 8), torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 6, dtype=torch.long), 8)
    a, b = parent.prepare(*args), m.prepare(*args)
    for age in range(1, 8):
        image, q = torch.randn_like(images), torch.randn(1, 8)
        x, _ = parent.predict(a, age, image, q)
        y, _ = m.predict(b, age, image, q)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    y.square().mean().backward()
    assert all(p.grad is None for p in m.early.parameters())
    assert m.late.condition_updater.reference.output.weight.grad.abs().sum() > 0


def test_invalid_arm():
    with pytest.raises(ValueError):
        TokenReference(64, 'unknown')
