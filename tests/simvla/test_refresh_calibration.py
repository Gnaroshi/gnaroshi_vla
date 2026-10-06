import copy
import pytest
import torch

from methods.refresh_calibration.model import RefreshCalibratedCondition, ridge_write, VARIANTS


def inputs(batch=2, dim=24, tokens=12):
    torch.manual_seed(9)
    return (torch.randn(batch, tokens, dim), torch.rand(batch, 2, 32, 32, 3),
            torch.randn(batch, 8), torch.randn(batch, dim),
            torch.ones(batch, tokens, dtype=torch.bool), torch.zeros(batch, tokens, dtype=torch.long))


def test_ridge_solves_regularized_normal_equations():
    torch.manual_seed(2)
    x, r = torch.randn(2, 9, 4), torch.randn(2, 9, 6)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[:, -2:] = False
    w = ridge_write(x, r, valid, .1)
    a, b = x[:, :7], r[:, :7]
    residual = a.transpose(1, 2) @ (a @ w - b) / 7 + .1 * w
    assert residual.abs().max() < 1e-6
    r[:, -2:] = 1e8
    torch.testing.assert_close(w, ridge_write(x, r, valid, .1), rtol=0, atol=0)


def test_ridge_is_differentiable_and_rank_deficiency_is_safe():
    x = torch.ones(2, 6, 12, requires_grad=True)
    r = torch.randn(2, 6, 4, requires_grad=True)
    w = ridge_write(x, r, torch.ones(2, 6, dtype=torch.bool), .01)
    w.square().mean().backward()
    assert torch.isfinite(w).all() and torch.isfinite(x.grad).all() and r.grad.abs().max() > 0
    assert torch.equal(ridge_write(x, r, torch.zeros(2, 6, dtype=torch.bool), .01), torch.zeros_like(w))


@pytest.mark.parametrize("variant", VARIANTS)
def test_prepare_predict_and_future_gradient(variant):
    torch.set_num_threads(1)
    model = RefreshCalibratedCondition(variant, dim=24, width=16)
    a, image, q, lang, valid, groups = inputs()
    valid[:, -1] = False
    state = model.prepare(a, image, q, lang, valid, groups)
    original = state.anchor.detach().clone()
    current = torch.rand_like(image)
    prediction = model.predict(state, current, q)
    prediction.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.features.parameters())
    assert torch.equal(state.anchor, original)
    assert torch.equal(prediction[:, -1], a[:, -1])
    assert not torch.equal(prediction[:, :-1], model.predict(state, image, q)[:, :-1])


def test_fixed_mapping_does_not_read_valid_anchor_values():
    model = RefreshCalibratedCondition("fixed", dim=24, width=16)
    a, im, q, lang, mask, group = inputs()
    first = model.prepare(a, im, q, lang, mask, group)
    other = model.prepare(a + 10, im, q, lang, mask, group)
    assert torch.equal(model.predict(first, im, q), model.predict(other, im, q))


def test_ridge_responds_to_anchor_and_shared_initialization_matches():
    common = RefreshCalibratedCondition("fixed", dim=24, width=16).state_dict()
    a, im, q, lang, mask, group = inputs()
    for variant in VARIANTS:
        model = RefreshCalibratedCondition(variant, dim=24, width=16)
        model.initialize_common(common)
        for key, value in common.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    first = model.prepare(a, im, q, lang, mask, group)
    second = model.prepare(a + 1, im, q, lang, mask, group)
    assert not torch.equal(model.predict(first, im, q), model.predict(second, im, q))


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_regularization(value):
    with pytest.raises(ValueError):
        RefreshCalibratedCondition("ridge", ridge_lambda=value)
