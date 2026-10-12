from types import SimpleNamespace

import pytest
import torch
from torch import nn

from methods.latentloop.modules.age_specialized_condition import AgeSpecializedCondition


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))

    def prepare(self, x):
        return SimpleNamespace(previous=x, age=0)

    def predict(self, ctx, age, images, proprio):
        assert age == ctx.age + 1
        ctx.previous = ctx.previous + self.weight * images
        ctx.age = age
        return ctx.previous, {'base': ctx.previous}


@pytest.mark.parametrize('protected', [True, False])
def test_initial_all_ages_match_original(protected):
    original = Toy()
    model = AgeSpecializedCondition(original, protected)
    a, b = original.prepare(torch.tensor(1.)), model.prepare(torch.tensor(1.))
    for age in range(1, 8):
        torch.testing.assert_close(original.predict(a, age, torch.tensor(float(age)), None)[0],
                                   model.predict(b, age, torch.tensor(float(age)), None)[0], rtol=0, atol=0)


def test_late_loss_preserves_early_weights_and_outputs():
    model = AgeSpecializedCondition(Toy(), True)
    original = model.early.weight.detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    ctx = model.prepare(torch.tensor(1.))
    for age in range(1, 8):
        y, _ = model.predict(ctx, age, torch.tensor(1.), None)
    y.square().backward()
    assert model.early.weight.grad is None
    assert model.late.weight.grad.abs() > 0
    optimizer.step()
    assert torch.equal(model.early.weight, original)
    assert not torch.equal(model.late.weight, original)
    ctx = model.prepare(torch.tensor(1.))
    for age in range(1, 4):
        y, _ = model.predict(ctx, age, torch.tensor(1.), None)
    torch.testing.assert_close(y, torch.tensor(1.) + original * 3)


def test_shared_control_allows_late_gradient_to_modify_early():
    model = AgeSpecializedCondition(Toy(), False)
    ctx = model.prepare(torch.tensor(1.))
    for age in range(1, 8):
        y, _ = model.predict(ctx, age, torch.tensor(1.), None)
    y.backward()
    assert model.early.weight.grad == 7
    assert model.late is None


@pytest.mark.parametrize('age', [0, 8, 1.5, True])
def test_invalid_age(age):
    with pytest.raises(ValueError):
        AgeSpecializedCondition(Toy(), True).expert(age)
