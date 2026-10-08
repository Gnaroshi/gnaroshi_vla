import torch
from torch import nn
import pytest

from tools.simvla.condition_capacity import fit_condition, output_basis, project


def test_fixed_span_and_free_fit_differ_on_unavailable_action_direction():
    start = torch.ones(1, 6, 3)
    target = torch.ones(1, 6, 1)*1.2
    valid = torch.ones(1, 6, dtype=torch.bool)
    basis = torch.tensor([[1.], [0.], [0.]])
    decode = lambda z: z[..., 1:2]
    restricted, details = fit_condition(start, target, decode, valid, basis=basis, steps=8)
    free, free_details = fit_condition(start, target, decode, valid, steps=8)
    assert torch.equal(restricted, start)
    assert free_details['trace'][-1]['action_l1'] < details['trace'][-1]['action_l1']
    assert details['forward_calls'] == free_details['forward_calls'] == 33
    assert details['backward_calls'] == free_details['backward_calls'] == 8


@pytest.mark.parametrize('restricted', [False, True])
def test_mask_radius_monotonicity_and_no_network_weight_updates(restricted):
    start = torch.ones(1, 6, 3)
    target = torch.ones(1, 6, 2)*3
    valid = torch.tensor([[True, True, True, True, True, False]])
    network = nn.Linear(3, 2, bias=False).requires_grad_(False)
    before = network.weight.detach().clone()
    basis = torch.eye(3)[:, :2] if restricted else None
    fitted, details = fit_condition(start, target, network, valid, basis=basis,
        steps=8, step_fraction=.05, radius_fraction=.1)
    assert torch.equal(network.weight, before) and network.weight.grad is None
    assert torch.equal(fitted[:, -1], start[:, -1])
    assert details['max_relative_token_change'] <= .100001
    errors = [row['action_l1'] for row in details['trace']]
    assert all(y <= x+1e-7 for x,y in zip(errors, errors[1:]))
    assert details['span_relative_error'] < 1e-5


def test_basis_includes_both_heads_and_biases():
    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.condition_updater = nn.Module()
            self.action_condition_updater = nn.Module()
            self.condition_updater.up = nn.Linear(1, 4)
            self.action_condition_updater.up = nn.Linear(1, 4)
    model = Heads()
    with torch.no_grad():
        model.condition_updater.up.weight.copy_(torch.tensor([[1.],[0.],[0.],[0.]]))
        model.condition_updater.up.bias.copy_(torch.tensor([0.,1.,0.,0.]))
        model.action_condition_updater.up.weight.copy_(torch.tensor([[0.],[0.],[1.],[0.]]))
        model.action_condition_updater.up.bias.copy_(torch.tensor([0.,0.,0.,1.]))
    basis, _ = output_basis(model)
    assert basis.shape == (4,4)
    x = torch.randn(2,3,4)
    torch.testing.assert_close(project(x,basis),x)


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        fit_condition(torch.ones(1,6,3), torch.ones(1,6,3), lambda z:z,
            torch.ones(1,6,dtype=torch.bool), steps=0)
