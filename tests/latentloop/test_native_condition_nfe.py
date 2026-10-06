import pytest
import torch

from tools.simvla.native_condition_nfe import NFES, condition_prediction, sample_indices
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement import condition_query


def test_identical_samples_across_solvers():
    rows = [sample_indices(7, 251, 1000) for _ in NFES]
    assert all(r == rows[0] for r in rows)
    assert rows[0] != sample_indices(7, 252, 1000)


@pytest.mark.parametrize('age', [1, 3])
def test_native_prediction_matches_frozen_k2(age):
    torch.manual_seed(7)
    adapter = NativeSimVLAV0().eval()
    # The real experiment starts from trained weights, not the zero residual initialization.
    torch.nn.init.normal_(adapter.condition_updater.up.weight, std=0.01)
    sequence = dict(anchor_condition=torch.randn(1, 122, 960),
        teacher_conditions=torch.randn(1, 3, 122, 960),
        image_sequence=torch.randint(0, 256, (1, 4, 2, 3, 224, 224), dtype=torch.uint8),
        proprio_sequence=torch.randn(1, 4, 8), valid_mask=torch.ones(1, 122, dtype=torch.bool),
        group_ids=torch.zeros(1, 122, dtype=torch.long))
    expected = condition_query(adapter, sequence, age).condition
    prediction, previous = condition_prediction(adapter, sequence, age)
    assert torch.equal(prediction, expected)
    assert torch.equal(previous, sequence['anchor_condition'] if age == 1 else sequence['teacher_conditions'][:, 1])
    prediction.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in adapter.delta_encoder.parameters())
    with pytest.raises(ValueError): condition_prediction(adapter, sequence, 2)
