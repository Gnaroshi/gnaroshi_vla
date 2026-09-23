import torch

from methods.decoder_guided_latent_dynamics.action_delta_predictor import (
    ObservationConditionedPredictor,
)
from methods.decoder_guided_latent_dynamics.predictor_losses import action_delta_loss


def test_predictor_shapes_and_finite_loss() -> None:
    model = ObservationConditionedPredictor(
        output_dim=7, hidden_dim=32, feature_dim=16, predict_gripper_switch=True
    )
    images = [torch.rand(2, 3, 64, 64) for _ in range(4)]
    proprio = [torch.rand(2, 8) for _ in range(2)]
    delta, switch = model(
        *images, *proprio, torch.rand(2, 7), torch.ones(2, 1)
    )
    loss, parts = action_delta_loss(
        delta, torch.rand_like(delta), switch, torch.zeros(2)
    )
    assert delta.shape == (2, 7)
    assert switch.shape == (2,)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
