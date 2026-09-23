import torch

from methods.decoder_guided_latent_dynamics.geometry import (
    damped_action_to_latent_lift,
    damped_latent_decomposition,
)


def test_dls_lift_recovers_small_linear_action_delta() -> None:
    torch.manual_seed(7)
    jacobian = torch.randn(4, 3, 12)
    desired = torch.randn(4, 3) * 0.05
    update = damped_action_to_latent_lift(jacobian, desired, 1e-6)
    achieved = (jacobian @ update.unsqueeze(-1)).squeeze(-1)
    assert update.shape == (4, 12)
    assert torch.isfinite(update).all()
    torch.testing.assert_close(achieved, desired, atol=2e-5, rtol=2e-5)


def test_damped_residual_has_low_action_leakage() -> None:
    torch.manual_seed(11)
    jacobian = torch.randn(3, 4, 16)
    delta = torch.randn(3, 16)
    parallel, residual = damped_latent_decomposition(jacobian, delta, 1e-6)
    original_action = (jacobian @ delta.unsqueeze(-1)).squeeze(-1)
    residual_action = (jacobian @ residual.unsqueeze(-1)).squeeze(-1)
    assert parallel.shape == residual.shape == delta.shape
    assert torch.linalg.vector_norm(residual_action) < 1e-4 * torch.linalg.vector_norm(
        original_action
    )
