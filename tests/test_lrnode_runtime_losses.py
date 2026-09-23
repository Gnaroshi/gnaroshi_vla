from __future__ import annotations

import torch

from architectures.seer.upstream.utils.lrnode_runtime_losses import (
    age_weighted_loss,
    differentiable_temporal_ensemble,
    runtime_aligned_action_losses,
)


def _chunk(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, -1, 1).expand(1, -1, 7)


def test_temporal_ensemble_uses_runtime_query_token_alignment() -> None:
    anchor = _chunk([10.0, 11.0, 12.0])
    recurrent = torch.stack(
        (
            _chunk([20.0, 21.0, 22.0]),
            _chunk([30.0, 31.0, 32.0]),
            _chunk([40.0, 41.0, 42.0]),
        ),
        dim=1,
    )
    result = differentiable_temporal_ensemble(anchor, recurrent, temperature=0.0)
    assert result.shape == (1, 3, 7)
    assert torch.allclose(result[:, 0], torch.full((1, 7), 15.5))
    assert torch.allclose(result[:, 1], torch.full((1, 7), 21.0))
    assert torch.allclose(result[:, 2], torch.full((1, 7), 31.0))


def test_runtime_losses_are_zero_for_exact_teacher_chunks() -> None:
    anchor = torch.rand(2, 3, 7)
    anchor[..., 6] = torch.sigmoid(anchor[..., 6])
    teacher = torch.rand(2, 3, 3, 7)
    teacher[..., 6] = torch.sigmoid(teacher[..., 6])
    predicted = teacher.clone().requires_grad_(True)
    losses = runtime_aligned_action_losses(predicted, teacher, anchor)
    assert torch.allclose(losses.overlap, torch.zeros_like(losses.overlap))
    assert torch.allclose(
        losses.predicted_executed, losses.teacher_executed, atol=1e-6, rtol=1e-6
    )
    # Soft-target BCE has non-zero entropy even for an exact probability target.
    total = losses.gripper_distill + losses.ensemble + losses.gripper_switch
    total.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_age_weighted_loss_emphasizes_age_three() -> None:
    prediction = torch.tensor([[[0.0], [0.0], [1.0]]])
    target = torch.zeros_like(prediction)
    weights = torch.tensor([1.0, 1.0, 2.0])
    assert torch.allclose(
        age_weighted_loss(prediction, target, weights, "mse"),
        torch.tensor(0.5),
    )
