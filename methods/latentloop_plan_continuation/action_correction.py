"""Parameter-matched full-horizon action-space correction baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def shift_action_horizon(horizon: Tensor) -> Tensor:
    """Shift ``[a_t^0,...,a_t^(P-1)]`` to the next query time.

    Verified overlap tokens move left by one. The unmatched terminal boundary
    token is repeated; it remains trainable through the residual predictor and
    is excluded from overlap metrics that require a shared intended time.
    """

    if horizon.ndim < 2 or horizon.shape[-2] < 1:
        raise ValueError(f"Expected [..., P, A] with P>=1, got {tuple(horizon.shape)}")
    if horizon.shape[-2] == 1:
        return horizon.clone()
    return torch.cat((horizon[..., 1:, :], horizon[..., -1:, :]), dim=-2)


@dataclass(frozen=True)
class ActionCorrectionOutput:
    """Continuous full-horizon output of the action-space baseline."""

    arm: Tensor
    gripper_logit: Tensor
    gripper_probability: Tensor
    arm_residual: Tensor
    gripper_logit_residual: Tensor


class _ScalarEmbedding(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, value: Tensor | float, leading: tuple[int, ...], reference: Tensor) -> Tensor:
        tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(*leading, 1)
        elif tensor.shape == leading:
            tensor = tensor.unsqueeze(-1)
        elif tensor.shape != leading + (1,):
            tensor = torch.broadcast_to(tensor, leading + (1,))
        return self.network(tensor)


class MatchedActionSpaceCorrection(nn.Module):
    """Correct every arm token and gripper logit from fresh feedback."""

    def __init__(
        self,
        action_pred_steps: int = 3,
        motion_dim: int = 128,
        hidden_dim: int = 256,
        time_dim: int = 32,
        token_dim: int = 32,
    ) -> None:
        super().__init__()
        if action_pred_steps < 1:
            raise ValueError("action_pred_steps must be positive")
        self.action_pred_steps = int(action_pred_steps)
        self.motion_dim = int(motion_dim)
        self.hidden_dim = int(hidden_dim)
        self.age_embedding = _ScalarEmbedding(time_dim)
        self.token_embedding = nn.Embedding(self.action_pred_steps, token_dim)
        input_dim = 7 + self.motion_dim + time_dim + token_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.arm_head = nn.Linear(self.hidden_dim, 6)
        self.gripper_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, shifted_arm: Tensor, shifted_gripper_logit: Tensor, u_delta: Tensor, age: Tensor | float) -> ActionCorrectionOutput:
        """Predict residuals for the complete ``P``-token horizon."""

        if shifted_arm.shape[-2:] != (self.action_pred_steps, 6):
            raise ValueError(
                f"shifted_arm must end in [{self.action_pred_steps},6], got {tuple(shifted_arm.shape)}"
            )
        if shifted_gripper_logit.shape != shifted_arm.shape[:-1] + (1,):
            raise ValueError(
                "shifted_gripper_logit must match arm leading/token dimensions; "
                f"got {tuple(shifted_gripper_logit.shape)}"
            )
        leading = tuple(shifted_arm.shape[:-2])
        if u_delta.shape != leading + (self.motion_dim,):
            raise ValueError(
                f"u_delta must have shape {leading + (self.motion_dim,)}, got {tuple(u_delta.shape)}"
            )
        age_feature = self.age_embedding(age, leading, shifted_arm).unsqueeze(-2).expand(
            *leading, self.action_pred_steps, -1
        )
        token_ids = torch.arange(self.action_pred_steps, device=shifted_arm.device)
        token_shape = (1,) * len(leading) + (self.action_pred_steps, -1)
        token_feature = self.token_embedding(token_ids).reshape(token_shape).expand(
            *leading, self.action_pred_steps, -1
        )
        motion = u_delta.unsqueeze(-2).expand(*leading, self.action_pred_steps, -1)
        base = torch.cat((shifted_arm, shifted_gripper_logit), dim=-1)
        hidden = self.trunk(torch.cat((base, motion, age_feature, token_feature), dim=-1))
        arm_residual = self.arm_head(hidden)
        gripper_residual = self.gripper_head(hidden)
        arm = torch.clamp(shifted_arm + arm_residual, -1.0, 1.0)
        gripper_logit = shifted_gripper_logit + gripper_residual
        return ActionCorrectionOutput(
            arm=arm,
            gripper_logit=gripper_logit,
            gripper_probability=torch.sigmoid(gripper_logit),
            arm_residual=arm_residual,
            gripper_logit_residual=gripper_residual,
        )


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def find_action_correction_hidden_dim(
    target_parameters: int,
    *,
    action_pred_steps: int = 3,
    motion_dim: int = 128,
    minimum: int = 16,
    maximum: int = 1024,
) -> tuple[int, int, float]:
    """Return the hidden size with parameter count closest to a target."""

    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    best: tuple[int, int, int] | None = None
    for hidden_dim in range(int(minimum), int(maximum) + 1):
        count = _parameter_count(
            MatchedActionSpaceCorrection(action_pred_steps, motion_dim, hidden_dim)
        )
        candidate = (abs(count - int(target_parameters)), hidden_dim, count)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, hidden_dim, count = best
    relative_error = abs(count - int(target_parameters)) / float(target_parameters)
    return hidden_dim, count, relative_error
