"""Matched direct action-space correction for a complete action horizon."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ShiftedActionHorizon:
    """Time-aligned previous horizon and its verified overlap mask."""

    values: Tensor
    valid_mask: Tensor


def shift_action_horizon_with_mask(horizon: Tensor) -> ShiftedActionHorizon:
    """Align a previous ``[..., P, A]`` horizon to the next query.

    For Seer's zero-offset labels, previous token ``h+1`` and current token
    ``h`` refer to the same environment time. The final token has no previous
    overlap, so its repeated boundary initialization is explicitly marked
    invalid. The predictor still learns a residual for every token.
    """

    if horizon.ndim < 2 or horizon.shape[-2] < 1:
        raise ValueError(f"Expected [..., P, A] with P>=1, got {tuple(horizon.shape)}")
    token_count = int(horizon.shape[-2])
    if token_count == 1:
        shifted = horizon.clone()
        mask = torch.zeros(
            horizon.shape[:-1] + (1,), device=horizon.device, dtype=torch.bool
        )
        return ShiftedActionHorizon(shifted, mask)
    shifted = torch.cat((horizon[..., 1:, :], horizon[..., -1:, :]), dim=-2)
    mask = torch.ones(
        horizon.shape[:-1] + (1,), device=horizon.device, dtype=torch.bool
    )
    mask[..., -1, :] = False
    return ShiftedActionHorizon(shifted, mask)


def shift_action_horizon(horizon: Tensor) -> Tensor:
    """Compatibility helper returning only the aligned horizon values."""

    return shift_action_horizon_with_mask(horizon).values


@dataclass(frozen=True)
class ActionCorrectionOutput:
    """Continuous output of the matched action-space baseline."""

    arm: Tensor
    gripper_logit: Tensor
    gripper_probability: Tensor
    arm_residual: Tensor
    gripper_logit_residual: Tensor
    shifted_overlap_mask: Tensor


class _ScalarEmbedding(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        value: Tensor | float,
        leading: tuple[int, ...],
        reference: Tensor,
    ) -> Tensor:
        tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(*leading, 1)
        elif tensor.shape == leading:
            tensor = tensor.unsqueeze(-1)
        elif tensor.shape != leading + (1,):
            tensor = torch.broadcast_to(tensor, leading + (1,))
        return self.network(tensor)


class MatchedActionSpaceCorrection(nn.Module):
    """Correct all arm tokens and continuous gripper logits from fresh input."""

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

    def forward(
        self,
        previous_arm: Tensor,
        previous_gripper_logit: Tensor,
        u_delta: Tensor,
        age: Tensor | float,
        *,
        inputs_are_shifted: bool = False,
    ) -> ActionCorrectionOutput:
        """Predict a residual for every token in the current action horizon."""

        if previous_arm.shape[-2:] != (self.action_pred_steps, 6):
            raise ValueError(
                "previous_arm must end in "
                f"[{self.action_pred_steps},6], got {tuple(previous_arm.shape)}"
            )
        if previous_gripper_logit.shape != previous_arm.shape[:-1] + (1,):
            raise ValueError(
                "previous_gripper_logit must match arm leading/token dimensions; "
                f"got {tuple(previous_gripper_logit.shape)}"
            )
        arm_shift = (
            ShiftedActionHorizon(
                previous_arm,
                torch.ones(
                    previous_arm.shape[:-1] + (1,),
                    device=previous_arm.device,
                    dtype=torch.bool,
                ),
            )
            if inputs_are_shifted
            else shift_action_horizon_with_mask(previous_arm)
        )
        grip_shift = (
            previous_gripper_logit
            if inputs_are_shifted
            else shift_action_horizon(previous_gripper_logit)
        )
        shifted_arm = arm_shift.values
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
        base = torch.cat((shifted_arm, grip_shift), dim=-1)
        hidden = self.trunk(torch.cat((base, motion, age_feature, token_feature), dim=-1))
        arm_residual = self.arm_head(hidden)
        gripper_residual = self.gripper_head(hidden)
        arm = torch.clamp(shifted_arm + arm_residual, -1.0, 1.0)
        gripper_logit = grip_shift + gripper_residual
        return ActionCorrectionOutput(
            arm=arm,
            gripper_logit=gripper_logit,
            gripper_probability=torch.sigmoid(gripper_logit),
            arm_residual=arm_residual,
            gripper_logit_residual=gripper_residual,
            shifted_overlap_mask=arm_shift.valid_mask,
        )


class ActionCorrectionCache:
    """Cache the baseline's own continuous horizon between full refreshes."""

    def __init__(self) -> None:
        self.arm: Tensor | None = None
        self.gripper_logit: Tensor | None = None

    @property
    def initialized(self) -> bool:
        return self.arm is not None and self.gripper_logit is not None

    def initialize_from_full(self, arm: Tensor, gripper_logit: Tensor) -> None:
        """Initialize at a full-Seer step without retaining autograd history."""

        if arm.shape[:-1] + (1,) != gripper_logit.shape:
            raise ValueError("arm and gripper_logit horizons do not align")
        self.arm = arm.detach().clone()
        self.gripper_logit = gripper_logit.detach().clone()

    def update_from_prediction(self, output: ActionCorrectionOutput) -> None:
        """Recursively cache the baseline's own predicted horizon."""

        self.arm = output.arm.detach().clone()
        self.gripper_logit = output.gripper_logit.detach().clone()

    def tensors(self) -> tuple[Tensor, Tensor]:
        """Return the current continuous horizon or fail before initialization."""

        if not self.initialized:
            raise RuntimeError("Action-correction cache has not been initialized")
        assert self.arm is not None and self.gripper_logit is not None
        return self.arm, self.gripper_logit


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def action_correction_parameter_count(
    hidden_dim: int,
    *,
    action_pred_steps: int = 3,
    motion_dim: int = 128,
    time_dim: int = 32,
    token_dim: int = 32,
) -> int:
    """Return the exact corrector count without constructing probe modules."""

    input_dim = 7 + int(motion_dim) + int(time_dim) + int(token_dim)
    age_parameters = 2 * time_dim + time_dim * time_dim + time_dim
    token_parameters = action_pred_steps * token_dim
    trunk_parameters = (
        input_dim * hidden_dim
        + hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim
    )
    head_parameters = hidden_dim * 7 + 7
    return int(age_parameters + token_parameters + trunk_parameters + head_parameters)


def find_action_correction_hidden_dim(
    target_parameters: int,
    *,
    action_pred_steps: int = 3,
    motion_dim: int = 128,
    minimum: int = 16,
    maximum: int = 1024,
) -> tuple[int, int, float]:
    """Return the hidden size whose parameter count is closest to a target."""

    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    best: tuple[int, int, int] | None = None
    for hidden_dim in range(int(minimum), int(maximum) + 1):
        count = action_correction_parameter_count(
            hidden_dim,
            action_pred_steps=action_pred_steps,
            motion_dim=motion_dim,
        )
        candidate = (abs(count - int(target_parameters)), hidden_dim, count)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, hidden_dim, count = best
    relative_error = abs(count - int(target_parameters)) / float(target_parameters)
    return hidden_dim, count, relative_error
