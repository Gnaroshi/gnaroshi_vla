"""Shared-feature joint action surrogate and parameter-matched wide control."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .alignment import align_immutable_anchor


class ScalarEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, value: Tensor | float, leading: tuple[int, ...], ref: Tensor) -> Tensor:
        tensor = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(*leading, 1)
        elif tensor.shape == leading:
            tensor = tensor.unsqueeze(-1)
        else:
            tensor = torch.broadcast_to(tensor, leading + (1,))
        return self.network(tensor)


@dataclass(frozen=True)
class SurrogateOutput:
    arm: Tensor
    gripper_logit: Tensor
    gripper_probability: Tensor
    horizon: Tensor
    residual: Tensor
    aligned_anchor: Tensor
    valid_mask: Tensor
    elapsed: Tensor


class JointLatentAnchoredActionSurrogate(nn.Module):
    """Predict a nonrecursive correction relative to one immutable exact anchor."""

    def __init__(
        self,
        *,
        latent_dim: int = 384,
        motion_dim: int = 128,
        action_pred_steps: int = 3,
        hidden_dim: int = 192,
        latent_projection_dim: int = 32,
        age_dim: int = 16,
        token_dim: int = 16,
    ) -> None:
        super().__init__()
        if action_pred_steps != 3:
            raise ValueError("The source-locked Seer contract requires P=3")
        self.latent_dim = int(latent_dim)
        self.motion_dim = int(motion_dim)
        self.action_pred_steps = int(action_pred_steps)
        self.hidden_dim = int(hidden_dim)
        self.latent_projection = nn.Linear(self.latent_dim, latent_projection_dim)
        self.age_embedding = ScalarEmbedding(age_dim)
        self.token_embedding = nn.Embedding(self.action_pred_steps, token_dim)
        input_dim = 7 + 1 + self.motion_dim + latent_projection_dim + age_dim + token_dim
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
        *,
        anchor_arm: Tensor,
        anchor_gripper_logit: Tensor,
        anchor_latent: Tensor,
        current_latent: Tensor,
        shared_feature: Tensor,
        elapsed: Tensor | int,
    ) -> SurrogateOutput:
        expected_arm = (self.action_pred_steps, 6)
        if anchor_arm.shape[-2:] != expected_arm:
            raise ValueError(f"anchor_arm must end in {expected_arm}, got {tuple(anchor_arm.shape)}")
        if anchor_gripper_logit.shape != anchor_arm.shape[:-1] + (1,):
            raise ValueError("anchor gripper logits do not align with anchor arm tokens")
        if anchor_latent.shape != current_latent.shape:
            raise ValueError("anchor/current latent shapes differ")
        if current_latent.shape[-2:] != (self.action_pred_steps, self.latent_dim):
            raise ValueError("current latent does not match [P,D]")
        leading = tuple(anchor_arm.shape[:-2])
        if shared_feature.shape != leading + (self.motion_dim,):
            raise ValueError(
                f"shared_feature must be {leading + (self.motion_dim,)}, got {tuple(shared_feature.shape)}"
            )

        anchor = torch.cat((anchor_arm, anchor_gripper_logit), dim=-1)
        aligned = align_immutable_anchor(anchor, elapsed)
        latent_delta = self.latent_projection(current_latent - anchor_latent)
        age = self.age_embedding(elapsed, leading, anchor).unsqueeze(-2).expand(
            *leading, self.action_pred_steps, -1
        )
        token_ids = torch.arange(self.action_pred_steps, device=anchor.device)
        token_view = (1,) * len(leading) + (self.action_pred_steps, -1)
        token = self.token_embedding(token_ids).reshape(token_view).expand(
            *leading, self.action_pred_steps, -1
        )
        motion = shared_feature.unsqueeze(-2).expand(*leading, self.action_pred_steps, -1)
        valid = aligned.valid_mask.to(dtype=anchor.dtype)
        hidden = self.trunk(
            torch.cat((aligned.values, valid, motion, latent_delta, age, token), dim=-1)
        )
        arm_residual = self.arm_head(hidden)
        gripper_residual = self.gripper_head(hidden)
        residual = torch.cat((arm_residual, gripper_residual), dim=-1)
        arm = torch.clamp(aligned.values[..., :6] + arm_residual, -1.0, 1.0)
        gripper_logit = aligned.values[..., 6:] + gripper_residual
        horizon = torch.cat((arm, torch.sigmoid(gripper_logit)), dim=-1)
        return SurrogateOutput(
            arm=arm,
            gripper_logit=gripper_logit,
            gripper_probability=torch.sigmoid(gripper_logit),
            horizon=horizon,
            residual=residual,
            aligned_anchor=aligned.values,
            valid_mask=aligned.valid_mask,
            elapsed=aligned.elapsed,
        )


class WideLatentCapacityControl(nn.Module):
    """Capacity-only latent residual; it has no fast action surrogate."""

    def __init__(
        self,
        *,
        latent_dim: int = 384,
        motion_dim: int = 128,
        action_pred_steps: int = 3,
        hidden_dim: int = 96,
        age_dim: int = 16,
        token_dim: int = 16,
    ) -> None:
        super().__init__()
        if action_pred_steps != 3:
            raise ValueError("The source-locked Seer contract requires P=3")
        self.latent_dim = int(latent_dim)
        self.motion_dim = int(motion_dim)
        self.action_pred_steps = int(action_pred_steps)
        self.age_embedding = ScalarEmbedding(age_dim)
        self.token_embedding = nn.Embedding(action_pred_steps, token_dim)
        input_dim = latent_dim + motion_dim + age_dim + token_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, latent: Tensor, shared_feature: Tensor, age: Tensor | float) -> Tensor:
        if latent.shape[-2:] != (self.action_pred_steps, self.latent_dim):
            raise ValueError("wide control latent does not match [P,D]")
        leading = tuple(latent.shape[:-2])
        if shared_feature.shape != leading + (self.motion_dim,):
            raise ValueError("wide control feature leading dimensions do not match")
        age_feature = self.age_embedding(age, leading, latent).unsqueeze(-2).expand(
            *leading, self.action_pred_steps, -1
        )
        token_ids = torch.arange(self.action_pred_steps, device=latent.device)
        token_view = (1,) * len(leading) + (self.action_pred_steps, -1)
        token = self.token_embedding(token_ids).reshape(token_view).expand(
            *leading, self.action_pred_steps, -1
        )
        motion = shared_feature.unsqueeze(-2).expand(*leading, self.action_pred_steps, -1)
        return latent + self.network(torch.cat((latent, motion, age_feature, token), dim=-1))


def _linear_count(input_dim: int, output_dim: int) -> int:
    return input_dim * output_dim + output_dim


def joint_surrogate_parameter_count(
    *,
    latent_dim: int = 384,
    motion_dim: int = 128,
    action_pred_steps: int = 3,
    hidden_dim: int = 192,
    latent_projection_dim: int = 32,
    age_dim: int = 16,
    token_dim: int = 16,
) -> int:
    input_dim = 7 + 1 + motion_dim + latent_projection_dim + age_dim + token_dim
    return int(
        _linear_count(latent_dim, latent_projection_dim)
        + _linear_count(1, age_dim)
        + _linear_count(age_dim, age_dim)
        + action_pred_steps * token_dim
        + _linear_count(input_dim, hidden_dim)
        + _linear_count(hidden_dim, hidden_dim)
        + _linear_count(hidden_dim, 6)
        + _linear_count(hidden_dim, 1)
    )


def wide_control_parameter_count(
    *,
    latent_dim: int = 384,
    motion_dim: int = 128,
    action_pred_steps: int = 3,
    hidden_dim: int = 96,
    age_dim: int = 16,
    token_dim: int = 16,
) -> int:
    input_dim = latent_dim + motion_dim + age_dim + token_dim
    return int(
        _linear_count(1, age_dim)
        + _linear_count(age_dim, age_dim)
        + action_pred_steps * token_dim
        + _linear_count(input_dim, hidden_dim)
        + _linear_count(hidden_dim, latent_dim)
    )
