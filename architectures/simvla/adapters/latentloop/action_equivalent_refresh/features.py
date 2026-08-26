"""SimVLA runtime features for counterfactual action-fidelity prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from methods.latentloop.modules.native_simvla_v0 import NativeV0UpdateOutput


GROUP_STAT_NAMES = (
    "applied_residual_mean_abs",
    "applied_residual_rms",
    "applied_residual_max_abs",
    "gate_mean",
    "gate_max",
)


@dataclass(frozen=True)
class SimVLAActionFidelityFeatureConfig:
    delta_dim: int = 128
    proprio_dim: int = 8
    action_dim: int = 7
    first_r: int = 5
    num_token_groups: int = 8
    max_age: int = 3

    @property
    def input_dim(self) -> int:
        return (
            self.delta_dim
            + self.num_token_groups * len(GROUP_STAT_NAMES)
            + self.first_r * self.action_dim
            + 1
            + 3 * self.proprio_dim
            + self.max_age
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "input_dim": self.input_dim}


def feature_names(
    config: SimVLAActionFidelityFeatureConfig,
) -> tuple[str, ...]:
    names = [f"delta_feature/{index}" for index in range(config.delta_dim)]
    names.extend(
        f"token_group_{group}/{stat}"
        for group in range(config.num_token_groups)
        for stat in GROUP_STAT_NAMES
    )
    names.extend(
        f"previous_action_{step}/{dimension}"
        for step in range(config.first_r)
        for dimension in range(config.action_dim)
    )
    names.append("previous_action_available")
    for prefix in ("previous_proprio", "current_proprio", "proprio_delta"):
        names.extend(f"{prefix}/{index}" for index in range(config.proprio_dim))
    names.extend(f"candidate_age/{age}" for age in range(1, config.max_age + 1))
    if len(names) != config.input_dim:
        raise RuntimeError("SimVLA action-fidelity feature schema drift")
    return tuple(names)


def _group_statistics(
    update: NativeV0UpdateOutput,
    valid_mask: Tensor,
    group_ids: Tensor,
    *,
    num_groups: int,
) -> Tensor:
    residual = update.residual.float()
    gate = update.gate.float()
    if residual.ndim != 3 or gate.shape != (*residual.shape[:2], 1):
        raise ValueError("update residual/gate must be [B,T,D] and [B,T,1]")
    if valid_mask.shape != residual.shape[:2] or group_ids.shape != residual.shape[:2]:
        raise ValueError("valid_mask and group_ids must match update [B,T]")
    applied = residual * gate
    batch, _, dimension = residual.shape
    groups: list[Tensor] = []
    for group in range(int(num_groups)):
        token_mask = (
            valid_mask.to(device=residual.device, dtype=torch.bool)
            & (group_ids.to(device=residual.device, dtype=torch.long) == group)
        )
        residual_mask = token_mask.unsqueeze(-1)
        token_count = token_mask.sum(dim=1).clamp_min(1).float()
        value_count = token_count * int(dimension)
        applied_abs = applied.abs() * residual_mask
        mean_abs = applied_abs.sum(dim=(1, 2)) / value_count
        rms = torch.sqrt(
            (applied.square() * residual_mask).sum(dim=(1, 2))
            / value_count
        )
        max_abs = applied_abs.flatten(1).amax(dim=1)
        gate_values = gate.squeeze(-1) * token_mask
        gate_mean = gate_values.sum(dim=1) / token_count
        gate_max = gate_values.amax(dim=1)
        groups.append(
            torch.stack((mean_abs, rms, max_abs, gate_mean, gate_max), dim=-1)
        )
    return torch.cat(groups, dim=-1).reshape(batch, -1)


def build_simvla_action_fidelity_features(
    *,
    delta_feature: Tensor,
    update: NativeV0UpdateOutput,
    valid_mask: Tensor,
    group_ids: Tensor,
    previous_action_chunk: Tensor | None,
    previous_proprio: Tensor,
    current_proprio: Tensor,
    candidate_age: Tensor | int,
    config: SimVLAActionFidelityFeatureConfig | None = None,
) -> Tensor:
    """Build a fixed feature vector without exact-current-condition leakage."""

    cfg = config or SimVLAActionFidelityFeatureConfig()
    if delta_feature.ndim != 2 or delta_feature.shape[-1] != cfg.delta_dim:
        raise ValueError(f"delta_feature must be [B,{cfg.delta_dim}]")
    batch = int(delta_feature.shape[0])
    expected_proprio = (batch, cfg.proprio_dim)
    if previous_proprio.shape != expected_proprio or current_proprio.shape != expected_proprio:
        raise ValueError(f"proprio tensors must be {expected_proprio}")

    if previous_action_chunk is None:
        action = delta_feature.new_zeros((batch, cfg.first_r, cfg.action_dim))
        action_available = delta_feature.new_zeros((batch, 1))
    else:
        if (
            previous_action_chunk.ndim != 3
            or previous_action_chunk.shape[0] != batch
            or previous_action_chunk.shape[-1] != cfg.action_dim
        ):
            raise ValueError("previous_action_chunk must be [B,H,7]")
        if int(previous_action_chunk.shape[1]) < cfg.first_r:
            raise ValueError("previous_action_chunk is shorter than execution horizon")
        action = previous_action_chunk[:, : cfg.first_r].float()
        action_available = delta_feature.new_ones((batch, 1))

    age = torch.as_tensor(
        candidate_age, device=delta_feature.device, dtype=torch.long
    )
    if age.ndim == 0:
        age = age.expand(batch)
    if age.shape != (batch,) or bool((age < 1).any()) or bool((age > cfg.max_age).any()):
        raise ValueError(f"candidate_age must be scalar or [B] in [1,{cfg.max_age}]")
    age_one_hot = F.one_hot(age - 1, num_classes=cfg.max_age).float()
    group_stats = _group_statistics(
        update,
        valid_mask,
        group_ids,
        num_groups=cfg.num_token_groups,
    )
    previous_q = previous_proprio.float()
    current_q = current_proprio.float()
    features = torch.cat(
        (
            delta_feature.float(),
            group_stats,
            action.reshape(batch, -1),
            action_available,
            previous_q,
            current_q,
            current_q - previous_q,
            age_one_hot,
        ),
        dim=-1,
    )
    if features.shape != (batch, cfg.input_dim):
        raise RuntimeError(
            f"feature builder produced {tuple(features.shape)}, expected "
            f"{(batch, cfg.input_dim)}"
        )
    return features


def control_risk_scores(
    *,
    update: NativeV0UpdateOutput,
    valid_mask: Tensor,
    candidate_age: Tensor | int,
    max_age: int = 3,
) -> dict[str, Tensor]:
    """Cheap controls for matched-budget ablations, not the learned router."""

    residual = (update.residual.float() * update.gate.float()).square()
    mask = valid_mask.to(device=residual.device, dtype=residual.dtype).unsqueeze(-1)
    denominator = (mask.sum(dim=(1, 2)) * residual.shape[-1]).clamp_min(1.0)
    residual_rms = torch.sqrt((residual * mask).sum(dim=(1, 2)) / denominator)
    gate_max = (update.gate.squeeze(-1).float() * valid_mask.float()).amax(dim=1)
    age = torch.as_tensor(candidate_age, device=residual.device, dtype=torch.float32)
    if age.ndim == 0:
        age = age.expand(residual.shape[0])
    return {
        "age_only": age / float(max_age),
        "applied_residual_rms": residual_rms,
        "gate_max": gate_max,
    }


def runtime_feature_contract(
    config: SimVLAActionFidelityFeatureConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SimVLAActionFidelityFeatureConfig()
    return {
        "feature_config": cfg.to_dict(),
        "feature_names": list(feature_names(cfg)),
        "runtime_inputs": [
            "U_C delta feature",
            "U_C residual and gate summaries",
            "previous action chunk",
            "previous/current proprio",
            "age since exact condition",
        ],
        "forbidden_runtime_inputs": [
            "current exact condition",
            "current exact action",
            "teacher success label",
            "new image encoder",
            "new action generator",
        ],
    }
