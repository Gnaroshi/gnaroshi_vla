"""GPU-native SimVLA runtime features for next-anchor recoverability."""

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
class SimVLARecoverabilityFeatureConfig:
    delta_dim: int = 128
    proprio_dim: int = 8
    action_dim: int = 7
    first_r: int = 5
    num_token_groups: int = 8
    max_age: int = 7
    condition_summary_bins: int = 32

    @property
    def input_dim(self) -> int:
        base = (
            self.delta_dim
            + self.num_token_groups * len(GROUP_STAT_NAMES)
            + self.first_r * self.action_dim
            + 1
            + 3 * self.proprio_dim
            + self.max_age
        )
        # Mean and RMS summaries for both the exact anchor and U_C candidate.
        return base + 4 * self.condition_summary_bins

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "input_dim": self.input_dim}


def _vectorized_group_statistics(
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
    valid = valid_mask.to(device=residual.device, dtype=residual.dtype)
    groups = F.one_hot(
        group_ids.to(device=residual.device, dtype=torch.long),
        num_classes=int(num_groups),
    ).to(residual.dtype)
    weights = groups * valid.unsqueeze(-1)
    token_count = weights.sum(dim=1).clamp_min(1.0)
    dimension = float(residual.shape[-1])
    applied = residual * gate
    mean_abs = torch.einsum("btd,btg->bg", applied.abs(), weights) / (
        token_count * dimension
    )
    rms = torch.sqrt(
        torch.einsum("btd,btg->bg", applied.square(), weights)
        / (token_count * dimension)
    )

    applied_token_max = applied.abs().amax(dim=-1)
    gate_token = gate.squeeze(-1)
    negative = torch.finfo(residual.dtype).min
    max_abs = (
        applied_token_max.unsqueeze(-1)
        .expand(-1, -1, int(num_groups))
        .masked_fill(weights == 0, negative)
        .amax(dim=1)
        .clamp_min(0.0)
    )
    gate_mean = torch.einsum("bt,btg->bg", gate_token, weights) / token_count
    gate_max = (
        gate_token.unsqueeze(-1)
        .expand(-1, -1, int(num_groups))
        .masked_fill(weights == 0, negative)
        .amax(dim=1)
        .clamp_min(0.0)
    )
    return torch.stack((mean_abs, rms, max_abs, gate_mean, gate_max), dim=-1).flatten(1)


def _condition_summary(
    condition: Tensor,
    valid_mask: Tensor,
    *,
    bins: int,
) -> Tensor:
    if condition.ndim != 3 or valid_mask.shape != condition.shape[:2]:
        raise ValueError("condition/valid_mask must be [B,T,D] and [B,T]")
    mask = valid_mask.to(device=condition.device, dtype=condition.dtype).unsqueeze(-1)
    count = mask.sum(dim=1).clamp_min(1.0)
    mean = (condition.float() * mask).sum(dim=1) / count
    rms = torch.sqrt((condition.float().square() * mask).sum(dim=1) / count)
    mean_bins = F.adaptive_avg_pool1d(mean.unsqueeze(1), int(bins)).squeeze(1)
    rms_bins = F.adaptive_avg_pool1d(rms.unsqueeze(1), int(bins)).squeeze(1)
    return torch.cat((mean_bins, rms_bins), dim=-1)


def build_simvla_recoverability_features(
    *,
    delta_feature: Tensor,
    update: NativeV0UpdateOutput,
    anchor_condition: Tensor,
    valid_mask: Tensor,
    group_ids: Tensor,
    previous_action_chunk: Tensor,
    previous_proprio: Tensor,
    current_proprio: Tensor,
    candidate_age: Tensor,
    config: SimVLARecoverabilityFeatureConfig | None = None,
) -> Tensor:
    """Build one shared feature vector for all three predicted compute modes.

    No current exact condition/action is accepted by this API.  Anchor and
    candidate summaries use deterministic pooling rather than a second image or
    language encoder.
    """

    cfg = config or SimVLARecoverabilityFeatureConfig()
    if delta_feature.ndim != 2 or delta_feature.shape[-1] != cfg.delta_dim:
        raise ValueError(f"delta_feature must be [B,{cfg.delta_dim}]")
    batch = int(delta_feature.shape[0])
    if anchor_condition.shape != update.condition.shape:
        raise ValueError("anchor and candidate conditions must have identical shapes")
    if previous_action_chunk.ndim != 3 or previous_action_chunk.shape[0] != batch:
        raise ValueError("previous_action_chunk must be [B,H,7]")
    if previous_action_chunk.shape[-1] != cfg.action_dim:
        raise ValueError("previous action dimension differs from feature config")
    if int(previous_action_chunk.shape[1]) < cfg.first_r:
        raise ValueError("previous action chunk is shorter than first_r")
    if previous_proprio.shape != (batch, cfg.proprio_dim) or current_proprio.shape != (
        batch,
        cfg.proprio_dim,
    ):
        raise ValueError("proprio tensors violate the configured dimension")
    age = candidate_age.to(device=delta_feature.device, dtype=torch.long).reshape(batch)
    if bool((age < 1).any()) or bool((age > cfg.max_age).any()):
        raise ValueError(f"candidate age must be in [1,{cfg.max_age}]")

    group_stats = _vectorized_group_statistics(
        update,
        valid_mask,
        group_ids,
        num_groups=cfg.num_token_groups,
    )
    anchor_summary = _condition_summary(
        anchor_condition,
        valid_mask,
        bins=cfg.condition_summary_bins,
    )
    candidate_summary = _condition_summary(
        update.condition,
        valid_mask,
        bins=cfg.condition_summary_bins,
    )
    previous_q = previous_proprio.float()
    current_q = current_proprio.float()
    features = torch.cat(
        (
            delta_feature.float(),
            group_stats,
            previous_action_chunk[:, : cfg.first_r].float().flatten(1),
            delta_feature.new_ones((batch, 1)),
            previous_q,
            current_q,
            current_q - previous_q,
            F.one_hot(age - 1, num_classes=cfg.max_age).float(),
            anchor_summary,
            candidate_summary,
        ),
        dim=-1,
    )
    if features.shape != (batch, cfg.input_dim):
        raise RuntimeError("recoverability feature schema drift")
    return features


def runtime_feature_contract(
    config: SimVLARecoverabilityFeatureConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SimVLARecoverabilityFeatureConfig()
    return {
        "provisional_name": "rewq_v0",
        "feature_config": cfg.to_dict(),
        "runtime_inputs": [
            "U_C observation delta feature",
            "U_C residual/gate summaries",
            "cached exact-anchor condition summary",
            "U_C candidate-condition summary",
            "previous action chunk",
            "previous/current proprio",
            "candidate age",
        ],
        "forbidden_runtime_inputs": [
            "current exact condition",
            "current exact action chunk",
            "future observation",
            "teacher success label",
            "simulator object state",
            "new image encoder",
            "new action generator",
        ],
    }
