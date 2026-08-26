"""Exact-cache sequence extraction for compact action-fidelity supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from architectures.simvla.adapters.latentloop.action_equivalent_refresh.features import (
    SimVLAActionFidelityFeatureConfig,
    build_simvla_action_fidelity_features,
)
from ..efficient_multirate.coupled_condition_generation import (
    condition_update_with_code,
)
from methods.latentloop.modules.action_equivalent_refresh import (
    CounterfactualActionTargets,
    counterfactual_action_targets,
)
from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0ObservationPair,
)


SameNoiseDecoder = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class CompactFidelityRecords:
    features: Tensor
    targets: CounterfactualActionTargets
    episode_first_candidates: Tensor
    episode_ids: tuple[str, ...]


def _batch_age_flatten(value: Tensor) -> Tensor:
    if value.ndim < 2:
        raise ValueError("batch-age tensor needs at least two dimensions")
    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])


def extract_compact_fidelity_records(
    *,
    condition_adapter: NativeSimVLAV0,
    decode_same_noise: SameNoiseDecoder,
    anchor_condition: Tensor,
    exact_conditions: Tensor,
    image_sequence: Tensor,
    proprio_sequence: Tensor,
    explicit_noises: Tensor,
    initial_action_chunk: Tensor,
    valid_mask: Tensor,
    group_ids: Tensor,
    episode_ids: Sequence[str],
    arm_scale: Tensor,
    feature_config: SimVLAActionFidelityFeatureConfig | None = None,
) -> CompactFidelityRecords:
    """Extract q1--q3 labels while exact-current tensors remain transient.

    ``decode_same_noise`` must run the fixed N_G=3 action-generation path for
    both candidate and exact conditions.  The explicit noise tensor is passed
    unchanged to both calls at every age.
    """

    cfg = feature_config or SimVLAActionFidelityFeatureConfig()
    batch = int(anchor_condition.shape[0])
    if exact_conditions.shape[:2] != (batch, cfg.max_age):
        raise ValueError("exact_conditions must be [B,3,T,D]")
    if image_sequence.shape[:2] != (batch, cfg.max_age + 1):
        raise ValueError("image_sequence must contain q0--q3")
    if proprio_sequence.shape != (batch, cfg.max_age + 1, cfg.proprio_dim):
        raise ValueError("proprio_sequence must be [B,4,Q]")
    if explicit_noises.shape[:2] != (batch, cfg.max_age):
        raise ValueError("explicit_noises must contain one q1--q3 noise per row")
    if initial_action_chunk.shape[:1] != (batch,):
        raise ValueError("initial_action_chunk batch size changed")
    if len(episode_ids) != batch:
        raise ValueError("episode IDs must align with sequence batch")

    previous_condition = anchor_condition
    previous_action = initial_action_chunk
    age_features: list[Tensor] = []
    age_targets: list[CounterfactualActionTargets] = []
    with torch.no_grad():
        for age in range(1, cfg.max_age + 1):
            pair = NativeV0ObservationPair(
                previous_images=image_sequence[:, age - 1],
                current_images=image_sequence[:, age],
                previous_proprio=proprio_sequence[:, age - 1],
                current_proprio=proprio_sequence[:, age],
            )
            exposed = condition_update_with_code(
                condition_adapter,
                previous_condition,
                pair,
                valid_mask=valid_mask,
                group_ids=group_ids,
                age=age,
            )
            features = build_simvla_action_fidelity_features(
                delta_feature=exposed.condition_change_code,
                update=exposed.update,
                valid_mask=valid_mask,
                group_ids=group_ids,
                previous_action_chunk=previous_action,
                previous_proprio=proprio_sequence[:, age - 1],
                current_proprio=proprio_sequence[:, age],
                candidate_age=age,
                config=cfg,
            )
            noise = explicit_noises[:, age - 1]
            candidate_action = decode_same_noise(
                exposed.update.condition, proprio_sequence[:, age], noise
            )
            exact_action = decode_same_noise(
                exact_conditions[:, age - 1], proprio_sequence[:, age], noise
            )
            targets = counterfactual_action_targets(
                candidate_action,
                exact_action,
                arm_scale=arm_scale,
                first_r=cfg.first_r,
            )
            age_features.append(features)
            age_targets.append(targets)
            previous_condition = exposed.update.condition
            previous_action = candidate_action

    # Preserve q1,q2,q3 order inside each exact-anchored sequence.
    features = _batch_age_flatten(torch.stack(age_features, dim=1))

    def flatten_target(name: str) -> Tensor:
        return _batch_age_flatten(
            torch.stack([getattr(value, name) for value in age_targets], dim=1)
        )

    targets = CounterfactualActionTargets(
        arm_normalized_l1=flatten_target("arm_normalized_l1"),
        direction_cosine_error=flatten_target("direction_cosine_error"),
        direction_valid=flatten_target("direction_valid"),
        gripper_mismatch=flatten_target("gripper_mismatch"),
    )
    first_candidates = torch.zeros(
        (batch, cfg.max_age), device=features.device, dtype=torch.bool
    )
    first_candidates[:, 0] = True
    expanded_ids = tuple(
        str(episode_id)
        for episode_id in episode_ids
        for _ in range(cfg.max_age)
    )
    return CompactFidelityRecords(
        features=features,
        targets=targets,
        episode_first_candidates=first_candidates.reshape(-1),
        episode_ids=expanded_ids,
    )
