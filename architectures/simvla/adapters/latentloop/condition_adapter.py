"""SimVLA-facing composition of architecture-neutral LatentLoop modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from methods.latentloop.modules import (
    ChunkAwareConditionUpdater,
    ExecutedActionEncoder,
    MatchedActionChunkCorrection,
    NonRecurrentConditionPredictor,
    ObservationChangeEncoder,
    ObservationPair,
    count_trainable_parameters,
    find_action_correction_hidden_dim,
)


LatentLoopVariant = Literal[
    "chunk_aware_latentloop",
    "old_observation_only",
    "no_observation",
    "nonrecurrent_condition",
    "action_chunk_correction",
]


@dataclass(frozen=True)
class LatentLoopAdapterConfig:
    """Serializable dimensions for one SimVLA LatentLoop adapter."""

    variant: LatentLoopVariant = "chunk_aware_latentloop"
    condition_dim: int = 960
    condition_tokens: int = 122
    action_horizon: int = 10
    action_dim: int = 7
    maximum_execution_horizon: int = 5
    proprio_dim: int = 8
    observation_dim: int = 128
    action_feature_dim: int = 128
    context_dim: int = 128
    fusion_hidden_dim: int = 128
    dynamics_hidden_dim: int = 128
    rank_dim: int = 64
    action_encoder_hidden_dim: int = 128
    action_correction_hidden_dim: int = 0
    gate_mode: str = "scalar"
    gate_bias: float = -4.0
    use_post_layernorm: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a checkpoint-safe configuration dictionary."""

        return asdict(self)


class SimVLAChunkAwareAdapter(nn.Module):
    """Compose observation/action encoders with one condition/action predictor."""

    def __init__(self, config: LatentLoopAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.variant = config.variant
        self.observation_encoder: ObservationChangeEncoder | None = None
        if config.variant != "no_observation":
            self.observation_encoder = ObservationChangeEncoder(
                proprio_dim=config.proprio_dim,
                output_dim=config.observation_dim,
            )
        self.action_encoder: ExecutedActionEncoder | None = None
        if config.variant != "old_observation_only":
            self.action_encoder = ExecutedActionEncoder(
                action_dim=config.action_dim,
                max_actions=config.maximum_execution_horizon,
                hidden_dim=config.action_encoder_hidden_dim,
                output_dim=config.action_feature_dim,
            )
        common = {
            "condition_dim": config.condition_dim,
            "observation_dim": config.observation_dim,
            "action_feature_dim": config.action_feature_dim,
            "context_dim": config.context_dim,
            "fusion_hidden_dim": config.fusion_hidden_dim,
            "dynamics_hidden_dim": config.dynamics_hidden_dim,
            "rank_dim": config.rank_dim,
            "gate_mode": config.gate_mode,
            "gate_bias": config.gate_bias,
            "use_post_layernorm": config.use_post_layernorm,
        }
        self.condition_updater: ChunkAwareConditionUpdater | None = None
        self.nonrecurrent_predictor: NonRecurrentConditionPredictor | None = None
        self.action_correction: MatchedActionChunkCorrection | None = None
        if config.variant in {"chunk_aware_latentloop", "old_observation_only", "no_observation"}:
            self.condition_updater = ChunkAwareConditionUpdater(**common)
        elif config.variant == "nonrecurrent_condition":
            self.nonrecurrent_predictor = NonRecurrentConditionPredictor(**common)
        elif config.variant == "action_chunk_correction":
            if config.action_correction_hidden_dim < 1:
                raise ValueError("action_correction_hidden_dim must be resolved before construction")
            self.action_correction = MatchedActionChunkCorrection(
                action_horizon=config.action_horizon,
                observation_dim=config.observation_dim,
                action_feature_dim=config.action_feature_dim,
                hidden_dim=config.action_correction_hidden_dim,
            )
        else:
            raise ValueError(f"Unsupported LatentLoop variant: {config.variant}")

    def encode_observation(
        self,
        previous_images: Tensor,
        current_images: Tensor,
        previous_proprio: Tensor,
        current_proprio: Tensor,
    ) -> Tensor:
        """Encode previous-query to current-query change, or zero for no-observation."""

        if self.observation_encoder is None:
            return previous_proprio.new_zeros(
                (previous_proprio.shape[0], self.config.observation_dim)
            )
        return self.observation_encoder(
            ObservationPair(
                previous_images=previous_images,
                current_images=current_images,
                previous_proprio=previous_proprio,
                current_proprio=current_proprio,
            ),
            use_observation=self.variant != "no_observation",
        )

    def encode_executed_actions(
        self,
        executed_actions: Tensor,
        execution_horizon: Tensor | int,
        elapsed_time: Tensor | float,
        *,
        reference_feature: Tensor,
    ) -> Tensor:
        """Encode only executed actions; old observation-only receives exact zeros."""

        if self.action_encoder is None:
            return reference_feature.new_zeros(
                (reference_feature.shape[0], self.config.action_feature_dim)
            )
        return self.action_encoder(
            executed_actions,
            execution_horizon,
            elapsed_time,
        ).feature

    def update_recurrent_condition(
        self,
        previous_condition: Tensor,
        observation_feature: Tensor,
        action_feature: Tensor,
        *,
        execution_horizon: Tensor | int,
        elapsed_time: Tensor | float,
        query_age: Tensor | int,
    ) -> Tensor:
        """Return one recursive updated condition for recurrent variants."""

        if self.condition_updater is None:
            raise RuntimeError(f"{self.variant} is not a recurrent condition variant")
        return self.condition_updater(
            previous_condition,
            observation_feature,
            action_feature,
            execution_horizon=execution_horizon,
            elapsed_time=elapsed_time,
            query_age=query_age,
        ).condition

    def predict_nonrecurrent_condition(
        self,
        anchor_condition: Tensor,
        observation_feature: Tensor,
        action_history_feature: Tensor,
        *,
        execution_horizon: Tensor | int,
        elapsed_time: Tensor | float,
        query_age: Tensor | int,
    ) -> Tensor:
        """Predict from the fixed full anchor without accepting a previous prediction."""

        if self.nonrecurrent_predictor is None:
            raise RuntimeError(f"{self.variant} is not the nonrecurrent variant")
        return self.nonrecurrent_predictor(
            anchor_condition,
            observation_feature,
            action_history_feature,
            execution_horizon=execution_horizon,
            elapsed_time=elapsed_time,
            query_age=query_age,
        ).condition

    def correct_action_chunk(
        self,
        previous_action_chunk: Tensor,
        observation_feature: Tensor,
        action_feature: Tensor,
        *,
        execution_horizon: Tensor | int,
        elapsed_time: Tensor | float,
        query_age: Tensor | int,
    ) -> Tensor:
        """Run the matched action-space baseline without producing a condition."""

        if self.action_correction is None:
            raise RuntimeError(f"{self.variant} is not the action-correction variant")
        return self.action_correction(
            previous_action_chunk,
            observation_feature,
            action_feature,
            execution_horizon=execution_horizon,
            elapsed_time=elapsed_time,
            query_age=query_age,
        ).action_chunk


def _base_config(variant: LatentLoopVariant, **overrides: object) -> LatentLoopAdapterConfig:
    payload = LatentLoopAdapterConfig(variant=variant).to_dict()
    if variant == "old_observation_only" and "action_feature_dim" not in overrides:
        payload["action_feature_dim"] = 0
    if variant == "old_observation_only" and "fusion_hidden_dim" not in overrides:
        payload["fusion_hidden_dim"] = 147
    if variant == "no_observation" and "observation_dim" not in overrides:
        payload["observation_dim"] = 0
    payload.update(overrides)
    return LatentLoopAdapterConfig(**payload)


def build_latentloop_adapter(
    variant: LatentLoopVariant,
    **overrides: object,
) -> SimVLAChunkAwareAdapter:
    """Build a variant and automatically parameter-match action correction."""

    config = _base_config(variant, **overrides)
    if variant == "action_chunk_correction" and config.action_correction_hidden_dim < 1:
        matched_overrides = dict(overrides)
        matched_overrides.pop("action_correction_hidden_dim", None)
        primary = SimVLAChunkAwareAdapter(
            _base_config("chunk_aware_latentloop", **matched_overrides)
        )
        target = count_trainable_parameters(primary)
        shared_probe = SimVLAChunkAwareAdapter(
            _base_config(
                "action_chunk_correction",
                action_correction_hidden_dim=16,
                **matched_overrides,
            )
        )
        assert shared_probe.action_correction is not None
        shared = count_trainable_parameters(shared_probe) - count_trainable_parameters(
            shared_probe.action_correction
        )
        hidden_dim, _, _ = find_action_correction_hidden_dim(
            target,
            shared_parameters=shared,
            action_horizon=config.action_horizon,
        )
        config = _base_config(
            variant,
            action_correction_hidden_dim=hidden_dim,
            **matched_overrides,
        )
    return SimVLAChunkAwareAdapter(config)


def parameter_budget_audit() -> dict[str, object]:
    """Count primary/matched variants and report the fixed +/-10% criterion."""

    variants: tuple[LatentLoopVariant, ...] = (
        "chunk_aware_latentloop",
        "old_observation_only",
        "no_observation",
        "nonrecurrent_condition",
        "action_chunk_correction",
    )
    modules = {variant: build_latentloop_adapter(variant) for variant in variants}
    primary = count_trainable_parameters(modules["chunk_aware_latentloop"])
    rows: dict[str, object] = {}
    for variant, module in modules.items():
        count = count_trainable_parameters(module)
        relative = abs(count - primary) / float(primary)
        match_required = variant in {
            "chunk_aware_latentloop",
            "old_observation_only",
            "nonrecurrent_condition",
            "action_chunk_correction",
        }
        rows[variant] = {
            "trainable_parameters": count,
            "relative_difference_from_primary": relative,
            "parameter_match_required": match_required,
            "within_10_percent": relative <= 0.10,
            "parameter_match_pass": (not match_required) or relative <= 0.10,
            "config": module.config.to_dict(),
        }
    return {
        "primary_variant": "chunk_aware_latentloop",
        "primary_trainable_parameters": primary,
        "seer_470146_is_reference_only": True,
        "variants": rows,
    }
