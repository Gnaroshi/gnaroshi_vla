"""Recurrent two-update model for native-R5 exact-q2 regeneration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import Tensor, nn

from methods.latentloop.modules import (
    ChunkAwareConditionUpdater,
    ExecutedActionEncoder,
    ObservationChangeEncoder,
    ObservationPair,
)


@dataclass(frozen=True)
class ExactQ2ModelConfig:
    """Shared 1x capacity and tensor dimensions for both exact-q2 candidates."""

    condition_dim: int = 960
    condition_tokens: int = 122
    action_horizon: int = 10
    action_dim: int = 7
    execution_horizon: int = 5
    proprio_dim: int = 8
    observation_dim: int = 128
    action_feature_dim: int = 128
    context_dim: int = 128
    fusion_hidden_dim: int = 128
    dynamics_hidden_dim: int = 128
    rank_dim: int = 64
    action_encoder_hidden_dim: int = 128
    gate_mode: str = "scalar"
    gate_bias: float = -4.0
    use_post_layernorm: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionFeatures:
    """One ordered query-boundary transition feature pair."""

    observation: Tensor
    executed_action: Tensor


@dataclass(frozen=True)
class ExactQ2Prediction:
    """q1 auxiliary and q2 primary condition predictions."""

    c1: Tensor
    c2: Tensor
    u1: TransitionFeatures
    u2: TransitionFeatures
    q2_input_condition: Tensor


class TransitionFeatureEncoder(nn.Module):
    """Shared source-compatible E_eta(o_j,o_j+1,X_j,dt_j)."""

    def __init__(self, config: ExactQ2ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_encoder = ObservationChangeEncoder(
            proprio_dim=config.proprio_dim,
            output_dim=config.observation_dim,
        )
        self.action_encoder = ExecutedActionEncoder(
            action_dim=config.action_dim,
            max_actions=config.execution_horizon,
            hidden_dim=config.action_encoder_hidden_dim,
            output_dim=config.action_feature_dim,
        )

    def forward(
        self,
        previous_images: Tensor,
        current_images: Tensor,
        previous_proprio: Tensor,
        current_proprio: Tensor,
        executed_actions: Tensor,
        elapsed_time: Tensor,
    ) -> TransitionFeatures:
        observation = self.observation_encoder(
            ObservationPair(
                previous_images=previous_images,
                current_images=current_images,
                previous_proprio=previous_proprio,
                current_proprio=current_proprio,
            )
        )
        action = self.action_encoder(
            executed_actions,
            self.config.execution_horizon,
            elapsed_time,
        ).feature
        return TransitionFeatures(observation=observation, executed_action=action)


def build_condition_updater(config: ExactQ2ModelConfig) -> ChunkAwareConditionUpdater:
    """Build the identical 1x updater used by both candidates."""

    return ChunkAwareConditionUpdater(
        condition_dim=config.condition_dim,
        observation_dim=config.observation_dim,
        action_feature_dim=config.action_feature_dim,
        context_dim=config.context_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        dynamics_hidden_dim=config.dynamics_hidden_dim,
        rank_dim=config.rank_dim,
        gate_mode=config.gate_mode,
        gate_bias=config.gate_bias,
        use_post_layernorm=config.use_post_layernorm,
    )


class RecurrentExactQ2(nn.Module):
    """Apply one shared updater twice; q2 consumes the model's predicted q1."""

    candidate_name = "recurrent_exact_q2"

    def __init__(self, config: ExactQ2ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExactQ2ModelConfig()
        self.transition_encoder = TransitionFeatureEncoder(self.config)
        self.updater = build_condition_updater(self.config)

    def forward(
        self,
        *,
        c0_full: Tensor,
        q0_raw_rgb: Tensor,
        q0_proprio: Tensor,
        q1_raw_rgb: Tensor,
        q1_proprio: Tensor,
        q2_raw_rgb: Tensor,
        q2_proprio: Tensor,
        x0_executed: Tensor,
        x1_executed: Tensor,
        elapsed_q0_to_q1: Tensor,
        elapsed_q1_to_q2: Tensor,
    ) -> ExactQ2Prediction:
        u1 = self.transition_encoder(
            q0_raw_rgb,
            q1_raw_rgb,
            q0_proprio,
            q1_proprio,
            x0_executed,
            elapsed_q0_to_q1,
        )
        u2 = self.transition_encoder(
            q1_raw_rgb,
            q2_raw_rgb,
            q1_proprio,
            q2_proprio,
            x1_executed,
            elapsed_q1_to_q2,
        )
        c1 = self.updater(
            c0_full,
            u1.observation,
            u1.executed_action,
            execution_horizon=self.config.execution_horizon,
            elapsed_time=elapsed_q0_to_q1,
            query_age=1,
        ).condition
        # This exact tensor, without teacher substitution or detach, is the q2 input.
        c2 = self.updater(
            c1,
            u2.observation,
            u2.executed_action,
            execution_horizon=self.config.execution_horizon,
            elapsed_time=elapsed_q1_to_q2,
            query_age=2,
        ).condition
        return ExactQ2Prediction(c1=c1, c2=c2, u1=u1, u2=u2, q2_input_condition=c1)
