"""Direct nonrecurrent q0-to-q2 model with ordered u1/u2 inputs."""

from __future__ import annotations

from torch import Tensor, nn

from .recurrent_model import (
    ExactQ2ModelConfig,
    ExactQ2Prediction,
    TransitionFeatureEncoder,
    TransitionFeatures,
    build_condition_updater,
)


class OrderedTransitionMixer(nn.Module):
    """Parameter-free ordered fusion that preserves the shared 1x parameter budget."""

    @staticmethod
    def one(u1: Tensor) -> Tensor:
        return u1

    @staticmethod
    def two(u1: Tensor, u2: Tensor) -> Tensor:
        # Fixed coefficients make swapping u1/u2 observable without adding capacity.
        return (u1 + 2.0 * u2) / 3.0


class DirectExactQ2(nn.Module):
    """Predict q2 directly from C0,u1,u2 and never accept predicted C1 as input."""

    candidate_name = "direct_exact_q2"

    def __init__(self, config: ExactQ2ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExactQ2ModelConfig()
        self.transition_encoder = TransitionFeatureEncoder(self.config)
        self.updater = build_condition_updater(self.config)
        self.mixer = OrderedTransitionMixer()

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
            self.mixer.one(u1.observation),
            self.mixer.one(u1.executed_action),
            execution_horizon=self.config.execution_horizon,
            elapsed_time=elapsed_q0_to_q1,
            query_age=1,
        ).condition
        mixed = TransitionFeatures(
            observation=self.mixer.two(u1.observation, u2.observation),
            executed_action=self.mixer.two(u1.executed_action, u2.executed_action),
        )
        elapsed_q0_to_q2 = elapsed_q0_to_q1 + elapsed_q1_to_q2
        c2 = self.updater(
            c0_full,
            mixed.observation,
            mixed.executed_action,
            # R remains the source-native per-query execution horizon. The two
            # ordered transitions are already explicit in the mixed u1/u2 input.
            execution_horizon=self.config.execution_horizon,
            elapsed_time=elapsed_q0_to_q2,
            query_age=2,
        ).condition
        return ExactQ2Prediction(c1=c1, c2=c2, u1=u1, u2=u2, q2_input_condition=c0_full)
