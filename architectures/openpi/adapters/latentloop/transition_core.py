"""pi0.5 projections around the shared variable-time transition core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from methods.variable_time_latentloop.action_grounding import ExecutedActionEncoder
from methods.variable_time_latentloop.transition import (
    TransitionConfig,
    VariableTimeTransitionCore,
    count_trainable_parameters,
)

from .kv_codec import LayerSharedKVCodec
from .prefix_kv_hook import PrefixEmbeddingState, PrefixKVState


@dataclass(frozen=True)
class OpenPIKVLatentLoopConfig:
    prefix_embedding_dim: int = 2048
    head_dim: int = 256
    action_dim: int = 7
    execution_horizon: int = 5
    action_horizon: int = 10
    parameter_cap: int = 19_000_000
    transition: TransitionConfig = TransitionConfig()


@dataclass(frozen=True)
class OpenPITransitionOutput:
    state: PrefixKVState
    encoded_state: Tensor
    encoded_delta: Tensor
    gate: Tensor
    action_feature: Tensor


class OpenPIKVLatentLoop(nn.Module):
    """Predict full image/language pre-RoPE KV deltas with shared projections."""

    def __init__(self, config: OpenPIKVLatentLoopConfig = OpenPIKVLatentLoopConfig()) -> None:
        super().__init__()
        self.config = config
        transition = config.transition
        self.prefix_projection = nn.Sequential(
            nn.LayerNorm(config.prefix_embedding_dim),
            nn.Linear(config.prefix_embedding_dim, transition.observation_width),
            nn.GELU(),
            nn.LayerNorm(transition.observation_width),
        )
        self.action_encoder = ExecutedActionEncoder(
            action_dim=config.action_dim,
            hidden_dim=transition.action_width,
            output_dim=transition.action_width,
        )
        self.prefix_history_encoder = nn.GRU(
            transition.observation_width,
            transition.observation_width,
            batch_first=True,
        )
        self.robot_history_encoder = nn.GRU(
            transition.robot_state_dim,
            transition.robot_width,
            batch_first=True,
        )
        self.codec = LayerSharedKVCodec(config.head_dim, transition.state_width)
        self.transition = VariableTimeTransitionCore(transition)
        parameters = count_trainable_parameters(self)
        if parameters > config.parameter_cap:
            raise ValueError(f"OpenPI LatentLoop has {parameters:,} trainable parameters, cap is {config.parameter_cap:,}")

    @property
    def trainable_parameters(self) -> int:
        return count_trainable_parameters(self)

    def _layout_template(self, layout: PrefixEmbeddingState, previous: PrefixKVState) -> PrefixKVState:
        return PrefixKVState(
            embeddings=layout.embeddings,
            pad_mask=layout.pad_mask,
            attention_pattern=layout.attention_pattern,
            position_ids=layout.position_ids,
            pre_rope_keys=previous.pre_rope_keys,
            values=previous.values,
        )

    def forward(
        self,
        previous_state: PrefixKVState,
        current_prefix: PrefixEmbeddingState,
        previous_prefix_embeddings: Tensor,
        executed_actions: Tensor,
        robot_state: Tensor,
        *,
        delta_q: Tensor | int,
        delta_a: Tensor | int,
        full_refresh_age: Tensor | int,
        executed_action_lengths: Tensor | None = None,
        intermediate_prefix_embeddings: Tensor | None = None,
        robot_state_history: Tensor | None = None,
    ) -> OpenPITransitionOutput:
        previous_state.validate()
        current_prefix.validate()
        if current_prefix.embeddings.shape != previous_prefix_embeddings.shape:
            raise ValueError("current and previous prefix embeddings must be aligned")
        if current_prefix.embeddings.shape[1] != previous_state.num_tokens:
            raise ValueError("prefix token count changed; explicit alignment is required")
        packed = self.codec.pack(previous_state)
        encoded = self.codec.encode(packed)
        adapter_dtype = self.prefix_projection[1].weight.dtype
        current_obs = self.prefix_projection(current_prefix.embeddings.to(dtype=adapter_dtype))
        previous_obs = self.prefix_projection(previous_prefix_embeddings.to(dtype=adapter_dtype))
        if intermediate_prefix_embeddings is None:
            interval_observation = current_obs - previous_obs
        else:
            if intermediate_prefix_embeddings.ndim != 4:
                raise ValueError("intermediate_prefix_embeddings must be [B,M,S,E]")
            batch, interval, tokens, width = intermediate_prefix_embeddings.shape
            if (batch, tokens, width) != (
                current_obs.shape[0],
                current_obs.shape[1],
                self.config.prefix_embedding_dim,
            ):
                raise ValueError("intermediate prefix history does not match the current prefix layout")
            projected = self.prefix_projection(
                intermediate_prefix_embeddings.reshape(batch * interval, tokens, width).to(dtype=adapter_dtype)
            ).reshape(batch, interval, tokens, -1)
            previous_sequence = torch.cat((previous_obs[:, None], projected[:, :-1]), dim=1)
            changes = projected - previous_sequence
            history_input = changes.permute(0, 2, 1, 3).reshape(batch * tokens, interval, -1)
            _, history_hidden = self.prefix_history_encoder(history_input)
            interval_observation = history_hidden[-1].reshape(batch, tokens, -1)
        if robot_state_history is None:
            robot_state_history = robot_state[:, None]
        if robot_state_history.ndim != 3 or robot_state_history.shape[0] != robot_state.shape[0]:
            raise ValueError("robot_state_history must be [B,M,robot_state_dim]")
        if robot_state_history.shape[-1] != self.config.transition.robot_state_dim:
            raise ValueError("robot_state_history has the wrong feature dimension")
        robot_state = robot_state.to(dtype=adapter_dtype)
        robot_state_history = robot_state_history.to(dtype=adapter_dtype)
        _, robot_history_hidden = self.robot_history_encoder(robot_state_history)
        action = self.action_encoder(executed_actions.to(dtype=adapter_dtype), executed_action_lengths)
        transition = self.transition(
            encoded,
            current_obs,
            previous_obs,
            action.feature,
            robot_state,
            interval_observation=interval_observation,
            robot_history_feature=robot_history_hidden[-1],
            delta_q=delta_q,
            delta_a=delta_a,
            full_refresh_age=full_refresh_age,
        )
        predicted_packed = self.codec.apply_delta(packed, transition.state - encoded)
        template = self._layout_template(current_prefix, previous_state)
        predicted = self.codec.unpack_like(predicted_packed, template)
        return OpenPITransitionOutput(
            state=predicted,
            encoded_state=transition.state,
            encoded_delta=transition.delta,
            gate=transition.gate,
            action_feature=action.feature,
        )


def adapter_config_from_dict(payload: dict[str, object]) -> OpenPIKVLatentLoopConfig:
    transition_payload = dict(payload.get("transition", {}))
    transition = TransitionConfig(**transition_payload)
    outer = {key: value for key, value in payload.items() if key != "transition"}
    return OpenPIKVLatentLoopConfig(transition=transition, **outer)


def adapter_config_to_dict(config: OpenPIKVLatentLoopConfig) -> dict[str, object]:
    result = dict(config.__dict__)
    result["transition"] = dict(config.transition.__dict__)
    return result
