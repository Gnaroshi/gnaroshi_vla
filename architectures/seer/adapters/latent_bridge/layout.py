"""Verified token layout for Seer's causal transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class SeerTokenLayout:
    sequence_length: int
    resampler_queries_per_camera: int
    action_tokens: int
    observation_prediction_tokens: int = 0

    text_tokens: int = 1
    state_tokens: int = 1
    camera_cls_tokens: int = 2

    @classmethod
    def from_model(cls, model) -> "SeerTokenLayout":
        obs_tokens = int(model.NUM_OBS_TOKEN) if bool(model.obs_pred) else 0
        return cls(
            sequence_length=int(model.sequence_length),
            resampler_queries_per_camera=int(model.NUM_RESAMPLER_QUERY),
            action_tokens=int(model.action_pred_steps),
            observation_prediction_tokens=obs_tokens,
        )

    @property
    def primary_tokens(self) -> int:
        return self.resampler_queries_per_camera + 1

    @property
    def wrist_tokens(self) -> int:
        return self.resampler_queries_per_camera + 1

    @property
    def conditioning_tokens(self) -> int:
        return self.text_tokens + self.state_tokens + self.primary_tokens + self.wrist_tokens

    @property
    def tokens_per_timestep(self) -> int:
        return self.conditioning_tokens + self.observation_prediction_tokens + self.action_tokens

    @property
    def flattened_tokens(self) -> int:
        return self.sequence_length * self.tokens_per_timestep

    @property
    def per_timestep_slices(self) -> dict[str, slice]:
        q = self.resampler_queries_per_camera
        text = slice(0, 1)
        state = slice(1, 2)
        primary = slice(2, 2 + q)
        wrist = slice(2 + q, 2 + 2 * q)
        primary_cls = slice(2 + 2 * q, 3 + 2 * q)
        wrist_cls = slice(3 + 2 * q, 4 + 2 * q)
        obs_start = self.conditioning_tokens
        action_start = obs_start + self.observation_prediction_tokens
        return {
            "text": text,
            "state": state,
            "primary_resampled": primary,
            "wrist_resampled": wrist,
            "primary_cls": primary_cls,
            "wrist_cls": wrist_cls,
            "visual": slice(primary.start, wrist_cls.stop),
            "multimodal_context": slice(0, self.conditioning_tokens),
            "observation_prediction": slice(obs_start, action_start),
            "action": slice(action_start, action_start + self.action_tokens),
            "all": slice(0, self.tokens_per_timestep),
        }

    def validate_flattened(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 3:
            raise ValueError(f"expected [B, S*T, D], got {tuple(tensor.shape)}")
        if tensor.shape[1] != self.flattened_tokens:
            raise ValueError(
                f"flattened token mismatch: expected {self.flattened_tokens}, got {tensor.shape[1]}"
            )

    def as_timesteps(self, tensor: torch.Tensor) -> torch.Tensor:
        self.validate_flattened(tensor)
        return tensor.reshape(
            tensor.shape[0], self.sequence_length, self.tokens_per_timestep, tensor.shape[-1]
        )

    def select(self, tensor: torch.Tensor, *, timestep: int, group: str) -> torch.Tensor:
        if group not in self.per_timestep_slices:
            raise KeyError(f"unknown token group {group!r}")
        if timestep < 0:
            timestep += self.sequence_length
        if timestep < 0 or timestep >= self.sequence_length:
            raise IndexError(f"timestep {timestep} outside sequence length {self.sequence_length}")
        return self.as_timesteps(tensor)[:, timestep, self.per_timestep_slices[group], :]

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            {
                "conditioning_tokens": self.conditioning_tokens,
                "tokens_per_timestep": self.tokens_per_timestep,
                "flattened_tokens": self.flattened_tokens,
                "per_timestep_slices": {
                    key: [value.start, value.stop] for key, value in self.per_timestep_slices.items()
                },
            }
        )
        return result
