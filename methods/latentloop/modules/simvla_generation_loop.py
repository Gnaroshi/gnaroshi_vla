"""Schedule-independent hidden updater for the SimVLA Generation Loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GenerationUpdateOutput:
    hidden: Tensor
    residual: Tensor
    gate: Tensor


@dataclass(frozen=True)
class GenerationFlowTrace:
    final_noisy_action: Tensor
    full_step_indices: tuple[int, ...]
    skipped_step_indices: tuple[int, ...]
    skipped_ages: tuple[int, ...]
    predicted_hidden: tuple[Tensor, ...]
    predicted_velocity: tuple[Tensor, ...]
    skipped_noisy_actions: tuple[Tensor, ...]
    skipped_times: tuple[Tensor, ...]


FullStep = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


class SimVLAGenerationHiddenUpdater(nn.Module):
    """Update action-token hidden state between frozen full-transformer calls.

    The same weights are used for every action token, flow position, skipped
    age, and supported N_G schedule. The compact condition-change code is an
    input; this module deliberately contains no observation encoder.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        condition_dim: int = 960,
        action_dim: int = 7,
        proprio_dim: int = 8,
        condition_code_dim: int = 128,
        rank_dim: int = 128,
        max_generator_age: int = 3,
        gate_bias: float = -4.0,
    ) -> None:
        super().__init__()
        if max_generator_age < 1 or max_generator_age > 9:
            raise ValueError("max_generator_age must be in [1,9]")
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = int(condition_dim)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.condition_code_dim = int(condition_code_dim)
        self.rank_dim = int(rank_dim)
        self.max_generator_age = int(max_generator_age)

        self.hidden_norm = nn.LayerNorm(self.hidden_dim)
        self.hidden_down = nn.Linear(self.hidden_dim, self.rank_dim)
        self.action_projection = nn.Linear(3 * self.action_dim, self.rank_dim)
        self.proprio_projection = nn.Linear(self.proprio_dim, self.rank_dim)
        self.condition_projection = nn.Linear(self.condition_dim, self.rank_dim)
        self.condition_code_projection = nn.Linear(self.condition_code_dim, self.rank_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(4, self.rank_dim),
            nn.GELU(),
            nn.Linear(self.rank_dim, self.rank_dim),
        )
        self.age_embedding = nn.Embedding(self.max_generator_age + 1, self.rank_dim)
        self.token_embedding = nn.Embedding(10, self.rank_dim)
        self.mixer = nn.Sequential(
            nn.LayerNorm(self.rank_dim),
            nn.Linear(self.rank_dim, 2 * self.rank_dim),
            nn.GELU(),
            nn.Linear(2 * self.rank_dim, self.rank_dim),
            nn.GELU(),
        )
        self.hidden_up = nn.Linear(self.rank_dim, self.hidden_dim)
        self.gate_head = nn.Linear(self.rank_dim, 1)
        nn.init.zeros_(self.hidden_up.weight)
        nn.init.zeros_(self.hidden_up.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))

    def forward(
        self,
        previous_hidden: Tensor,
        noisy_action_before: Tensor,
        noisy_action_after: Tensor,
        *,
        tau_before: Tensor,
        tau_after: Tensor,
        proprio: Tensor,
        condition_change_code: Tensor,
        condition: Tensor,
        condition_valid_mask: Tensor | None,
        generator_age: Tensor | int,
    ) -> GenerationUpdateOutput:
        if previous_hidden.ndim != 3 or previous_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"previous_hidden must be [B,A,{self.hidden_dim}]")
        batch, action_tokens, _ = previous_hidden.shape
        expected_action = (batch, action_tokens, self.action_dim)
        if noisy_action_before.shape != expected_action or noisy_action_after.shape != expected_action:
            raise ValueError(f"noisy actions must both be {expected_action}")
        if proprio.shape != (batch, self.proprio_dim):
            raise ValueError(f"proprio must be {(batch, self.proprio_dim)}")
        if condition_change_code.shape != (batch, self.condition_code_dim):
            raise ValueError(
                f"condition_change_code must be {(batch, self.condition_code_dim)}"
            )
        if condition.ndim != 3 or condition.shape[0] != batch or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition must be [B,T,{self.condition_dim}], got {tuple(condition.shape)}"
            )
        if condition_valid_mask is None:
            condition_summary = condition.mean(dim=1)
        else:
            if condition_valid_mask.shape != condition.shape[:2]:
                raise ValueError("condition_valid_mask must match condition [B,T]")
            condition_weight = condition_valid_mask.to(
                device=condition.device, dtype=condition.dtype
            ).unsqueeze(-1)
            condition_summary = (condition * condition_weight).sum(dim=1) / condition_weight.sum(
                dim=1
            ).clamp_min(1.0)
        if action_tokens > self.token_embedding.num_embeddings:
            raise ValueError("action token count exceeds the native H=10 contract")

        tau_before = torch.as_tensor(
            tau_before, device=previous_hidden.device, dtype=previous_hidden.dtype
        ).reshape(batch)
        tau_after = torch.as_tensor(
            tau_after, device=previous_hidden.device, dtype=previous_hidden.dtype
        ).reshape(batch)
        age = torch.as_tensor(
            generator_age, device=previous_hidden.device, dtype=torch.long
        )
        if age.ndim == 0:
            age = age.expand(batch)
        if (
            age.shape != (batch,)
            or bool((age < 1).any())
            or bool((age > self.max_generator_age).any())
        ):
            raise ValueError(
                "generator_age must be scalar or [B] with values "
                f"1..{self.max_generator_age}"
            )

        action_features = torch.cat(
            (
                noisy_action_before,
                noisy_action_after,
                noisy_action_after - noisy_action_before,
            ),
            dim=-1,
        )
        time_features = torch.stack(
            (
                tau_before,
                tau_after,
                tau_after - tau_before,
                tau_before * tau_after,
            ),
            dim=-1,
        )
        token_ids = torch.arange(action_tokens, device=previous_hidden.device)
        latent = self.hidden_down(self.hidden_norm(previous_hidden))
        latent = latent + self.action_projection(action_features)
        latent = latent + self.proprio_projection(proprio).unsqueeze(1)
        latent = latent + self.condition_projection(condition_summary).unsqueeze(1)
        latent = latent + self.condition_code_projection(condition_change_code).unsqueeze(1)
        latent = latent + self.time_projection(time_features).unsqueeze(1)
        latent = latent + self.age_embedding(age).unsqueeze(1)
        latent = latent + self.token_embedding(token_ids).unsqueeze(0)
        latent = self.mixer(F.gelu(latent))
        residual = self.hidden_up(latent)
        gate = torch.sigmoid(self.gate_head(latent))
        return GenerationUpdateOutput(
            hidden=previous_hidden + gate * residual,
            residual=residual,
            gate=gate,
        )

    def parameter_audit(self) -> dict[str, int | bool | list[int]]:
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "trainable_parameters": trainable,
            "hard_cap": 1_000_000,
            "under_hard_cap": trainable <= 1_000_000,
            "contains_observation_encoder": False,
            "uses_full_condition_q_j": True,
            "uses_condition_change_code_c_j": True,
            "schedule_independent": True,
            "trained_generator_ages": list(range(1, self.max_generator_age + 1)),
        }


class SimVLAGenerationLoop(nn.Module):
    """Integrate all ten native flow steps while replacing selected hidden calls."""

    def __init__(self, updater: SimVLAGenerationHiddenUpdater, decoder: nn.Module) -> None:
        super().__init__()
        self.updater = updater
        self.decoder = decoder
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.decoder.eval()

    def forward(
        self,
        initial_noisy_action: Tensor,
        *,
        full_step: FullStep,
        full_step_indices: Sequence[int],
        proprio: Tensor,
        condition: Tensor,
        condition_valid_mask: Tensor | None,
        condition_change_code: Tensor,
    ) -> GenerationFlowTrace:
        full_indices = tuple(int(value) for value in full_step_indices)
        if not full_indices or full_indices[0] != 0:
            raise ValueError("Generation Loop schedules must start with a full step at index 0")
        if tuple(sorted(set(full_indices))) != full_indices:
            raise ValueError("full step indices must be unique and increasing")
        if any(value < 0 or value >= 10 for value in full_indices):
            raise ValueError("full step indices must be in [0,9]")
        x = initial_noisy_action
        batch = x.shape[0]
        dt = -0.1
        previous_hidden: Tensor | None = None
        previous_x: Tensor | None = None
        skipped_indices: list[int] = []
        skipped_ages: list[int] = []
        predicted_hidden: list[Tensor] = []
        predicted_velocity: list[Tensor] = []
        skipped_actions: list[Tensor] = []
        skipped_times: list[Tensor] = []
        age = 0
        full_set = set(full_indices)
        for step in range(10):
            tau = x.new_full((batch,), 1.0 - step / 10.0)
            if step in full_set:
                hidden, velocity = full_step(x, tau)
                age = 0
            else:
                if previous_hidden is None or previous_x is None:
                    raise RuntimeError("skipped flow step has no preceding full/predicted hidden")
                age += 1
                if age > self.updater.max_generator_age:
                    raise RuntimeError(
                        "Generation Loop schedule requires generator age "
                        f"{age}, but updater supports only "
                        f"1..{self.updater.max_generator_age}"
                    )
                update = self.updater(
                    previous_hidden,
                    previous_x,
                    x,
                    tau_before=x.new_full((batch,), 1.0 - (step - 1) / 10.0),
                    tau_after=tau,
                    proprio=proprio,
                    condition=condition,
                    condition_valid_mask=condition_valid_mask,
                    condition_change_code=condition_change_code,
                    generator_age=age,
                )
                hidden = update.hidden
                velocity = self.decoder(hidden)
                skipped_indices.append(step)
                skipped_ages.append(age)
                predicted_hidden.append(hidden)
                predicted_velocity.append(velocity)
                skipped_actions.append(x)
                skipped_times.append(tau)
            previous_hidden = hidden
            previous_x = x
            x = x + dt * velocity
        return GenerationFlowTrace(
            final_noisy_action=x,
            full_step_indices=full_indices,
            skipped_step_indices=tuple(skipped_indices),
            skipped_ages=tuple(skipped_ages),
            predicted_hidden=tuple(predicted_hidden),
            predicted_velocity=tuple(predicted_velocity),
            skipped_noisy_actions=tuple(skipped_actions),
            skipped_times=tuple(skipped_times),
        )
