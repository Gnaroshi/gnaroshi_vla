"""Architecture-neutral modules for the corrected native SimVLA V0 path.

V0 predicts the current action condition from the previous action condition and
the latest observation/proprioception change.  It deliberately has no executed
action input and no action-correction head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ImageViews = Tensor | Sequence[Tensor]


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of parameters visible to an optimizer."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _as_ordered_views(images: ImageViews) -> list[Tensor]:
    if torch.is_tensor(images):
        if images.ndim == 5:
            return [images[:, index] for index in range(images.shape[1])]
        if images.ndim == 4:
            return [images]
        if images.ndim == 3:
            return [images.unsqueeze(0)]
        raise ValueError(f"Unsupported image shape: {tuple(images.shape)}")
    views = list(images)
    if not views:
        raise ValueError("at least one camera view is required")
    return views


def _float_chw(image: Tensor) -> Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected batched image [B,C,H,W] or [B,H,W,C], got {tuple(image.shape)}")
    if image.shape[1] in {1, 3, 4}:
        result = image
    elif image.shape[-1] in {1, 3, 4}:
        result = image.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Cannot infer image channels from {tuple(image.shape)}")
    result = result.float()
    if result.numel() and float(result.detach().amax().item()) > 2.0:
        result = result / 255.0
    if result.shape[1] == 4:
        result = result[:, :3]
    if result.shape[1] == 1:
        result = result.repeat(1, 3, 1, 1)
    return result


@dataclass(frozen=True)
class NativeV0ObservationPair:
    """Only inputs allowed to the corrected V0 delta encoder."""

    previous_images: ImageViews
    current_images: ImageViews
    previous_proprio: Tensor
    current_proprio: Tensor


class NativeV0DeltaEncoder(nn.Module):
    """Encode ordered multi-view RGB and proprioception changes.

    A shared CNN is applied independently to every camera view.  View features
    are concatenated in source order, rather than averaged, so swapping cameras
    changes the result while sharing all image weights.
    """

    def __init__(
        self,
        *,
        num_views: int = 2,
        proprio_dim: int = 8,
        image_size: int = 64,
        per_view_dim: int = 128,
        proprio_feature_dim: int = 128,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be positive")
        self.num_views = int(num_views)
        self.proprio_dim = int(proprio_dim)
        self.image_size = int(image_size)
        self.per_view_dim = int(per_view_dim)
        self.output_dim = int(output_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(12, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, self.per_view_dim),
            nn.GELU(),
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(3 * self.proprio_dim, proprio_feature_dim),
            nn.GELU(),
            nn.Linear(proprio_feature_dim, proprio_feature_dim),
            nn.GELU(),
        )
        fusion_dim = self.num_views * self.per_view_dim + proprio_feature_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, pair: NativeV0ObservationPair) -> Tensor:
        previous_views = _as_ordered_views(pair.previous_images)
        current_views = _as_ordered_views(pair.current_images)
        if len(previous_views) != self.num_views or len(current_views) != self.num_views:
            raise ValueError(
                f"expected exactly {self.num_views} ordered camera views, got "
                f"{len(previous_views)} and {len(current_views)}"
            )
        previous_q = pair.previous_proprio
        current_q = pair.current_proprio
        if previous_q.ndim != 2 or current_q.shape != previous_q.shape:
            raise ValueError("previous/current proprio must have identical [B,Q] shapes")
        if previous_q.shape[-1] != self.proprio_dim:
            raise ValueError(f"expected proprio dim {self.proprio_dim}, got {previous_q.shape[-1]}")

        ordered_features: list[Tensor] = []
        for previous, current in zip(previous_views, current_views):
            previous_f = _float_chw(previous)
            current_f = _float_chw(current).to(previous_f.device)
            if previous_f.shape != current_f.shape:
                raise ValueError("previous/current image shapes must match for every view")
            previous_f = F.interpolate(
                previous_f,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            current_f = F.interpolate(
                current_f,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            ordered_features.append(
                self.image_encoder(
                    torch.cat((previous_f, current_f, current_f - previous_f), dim=1)
                )
            )
        reference = ordered_features[0]
        previous_q = previous_q.to(device=reference.device, dtype=reference.dtype)
        current_q = current_q.to(device=reference.device, dtype=reference.dtype)
        proprio_feature = self.proprio_encoder(
            torch.cat((previous_q, current_q, current_q - previous_q), dim=-1)
        )
        return self.fusion(torch.cat((*ordered_features, proprio_feature), dim=-1))


@dataclass(frozen=True)
class NativeV0UpdateOutput:
    """One token-shared condition update and diagnostics."""

    condition: Tensor
    residual: Tensor
    gate: Tensor


class TokenSharedConditionUpdater(nn.Module):
    """Apply the same low-rank transition weights to every valid token."""

    def __init__(
        self,
        *,
        condition_dim: int = 960,
        delta_dim: int = 128,
        rank_dim: int = 64,
        max_tokens: int = 256,
        num_token_groups: int = 8,
        max_age: int = 3,
        gate_bias: float = -4.0,
    ) -> None:
        super().__init__()
        if rank_dim not in {64, 96}:
            raise ValueError("only the primary rank 64 or disabled diagnostic rank 96 is defined")
        self.condition_dim = int(condition_dim)
        self.delta_dim = int(delta_dim)
        self.rank_dim = int(rank_dim)
        self.max_tokens = int(max_tokens)
        self.num_token_groups = int(num_token_groups)
        self.max_age = int(max_age)
        self.norm = nn.LayerNorm(self.condition_dim)
        self.down = nn.Linear(self.condition_dim, self.rank_dim)
        self.delta_projection = nn.Linear(self.delta_dim, self.rank_dim)
        self.token_embedding = nn.Embedding(self.max_tokens, self.rank_dim)
        self.group_embedding = nn.Embedding(self.num_token_groups, self.rank_dim)
        self.age_embedding = nn.Embedding(self.max_age + 1, self.rank_dim)
        self.up = nn.Linear(self.rank_dim, self.condition_dim)
        self.gate_head = nn.Linear(self.rank_dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))

    def forward(
        self,
        previous_condition: Tensor,
        delta_feature: Tensor,
        *,
        valid_mask: Tensor,
        group_ids: Tensor,
        age: Tensor | int,
    ) -> NativeV0UpdateOutput:
        if previous_condition.ndim != 3 or previous_condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"previous_condition must be [B,T,{self.condition_dim}], got "
                f"{tuple(previous_condition.shape)}"
            )
        batch_size, token_count, _ = previous_condition.shape
        if token_count > self.max_tokens:
            raise ValueError(f"condition token count {token_count} exceeds {self.max_tokens}")
        if delta_feature.shape != (batch_size, self.delta_dim):
            raise ValueError(
                f"delta_feature must be {(batch_size, self.delta_dim)}, got {tuple(delta_feature.shape)}"
            )
        valid_mask = valid_mask.to(device=previous_condition.device, dtype=torch.bool)
        group_ids = group_ids.to(device=previous_condition.device, dtype=torch.long)
        if valid_mask.shape != (batch_size, token_count):
            raise ValueError("valid_mask must be [B,T]")
        if group_ids.shape != (batch_size, token_count):
            raise ValueError("group_ids must be [B,T]")
        if bool((group_ids < 0).any()) or bool((group_ids >= self.num_token_groups).any()):
            raise ValueError("group_ids are outside the configured embedding table")
        age_tensor = torch.as_tensor(age, device=previous_condition.device, dtype=torch.long)
        if age_tensor.ndim == 0:
            age_tensor = age_tensor.expand(batch_size)
        if age_tensor.shape != (batch_size,):
            raise ValueError("age must be scalar or [B]")
        if bool((age_tensor < 1).any()) or bool((age_tensor > self.max_age).any()):
            raise ValueError(f"age must be in [1,{self.max_age}]")

        token_ids = torch.arange(token_count, device=previous_condition.device)
        hidden = self.down(self.norm(previous_condition))
        hidden = hidden + self.delta_projection(delta_feature).unsqueeze(1)
        hidden = hidden + self.token_embedding(token_ids).unsqueeze(0)
        hidden = hidden + self.group_embedding(group_ids)
        hidden = hidden + self.age_embedding(age_tensor).unsqueeze(1)
        hidden = F.gelu(hidden)
        residual = self.up(hidden)
        gate = torch.sigmoid(self.gate_head(hidden))
        candidate = previous_condition + gate * residual
        condition = torch.where(valid_mask.unsqueeze(-1), candidate, previous_condition)
        residual = torch.where(valid_mask.unsqueeze(-1), residual, torch.zeros_like(residual))
        gate = torch.where(valid_mask.unsqueeze(-1), gate, torch.zeros_like(gate))
        return NativeV0UpdateOutput(condition=condition, residual=residual, gate=gate)


@dataclass(frozen=True)
class NativeV0UnrollOutput:
    """Recursive age-1/2/3 conditions and per-age diagnostics."""

    conditions: tuple[Tensor, Tensor, Tensor]
    delta_features: tuple[Tensor, Tensor, Tensor]
    updates: tuple[NativeV0UpdateOutput, NativeV0UpdateOutput, NativeV0UpdateOutput]


class NativeSimVLAV0(nn.Module):
    """Correct V0 composition for native SimVLA K=4."""

    def __init__(
        self,
        *,
        num_views: int = 2,
        proprio_dim: int = 8,
        condition_dim: int = 960,
        delta_dim: int = 128,
        rank_dim: int = 64,
        max_tokens: int = 256,
        num_token_groups: int = 8,
    ) -> None:
        super().__init__()
        self.delta_encoder = NativeV0DeltaEncoder(
            num_views=num_views,
            proprio_dim=proprio_dim,
            output_dim=delta_dim,
        )
        self.condition_updater = TokenSharedConditionUpdater(
            condition_dim=condition_dim,
            delta_dim=delta_dim,
            rank_dim=rank_dim,
            max_tokens=max_tokens,
            num_token_groups=num_token_groups,
            max_age=3,
        )
        self.num_views = int(num_views)
        self.proprio_dim = int(proprio_dim)
        self.condition_dim = int(condition_dim)
        self.delta_dim = int(delta_dim)
        self.rank_dim = int(rank_dim)

    def update_once(
        self,
        previous_condition: Tensor,
        pair: NativeV0ObservationPair,
        *,
        valid_mask: Tensor,
        group_ids: Tensor,
        age: Tensor | int,
    ) -> NativeV0UpdateOutput:
        delta = self.delta_encoder(pair)
        return self.condition_updater(
            previous_condition,
            delta,
            valid_mask=valid_mask,
            group_ids=group_ids,
            age=age,
        )

    def unroll_k4(
        self,
        anchor_condition: Tensor,
        image_sequence: Tensor,
        proprio_sequence: Tensor,
        *,
        valid_mask: Tensor,
        group_ids: Tensor,
    ) -> NativeV0UnrollOutput:
        """Unroll q0->q1->q2->q3 with full BPTT and no teacher input.

        ``image_sequence`` is ``[B,4,V,...]`` and ``proprio_sequence`` is
        ``[B,4,Q]``.  Teacher conditions are intentionally absent from this
        signature, making accidental teacher forcing impossible.
        """

        if image_sequence.ndim not in {6, 7} or image_sequence.shape[1] != 4:
            raise ValueError("image_sequence must have query axis [B,4,V,...]")
        if proprio_sequence.ndim != 3 or proprio_sequence.shape[1] != 4:
            raise ValueError("proprio_sequence must be [B,4,Q]")
        previous = anchor_condition
        conditions: list[Tensor] = []
        deltas: list[Tensor] = []
        updates: list[NativeV0UpdateOutput] = []
        for age in (1, 2, 3):
            pair = NativeV0ObservationPair(
                previous_images=image_sequence[:, age - 1],
                current_images=image_sequence[:, age],
                previous_proprio=proprio_sequence[:, age - 1],
                current_proprio=proprio_sequence[:, age],
            )
            delta = self.delta_encoder(pair)
            update = self.condition_updater(
                previous,
                delta,
                valid_mask=valid_mask,
                group_ids=group_ids,
                age=age,
            )
            deltas.append(delta)
            updates.append(update)
            conditions.append(update.condition)
            previous = update.condition
        return NativeV0UnrollOutput(
            conditions=(conditions[0], conditions[1], conditions[2]),
            delta_features=(deltas[0], deltas[1], deltas[2]),
            updates=(updates[0], updates[1], updates[2]),
        )

    def forward(
        self,
        anchor_condition: Tensor,
        image_sequence: Tensor,
        proprio_sequence: Tensor,
        *,
        valid_mask: Tensor,
        group_ids: Tensor,
    ) -> NativeV0UnrollOutput:
        """DDP-visible alias for the fixed recursive K=4 unroll."""

        return self.unroll_k4(
            anchor_condition,
            image_sequence,
            proprio_sequence,
            valid_mask=valid_mask,
            group_ids=group_ids,
        )

    def parameter_audit(self) -> dict[str, int | bool]:
        observation = count_trainable_parameters(self.delta_encoder)
        updater = self.condition_updater
        gates = sum(parameter.numel() for parameter in updater.gate_head.parameters())
        embeddings = sum(
            parameter.numel()
            for module in (updater.token_embedding, updater.group_embedding, updater.age_embedding)
            for parameter in module.parameters()
        )
        transition = count_trainable_parameters(updater) - gates - embeddings
        total = observation + transition + gates + embeddings
        return {
            "observation_change_encoder": observation,
            "token_transition": transition,
            "gates": gates,
            "embeddings": embeddings,
            "total": total,
            "under_hard_cap_1000000": total <= 1_000_000,
            "in_target_range_500000_1000000": 500_000 <= total <= 1_000_000,
        }
