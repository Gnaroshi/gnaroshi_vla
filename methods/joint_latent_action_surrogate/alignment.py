"""Exact Seer action-token alignment for an immutable horizon anchor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlignedActionAnchor:
    """Anchor values aligned to the current query and their validity mask."""

    values: Tensor
    valid_mask: Tensor
    elapsed: Tensor


class ImmutableActionAnchorCache:
    """Small runtime state whose anchor changes only on an exact write."""

    def __init__(self) -> None:
        self.arm: Tensor | None = None
        self.gripper_logit: Tensor | None = None
        self.latent: Tensor | None = None
        self.query_time: int | None = None
        self.generation = -1

    @property
    def initialized(self) -> bool:
        return self.arm is not None

    def write_exact(
        self,
        arm: Tensor,
        gripper_logit: Tensor,
        latent: Tensor,
        query_time: int,
    ) -> None:
        if arm.shape[:-1] + (1,) != gripper_logit.shape:
            raise ValueError("Exact anchor arm/gripper horizons do not align")
        self.arm = arm.detach().clone()
        self.gripper_logit = gripper_logit.detach().clone()
        self.latent = latent.detach().clone()
        self.query_time = int(query_time)
        self.generation += 1

    def snapshot(self) -> tuple[Tensor, Tensor, Tensor, int, int]:
        if not self.initialized:
            raise RuntimeError("Exact anchor is not initialized")
        assert self.arm is not None
        assert self.gripper_logit is not None
        assert self.latent is not None
        assert self.query_time is not None
        return (
            self.arm.detach().clone(),
            self.gripper_logit.detach().clone(),
            self.latent.detach().clone(),
            self.query_time,
            self.generation,
        )

    def state_dict(self) -> dict[str, Tensor | int | None]:
        if not self.initialized:
            return {"query_time": None, "generation": self.generation}
        arm, gripper, latent, query_time, generation = self.snapshot()
        return {
            "arm": arm,
            "gripper_logit": gripper,
            "latent": latent,
            "query_time": query_time,
            "generation": generation,
        }

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        if state.get("query_time") is None:
            self.__init__()
            return
        self.arm = torch.as_tensor(state["arm"]).detach().clone()
        self.gripper_logit = torch.as_tensor(state["gripper_logit"]).detach().clone()
        self.latent = torch.as_tensor(state["latent"]).detach().clone()
        self.query_time = int(state["query_time"])
        self.generation = int(state["generation"])


def _elapsed_tensor(elapsed: Tensor | int, leading: tuple[int, ...], ref: Tensor) -> Tensor:
    value = torch.as_tensor(elapsed, device=ref.device, dtype=torch.long)
    if value.ndim == 0:
        value = value.expand(leading)
    else:
        value = torch.broadcast_to(value, leading)
    if torch.any(value < 0):
        raise ValueError("Anchor elapsed time must be non-negative")
    return value


def align_immutable_anchor(anchor: Tensor, elapsed: Tensor | int) -> AlignedActionAnchor:
    """Shift ``[..., P, A]`` by an arbitrary elapsed query count.

    Seer's token ``h`` predicts the action at query-relative offset ``h``.  At
    elapsed age ``m``, current token ``h`` overlaps anchor token ``h + m``.
    Non-overlapping tail entries are exactly zero and explicitly invalid; the
    last valid token is never copied into the tail.
    """

    if anchor.ndim < 2 or anchor.shape[-2] < 1:
        raise ValueError(f"Expected anchor [..., P, A], got {tuple(anchor.shape)}")
    leading = tuple(anchor.shape[:-2])
    token_count = int(anchor.shape[-2])
    elapsed_tensor = _elapsed_tensor(elapsed, leading, anchor)
    token = torch.arange(token_count, device=anchor.device, dtype=torch.long)
    source = elapsed_tensor.unsqueeze(-1) + token.reshape((1,) * len(leading) + (-1,))
    valid = source < token_count
    clamped = source.clamp(max=token_count - 1)
    gather_index = clamped.unsqueeze(-1).expand(*leading, token_count, anchor.shape[-1])
    shifted = torch.gather(anchor, dim=-2, index=gather_index)
    shifted = torch.where(valid.unsqueeze(-1), shifted, torch.zeros_like(shifted))
    return AlignedActionAnchor(
        values=shifted,
        valid_mask=valid.unsqueeze(-1),
        elapsed=elapsed_tensor,
    )
