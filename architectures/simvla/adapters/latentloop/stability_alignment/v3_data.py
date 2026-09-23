"""Training-only hard pools and cross-query gripper supervision for V3."""

from __future__ import annotations

import math
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Sampler

from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    collate_exact_teacher_sequences,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    canonical_sha256,
)
from architectures.simvla.adapters.latentloop.stability_alignment.data import (
    StabilityExactTeacherDataset,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_contracts import (
    V3_HARD_POOL_SCHEMA,
)


class V3StabilityExactTeacherDataset(StabilityExactTeacherDataset):
    """Add q0 teacher action so age-1 query-boundary events are observable."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        anchor = self.store.query(self.windows[index][0])
        item["anchor_teacher_action"] = anchor["teacher_action"]
        return item

    def contract(self) -> dict[str, Any]:
        payload = super().contract()
        payload.update(
            {
                "schema_version": "simvla_stability_v3_dataset_v1",
                "anchor_teacher_action_included": True,
                "cross_query_gripper_boundary": "previous query executed index R-1 to current index 0",
            }
        )
        return payload


def collate_v3_sequences(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_exact_teacher_sequences(items)
    batch["anchor_teacher_action"] = torch.stack(
        [item["anchor_teacher_action"] for item in items]
    )
    return batch


def gripper_event_contract(
    anchor_action: torch.Tensor,
    teacher_actions: torch.Tensor,
    *,
    first_r: int = 5,
) -> dict[str, bool | int]:
    if anchor_action.ndim != 2 or anchor_action.shape[-1] != 7:
        raise ValueError("anchor_action must be [H,7]")
    if teacher_actions.ndim != 3 or teacher_actions.shape[-1] != 7:
        raise ValueError("teacher_actions must be [age,H,7]")
    current = teacher_actions[:, : int(first_r), 6].float() >= 0.0
    within = current[:, 1:] != current[:, :-1]
    previous_last = torch.cat(
        (
            (anchor_action[int(first_r) - 1, 6] >= 0.0).reshape(1),
            current[:-1, int(first_r) - 1],
        )
    )
    boundary = current[:, 0] != previous_last
    return {
        "has_any_event": bool(within.any().item() or boundary.any().item()),
        "has_within_query_event": bool(within.any().item()),
        "has_cross_query_event": bool(boundary.any().item()),
        "within_query_event_positions": int(within.sum().item()),
        "cross_query_event_positions": int(boundary.sum().item()),
    }


def build_v3_hard_pool_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_contract: Mapping[str, Any],
    source_lock: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("hard-pool construction needs score rows")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["dataset_index"]))
    indices = [int(row["dataset_index"]) for row in ordered]
    if len(indices) != len(set(indices)):
        raise ValueError("hard-pool rows contain duplicate dataset indices")
    hard_count = max(1, int(math.ceil(0.10 * len(ordered))))
    hard_rows = sorted(
        ordered,
        key=lambda row: (-float(row["tail_score"]), int(row["dataset_index"])),
    )[:hard_count]
    hard = {int(row["dataset_index"]) for row in hard_rows}
    gripper = {
        int(row["dataset_index"])
        for row in ordered
        if bool(row["has_any_gripper_event"]) and int(row["dataset_index"]) not in hard
    }
    base = {value for value in indices if value not in hard and value not in gripper}
    if not base or not gripper or not hard:
        raise RuntimeError("V3 needs nonempty base, gripper, and hard pools")
    cross_query = sum(bool(row["has_cross_query_gripper_event"]) for row in ordered)
    any_event = sum(bool(row["has_any_gripper_event"]) for row in ordered)
    payload: dict[str, Any] = {
        "schema_version": V3_HARD_POOL_SCHEMA,
        "split": "train",
        "dataset_contract": dict(dataset_contract),
        "source_lock": dict(source_lock),
        "pool_indices": {
            "base": sorted(base),
            "gripper_transition": sorted(gripper),
            "recurrence_action_tail": sorted(hard),
        },
        "pool_counts": {
            "base": len(base),
            "gripper_transition": len(gripper),
            "recurrence_action_tail": len(hard),
            "total_unique_windows": len(ordered),
        },
        "sampling_ratio": {
            "base": 0.70,
            "gripper_transition": 0.15,
            "recurrence_action_tail": 0.15,
        },
        "base_excludes_hard_pool": not bool(base & hard),
        "base_excludes_gripper_pool": not bool(base & gripper),
        "gripper_explicit_pool_excludes_hard_pool": not bool(gripper & hard),
        "explicit_pools_pairwise_disjoint": not bool(
            (base & gripper) or (base & hard) or (gripper & hard)
        ),
        "event_coverage": {
            "any_event_sequences": int(any_event),
            "cross_query_event_sequences": int(cross_query),
            "any_event_fraction": any_event / len(ordered),
            "cross_query_event_fraction": cross_query / len(ordered),
            "within_query_event_positions": int(
                sum(int(row["within_query_event_positions"]) for row in ordered)
            ),
            "cross_query_event_positions": int(
                sum(int(row["cross_query_event_positions"]) for row in ordered)
            ),
            "parent_age3_mismatch_sequences": int(
                sum(bool(row["parent_age3_sign_mismatch_sequence"]) for row in ordered)
            ),
            "parent_age3_mismatched_positions": int(
                sum(int(row["parent_age3_sign_mismatch_positions"]) for row in ordered)
            ),
        },
        "hard_score": (
            "mean of percentile ranks for frozen-parent age3 recurrence divergence "
            "and frozen-parent age3 learned-N_G3 first-R action error"
        ),
        "hard_quantile": 0.90,
        "hard_threshold": float(min(float(row["tail_score"]) for row in hard_rows)),
        "score_rows_sha256": canonical_sha256(ordered),
    }
    payload["combined_sha256"] = canonical_sha256(payload)
    return payload


class V3PoolSampler(Sampler[int]):
    """Deterministic explicit 70/15/15 stream over source-locked pools."""

    # Three gripper and three hard slots per 20 optimizer steps.
    _GRIPPER_SLOTS = frozenset({2, 9, 16})
    _HARD_SLOTS = frozenset({5, 12, 19})

    def __init__(
        self,
        contract: Mapping[str, Any],
        *,
        seed: int,
        start_step: int,
        stop_step: int,
    ) -> None:
        if contract.get("schema_version") != V3_HARD_POOL_SCHEMA:
            raise ValueError("V3 pool schema changed")
        if int(start_step) < 0 or int(stop_step) <= int(start_step):
            raise ValueError("invalid V3 sampler step interval")
        pools = contract["pool_indices"]
        self.pools = {
            "base": tuple(int(value) for value in pools["base"]),
            "gripper_transition": tuple(
                int(value) for value in pools["gripper_transition"]
            ),
            "recurrence_action_tail": tuple(
                int(value) for value in pools["recurrence_action_tail"]
            ),
        }
        if any(not values for values in self.pools.values()):
            raise ValueError("V3 sampler received an empty pool")
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self._permutation_cache: dict[tuple[str, int], tuple[int, ...]] = {}

    @classmethod
    def stream_name(cls, optimizer_step: int) -> str:
        slot = int(optimizer_step) % 20
        if slot in cls._GRIPPER_SLOTS:
            return "gripper_transition"
        if slot in cls._HARD_SLOTS:
            return "recurrence_action_tail"
        return "base"

    def _permuted(self, name: str, cycle: int) -> tuple[int, ...]:
        key = (name, int(cycle))
        if key in self._permutation_cache:
            return self._permutation_cache[key]
        stream = {
            "base": 0,
            "gripper_transition": 1,
            "recurrence_action_tail": 2,
        }[name]
        values = self.pools[name]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + stream * 1_000_003 + int(cycle))
        order = torch.randperm(len(values), generator=generator).tolist()
        result = tuple(values[index] for index in order)
        self._permutation_cache[key] = result
        return result

    def index(self, optimizer_step: int) -> tuple[int, str]:
        step = int(optimizer_step)
        name = self.stream_name(step)
        full_cycles, slot = divmod(step, 20)
        per_cycle = {
            "base": 14,
            "gripper_transition": 3,
            "recurrence_action_tail": 3,
        }
        logical = full_cycles * per_cycle[name]
        logical += sum(self.stream_name(value) == name for value in range(slot + 1)) - 1
        values = self.pools[name]
        cycle, offset = divmod(logical, len(values))
        return self._permuted(name, cycle)[offset], name

    def __iter__(self) -> Iterator[int]:
        for step in range(self.start_step, self.stop_step):
            yield self.index(step)[0]

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def state_dict(self, next_step: int) -> dict[str, Any]:
        counts = {name: 0 for name in self.pools}
        for step in range(self.start_step, int(next_step)):
            counts[self.stream_name(step)] += 1
        return {
            "schema_version": "simvla_stability_v3_sampler_v1",
            "seed": self.seed,
            "next_optimizer_step": int(next_step),
            "explicit_stream_counts": counts,
            "explicit_sampling_ratio": {
                name: count / max(sum(counts.values()), 1)
                for name, count in counts.items()
            },
        }
