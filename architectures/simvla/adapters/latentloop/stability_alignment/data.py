"""Episode-disjoint exact-cache datasets and event-aware sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import Sampler

from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherSequenceDataset,
    ExactTeacherStore,
)
from architectures.simvla.adapters.latentloop.native_v0_dataset import (
    stable_episode_partition,
)


SPLITS = ("train", "checkpoint_validation", "final_offline", "all")


def compose_q0_q7_windows(store: ExactTeacherStore) -> tuple[tuple[str, ...], ...]:
    """Join adjacent source q0-q3 chunks without duplicating query tensors."""

    grouped: dict[tuple[int, str], dict[int, tuple[str, ...]]] = {}
    for raw_window in store.manifest["windows"]:
        window = tuple(str(value) for value in raw_window)
        if len(window) != 4:
            raise ValueError("source exact-cache windows must contain q0-q3")
        metadata = store.query(window[0])["metadata"]
        identity = (int(metadata["task_id"]), str(metadata["episode_id"]))
        anchor = int(metadata["query_index"])
        grouped.setdefault(identity, {})[anchor] = window
    result: list[tuple[str, ...]] = []
    for chunks in grouped.values():
        for anchor in sorted(chunks):
            if anchor + 4 not in chunks:
                continue
            combined = chunks[anchor] + chunks[anchor + 4]
            indices = [
                int(store.query(query_id)["metadata"]["query_index"])
                for query_id in combined
            ]
            if indices != list(range(anchor, anchor + 8)):
                raise RuntimeError("q0-q7 cache composition is not query-contiguous")
            result.append(combined)
    if not result:
        raise RuntimeError("exact cache contains no adjacent q0-q7 sequences")
    return tuple(result)


def episode_split(task_id: int, episode_id: str, split_seed: int) -> str:
    """Keep the historical 20% P1 set intact as final_offline."""

    value = stable_episode_partition(int(task_id), str(episode_id), int(split_seed))
    if value < 0.20:
        return "final_offline"
    if value < 0.30:
        return "checkpoint_validation"
    return "train"


class StabilityExactTeacherDataset(ExactTeacherSequenceDataset):
    """Exact q0-q3 windows with a three-way episode-disjoint split."""

    def __init__(
        self,
        cache: str | Path,
        *,
        split: str,
        split_seed: int = 20260822,
        max_age: int = 3,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.store = ExactTeacherStore(cache)
        if int(max_age) not in {3, 7}:
            raise ValueError("max_age must be 3 or 7")
        source_windows = (
            tuple(tuple(str(value) for value in raw) for raw in self.store.manifest["windows"])
            if int(max_age) == 3
            else compose_q0_q7_windows(self.store)
        )
        windows: list[tuple[str, ...]] = []
        identities: list[tuple[int, str, int]] = []
        for window in source_windows:
            first = self.store.query(window[0])["metadata"]
            assigned = episode_split(
                int(first["task_id"]), str(first["episode_id"]), int(split_seed)
            )
            if split != "all" and assigned != split:
                continue
            windows.append(window)
            identities.append(
                (
                    int(first["task_id"]),
                    str(first["episode_id"]),
                    int(first["query_index"]),
                )
            )
        if not windows:
            raise ValueError(f"no exact teacher windows found for split={split}")
        self.windows = tuple(windows)
        self.identities = tuple(identities)
        self.split = str(split)
        self.split_seed = int(split_seed)
        self.max_age = int(max_age)
        self.heldout_fraction = 0.20
        encoded = json.dumps(
            self.identities, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        self.split_sha256 = hashlib.sha256(encoded).hexdigest()

    def contract(self) -> dict[str, Any]:
        parent = super().contract()
        parent.update(
            {
                "split": self.split,
                "split_contract": {
                    "train": "stable episode hash in [0.30,1.00)",
                    "checkpoint_validation": "stable episode hash in [0.20,0.30)",
                    "final_offline": "historical P1 stable episode hash in [0.00,0.20)",
                },
                "condition_ages": list(range(1, self.max_age + 1)),
                "sequence_queries": self.max_age + 1,
                "fixed_k_c": self.max_age + 1,
                "long_span_query_tensors_reused_without_duplication": self.max_age == 7,
            }
        )
        return parent


def teacher_gripper_event(teacher_actions: torch.Tensor, first_r: int = 5) -> bool:
    if teacher_actions.ndim != 3 or teacher_actions.shape[-1] != 7:
        raise ValueError("teacher_actions must be [age,H,7]")
    gripper = teacher_actions[:, : int(first_r), 6].float()
    sign = gripper >= 0.0
    within = sign[:, 1:] != sign[:, :-1]
    across_age = sign[1:, 0] != sign[:-1, -1]
    return bool(within.any().item() or across_age.any().item())


def build_event_index(dataset: StabilityExactTeacherDataset) -> dict[str, Any]:
    natural: list[int] = []
    event: list[int] = []
    for index, window in enumerate(dataset.windows):
        actions = torch.stack(
            [dataset.store.query(query_id)["teacher_action"] for query_id in window[1:]]
        )
        natural.append(index)
        if teacher_gripper_event(actions):
            event.append(index)
    if not event:
        raise RuntimeError("exact cache contains no gripper event windows")
    return {
        "schema_version": "simvla_stability_event_index_v1",
        "split": dataset.split,
        "split_sha256": dataset.split_sha256,
        "natural_indices": natural,
        "event_indices": event,
        "natural_count": len(natural),
        "event_count": len(event),
        "sampling_contract": "three natural optimizer steps followed by one event step",
    }


class ReplicatedEventAwareSampler(Sampler[int]):
    """Deterministic 75/25 sampler shared by both DDP replicas and branches."""

    def __init__(
        self,
        event_index: dict[str, Any],
        *,
        seed: int,
        start_step: int,
        stop_step: int,
    ) -> None:
        if start_step < 0 or stop_step <= start_step:
            raise ValueError("invalid optimizer-step interval")
        self.natural = tuple(int(value) for value in event_index["natural_indices"])
        self.events = tuple(int(value) for value in event_index["event_indices"])
        if not self.natural or not self.events:
            raise ValueError("event-aware sampler needs natural and event indices")
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)

    def _permuted(self, values: Sequence[int], cycle: int, stream: int) -> tuple[int, ...]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 1_000_003 * int(stream) + int(cycle))
        order = torch.randperm(len(values), generator=generator).tolist()
        return tuple(values[index] for index in order)

    def index(self, optimizer_step: int) -> int:
        step = int(optimizer_step)
        event_step = step % 4 == 3
        sequence = self.events if event_step else self.natural
        stream = 1 if event_step else 0
        logical = step // 4 if event_step else step - (step // 4)
        cycle, offset = divmod(logical, len(sequence))
        return self._permuted(sequence, cycle, stream)[offset]

    def __iter__(self) -> Iterator[int]:
        for step in range(self.start_step, self.stop_step):
            yield self.index(step)

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def state_dict(self, next_step: int) -> dict[str, Any]:
        return {
            "sampler": "replicated_event_aware_75_25",
            "seed": self.seed,
            "next_optimizer_step": int(next_step),
            "natural_count": len(self.natural),
            "event_count": len(self.events),
        }
