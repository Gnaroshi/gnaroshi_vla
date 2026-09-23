"""Episode-disjoint native-R5 q0/q1/q2/q3 dataset for corrected SimVLA V0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from architectures.simvla.adapters.latentloop.source_lock import sha256_file
from architectures.simvla.adapters.latentloop.native_v0_training_cache import (
    TRAINING_CACHE_SCHEMA,
    load_compact_sequence,
    load_raw_rgb_sequence,
    load_training_manifest,
    sequence_explicit_noises,
)


SEQUENCE_SCHEMA_VERSION = "simvla_native_v0_q0_q3_v1"


def stable_episode_partition(task_id: int, episode_id: str, seed: int) -> float:
    payload = f"{seed}|{task_id}|{episode_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


class NativeV0SequenceDataset(Dataset[dict[str, Any]]):
    """Return four contiguous query observations and three teacher targets."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split: str,
        heldout_fraction: float = 0.2,
        split_seed: int = 20260822,
        anchor_modulus: int = 4,
    ) -> None:
        if split not in {"train", "heldout", "all"}:
            raise ValueError("split must be train, heldout, or all")
        if not 0.0 < heldout_fraction < 1.0:
            raise ValueError("heldout_fraction must be in (0,1)")
        if anchor_modulus != 4:
            raise ValueError("correct native V0 uses fixed K=4 anchors")
        self.cache_dir = Path(cache_dir).resolve()
        self.manifest = load_training_manifest(self.cache_dir)
        if int(self.manifest["execution_horizon"]) != 5:
            raise ValueError("correct native SimVLA V0 requires an R=5 cache")
        if self.manifest.get("data_role") != "official_libero_training_demonstrations":
            raise ValueError("correct native V0 forbids evaluation-rollout cache training")
        selected: list[int] = []
        selected_identities: list[tuple[int, str, int]] = []
        for index, entry in enumerate(self.manifest["sequences"]):
            task_id = int(entry["task_id"])
            episode_id = str(entry["episode_id"])
            query_index = int(entry["anchor_query_index"])
            if query_index % anchor_modulus:
                raise ValueError(f"cache contains a non-K4 anchor: {entry}")
            heldout = stable_episode_partition(task_id, episode_id, split_seed) < heldout_fraction
            if split == "train" and heldout:
                continue
            if split == "heldout" and not heldout:
                continue
            selected.append(index)
            selected_identities.append((task_id, episode_id, query_index))
        self.indices = tuple(selected)
        self.sequence_identities = tuple(selected_identities)
        if not self.indices:
            raise ValueError(f"no native q0-q3 sequences found for split={split}")
        self.split = split
        self.heldout_fraction = float(heldout_fraction)
        self.split_seed = int(split_seed)
        split_payload = json.dumps(
            self.sequence_identities,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.split_sha256 = hashlib.sha256(split_payload).hexdigest()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.manifest["sequences"][self.indices[index]]
        sequence = load_compact_sequence(self.cache_dir, entry)
        identity = (int(sequence["task_id"]), str(sequence["episode_id"]))
        q0 = int(sequence["anchor_query_index"])
        conditions = sequence["condition_sequence"]
        proprio = sequence["proprio_sequence"]
        if conditions.shape != (4, 122, 960) or proprio.shape != (4, 8):
            raise AssertionError("compact native sequence shapes changed")
        return {
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            "task_id": identity[0],
            "episode_id": identity[1],
            "anchor_query_index": q0,
            "language_instruction": str(sequence["language_instruction"]),
            "image_sequence": load_raw_rgb_sequence(sequence),
            "proprio_sequence": proprio,
            "anchor_condition": conditions[0],
            "teacher_conditions": conditions[1:],
            "explicit_noises": sequence_explicit_noises(sequence, self.manifest),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            "cache_schema_version": TRAINING_CACHE_SCHEMA,
            "cache_dir": str(self.cache_dir),
            "cache_manifest_sha256": sha256_file(self.cache_dir / "manifest.json"),
            "execution_horizon": 5,
            "action_horizon": 10,
            "fixed_k": 4,
            "split": self.split,
            "split_unit": "task_id+episode_id",
            "heldout_fraction": self.heldout_fraction,
            "split_seed": self.split_seed,
            "split_sha256": self.split_sha256,
            "sequence_count": len(self),
            "updater_inputs": ["anchor_or_previous_predicted_condition", "image_sequence", "proprio_sequence"],
            "teacher_conditions_are_targets_only": True,
            "cached_teacher_actions_are_objective_targets": False,
            "action_targets_are_regenerated_with_runtime_norm": True,
            "executed_actions_present": False,
        }


def collate_native_v0_sequences(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    tensor_keys = (
        "image_sequence",
        "proprio_sequence",
        "anchor_condition",
        "teacher_conditions",
        "explicit_noises",
    )
    return {
        **{key: torch.stack(tuple(item[key] for item in items), dim=0) for key in tensor_keys},
        "task_id": torch.tensor([int(item["task_id"]) for item in items], dtype=torch.long),
        "episode_id": [str(item["episode_id"]) for item in items],
        "anchor_query_index": torch.tensor(
            [int(item["anchor_query_index"]) for item in items], dtype=torch.long
        ),
        "language_instruction": [str(item["language_instruction"]) for item in items],
    }


def write_sequence_contract(path: str | Path, dataset: NativeV0SequenceDataset) -> None:
    Path(path).write_text(
        json.dumps(dataset.contract(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
