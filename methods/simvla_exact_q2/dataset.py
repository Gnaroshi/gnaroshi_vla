"""Immutable q0-q1-q2 indexing over a native-R5 SimVLA query cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from methods.latentloop.training.query_cache_dataset import (
    QueryCacheDataset,
    load_manifest,
)


EXACT_Q2_DATASET_SCHEMA = "simvla_r5_exact_q2_dataset_v1"
EXACT_Q2_SPLIT_SCHEMA = "simvla_r5_exact_q2_episode_split_v1"


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_key(task_id: int, episode_id: str) -> str:
    """Return the stable, human-readable split key."""

    return f"{int(task_id)}|{episode_id}"


def episode_is_heldout(
    task_id: int,
    episode_id: str,
    *,
    heldout_fraction: float,
    split_seed: int,
) -> bool:
    """Apply the existing LatentLoop episode-hash split contract."""

    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    payload = f"{int(task_id)}|{episode_id}|{int(split_seed)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10_000
    return value < int(heldout_fraction * 10_000)


def validate_exact_q2_pair(
    q0_record: Mapping[str, Any],
    q1_record: Mapping[str, Any],
    *,
    native_execution_horizon: int = 5,
) -> list[str]:
    """Validate one q0->q1 and q1->q2 pair without mutating cache tensors."""

    errors: list[str] = []
    if int(q0_record["task_id"]) != int(q1_record["task_id"]):
        errors.append("task_id differs")
    if str(q0_record["episode_id"]) != str(q1_record["episode_id"]):
        errors.append("episode_id differs")
    q0 = int(q0_record["query_index"])
    q1 = int(q1_record["query_index"])
    q2 = int(q1_record["next_query_index"])
    if q1 != q0 + 1 or q2 != q1 + 1:
        errors.append(f"query indices are not consecutive: {q0},{q1},{q2}")
    for record_name, record in (("q0", q0_record), ("q1", q1_record)):
        if int(record["execution_horizon"]) != int(native_execution_horizon):
            errors.append(f"{record_name} execution horizon is not R={native_execution_horizon}")
        if tuple(record["executed_subchunk"].shape) != (native_execution_horizon, 7):
            errors.append(f"{record_name} executed subchunk has the wrong shape")
    for previous_key, current_key in (
        ("next_raw_rgb", "raw_rgb"),
        ("next_proprio", "proprio"),
        ("next_full_condition", "full_condition"),
        ("next_teacher_action_chunk", "teacher_action_chunk"),
        ("next_initial_noise", "initial_noise"),
    ):
        if not torch.equal(q0_record[previous_key], q1_record[current_key]):
            errors.append(f"boundary mismatch {previous_key}->{current_key}")
    if str(q0_record["next_action_noise_hash"]) != str(q1_record["action_noise_hash"]):
        errors.append("q1 action-noise hash differs across the record boundary")
    return errors


def _source_signature(cache_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = cache_root / "source_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checkpoint = lock["checkpoint"]
    protocol = manifest.get("metadata", {}).get("protocol", {})
    return {
        "root_commit": lock["root_commit"],
        "simvla_upstream_commit": lock["simvla_upstream_commit"],
        "checkpoint_identifier": checkpoint["identifier"],
        "checkpoint_revision": checkpoint["revision"],
        "checkpoint_blob_sha256": checkpoint["hf_blob_key_sha256"],
        "norm_stats_sha256": lock["norm_stats_sha256"],
        "preprocessing": {
            "client_resize_size": int(protocol["client_resize_size"]),
            "image_size": int(protocol["image_size"]),
            "task_order": protocol["task_order"],
        },
        "environment": {
            "experiment_seed": int(protocol["experiment_seed"]),
            "render_backend": protocol["render_backend"],
            "environment_lifecycle": protocol["environment_lifecycle"],
            "mujoco": lock["packages"]["mujoco"],
            "torch": lock["torch"],
            "transformers": lock["packages"]["transformers"],
        },
    }


def build_exact_q2_index(
    cache_root: str | Path,
    *,
    full_refresh_interval: int = 4,
    split_seed: int = 20260804,
    heldout_fraction: float = 0.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build tuple locators and an episode-disjoint split over an immutable cache."""

    cache = Path(cache_root).resolve()
    manifest = load_manifest(cache)
    if int(manifest["execution_horizon"]) != 5:
        raise ValueError("exact-q2 dataset requires the native R5 production cache")
    protocol = manifest.get("metadata", {}).get("protocol", {})
    if int(protocol.get("action_horizon_H_a", -1)) != 10:
        raise ValueError("exact-q2 dataset requires H=10")
    if int(full_refresh_interval) < 3:
        raise ValueError("full_refresh_interval must leave a q2 boundary")

    cache_dataset = QueryCacheDataset(cache)
    tuples: list[dict[str, Any]] = []
    episodes: set[tuple[int, str]] = set()
    seen_tuple_ids: set[str] = set()
    previous_index: int | None = None
    previous_record: Mapping[str, Any] | None = None
    previous_episode: tuple[int, str] | None = None
    for record_index in range(len(cache_dataset)):
        record = cache_dataset[record_index]
        current_episode = (int(record["task_id"]), str(record["episode_id"]))
        episodes.add(current_episode)
        if current_episode != previous_episode:
            previous_index = None
            previous_record = None
        if previous_record is not None and previous_index is not None:
            if int(previous_record["query_index"]) % int(full_refresh_interval) == 0:
                errors = validate_exact_q2_pair(previous_record, record)
                if errors:
                    raise RuntimeError(
                        f"invalid exact-q2 pair at indices {previous_index},{record_index}: {errors}"
                    )
                q0 = int(previous_record["query_index"])
                q1 = int(record["query_index"])
                q2 = int(record["next_query_index"])
                tuple_id = f"task{current_episode[0]:02d}/{current_episode[1]}/q{q0:04d}-q{q2:04d}"
                if tuple_id in seen_tuple_ids:
                    raise RuntimeError(f"duplicate exact-q2 tuple: {tuple_id}")
                seen_tuple_ids.add(tuple_id)
                tuples.append(
                    {
                        "tuple_id": tuple_id,
                        "task_id": current_episode[0],
                        "episode_id": current_episode[1],
                        "q0_query_index": q0,
                        "q1_query_index": q1,
                        "q2_query_index": q2,
                        "q0_record_index": previous_index,
                        "q1_record_index": record_index,
                        "language_instruction": str(record["language_instruction"]),
                        "task_identifier": str(record["task_identifier"]),
                        "epsilon1_sha256": str(previous_record["next_action_noise_hash"]),
                        "epsilon2_sha256": str(record["next_action_noise_hash"]),
                        "elapsed_q0_to_q1": float(previous_record["elapsed_time"]),
                        "elapsed_q1_to_q2": float(record["elapsed_time"]),
                    }
                )
        previous_index = record_index
        previous_record = record
        previous_episode = current_episode

    train_episodes = sorted(
        episode_key(*item)
        for item in episodes
        if not episode_is_heldout(
            *item,
            heldout_fraction=heldout_fraction,
            split_seed=split_seed,
        )
    )
    validation_episodes = sorted(
        episode_key(*item)
        for item in episodes
        if episode_is_heldout(
            *item,
            heldout_fraction=heldout_fraction,
            split_seed=split_seed,
        )
    )
    if not train_episodes or not validation_episodes:
        raise RuntimeError("episode split produced an empty partition")
    train_set = set(train_episodes)
    validation_set = set(validation_episodes)
    if train_set & validation_set:
        raise AssertionError("train and validation episodes overlap")
    train_tuple_ids = [
        row["tuple_id"]
        for row in tuples
        if episode_key(row["task_id"], row["episode_id"]) in train_set
    ]
    validation_tuple_ids = [
        row["tuple_id"]
        for row in tuples
        if episode_key(row["task_id"], row["episode_id"]) in validation_set
    ]
    manifest_sha = sha256_file(cache / "manifest.json")
    dataset_manifest = {
        "schema_version": EXACT_Q2_DATASET_SCHEMA,
        "experiment_identifier": "simvla_r5_exact_q2_regeneration",
        "cache_root": str(cache),
        "cache_manifest_sha256": manifest_sha,
        "cache_schema_version": manifest["schema_version"],
        "cache_total_records": int(manifest["total_records"]),
        "source_signature": _source_signature(cache, manifest),
        "native_semantics": {
            "action_horizon_H": 10,
            "execution_horizon_R": 5,
            "full_refresh_interval_queries": int(full_refresh_interval),
            "anchor_rule": f"q0_query_index % {int(full_refresh_interval)} == 0",
            "condition_shape": [122, 960],
            "action_chunk_shape": [10, 7],
            "camera_views": 2,
            "proprio_dim": 8,
        },
        "episodes": len(episodes),
        "tuples": tuples,
        "tuple_count": len(tuples),
    }
    split = {
        "schema_version": EXACT_Q2_SPLIT_SCHEMA,
        "cache_manifest_sha256": manifest_sha,
        "dataset_schema_version": EXACT_Q2_DATASET_SCHEMA,
        "split_unit": "task_id+episode_id",
        "split_contract": "sha256(task_id|episode_id|split_seed) mod 10000",
        "split_seed": int(split_seed),
        "heldout_fraction": float(heldout_fraction),
        "train_episode_ids": train_episodes,
        "validation_episode_ids": validation_episodes,
        "train_tuple_ids": train_tuple_ids,
        "validation_tuple_ids": validation_tuple_ids,
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "train_tuple_count": len(train_tuple_ids),
        "validation_tuple_count": len(validation_tuple_ids),
    }
    return dataset_manifest, split


class ExactQ2TupleDataset(Dataset[dict[str, Any]]):
    """Random-access exact-q2 tuples backed by the immutable query cache."""

    def __init__(
        self,
        dataset_manifest: str | Path | Mapping[str, Any],
        split: str | Path | Mapping[str, Any],
        *,
        partition: str,
    ) -> None:
        self.dataset_manifest = self._load_json(dataset_manifest)
        self.split = self._load_json(split)
        if self.dataset_manifest.get("schema_version") != EXACT_Q2_DATASET_SCHEMA:
            raise ValueError("unsupported exact-q2 dataset manifest")
        if self.split.get("schema_version") != EXACT_Q2_SPLIT_SCHEMA:
            raise ValueError("unsupported exact-q2 split")
        if partition not in {"train", "validation", "all"}:
            raise ValueError("partition must be train, validation, or all")
        self.partition = partition
        self.cache_root = Path(self.dataset_manifest["cache_root"])
        if sha256_file(self.cache_root / "manifest.json") != self.dataset_manifest[
            "cache_manifest_sha256"
        ]:
            raise RuntimeError("query cache manifest changed after exact-q2 indexing")
        self.cache_dataset = QueryCacheDataset(self.cache_root)
        selected = None if partition == "all" else set(self.split[f"{partition}_tuple_ids"])
        self.rows = [
            row
            for row in self.dataset_manifest["tuples"]
            if selected is None or row["tuple_id"] in selected
        ]
        if not self.rows:
            raise RuntimeError(f"exact-q2 partition {partition} is empty")

    @staticmethod
    def _load_json(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return json.loads(Path(value).read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        locator = self.rows[index]
        q0 = self.cache_dataset[int(locator["q0_record_index"])]
        q1 = self.cache_dataset[int(locator["q1_record_index"])]
        errors = validate_exact_q2_pair(q0, q1)
        if errors:
            raise RuntimeError(f"cache tuple {locator['tuple_id']} changed: {errors}")
        return {
            "tuple_id": locator["tuple_id"],
            "task_id": int(locator["task_id"]),
            "episode_id": locator["episode_id"],
            "q0_query_index": int(locator["q0_query_index"]),
            "q1_query_index": int(locator["q1_query_index"]),
            "q2_query_index": int(locator["q2_query_index"]),
            "language_instruction": q0["language_instruction"],
            "task_identifier": q0["task_identifier"],
            "q0_raw_rgb": q0["raw_rgb"],
            "q0_proprio": q0["proprio"],
            "c0_full": q0["full_condition"],
            "q1_raw_rgb": q0["next_raw_rgb"],
            "q1_proprio": q0["next_proprio"],
            "c1_full": q0["next_full_condition"],
            "a1_full": q0["next_teacher_action_chunk"],
            "epsilon1": q0["next_initial_noise"],
            "epsilon1_sha256": q0["next_action_noise_hash"],
            "q2_raw_rgb": q1["next_raw_rgb"],
            "q2_proprio": q1["next_proprio"],
            "c2_full": q1["next_full_condition"],
            "a2_full": q1["next_teacher_action_chunk"],
            "epsilon2": q1["next_initial_noise"],
            "epsilon2_sha256": q1["next_action_noise_hash"],
            "x0_executed": q0["executed_subchunk"],
            "x1_executed": q1["executed_subchunk"],
            "elapsed_q0_to_q1": float(q0["elapsed_time"]),
            "elapsed_q1_to_q2": float(q1["elapsed_time"]),
            "execution_horizon": int(q0["execution_horizon"]),
            "provenance": {
                "q0": q0["provenance"],
                "q1": q1["provenance"],
                "source_signature": self.dataset_manifest["source_signature"],
            },
        }


def collate_exact_q2(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack exact-q2 tensors/numbers and preserve identifiers as lists."""

    if not records:
        raise ValueError("cannot collate an empty exact-q2 batch")
    output: dict[str, Any] = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if all(torch.is_tensor(value) for value in values):
            output[key] = torch.stack(values)
        elif all(isinstance(value, bool) for value in values):
            output[key] = torch.tensor(values, dtype=torch.bool)
        elif all(isinstance(value, int) for value in values):
            output[key] = torch.tensor(values, dtype=torch.long)
        elif all(isinstance(value, float) for value in values):
            output[key] = torch.tensor(values, dtype=torch.float32)
        else:
            output[key] = values
    return output
