"""Cacheless V0 teacher windows generated online from frozen pi0.5.

Only four consecutive query states are live in an iterator. Teacher tensors are
never serialized, and deterministic query noise matches the schema-v2 cache
generator.
"""

from __future__ import annotations

from collections import Counter
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .cache_contract_v2 import resolve_task_identity
from .policy_io import explicit_policy_noise
from .policy_io import policy_noise_seed
from .policy_io import prepare_policy_observation
from .prefix_kv_hook import PrefixKVHook
from .prefix_kv_hook import PrefixKVState

STREAMING_SOURCE_SCHEMA_VERSION = 1
STREAMING_SOURCE_MODE = "online_frozen_teacher_rolling_v0"


@dataclass(frozen=True)
class StreamingTeacherConfig:
    action_horizon: int = 10
    execution_horizon: int = 5
    flow_steps: int = 10
    noise_seed_base: int = 20260820
    episode_order_seed: int = 42

    def validate(self) -> None:
        if (self.action_horizon, self.execution_horizon, self.flow_steps) != (10, 5, 10):
            raise ValueError("pi0.5 streaming V0 is pinned to H=10, R=5, and ten flow steps")


@dataclass(frozen=True)
class StreamingEpisode:
    suite: str
    benchmark_task_index: int
    episode_id: int
    role: str
    dataset_frame_start: int
    dataset_frame_stop: int
    query_count: int
    window_count: int

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.suite, self.benchmark_task_index, self.episode_id)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _expected_query_count(start: int, stop: int, execution_horizon: int) -> int:
    length = int(stop) - int(start)
    return max(0, (length - 1) // execution_horizon)


def build_streaming_episode_plan(
    split_contract: dict[str, Any],
    *,
    execution_horizon: int = 5,
) -> tuple[StreamingEpisode, ...]:
    episodes = []
    for assignment in split_contract.get("assignments", []):
        start = int(assignment["dataset_frame_start"])
        stop = int(assignment["dataset_frame_stop"])
        query_count = _expected_query_count(start, stop, execution_horizon)
        if query_count < 4:
            raise ValueError("every streaming V0 episode must provide anchor plus ages 1,2,3")
        episodes.append(
            StreamingEpisode(
                suite=str(assignment["suite"]),
                benchmark_task_index=int(assignment["benchmark_task_index"]),
                episode_id=int(assignment["episode_id"]),
                role=str(assignment["role"]),
                dataset_frame_start=start,
                dataset_frame_stop=stop,
                query_count=query_count,
                window_count=query_count - 3,
            )
        )
    episodes.sort(key=lambda row: row.identity)
    identities = [row.identity for row in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("streaming plan contains duplicate teacher episodes")
    return tuple(episodes)


def deterministic_episode_order(
    episodes: Iterable[StreamingEpisode],
    *,
    seed: int,
    epoch: int,
) -> tuple[StreamingEpisode, ...]:
    def order_key(row: StreamingEpisode) -> bytes:
        key = f"{seed}:{epoch}:{row.suite}:{row.benchmark_task_index}:{row.episode_id}"
        return hashlib.sha256(key.encode("utf-8")).digest()

    return tuple(sorted(episodes, key=order_key))


def iter_rolling_v0_examples(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    window: deque[dict[str, Any]] = deque(maxlen=4)
    for record in records:
        window.append(record)
        if len(window) == 4:
            yield {"records": tuple(window), "delta_q": 3}


def build_streaming_provenance(
    *,
    source_lock: dict[str, Any],
    checkpoint: str | Path,
    final_manifest: dict[str, Any],
    final_manifest_path: str | Path,
    split_contract: dict[str, Any],
    split_contract_path: str | Path,
    config: StreamingTeacherConfig,
) -> dict[str, Any]:
    config.validate()
    if final_manifest.get("source_lock_id") != source_lock.get("source_lock_id"):
        raise ValueError("final manifest and source lock differ")
    if split_contract.get("source_lock_id") != source_lock.get("source_lock_id"):
        raise ValueError("split contract and source lock differ")
    if split_contract.get("final_manifest_id") != final_manifest.get("manifest_id"):
        raise ValueError("split contract does not name the supplied final manifest")
    episodes = build_streaming_episode_plan(split_contract, execution_horizon=config.execution_horizon)
    role_episodes = Counter(row.role for row in episodes)
    role_queries = Counter()
    role_windows = Counter()
    for row in episodes:
        role_queries[row.role] += row.query_count
        role_windows[row.role] += row.window_count
    identity = {
        "schema_version": STREAMING_SOURCE_SCHEMA_VERSION,
        "training_source_mode": STREAMING_SOURCE_MODE,
        "source_lock_id": source_lock["source_lock_id"],
        "checkpoint_model_sha256": source_lock["checkpoint"]["model_sha256"],
        "checkpoint_config_sha256": source_lock["checkpoint"]["config_sha256"],
        "normalization_sha256": source_lock["normalization"]["sha256"],
        "final_manifest_id": final_manifest["manifest_id"],
        "final_manifest_sha256": _sha256(final_manifest_path),
        "split_contract_id": split_contract["split_contract_id"],
        "split_contract_sha256": _sha256(split_contract_path),
        "dataset_config": "pi05_libero_lora_pytorch",
        "teacher_checkpoint": str(Path(checkpoint).resolve()),
        "protocol": asdict(config),
    }
    return {
        **identity,
        "training_source_id": _canonical_hash(identity),
        "persistent_teacher_tensor_bytes": 0,
        "rolling_window_teacher_records": 4,
        "maximum_transient_teacher_records_per_active_iterator": 5,
        "maximum_concurrent_iterators_during_training_validation": 2,
        "maximum_transient_teacher_records_in_trainer": 10,
        "teacher_tensor_lifetime": "one rolling V0 window; discarded after iterator advance",
        "episode_order": "deterministic SHA256 permutation per epoch",
        "statistics": {
            "episodes": len(episodes),
            "queries": sum(row.query_count for row in episodes),
            "v0_windows": sum(row.window_count for row in episodes),
            "episodes_by_role": dict(sorted(role_episodes.items())),
            "queries_by_role": dict(sorted(role_queries.items())),
            "v0_windows_by_role": dict(sorted(role_windows.items())),
        },
    }


def _image_uint8(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _raw_policy_observation(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation/image": _image_uint8(sample["image"]),
        "observation/wrist_image": _image_uint8(sample["wrist_image"]),
        "observation/state": np.asarray(sample["state"]),
        "prompt": str(sample["prompt"]),
    }


def _training_state_payload(state: PrefixKVState) -> dict[str, torch.Tensor]:
    state.validate()
    return {
        "prefix_embeddings": state.embeddings.detach(),
        "prefix_pad_mask": state.pad_mask.detach(),
        "prefix_attention_pattern": state.attention_pattern.detach(),
        "prefix_position_ids": state.position_ids.detach(),
        "pre_rope_keys": torch.stack(state.pre_rope_keys, dim=1).detach(),
        "values": torch.stack(state.values, dim=1).detach(),
    }


class OnlineV0TeacherSource:
    """Generate exact frozen-teacher V0 windows without persistent tensor files."""

    def __init__(
        self,
        *,
        policy: Any,
        dataset: Any,
        source_lock: dict[str, Any],
        checkpoint: str | Path,
        final_manifest: dict[str, Any],
        final_manifest_path: str | Path,
        split_contract: dict[str, Any],
        split_contract_path: str | Path,
        config: StreamingTeacherConfig,
        device: str | torch.device,
    ) -> None:
        config.validate()
        self.policy = policy
        self.model = policy._model  # noqa: SLF001
        self.dataset = dataset
        self.device = torch.device(device)
        self.config = config
        self.final_manifest = final_manifest
        self.episodes = build_streaming_episode_plan(split_contract, execution_horizon=config.execution_horizon)
        self._episodes_by_role = {
            role: tuple(row for row in self.episodes if row.role == role)
            for role in {row.role for row in self.episodes}
        }
        if not self._episodes_by_role.get("train") or not self._episodes_by_role.get("checkpoint_validation"):
            raise ValueError("streaming source requires train and checkpoint_validation episodes")
        self.provenance = build_streaming_provenance(
            source_lock=source_lock,
            checkpoint=checkpoint,
            final_manifest=final_manifest,
            final_manifest_path=final_manifest_path,
            split_contract=split_contract,
            split_contract_path=split_contract_path,
            config=config,
        )
        self.hook = PrefixKVHook(self.model)
        base_dataset = getattr(dataset, "_dataset", dataset)
        self._starts = [int(value) for value in base_dataset.episode_data_index["from"].tolist()]
        self._stops = [int(value) for value in base_dataset.episode_data_index["to"].tolist()]
        self._counters: Counter[str] = Counter()
        self._timing_seconds: Counter[str] = Counter()
        self._verify_dataset_frame_contract()

    @classmethod
    def from_openpi(
        cls,
        *,
        policy: Any,
        source_lock: dict[str, Any],
        checkpoint: str | Path,
        final_manifest: dict[str, Any],
        final_manifest_path: str | Path,
        split_contract: dict[str, Any],
        split_contract_path: str | Path,
        config: StreamingTeacherConfig,
        device: str | torch.device,
    ) -> OnlineV0TeacherSource:
        from openpi.training import config as config_api
        from openpi.training import data_loader

        train_config = config_api.get_config("pi05_libero_lora_pytorch")
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        dataset = data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
        return cls(
            policy=policy,
            dataset=dataset,
            source_lock=source_lock,
            checkpoint=checkpoint,
            final_manifest=final_manifest,
            final_manifest_path=final_manifest_path,
            split_contract=split_contract,
            split_contract_path=split_contract_path,
            config=config,
            device=device,
        )

    def _verify_dataset_frame_contract(self) -> None:
        for row in self.episodes:
            if not 0 <= row.episode_id < len(self._starts):
                raise ValueError(f"teacher episode is absent from dataset: {row.identity}")
            observed = (self._starts[row.episode_id], self._stops[row.episode_id])
            expected = (row.dataset_frame_start, row.dataset_frame_stop)
            if observed != expected:
                raise ValueError(f"dataset frame bounds differ from split contract: {row.identity}")

    def _materialize_record(self, row: StreamingEpisode, query_index: int) -> dict[str, Any]:
        frame = row.dataset_frame_start + query_index * self.config.execution_horizon
        current = self.dataset[frame]
        if bool(torch.as_tensor(current["actions_is_pad"][: self.config.execution_horizon]).any()):
            raise RuntimeError(f"streaming query crosses padded actions: {row.identity}/{query_index}")
        if int(current["frame_index"]) != query_index * self.config.execution_horizon:
            raise RuntimeError(f"streaming query violates native R progression: {row.identity}/{query_index}")
        task = resolve_task_identity(int(current["task_index"]), str(current["prompt"]), self.final_manifest)
        if (str(task["suite"]), int(task["benchmark_task_index"])) != (
            row.suite,
            row.benchmark_task_index,
        ):
            raise RuntimeError(f"dataset task identity differs from split contract: {row.identity}")

        started = time.perf_counter()
        observation, _ = prepare_policy_observation(self.policy, _raw_policy_observation(current))
        seed = policy_noise_seed(
            self.config.noise_seed_base,
            row.suite,
            row.benchmark_task_index,
            row.episode_id,
            query_index,
        )
        with torch.no_grad():
            extraction = self.hook.extract(observation)
            noise = explicit_policy_noise(
                (1, self.model.config.action_horizon, self.model.config.action_dim),
                seed=seed,
                device=self.device,
            )
            teacher, action_timing = self.hook.sample_actions_from_state(
                extraction.state,
                extraction.robot_state,
                noise,
                num_steps=self.config.flow_steps,
            )
        executed = torch.as_tensor(
            current["actions"][: self.config.execution_horizon],
            device=self.device,
            dtype=torch.float32,
        )[..., :7]
        self._counters["teacher_queries_materialized"] += 1
        self._counters[f"teacher_queries_{row.role}"] += 1
        self._timing_seconds["teacher_materialization"] += time.perf_counter() - started
        return {
            "suite": row.suite,
            "benchmark_task_index": row.benchmark_task_index,
            "episode_id": row.episode_id,
            "query_index": query_index,
            "policy_query_index": query_index,
            "absolute_environment_step": query_index * self.config.execution_horizon,
            "robot_state_normalized": extraction.robot_state.detach(),
            **_training_state_payload(extraction.state),
            "action_noise": noise.detach(),
            "action_noise_seed": seed,
            "teacher_action_chunk_normalized": teacher.detach(),
            "executed_actions_postprocessed": executed.detach(),
            "timing_ms": {
                "prefix_embedding": extraction.prefix_embedding_ms,
                "full_prefix": extraction.full_prefix_ms,
                **action_timing,
            },
        }

    def _iter_episode(self, row: StreamingEpisode) -> Iterator[dict[str, Any]]:
        records = (self._materialize_record(row, query) for query in range(row.query_count))
        yielded = 0
        for example in iter_rolling_v0_examples(records):
            yielded += 1
            self._counters["v0_windows_yielded"] += 1
            self._counters[f"v0_windows_{row.role}"] += 1
            yield example
        if yielded != row.window_count:
            raise RuntimeError(f"streaming V0 window count differs from plan: {row.identity}")

    def iter_train_examples(self) -> Iterator[dict[str, Any]]:
        epoch = 0
        episodes = self._episodes_by_role["train"]
        while True:
            self._counters["train_epochs_started"] += 1
            for row in deterministic_episode_order(episodes, seed=self.config.episode_order_seed, epoch=epoch):
                yield from self._iter_episode(row)
            epoch += 1

    def iter_validation_examples(self, limit: int) -> Iterator[dict[str, Any]]:
        if limit < 1:
            raise ValueError("validation limit must be positive")
        yielded = 0
        for row in deterministic_episode_order(
            self._episodes_by_role["checkpoint_validation"],
            seed=self.config.episode_order_seed,
            epoch=0,
        ):
            for example in self._iter_episode(row):
                yield example
                yielded += 1
                if yielded >= limit:
                    return
        if yielded < limit:
            raise RuntimeError(f"streaming validation produced {yielded} of {limit} examples")

    def statistics(self) -> dict[str, Any]:
        return {
            "persistent_teacher_tensor_bytes": 0,
            "rolling_window_teacher_records": 4,
            "maximum_transient_teacher_records_per_active_iterator": 5,
            "maximum_concurrent_iterators_during_training_validation": 2,
            "maximum_transient_teacher_records_in_trainer": 10,
            "counters": dict(sorted(self._counters.items())),
            "timing_seconds": dict(sorted(self._timing_seconds.items())),
        }
