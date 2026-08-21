"""Versioned query-boundary cache used by SimVLA LatentLoop training."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


QUERY_CACHE_SCHEMA_VERSION = "simvla_query_boundary_v2"
TENSOR_KEYS = (
    "raw_rgb",
    "proprio",
    "full_condition",
    "teacher_action_chunk",
    "initial_noise",
    "executed_subchunk",
    "executed_env_actions",
    "next_raw_rgb",
    "next_proprio",
    "next_full_condition",
    "next_teacher_action_chunk",
    "next_initial_noise",
)
REQUIRED_KEYS = (
    "task_id",
    "episode_id",
    "query_index",
    "next_query_index",
    "absolute_env_timestep",
    "next_absolute_env_timestep",
    "language_instruction",
    "task_identifier",
    "execution_horizon",
    "elapsed_time",
    "action_noise_hash",
    "next_action_noise_hash",
    "provenance",
    *TENSOR_KEYS,
)


def tensor_sha256(tensor: Tensor) -> str:
    """Hash a tensor's dtype, shape, and contiguous CPU bytes."""

    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item()) if torch.is_floating_point(tensor) else True


def validate_query_record(
    record: Mapping[str, Any],
    *,
    expected_execution_horizon: int | None = None,
    condition_shape: tuple[int, int] = (122, 960),
    action_shape: tuple[int, int] = (10, 7),
) -> list[str]:
    """Return schema/semantic errors for one query-to-next-query record."""

    errors = [f"missing key: {key}" for key in REQUIRED_KEYS if key not in record]
    if errors:
        return errors
    execution_horizon = int(record["execution_horizon"])
    if expected_execution_horizon is not None and execution_horizon != expected_execution_horizon:
        errors.append(
            f"execution_horizon={execution_horizon} expected={expected_execution_horizon}"
        )
    if execution_horizon not in {1, 2, 5}:
        errors.append("execution_horizon must be one of 1, 2, 5")
    if int(record["next_query_index"]) != int(record["query_index"]) + 1:
        errors.append("next_query_index must equal query_index + 1")
    env_gap = int(record["next_absolute_env_timestep"]) - int(record["absolute_env_timestep"])
    if env_gap != execution_horizon:
        errors.append(f"query environment timestep gap={env_gap}, expected R={execution_horizon}")
    expected_shapes = {
        "raw_rgb": None,
        "proprio": (8,),
        "full_condition": condition_shape,
        "teacher_action_chunk": action_shape,
        "initial_noise": action_shape,
        "executed_subchunk": (execution_horizon, action_shape[-1]),
        "executed_env_actions": (execution_horizon, action_shape[-1]),
        "next_raw_rgb": None,
        "next_proprio": (8,),
        "next_full_condition": condition_shape,
        "next_teacher_action_chunk": action_shape,
        "next_initial_noise": action_shape,
    }
    for key, expected_shape in expected_shapes.items():
        value = record[key]
        if not torch.is_tensor(value):
            errors.append(f"{key} is not a tensor")
            continue
        if expected_shape is not None and tuple(value.shape) != expected_shape:
            errors.append(f"{key} shape={tuple(value.shape)} expected={expected_shape}")
        if key.endswith("raw_rgb") or key == "raw_rgb":
            if value.ndim != 4 or value.shape[0] != 2 or value.shape[-1] != 3:
                errors.append(f"{key} must be [2,H,W,3]")
        if not _finite(value):
            errors.append(f"{key} contains non-finite values")
    if torch.is_tensor(record["executed_subchunk"]) and torch.is_tensor(record["executed_env_actions"]):
        if not torch.equal(record["executed_subchunk"], record["executed_env_actions"]):
            errors.append("executed_subchunk does not exactly match actions sent to env")
    if torch.is_tensor(record["initial_noise"]):
        if tensor_sha256(record["initial_noise"]) != str(record["action_noise_hash"]):
            errors.append("initial_noise hash mismatch after serialization")
    if torch.is_tensor(record["next_initial_noise"]):
        if tensor_sha256(record["next_initial_noise"]) != str(record["next_action_noise_hash"]):
            errors.append("next_initial_noise hash mismatch after serialization")
    return errors


class QueryCacheShardWriter:
    """Write validated records to immutable torch shards and a JSON manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        execution_horizon: int,
        metadata: Mapping[str, Any],
        records_per_shard: int = 128,
    ) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty cache: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.execution_horizon = int(execution_horizon)
        self.metadata = dict(metadata)
        self.records_per_shard = int(records_per_shard)
        self._buffer: list[dict[str, Any]] = []
        self._shards: list[dict[str, Any]] = []
        self._total_records = 0

    def add(self, record: Mapping[str, Any]) -> None:
        """Validate and stage one record."""

        errors = validate_query_record(
            record,
            expected_execution_horizon=self.execution_horizon,
        )
        if errors:
            raise ValueError("Invalid query cache record: " + "; ".join(errors))
        self._buffer.append(dict(record))
        if len(self._buffer) >= self.records_per_shard:
            self.flush()

    def flush(self) -> None:
        """Persist the current shard and update the manifest."""

        if not self._buffer:
            return
        filename = f"query_shard_{len(self._shards):06d}.pt"
        path = self.output_dir / filename
        torch.save(self._buffer, path)
        digest = _file_sha256(path)
        count = len(self._buffer)
        self._shards.append({"file": filename, "records": count, "sha256": digest})
        self._total_records += count
        self._buffer = []
        self.write_manifest()

    def write_manifest(self) -> None:
        """Write a crash-tolerant manifest after every completed shard."""

        payload = {
            "schema_version": QUERY_CACHE_SCHEMA_VERSION,
            "execution_horizon": self.execution_horizon,
            "records_per_shard": self.records_per_shard,
            "total_records": self._total_records,
            "metadata": self.metadata,
            "shards": self._shards,
        }
        manifest_path = self.output_dir / "manifest.json"
        temporary_path = self.output_dir / ".manifest.json.tmp"
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)

    def close(self) -> None:
        """Flush all records and finalize the manifest."""

        self.flush()
        self.write_manifest()


def load_manifest(cache_dir: str | Path) -> dict[str, Any]:
    """Load and validate the top-level cache manifest version."""

    path = Path(cache_dir) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != QUERY_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache schema: {payload.get('schema_version')}")
    return payload


def merge_query_cache_parts(
    output_dir: str | Path,
    part_dirs: Sequence[str | Path],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one copy-free manifest over cache parts below ``output_dir``."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_parts = [Path(part).resolve() for part in part_dirs]
    if not resolved_parts:
        raise ValueError("at least one query-cache part is required")

    manifests = [load_manifest(part) for part in resolved_parts]
    execution_horizons = {int(item["execution_horizon"]) for item in manifests}
    if len(execution_horizons) != 1:
        raise ValueError(f"cache parts disagree on execution horizon: {execution_horizons}")

    merged_shards: list[dict[str, Any]] = []
    part_summaries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    total_records = 0
    for part, manifest in zip(resolved_parts, manifests):
        try:
            relative_part = part.relative_to(output)
        except ValueError as exc:
            raise ValueError(f"cache part must be below merged output: {part}") from exc
        part_records = 0
        for shard in manifest["shards"]:
            source_path = (part / shard["file"]).resolve()
            try:
                relative_path = source_path.relative_to(output).as_posix()
            except ValueError as exc:
                raise ValueError(f"cache shard escapes merged output: {source_path}") from exc
            if relative_path in seen_paths:
                raise ValueError(f"duplicate shard path while merging: {relative_path}")
            seen_paths.add(relative_path)
            merged = dict(shard)
            merged["file"] = relative_path
            merged["part"] = relative_part.as_posix()
            merged_shards.append(merged)
            part_records += int(shard["records"])
        if part_records != int(manifest["total_records"]):
            raise ValueError(
                f"part manifest count mismatch for {part}: "
                f"shards={part_records} manifest={manifest['total_records']}"
            )
        total_records += part_records
        part_summaries.append(
            {
                "path": relative_part.as_posix(),
                "records": part_records,
                "shards": len(manifest["shards"]),
            }
        )

    first_metadata = dict(manifests[0].get("metadata", {}))
    merged_metadata = {
        **first_metadata,
        **dict(metadata or {}),
        "cache_layout": "copy_free_merged_parts",
        "parts": part_summaries,
    }
    payload = {
        "schema_version": QUERY_CACHE_SCHEMA_VERSION,
        "execution_horizon": execution_horizons.pop(),
        "records_per_shard": int(manifests[0].get("records_per_shard", 0)),
        "total_records": total_records,
        "metadata": merged_metadata,
        "shards": merged_shards,
    }
    manifest_path = output / "manifest.json"
    temporary_path = output / ".manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return payload


def iter_query_records(cache_dir: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records without retaining the complete cache in memory."""

    root = Path(cache_dir)
    manifest = load_manifest(root)
    for shard in manifest["shards"]:
        records = torch.load(root / shard["file"], map_location="cpu", weights_only=False)
        yield from records


def validate_query_cache(cache_dir: str | Path, *, verify_shard_hashes: bool = True) -> dict[str, Any]:
    """Validate shards, records, episode query continuity, and exact noise reload."""

    root = Path(cache_dir)
    manifest = load_manifest(root)
    errors: list[str] = []
    episode_boundaries: dict[tuple[int, str], dict[str, Any]] = {}
    observed = 0
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if not path.exists():
            errors.append(f"missing shard: {path}")
            continue
        if verify_shard_hashes:
            digest = _file_sha256(path)
            if digest != shard.get("sha256"):
                errors.append(f"shard hash mismatch: {path.name}")
        records = torch.load(path, map_location="cpu", weights_only=False)
        if len(records) != int(shard["records"]):
            errors.append(f"shard record count mismatch: {path.name}")
        for record in records:
            observed += 1
            record_errors = validate_query_record(
                record,
                expected_execution_horizon=int(manifest["execution_horizon"]),
            )
            errors.extend(
                f"task={record.get('task_id')} episode={record.get('episode_id')} "
                f"query={record.get('query_index')}: {error}"
                for error in record_errors
            )
            key = (int(record["task_id"]), str(record["episode_id"]))
            previous = episode_boundaries.get(key)
            if previous is not None:
                if int(previous["next_query_index"]) != int(record["query_index"]):
                    errors.append(
                        f"non-contiguous query index for episode {key}: "
                        f"previous_next={previous['next_query_index']} "
                        f"current={record['query_index']}"
                    )
                for previous_key, current_key in (
                    ("next_raw_rgb", "raw_rgb"),
                    ("next_proprio", "proprio"),
                    ("next_full_condition", "full_condition"),
                    ("next_teacher_action_chunk", "teacher_action_chunk"),
                    ("next_initial_noise", "initial_noise"),
                ):
                    if not torch.equal(previous[previous_key], record[current_key]):
                        errors.append(
                            f"boundary tensor mismatch {previous_key}->{current_key} "
                            f"in episode {key}"
                        )
            episode_boundaries[key] = {
                "next_query_index": int(record["next_query_index"]),
                "next_raw_rgb": record["next_raw_rgb"],
                "next_proprio": record["next_proprio"],
                "next_full_condition": record["next_full_condition"],
                "next_teacher_action_chunk": record["next_teacher_action_chunk"],
                "next_initial_noise": record["next_initial_noise"],
            }
    if observed != int(manifest.get("total_records", -1)):
        errors.append(
            f"manifest total_records={manifest.get('total_records')} observed={observed}"
        )
    return {
        "schema_version": QUERY_CACHE_SCHEMA_VERSION,
        "cache_dir": str(root),
        "execution_horizon": int(manifest["execution_horizon"]),
        "records": observed,
        "episodes": len(episode_boundaries),
        "errors": errors,
        "passed": not errors,
    }


class QueryCacheDataset(Dataset[dict[str, Any]]):
    """Random-access query cache with one-shard memory residency."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = load_manifest(self.cache_dir)
        self.shards = list(self.manifest["shards"])
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["records"])
            self.cumulative.append(total)
        self._cached_shard_index = -1
        self._cached_records: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative, index)
        start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        if shard_index != self._cached_shard_index:
            self._cached_records = torch.load(
                self.cache_dir / self.shards[shard_index]["file"],
                map_location="cpu",
                weights_only=False,
            )
            self._cached_shard_index = shard_index
        return self._cached_records[index - start]


def collate_query_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack tensor/numeric fields and preserve text/provenance as lists."""

    if not records:
        raise ValueError("cannot collate an empty record list")
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


def deterministic_episode_split_indices(
    dataset: Dataset[Mapping[str, Any]],
    *,
    heldout_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split complete task/episode groups deterministically into train/held-out indices."""

    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0,1)")
    threshold = int(heldout_fraction * 10_000)
    train: list[int] = []
    heldout: list[int] = []
    for index in range(len(dataset)):
        record = dataset[index]
        split_episode = record.get("rollout_episode_id", record["episode_id"])
        key = f"{record['task_id']}|{split_episode}|{seed}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10_000
        (heldout if value < threshold else train).append(index)
    if not train or not heldout:
        raise RuntimeError("deterministic episode split produced an empty partition")
    return train, heldout
