"""Deduplicated exact teacher cache for native-R5 SimVLA V0.

The source compact cache already contains source-verified full VLM conditions.
This module reorganizes each query tensor exactly once and adds the frozen
same-noise teacher action. Production generation is two-rank, explicitly
approved, atomic, checksummed, and never writes one file per query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.native_v0_dataset import (
    stable_episode_partition,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    cached_batch_token_layout,
    freeze_module,
    load_frozen_simvla,
)
from architectures.simvla.adapters.latentloop.native_v0_training_cache import (
    TRAINING_CACHE_SCHEMA,
)
from architectures.simvla.adapters.latentloop.native_v0_condition_hook import (
    extract_action_condition,
)
from architectures.simvla.adapters.latentloop.native_v0_prepare import (
    _official_training_image_inputs,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    ACTION_DIM,
    ACTION_HORIZON,
    CACHE_MARKER_SCHEMA,
    CACHE_SCHEMA,
    CACHE_SHARD_SCHEMA,
    CONDITION_DIM,
    CONDITION_TOKENS,
    FIXED_K_C,
    PROPRIO_DIM,
    atomic_write_json,
    canonical_sha256,
    project_exact_teacher_cache,
    query_identity,
    require_gate_payload,
    sha256_file,
    validate_query_windows,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _gpu_contract_from_environment() -> dict[str, Any]:
    path = os.environ.get("SIMVLA_GPU_CONTRACT_JSON")
    if not path:
        raise RuntimeError("SIMVLA_GPU_CONTRACT_JSON is required")
    payload = _load_json(path)
    if payload.get("verdict") != "TWO_SELECTED_GPUS_IDLE":
        raise RuntimeError("two-GPU launch contract did not pass")
    return payload


def _verify_source_inputs(
    source: dict[str, Any],
    *,
    compact_root: Path,
    checkpoint: str,
    norm_stats: str | Path,
    action_noise_seed_base: int,
) -> None:
    observed_manifest = sha256_file(compact_root / "manifest.json")
    if source.get("compact_cache_manifest_sha256") != observed_manifest:
        raise RuntimeError("compact cache manifest differs from the child source lock")
    if source.get("checkpoint") != str(checkpoint):
        raise RuntimeError("checkpoint identifier differs from the child source lock")
    if source.get("norm_stats_sha256") != sha256_file(norm_stats):
        raise RuntimeError("normalization differs from the child source lock")
    if int(source.get("action_noise_seed_base", -1)) != int(action_noise_seed_base):
        raise RuntimeError("action-noise seed base differs from the child source lock")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compact_window_catalog(compact_cache: str | Path) -> list[dict[str, Any]]:
    root = Path(compact_cache).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema_version") != TRAINING_CACHE_SCHEMA:
        raise ValueError("exact cache requires the corrected native-R5 compact cache")
    if (
        int(manifest.get("fixed_k", -1)) != 4
        or int(manifest.get("execution_horizon", -1)) != 5
        or int(manifest.get("action_horizon", -1)) != 10
    ):
        raise ValueError("compact cache must use K_C=4, H=10, R=5")
    windows: list[dict[str, Any]] = []
    for window_id, entry in enumerate(manifest["sequences"]):
        task_id = int(entry["task_id"])
        episode_id = str(entry["episode_id"])
        anchor = int(entry["anchor_query_index"])
        query_ids = [query_identity(task_id, episode_id, anchor + age) for age in range(4)]
        windows.append(
            {
                "window_id": window_id,
                "task_id": task_id,
                "episode_id": episode_id,
                "anchor_query_index": anchor,
                "query_ids": query_ids,
                "source_sequence_file": str(entry["file"]),
                "source_sequence_sha256": str(entry["sha256"]),
            }
        )
    validation = validate_query_windows([item["query_ids"] for item in windows])
    if not validation["passed"]:
        raise RuntimeError(f"compact cache query catalog is not deduplicated: {validation}")
    return windows


def exact_cache_projection(
    compact_cache: str | Path,
    *,
    shared_storage_path: str | Path,
    shard_queries: int = 1024,
) -> dict[str, Any]:
    windows = compact_window_catalog(compact_cache)
    free_bytes = os.statvfs(Path(shared_storage_path).expanduser().resolve())
    available = int(free_bytes.f_bavail * free_bytes.f_frsize)
    projection = project_exact_teacher_cache(
        query_count=len(windows) * FIXED_K_C,
        window_count=len(windows),
        shard_queries=shard_queries,
        free_bytes_before=available,
    ).to_dict()
    projection.update(
        {
            "verdict": (
                "EXACT_FP32_STORAGE_GATE_PASS"
                if projection["exact_fp32_storage_gate_pass"]
                else "CACHE_GENERATION_BLOCKED"
            ),
            "compact_cache": str(Path(compact_cache).expanduser().resolve()),
            "compact_manifest_sha256": sha256_file(
                Path(compact_cache).expanduser().resolve() / "manifest.json"
            ),
            "query_deduplication": validate_query_windows(
                [item["query_ids"] for item in windows]
            ),
            "condition_origin": (
                "bitwise copy of source-verified full VLM condition in corrected native-R5 cache"
            ),
            "images": "stable HDF5 references; no image tensor duplication",
            "noise": "63-bit query key only; full noise tensor regenerated exactly",
            "windows": "ordered query IDs only",
        }
    )
    return projection


def _source_sequence(compact_root: Path, window: dict[str, Any]) -> dict[str, Any]:
    path = compact_root / str(window["source_sequence_file"])
    if sha256_file(path) != window["source_sequence_sha256"]:
        raise RuntimeError(f"source sequence hash changed: {path}")
    sequence = torch.load(path, map_location="cpu", weights_only=False)
    if sequence.get("schema_version") != TRAINING_CACHE_SCHEMA:
        raise ValueError(f"unexpected source sequence schema: {path}")
    return sequence


def _query_key(
    *,
    checkpoint: str,
    task_id: int,
    episode_id: str,
    query_index: int,
    seed_base: int,
) -> ActionNoiseKey:
    return ActionNoiseKey(
        checkpoint=checkpoint,
        task_id=task_id,
        episode_id=episode_id,
        policy_query_index=query_index,
        seed_base=seed_base,
    )


def _drop_unused_vlm(model: Any) -> dict[str, Any]:
    """Release modules that the exact cache action decoder never calls."""

    removed: list[str] = []
    for name in ("vlm", "vlm_processor"):
        if hasattr(model, name):
            delattr(model, name)
            removed.append(name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not hasattr(model, "transformer") or not hasattr(model, "action_space"):
        raise RuntimeError("dropping VLM damaged the frozen action path")
    return {
        "removed_unused_modules": removed,
        "transformer_preserved": True,
        "action_space_preserved": True,
    }


def _decode_batches(
    action_adapter: Any,
    *,
    conditions: torch.Tensor,
    proprio: torch.Tensor,
    noises: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, conditions.shape[0], batch_size):
        end = min(start + batch_size, conditions.shape[0])
        outputs.append(
            action_adapter.decode_action_from_condition(
                conditions[start:end].to(device, non_blocking=True),
                proprio[start:end].to(device, non_blocking=True),
                steps=10,
                initial_noise=noises[start:end].to(device, non_blocking=True),
                requires_grad=False,
            ).detach().cpu()
        )
    return torch.cat(outputs, dim=0).contiguous()


def _materialize_window_queries(
    *,
    compact_root: Path,
    windows: Sequence[dict[str, Any]],
    checkpoint: str,
    seed_base: int,
    processor: Any,
) -> dict[str, Any]:
    conditions: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    groups: list[torch.Tensor] = []
    proprio: list[torch.Tensor] = []
    noises: list[torch.Tensor] = []
    seeds: list[int] = []
    metadata: list[dict[str, Any]] = []
    for window in windows:
        sequence = _source_sequence(compact_root, window)
        sequence_conditions = sequence["condition_sequence"].float().contiguous()
        sequence_proprio = sequence["proprio_sequence"].float().contiguous()
        if sequence_conditions.shape != (4, CONDITION_TOKENS, CONDITION_DIM):
            raise ValueError("source condition shape changed")
        if sequence_proprio.shape != (4, PROPRIO_DIM):
            raise ValueError("source proprio shape changed")
        layout = cached_batch_token_layout(
            condition=sequence_conditions,
            language_instructions=[str(sequence["language_instruction"])] * 4,
            processor=processor,
        )
        for age in range(4):
            query_index = int(window["anchor_query_index"]) + age
            key = _query_key(
                checkpoint=checkpoint,
                task_id=int(window["task_id"]),
                episode_id=str(window["episode_id"]),
                query_index=query_index,
                seed_base=seed_base,
            )
            noise = explicit_action_noise(
                key,
                batch_size=1,
                action_horizon=ACTION_HORIZON,
                action_dim=ACTION_DIM,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )[0]
            conditions.append(sequence_conditions[age])
            masks.append(layout.valid_mask[age].cpu())
            groups.append(layout.group_ids[age].cpu().to(torch.uint8))
            proprio.append(sequence_proprio[age])
            noises.append(noise)
            seeds.append(key.seed())
            metadata.append(
                {
                    "query_id": window["query_ids"][age],
                    "window_id": int(window["window_id"]),
                    "age_in_window": age,
                    "task_id": int(window["task_id"]),
                    "episode_id": str(window["episode_id"]),
                    "query_index": query_index,
                    "language_instruction": str(sequence["language_instruction"]),
                    "raw_rgb_ref": dict(sequence["raw_rgb_refs"][age]),
                    "noise_key": {
                        "checkpoint": checkpoint,
                        "task_id": int(window["task_id"]),
                        "episode_id": str(window["episode_id"]),
                        "policy_query_index": query_index,
                        "seed_base": int(seed_base),
                        "seed": key.seed(),
                    },
                }
            )
    return {
        "conditions": torch.stack(conditions),
        "valid_masks": torch.stack(masks).to(torch.bool),
        "group_ids": torch.stack(groups).to(torch.uint8),
        "proprio": torch.stack(proprio),
        "noises": torch.stack(noises),
        "noise_seeds": torch.tensor(seeds, dtype=torch.int64),
        "metadata": metadata,
    }


def _write_shard(
    *,
    output: Path,
    rank: int,
    shard_index: int,
    source_hash: str,
    materialized: dict[str, Any],
    teacher_actions: torch.Tensor,
) -> dict[str, Any]:
    rank_root = output / f"rank_{rank:02d}"
    relative = Path(f"rank_{rank:02d}") / f"queries_{shard_index:05d}.pt"
    path = output / relative
    payload = {
        "schema_version": CACHE_SHARD_SCHEMA,
        "source_combined_sha256": source_hash,
        "rank": rank,
        "shard_index": shard_index,
        "query_count": len(materialized["metadata"]),
        "conditions": materialized["conditions"].contiguous(),
        "valid_masks": materialized["valid_masks"].contiguous(),
        "group_ids": materialized["group_ids"].contiguous(),
        "teacher_actions": teacher_actions.float().contiguous(),
        "proprio": materialized["proprio"].float().contiguous(),
        "noise_seeds": materialized["noise_seeds"].contiguous(),
        "metadata": materialized["metadata"],
    }
    _atomic_torch_save(payload, path)
    digest = sha256_file(path)
    marker = {
        "schema_version": CACHE_MARKER_SCHEMA,
        "source_combined_sha256": source_hash,
        "file": str(relative),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "rank": rank,
        "shard_index": shard_index,
        "query_count": len(materialized["metadata"]),
        "first_query_id": materialized["metadata"][0]["query_id"],
        "last_query_id": materialized["metadata"][-1]["query_id"],
        "query_ids": [item["query_id"] for item in materialized["metadata"]],
        "complete": True,
        "recovery": "keep completed shard and marker; regenerate only missing shard indices",
    }
    atomic_write_json(rank_root / f"queries_{shard_index:05d}.complete.json", marker)
    return marker


def generate_exact_cache(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("SIMVLA_EXACT_CACHE_GENERATION_APPROVED") != "1":
        raise RuntimeError(
            "production cache generation requires SIMVLA_EXACT_CACHE_GENERATION_APPROVED=1"
        )
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("production exact cache generation requires torchrun WORLD_SIZE=2")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    output = Path(args.output).expanduser().resolve()
    compact_root = Path(args.compact_cache).expanduser().resolve()
    source = _load_json(args.source_lock)
    _verify_source_inputs(
        source,
        compact_root=compact_root,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        action_noise_seed_base=args.action_noise_seed_base,
    )
    source_hash = str(source["combined_sha256"])
    gpu_contract = _gpu_contract_from_environment()
    require_gate_payload(
        args.pilot_gate,
        verdicts=("EXACT_TEACHER_CACHE_PASS",),
        source_combined_sha256=source_hash,
    )
    projection = exact_cache_projection(
        compact_root,
        shared_storage_path=output.parent,
        shard_queries=args.shard_queries,
    )
    if projection["verdict"] != "EXACT_FP32_STORAGE_GATE_PASS":
        raise RuntimeError(f"exact cache storage gate failed: {projection}")
    exists = torch.tensor([int(output.exists())], device=device)
    dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if int(exists.item()):
        raise FileExistsError(f"refusing existing exact cache output: {output}")
    if rank == 0:
        output.mkdir(parents=True)
        atomic_write_json(output / "source_lock.json", source)
        atomic_write_json(output / "storage_projection.json", projection)
    dist.barrier()

    windows = compact_window_catalog(compact_root)
    windows_per_rank = math.ceil(len(windows) / 2)
    rank_start = rank * windows_per_rank
    rank_end = min(rank_start + windows_per_rank, len(windows))
    assigned = windows[rank_start:rank_end]
    if not assigned:
        raise RuntimeError(f"rank {rank} received no windows")
    shard_windows = max(1, int(args.shard_queries) // FIXED_K_C)

    model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    freeze_module(model)
    dropped = _drop_unused_vlm(model)
    markers: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(assigned), shard_windows)):
        selected = assigned[start : start + shard_windows]
        materialized = _materialize_window_queries(
            compact_root=compact_root,
            windows=selected,
            checkpoint=args.checkpoint,
            seed_base=args.action_noise_seed_base,
            processor=processor,
        )
        actions = _decode_batches(
            action_adapter,
            conditions=materialized["conditions"],
            proprio=materialized["proprio"],
            noises=materialized["noises"],
            device=device,
            batch_size=args.decode_batch_size,
        )
        markers.append(
            _write_shard(
                output=output,
                rank=rank,
                shard_index=shard_index,
                source_hash=source_hash,
                materialized=materialized,
                teacher_actions=actions,
            )
        )
        del materialized, actions
    rank_manifest = {
        "schema_version": CACHE_SCHEMA,
        "source_combined_sha256": source_hash,
        "rank": rank,
        "window_range": [rank_start, rank_end],
        "windows": len(assigned),
        "queries": len(assigned) * FIXED_K_C,
        "shards": markers,
        "unused_vlm_release": dropped,
    }
    atomic_write_json(output / f"rank_{rank:02d}" / "rank_manifest.json", rank_manifest)
    dist.barrier()

    result: dict[str, Any] = {}
    if rank == 0:
        ranks = [_load_json(output / f"rank_{value:02d}" / "rank_manifest.json") for value in range(2)]
        all_markers = [marker for item in ranks for marker in item["shards"]]
        query_index: list[dict[str, Any]] = []
        for marker in all_markers:
            for offset, query_id_value in enumerate(marker["query_ids"]):
                query_index.append(
                    {
                        "query_id": query_id_value,
                        "file": marker["file"],
                        "offset": offset,
                    }
                )
        windows_payload = [item["query_ids"] for item in windows]
        validation = validate_query_windows(windows_payload)
        if not validation["passed"] or len(query_index) != validation["unique_queries"]:
            raise RuntimeError("merged exact cache query/window identity failed")
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "source_combined_sha256": source_hash,
            "checkpoint": args.checkpoint,
            "norm_stats_sha256": sha256_file(args.norm_stats),
            "compact_cache_manifest_sha256": sha256_file(compact_root / "manifest.json"),
            "condition_shape": [CONDITION_TOKENS, CONDITION_DIM],
            "condition_dtype": "torch.float32",
            "action_shape": [ACTION_HORIZON, ACTION_DIM],
            "action_dtype": "torch.float32",
            "execution_horizon": 5,
            "fixed_k_c": 4,
            "flow_steps": 10,
            "action_noise_seed_base": int(args.action_noise_seed_base),
            "condition_origin": "bitwise copied from source-verified compact training cache",
            "images": "stable HDF5 references",
            "query_count": len(query_index),
            "window_count": len(windows_payload),
            "query_index": query_index,
            "windows": windows_payload,
            "shards": all_markers,
            "query_deduplication": validation,
            "production_gpu_count": 2,
            "gpu_contract": gpu_contract,
            "complete": True,
        }
        manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
        atomic_write_json(output / "manifest.json", manifest)
        result = {
            "verdict": "EXACT_TEACHER_CACHE_COMPLETE",
            "source_combined_sha256": source_hash,
            "manifest": str(output / "manifest.json"),
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "queries": len(query_index),
            "windows": len(windows_payload),
            "shards": len(all_markers),
            "selected_physical_gpu_ids": gpu_contract["selected_physical_gpu_ids"],
        }
        atomic_write_json(output / "cache_generation_complete.json", result)
    dist.barrier()
    dist.destroy_process_group()
    return result


def _load_rgb_ref(ref: dict[str, Any]) -> torch.Tensor:
    with h5py.File(str(ref["hdf5_path"]), "r") as handle:
        demo = handle["data"][str(ref["demo_key"])]
        views = []
        for camera in ref["camera_names"]:
            image = np.asarray(demo[f"obs/{camera}"][int(ref["timestep"])])
            if bool(ref.get("rotate_180", True)):
                image = image[::-1, ::-1]
            views.append(np.ascontiguousarray(image))
    return torch.from_numpy(np.stack(views)).to(torch.uint8)


class ExactTeacherStore:
    """Lazy mmap-backed reader; checksums are validated before training, not per step."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = _load_json(self.root / "manifest.json")
        if self.manifest.get("schema_version") != CACHE_SCHEMA or not self.manifest.get("complete"):
            raise ValueError("exact teacher cache is incomplete or incompatible")
        self.locators = {
            str(item["query_id"]): (str(item["file"]), int(item["offset"]))
            for item in self.manifest["query_index"]
        }
        if len(self.locators) != int(self.manifest["query_count"]):
            raise RuntimeError("exact cache query index contains duplicates")
        self._loaded: dict[str, dict[str, Any]] = {}

    def _shard(self, relative: str) -> dict[str, Any]:
        if relative not in self._loaded:
            path = self.root / relative
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != CACHE_SHARD_SCHEMA:
                raise ValueError(f"invalid exact cache shard: {path}")
            self._loaded[relative] = payload
        return self._loaded[relative]

    def query(self, query_id: str) -> dict[str, Any]:
        relative, offset = self.locators[str(query_id)]
        shard = self._shard(relative)
        metadata = dict(shard["metadata"][offset])
        return {
            "metadata": metadata,
            "condition": shard["conditions"][offset],
            "valid_mask": shard["valid_masks"][offset],
            "group_ids": shard["group_ids"][offset].to(torch.long),
            "teacher_action": shard["teacher_actions"][offset],
            "proprio": shard["proprio"][offset],
            "noise_seed": int(shard["noise_seeds"][offset]),
        }


class ExactTeacherSequenceDataset(Dataset[dict[str, Any]]):
    """Native q0-q1-q2-q3 windows backed only by deduplicated query IDs."""

    def __init__(
        self,
        cache: str | Path,
        *,
        split: str,
        heldout_fraction: float = 0.2,
        split_seed: int = 20260822,
    ) -> None:
        if split not in {"train", "heldout", "all"}:
            raise ValueError("split must be train, heldout, or all")
        self.store = ExactTeacherStore(cache)
        selected: list[list[str]] = []
        identities: list[tuple[int, str, int]] = []
        for window in self.store.manifest["windows"]:
            first = self.store.query(window[0])["metadata"]
            heldout = stable_episode_partition(
                int(first["task_id"]), str(first["episode_id"]), int(split_seed)
            ) < float(heldout_fraction)
            if split == "train" and heldout:
                continue
            if split == "heldout" and not heldout:
                continue
            selected.append(list(window))
            identities.append(
                (int(first["task_id"]), str(first["episode_id"]), int(first["query_index"]))
            )
        if not selected:
            raise ValueError(f"no exact teacher windows found for split={split}")
        self.windows = tuple(tuple(item) for item in selected)
        self.identities = tuple(identities)
        self.split = split
        self.heldout_fraction = float(heldout_fraction)
        self.split_seed = int(split_seed)
        # Preserve the parent NativeV0SequenceDataset byte contract exactly.
        split_payload = json.dumps(
            self.identities,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.split_sha256 = hashlib.sha256(split_payload).hexdigest()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        queries = [self.store.query(value) for value in self.windows[index]]
        first = queries[0]["metadata"]
        images = torch.stack([_load_rgb_ref(item["metadata"]["raw_rgb_ref"]) for item in queries])
        noises = []
        for item in queries[1:]:
            metadata = item["metadata"]
            key_fields = metadata["noise_key"]
            key = ActionNoiseKey(
                checkpoint=str(key_fields["checkpoint"]),
                task_id=int(key_fields["task_id"]),
                episode_id=str(key_fields["episode_id"]),
                policy_query_index=int(key_fields["policy_query_index"]),
                seed_base=int(key_fields["seed_base"]),
            )
            if key.seed() != int(item["noise_seed"]):
                raise RuntimeError("cached query noise key changed")
            noises.append(
                explicit_action_noise(
                    key,
                    batch_size=1,
                    action_horizon=10,
                    action_dim=7,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )[0]
            )
        return {
            "task_id": int(first["task_id"]),
            "episode_id": str(first["episode_id"]),
            "anchor_query_index": int(first["query_index"]),
            "language_instruction": str(first["language_instruction"]),
            "query_ids": list(self.windows[index]),
            "image_sequence": images,
            "proprio_sequence": torch.stack([item["proprio"] for item in queries]),
            "anchor_condition": queries[0]["condition"],
            "teacher_conditions": torch.stack([item["condition"] for item in queries[1:]]),
            "valid_mask": queries[0]["valid_mask"],
            "group_ids": queries[0]["group_ids"],
            "teacher_actions": torch.stack([item["teacher_action"] for item in queries[1:]]),
            "explicit_noises": torch.stack(noises),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "cache_schema_version": CACHE_SCHEMA,
            "cache_manifest_sha256": sha256_file(self.store.root / "manifest.json"),
            "source_combined_sha256": self.store.manifest["source_combined_sha256"],
            "split": self.split,
            "split_unit": "task_id+episode_id",
            "split_seed": self.split_seed,
            "heldout_fraction": self.heldout_fraction,
            "split_sha256": self.split_sha256,
            "windows": len(self.windows),
            "teacher_conditions_are_targets_only": True,
            "teacher_actions_are_exact_cached_targets": True,
            "query_tensors_deduplicated": True,
            "action_horizon": 10,
            "execution_horizon": 5,
            "fixed_k_c": 4,
        }


def collate_exact_teacher_sequences(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    tensor_keys = (
        "image_sequence",
        "proprio_sequence",
        "anchor_condition",
        "teacher_conditions",
        "valid_mask",
        "group_ids",
        "teacher_actions",
        "explicit_noises",
    )
    return {
        **{key: torch.stack([item[key] for item in items]) for key in tensor_keys},
        "task_id": torch.tensor([item["task_id"] for item in items], dtype=torch.long),
        "episode_id": [item["episode_id"] for item in items],
        "anchor_query_index": torch.tensor(
            [item["anchor_query_index"] for item in items], dtype=torch.long
        ),
        "language_instruction": [item["language_instruction"] for item in items],
        "query_ids": [item["query_ids"] for item in items],
    }


def validate_exact_cache(cache: str | Path, *, verify_checksums: bool) -> dict[str, Any]:
    root = Path(cache).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    errors: list[str] = []
    if manifest.get("schema_version") != CACHE_SCHEMA or not manifest.get("complete"):
        errors.append("manifest schema/complete marker failed")
    for shard in manifest.get("shards", []):
        path = root / shard["file"]
        marker = path.with_suffix(".complete.json")
        if not path.is_file() or not marker.is_file():
            errors.append(f"missing shard or completion marker: {path}")
            continue
        observed = _load_json(marker)
        if observed.get("sha256") != shard.get("sha256"):
            errors.append(f"marker/manifest checksum mismatch: {path}")
        if verify_checksums and sha256_file(path) != shard.get("sha256"):
            errors.append(f"shard checksum mismatch: {path}")
    windows = validate_query_windows(manifest.get("windows", []))
    if not windows["passed"]:
        errors.extend(windows["errors"])
    return {
        "verdict": "EXACT_TEACHER_CACHE_VALID" if not errors else "EXACT_TEACHER_CACHE_INVALID",
        "source_combined_sha256": manifest.get("source_combined_sha256"),
        "queries": manifest.get("query_count"),
        "windows": manifest.get("window_count"),
        "shards": len(manifest.get("shards", [])),
        "checksums_verified": bool(verify_checksums),
        "errors": errors,
        "passed": not errors,
    }


def command_projection(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing projection output: {output}")
    result = exact_cache_projection(
        args.compact_cache,
        shared_storage_path=args.shared_storage_path,
        shard_queries=args.shard_queries,
    )
    source = _load_json(args.source_lock)
    compact_root = Path(args.compact_cache).expanduser().resolve()
    if source.get("compact_cache_manifest_sha256") != sha256_file(
        compact_root / "manifest.json"
    ):
        raise RuntimeError("projection compact cache differs from child source lock")
    result.update(
        {
            "source_combined_sha256": source["combined_sha256"],
            "checkpoint": source["checkpoint"],
            "checkpoint_revision": source["checkpoint_revision"],
            "norm_stats_sha256": source["norm_stats_sha256"],
            "dataset_splits": source["dataset_splits"],
            "action_noise_seed_base": source["action_noise_seed_base"],
        }
    )
    atomic_write_json(output, result)
    return result


def command_pilot(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing pilot output: {output}")
    output.mkdir(parents=True)
    source = _load_json(args.source_lock)
    source_hash = str(source["combined_sha256"])
    gpu_contract = _gpu_contract_from_environment()
    compact_root = Path(args.compact_cache).expanduser().resolve()
    _verify_source_inputs(
        source,
        compact_root=compact_root,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        action_noise_seed_base=args.action_noise_seed_base,
    )
    windows = compact_window_catalog(compact_root)[: int(args.pilot_windows)]
    if not windows:
        raise RuntimeError("pilot selected no windows")
    device = torch.device(args.device)
    model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    freeze_module(model)
    materialized = _materialize_window_queries(
        compact_root=compact_root,
        windows=windows,
        checkpoint=args.checkpoint,
        seed_base=args.action_noise_seed_base,
        processor=processor,
    )
    streaming_actions = _decode_batches(
        action_adapter,
        conditions=materialized["conditions"],
        proprio=materialized["proprio"],
        noises=materialized["noises"],
        device=device,
        batch_size=args.decode_batch_size,
    )
    _write_shard(
        output=output,
        rank=0,
        shard_index=0,
        source_hash=source_hash,
        materialized=materialized,
        teacher_actions=streaming_actions,
    )
    shard_path = output / "rank_00" / "queries_00000.pt"
    cached = torch.load(shard_path, map_location="cpu", weights_only=False)
    condition_diff = (materialized["conditions"] - cached["conditions"]).abs()
    action_diff = (streaming_actions - cached["teacher_actions"]).abs()
    reencoded_conditions: list[torch.Tensor] = []
    reencoded_masks: list[torch.Tensor] = []
    reencoded_groups: list[torch.Tensor] = []
    tokenizer = processor.tokenizer
    for metadata in materialized["metadata"]:
        raw_rgb = _load_rgb_ref(metadata["raw_rgb_ref"])
        processed = _official_training_image_inputs(
            raw_rgb,
            image_size=int(getattr(processor, "image_size", 384)),
            num_views=int(getattr(processor, "num_views", 3)),
        )
        processed.update(processor.encode_language([metadata["language_instruction"]]))
        processed = {key: value.to(device) for key, value in processed.items()}
        with torch.no_grad():
            extracted = extract_action_condition(
                model,
                input_ids=processed["input_ids"],
                image_input=processed["image_input"],
                image_mask=processed["image_mask"],
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                special_token_ids=getattr(tokenizer, "all_special_ids", ()),
            )
        reencoded_conditions.append(extracted.condition[0].detach().cpu())
        reencoded_masks.append(extracted.layout.valid_mask[0].detach().cpu())
        reencoded_groups.append(extracted.layout.group_ids[0].detach().cpu().to(torch.uint8))
    reencoded_condition = torch.stack(reencoded_conditions)
    reencoded_mask = torch.stack(reencoded_masks)
    reencoded_group = torch.stack(reencoded_groups)
    reencode_diff = (reencoded_condition.float() - materialized["conditions"].float()).abs()

    regenerated_noises: list[torch.Tensor] = []
    for metadata in cached["metadata"]:
        fields = metadata["noise_key"]
        regenerated_noises.append(
            explicit_action_noise(
                ActionNoiseKey(
                    checkpoint=str(fields["checkpoint"]),
                    task_id=int(fields["task_id"]),
                    episode_id=str(fields["episode_id"]),
                    policy_query_index=int(fields["policy_query_index"]),
                    seed_base=int(fields["seed_base"]),
                ),
                batch_size=1,
                action_horizon=ACTION_HORIZON,
                action_dim=ACTION_DIM,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )[0]
        )
    regenerated_noise = torch.stack(regenerated_noises)
    image_identity = []
    for metadata in materialized["metadata"]:
        first = _load_rgb_ref(metadata["raw_rgb_ref"])
        second = _load_rgb_ref(cached["metadata"][len(image_identity)]["raw_rgb_ref"])
        image_identity.append(bool(torch.equal(first, second)))
    checks = {
        "condition_bitwise_equal": torch.equal(materialized["conditions"], cached["conditions"]),
        "mask_bitwise_equal": torch.equal(materialized["valid_masks"], cached["valid_masks"]),
        "group_ids_bitwise_equal": torch.equal(materialized["group_ids"], cached["group_ids"]),
        "action_bitwise_equal": torch.equal(streaming_actions, cached["teacher_actions"]),
        "noise_seed_bitwise_equal": torch.equal(materialized["noise_seeds"], cached["noise_seeds"]),
        "delta_encoder_image_input_equal": all(image_identity),
        "proprio_bitwise_equal": torch.equal(materialized["proprio"], cached["proprio"]),
        "source_native_fp32": cached["conditions"].dtype == torch.float32,
        "streaming_vlm_reencode_allclose_1e_5": torch.allclose(
            reencoded_condition.float(),
            materialized["conditions"].float(),
            atol=1e-5,
            rtol=1e-5,
        ),
        "streaming_vlm_mask_identity": torch.equal(
            reencoded_mask, materialized["valid_masks"]
        ),
        "streaming_vlm_group_identity": torch.equal(
            reencoded_group, materialized["group_ids"]
        ),
        "noise_tensor_regeneration_bitwise_equal": torch.equal(
            regenerated_noise, materialized["noises"]
        ),
    }
    passed = all(checks.values())
    result = {
        "verdict": "EXACT_TEACHER_CACHE_PASS" if passed else "EXACT_TEACHER_CACHE_FAIL",
        "source_combined_sha256": source_hash,
        "queries": len(materialized["metadata"]),
        "windows": len(windows),
        "checks": checks,
        "condition_max_abs_difference": float(condition_diff.max().item()),
        "condition_mean_abs_difference": float(condition_diff.mean().item()),
        "action_max_abs_difference": float(action_diff.max().item()),
        "action_mean_abs_difference": float(action_diff.mean().item()),
        "streaming_vlm_reencode_max_abs_difference": float(reencode_diff.max().item()),
        "streaming_vlm_reencode_mean_abs_difference": float(reencode_diff.mean().item()),
        "condition_origin": (
            "cache payload is a bitwise copy of the source-verified full VLM output; "
            "the pilot also reruns the frozen VLM on the exact source image references"
        ),
        "pilot_only": True,
        "production_cache_generated": False,
        "gpu_contract": gpu_contract,
    }
    atomic_write_json(output / "exact_teacher_cache_pilot.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    projection = subparsers.add_parser("project")
    projection.add_argument("--output", required=True)
    projection.add_argument("--compact-cache", required=True)
    projection.add_argument("--source-lock", required=True)
    projection.add_argument("--shared-storage-path", required=True)
    projection.add_argument("--shard-queries", type=int, default=1024)

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--compact-cache", required=True)
    pilot.add_argument("--source-lock", required=True)
    pilot.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    pilot.add_argument("--norm-stats", required=True)
    pilot.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    pilot.add_argument("--pilot-windows", type=int, default=2)
    pilot.add_argument("--decode-batch-size", type=int, default=4)
    pilot.add_argument("--action-noise-seed-base", type=int, default=20260822)
    pilot.add_argument("--device", default="cuda")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--output", required=True)
    generate.add_argument("--compact-cache", required=True)
    generate.add_argument("--source-lock", required=True)
    generate.add_argument("--pilot-gate", required=True)
    generate.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    generate.add_argument("--norm-stats", required=True)
    generate.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    generate.add_argument("--shard-queries", type=int, default=1024)
    generate.add_argument("--decode-batch-size", type=int, default=4)
    generate.add_argument("--action-noise-seed-base", type=int, default=20260822)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--cache", required=True)
    validate.add_argument("--verify-checksums", action="store_true")
    validate.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "project":
        result = command_projection(args)
    elif args.command == "pilot":
        result = command_pilot(args)
    elif args.command == "generate":
        result = generate_exact_cache(args)
    else:
        result = validate_exact_cache(args.cache, verify_checksums=args.verify_checksums)
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing existing validation output: {output}")
        atomic_write_json(output, result)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
