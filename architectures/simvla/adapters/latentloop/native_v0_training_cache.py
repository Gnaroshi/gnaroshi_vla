"""Compact official-training sequence cache for corrected native SimVLA V0."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import torch

from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    native_v0_source_manifest,
    write_json,
)
from architectures.simvla.adapters.latentloop.source_lock import sha256_file


TRAINING_CACHE_SCHEMA = "simvla_native_v0_training_sequences_r5_v1"


def _resolve_hdf5(raw_path: str, dataset_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path.resolve()
    candidates = (
        dataset_root / path.parent.name / path.name,
        dataset_root / "libero_10" / path.name,
        dataset_root / path.name,
    )
    matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(set(matches)) != 1:
        raise FileNotFoundError(
            f"cannot uniquely relocate training HDF5 {raw_path!r} under {dataset_root}: {matches}"
        )
    return matches[0]


def _condition(record: dict[str, Any]) -> torch.Tensor:
    value = record["condition"].detach().cpu()
    if value.shape == (1, 122, 960):
        value = value[0]
    if value.shape != (122, 960):
        raise ValueError(f"unexpected cached SimVLA condition shape: {tuple(value.shape)}")
    return value.contiguous()


def _proprio(record: dict[str, Any]) -> torch.Tensor:
    value = record["proprio"].detach().cpu()
    if value.shape == (1, 8):
        value = value[0]
    if value.shape != (8,):
        raise ValueError(f"unexpected cached proprio shape: {tuple(value.shape)}")
    return value.float().contiguous()


def _flush_episode(
    records: list[dict[str, Any]],
    *,
    output: Path,
    dataset_root: Path,
    next_sequence_id: int,
) -> tuple[list[dict[str, Any]], int]:
    if not records:
        return [], next_sequence_id
    records.sort(key=lambda record: int(record["timestep"]))
    episode_id = str(records[0]["episode_id"])
    if any(str(record["episode_id"]) != episode_id for record in records):
        raise ValueError("episode flush received mixed episode IDs")
    by_timestep = {int(record["timestep"]): record for record in records}
    if len(by_timestep) != len(records):
        raise ValueError(f"duplicate timestep in training episode {episode_id}")
    task_id = int(episode_id.split(":", 1)[0])
    entries: list[dict[str, Any]] = []
    for anchor_timestep in sorted(by_timestep):
        if anchor_timestep % 20:
            continue
        timesteps = [anchor_timestep + 5 * offset for offset in range(4)]
        if any(timestep not in by_timestep for timestep in timesteps):
            continue
        selected = [by_timestep[timestep] for timestep in timesteps]
        raw_refs: list[dict[str, Any]] = []
        for record in selected:
            raw_ref = dict(record["raw_rgb_ref"])
            raw_ref["hdf5_path"] = str(
                _resolve_hdf5(str(raw_ref["hdf5_path"]), dataset_root)
            )
            raw_refs.append(raw_ref)
        sequence = {
            "schema_version": TRAINING_CACHE_SCHEMA,
            "task_id": task_id,
            "episode_id": episode_id,
            "anchor_query_index": anchor_timestep // 5,
            "anchor_timestep": anchor_timestep,
            "timesteps": timesteps,
            "language_instruction": str(selected[0]["language_instruction"]),
            "raw_rgb_refs": raw_refs,
            "proprio_sequence": torch.stack([_proprio(record) for record in selected]),
            "condition_sequence": torch.stack([_condition(record) for record in selected]),
            "source_cached_actions_used": False,
            "executed_actions_present": False,
        }
        relative = Path("sequences") / f"sequence_{next_sequence_id:07d}.pt"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sequence, path)
        entries.append(
            {
                "file": str(relative),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "task_id": task_id,
                "episode_id": episode_id,
                "anchor_query_index": anchor_timestep // 5,
                "anchor_timestep": anchor_timestep,
            }
        )
        next_sequence_id += 1
    return entries, next_sequence_id


def build_training_cache(
    *,
    output: str | Path,
    teacher_cache: str | Path,
    dataset_root: str | Path,
    checkpoint: str,
    norm_stats: str | Path,
    action_noise_seed_base: int,
) -> dict[str, Any]:
    """Extract non-overlapping q0/q1/q2/q3 sequences without running a model."""

    final_output = Path(output).expanduser().resolve()
    if final_output.exists():
        raise FileExistsError(f"refusing existing compact cache: {final_output}")
    output = final_output.with_name(f".{final_output.name}.tmp-{os.getpid()}")
    if output.exists():
        raise FileExistsError(f"refusing existing temporary compact cache: {output}")
    output.mkdir(parents=True)
    teacher_root = Path(teacher_cache).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    teacher_manifest_path = teacher_root / "manifest.json"
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    metadata = teacher_manifest.get("metadata", {})
    extra = metadata.get("extra", {})
    if metadata.get("architecture") != "simvla":
        raise ValueError("source teacher cache is not SimVLA")
    if metadata.get("checkpoint") != checkpoint:
        raise ValueError("source teacher cache checkpoint identifier differs")
    if extra.get("suite") != "libero_10":
        raise ValueError("correct native V0 first campaign requires libero_10 training data")
    if int(extra.get("action_horizon", -1)) != 10:
        raise ValueError("source teacher cache action horizon is not H=10")

    source = native_v0_source_manifest(
        checkpoint=checkpoint,
        norm_stats=norm_stats,
        cache=None,
    )
    source_shards: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    current_episode: str | None = None
    episode_records: list[dict[str, Any]] = []
    next_sequence_id = 0
    observed_samples = 0

    for shard_name in teacher_manifest["shards"]:
        shard_path = teacher_root / shard_name
        payload = shard_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        records = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        source_shards.append(
            {
                "file": str(shard_name),
                "sha256": digest,
                "size_bytes": len(payload),
                "records": len(records),
            }
        )
        for record in records:
            observed_samples += 1
            episode_id = str(record["episode_id"])
            if current_episode is None:
                current_episode = episode_id
            if episode_id != current_episode:
                new_entries, next_sequence_id = _flush_episode(
                    episode_records,
                    output=output,
                    dataset_root=dataset_root,
                    next_sequence_id=next_sequence_id,
                )
                sequences.extend(new_entries)
                episode_records = []
                current_episode = episode_id
            if int(record["timestep"]) % 5 == 0:
                episode_records.append(record)
    new_entries, next_sequence_id = _flush_episode(
        episode_records,
        output=output,
        dataset_root=dataset_root,
        next_sequence_id=next_sequence_id,
    )
    sequences.extend(new_entries)
    if not sequences:
        raise RuntimeError("no native-R5 q0/q1/q2/q3 training sequences were produced")

    generation_meta = Path(str(extra.get("generation_meta_path", "")))
    generation_meta_hash = (
        sha256_file(generation_meta) if generation_meta.is_file() else None
    )
    payload: dict[str, Any] = {
        "schema_version": TRAINING_CACHE_SCHEMA,
        "data_role": "official_libero_training_demonstrations",
        "suite": "libero_10",
        "action_horizon": 10,
        "execution_horizon": 5,
        "fixed_k": 4,
        "query_timesteps": [0, 5, 10, 15],
        "sequence_stride_timesteps": 20,
        "source_teacher_cache_root": str(teacher_root),
        "source_teacher_manifest_path": str(teacher_manifest_path),
        "source_teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "source_teacher_generation_meta_path": (
            str(generation_meta.resolve()) if generation_meta.is_file() else None
        ),
        "source_teacher_generation_meta_sha256": generation_meta_hash,
        "source_teacher_shards": source_shards,
        "source_teacher_samples": observed_samples,
        "dataset_root": str(dataset_root),
        "checkpoint": checkpoint,
        "checkpoint_revision": source["checkpoint"].get("revision"),
        "simvla_upstream_commit": source["simvla_upstream_commit"],
        "source_generation_combined_sha256": source["combined_sha256"],
        "selected_physical_gpu_ids": source["selected_physical_gpu_ids"],
        "metadata": {
            "source_lock": {
                "norm_stats_sha256": metadata.get("norm_stats_sha256"),
            },
            "source_teacher_cache_norm_stats_path": metadata.get("norm_stats_path"),
            "source_cached_teacher_actions_used": False,
            "final_libero_long_evaluation_episodes_used": False,
            "train_test_episode_overlap": False,
            "raw_rgb_storage": "relocated_hdf5_references",
            "condition_storage": "copy_once_from_official_training_teacher_cache",
            "action_noise_seed_base": int(action_noise_seed_base),
        },
        "sequences": sequences,
        "total_sequences": len(sequences),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(output / "manifest.json", payload)
    os.replace(output, final_output)
    return {
        "verdict": "NATIVE_V0_TRAINING_CACHE_BUILT",
        "output": str(final_output),
        "manifest_sha256": sha256_file(final_output / "manifest.json"),
        "source_teacher_samples": observed_samples,
        "sequences": len(sequences),
        "bytes": sum(int(item["size_bytes"]) for item in sequences),
        "training_demonstrations_only": True,
        "final_eval_episode_overlap": False,
    }


def load_training_manifest(cache_dir: str | Path) -> dict[str, Any]:
    root = Path(cache_dir).expanduser().resolve()
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != TRAINING_CACHE_SCHEMA:
        raise ValueError(
            "correct native V0 requires the compact official-training cache; "
            f"got {payload.get('schema_version')!r}"
        )
    return payload


def load_compact_sequence(
    cache_dir: str | Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    root = Path(cache_dir).expanduser().resolve()
    path = root / entry["file"]
    if path.stat().st_size != int(entry["size_bytes"]):
        raise RuntimeError(f"compact sequence size changed: {path}")
    sequence = torch.load(path, map_location="cpu", weights_only=False)
    if sequence.get("schema_version") != TRAINING_CACHE_SCHEMA:
        raise ValueError(f"invalid compact sequence schema: {path}")
    return sequence


def load_raw_rgb_sequence(sequence: dict[str, Any]) -> torch.Tensor:
    """Load ordered, rotated training RGB views as ``[4,V,H,W,C]`` uint8."""

    refs = list(sequence["raw_rgb_refs"])
    if len(refs) != 4:
        raise ValueError("native V0 sequence requires exactly four query observations")
    identities = {
        (str(ref["hdf5_path"]), str(ref["demo_key"])) for ref in refs
    }
    if len(identities) != 1:
        raise ValueError("native V0 sequence crossed an HDF5 demonstration boundary")
    hdf5_path, demo_key = identities.pop()
    frames: list[np.ndarray] = []
    with h5py.File(hdf5_path, "r") as h5:
        demo = h5["data"][demo_key]
        for ref in refs:
            views: list[np.ndarray] = []
            for camera in ref["camera_names"]:
                image = np.asarray(demo[f"obs/{camera}"][int(ref["timestep"])])
                if bool(ref.get("rotate_180", True)):
                    image = image[::-1, ::-1]
                views.append(np.ascontiguousarray(image))
            frames.append(np.stack(views, axis=0))
    return torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)


def sequence_explicit_noises(
    sequence: dict[str, Any],
    manifest: dict[str, Any],
) -> torch.Tensor:
    """Materialize q1/q2/q3 flow noise without consuming global RNG state."""

    base = int(manifest["metadata"]["action_noise_seed_base"])
    checkpoint = str(manifest["checkpoint"])
    noises = []
    for age in (1, 2, 3):
        key = ActionNoiseKey(
            checkpoint=checkpoint,
            task_id=int(sequence["task_id"]),
            episode_id=str(sequence["episode_id"]),
            policy_query_index=int(sequence["anchor_query_index"]) + age,
            seed_base=base,
        )
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
    return torch.stack(noises)


def first_training_query_record(cache_dir: str | Path) -> dict[str, Any]:
    """Materialize q0 from the first compact training sequence for bounded gates."""

    manifest = load_training_manifest(cache_dir)
    sequence = load_compact_sequence(cache_dir, manifest["sequences"][0])
    key = ActionNoiseKey(
        checkpoint=str(manifest["checkpoint"]),
        task_id=int(sequence["task_id"]),
        episode_id=str(sequence["episode_id"]),
        policy_query_index=int(sequence["anchor_query_index"]),
        seed_base=int(manifest["metadata"]["action_noise_seed_base"]),
    )
    return {
        "task_id": int(sequence["task_id"]),
        "episode_id": str(sequence["episode_id"]),
        "query_index": int(sequence["anchor_query_index"]),
        "language_instruction": str(sequence["language_instruction"]),
        "raw_rgb": load_raw_rgb_sequence(sequence)[0],
        "proprio": sequence["proprio_sequence"][0],
        "full_condition": sequence["condition_sequence"][0],
        "initial_noise": explicit_action_noise(
            key,
            batch_size=1,
            action_horizon=10,
            action_dim=7,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )[0],
    }


def validate_training_cache(
    cache_dir: str | Path,
    *,
    verify_sequence_hashes: bool = True,
) -> dict[str, Any]:
    root = Path(cache_dir).expanduser().resolve()
    manifest = load_training_manifest(root)
    errors: list[str] = []
    identities: set[tuple[int, str, int]] = set()
    for entry in manifest["sequences"]:
        identity = (
            int(entry["task_id"]),
            str(entry["episode_id"]),
            int(entry["anchor_query_index"]),
        )
        if identity in identities:
            errors.append(f"duplicate sequence identity: {identity}")
        identities.add(identity)
        path = root / entry["file"]
        if not path.is_file():
            errors.append(f"missing sequence: {path}")
            continue
        if path.stat().st_size != int(entry["size_bytes"]):
            errors.append(f"sequence size mismatch: {path.name}")
        if verify_sequence_hashes and sha256_file(path) != entry["sha256"]:
            errors.append(f"sequence hash mismatch: {path.name}")
    metadata = manifest.get("metadata", {})
    contract_checks = {
        "official_training_demonstrations": (
            manifest.get("data_role") == "official_libero_training_demonstrations"
        ),
        "no_final_eval_episodes": (
            metadata.get("final_libero_long_evaluation_episodes_used") is False
        ),
        "no_train_test_overlap": metadata.get("train_test_episode_overlap") is False,
        "native_h10_r5_k4": (
            manifest.get("action_horizon") == 10
            and manifest.get("execution_horizon") == 5
            and manifest.get("fixed_k") == 4
        ),
        "cached_actions_excluded": metadata.get("source_cached_teacher_actions_used") is False,
        "sequence_count_matches": len(identities) == int(manifest.get("total_sequences", -1)),
    }
    errors.extend(name for name, passed in contract_checks.items() if not passed)
    return {
        "schema_version": TRAINING_CACHE_SCHEMA,
        "cache_dir": str(root),
        "sequences": len(identities),
        "contract_checks": contract_checks,
        "sequence_hashes_verified": bool(verify_sequence_hashes),
        "errors": errors,
        "passed": not errors,
    }


def iter_training_sequences(cache_dir: str | Path) -> Iterator[dict[str, Any]]:
    root = Path(cache_dir).expanduser().resolve()
    manifest = load_training_manifest(root)
    for entry in manifest["sequences"]:
        yield torch.load(root / entry["file"], map_location="cpu", weights_only=False)
