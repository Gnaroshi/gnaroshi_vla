"""Migrate a legacy real Condition cache after label-only dataset correction.

The frozen VLM condition depends on RGB observations and language, not action
labels.  This migration reuses the large condition array only after proving
that every condition-relevant input and every cache record is unchanged.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .condition_cache import (
    CACHE_SCHEMA,
    CONDITION_DIM,
    CONDITION_TOKENS,
    _AllSplits,
    canonical_condition_token_layout,
    validate_real_condition_cache,
)
from .dataset import DATASET_SCHEMA
from .io_utils import atomic_write_json, sha256_file, sha256_text


LEGACY_DATASET_SCHEMA = "simvla_real_hdf5_v2"
LEGACY_CACHE_SCHEMA = "simvla_real_condition_cache_v1"
MIGRATION_SCHEMA = "simvla_real_condition_cache_label_migration_v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_dataset_integrity(
    manifest: Mapping[str, Any], manifest_path: Path
) -> None:
    episode_ids = [str(row["episode_id"]) for row in manifest["episodes"]]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("dataset contains duplicate episode IDs")
    train = [str(value) for value in manifest["splits"]["train"]]
    validation = [str(value) for value in manifest["splits"]["validation"]]
    if set(train) & set(validation) or set(train) | set(validation) != set(episode_ids):
        raise ValueError("dataset split is not a disjoint cover of its episodes")
    episode_sha256: dict[str, str] = {}
    for row in manifest["episodes"]:
        path = (manifest_path.parent / str(row["path"])).resolve()
        observed = sha256_file(path)
        if observed != row.get("sha256"):
            raise ValueError(f"dataset episode checksum mismatch: {row['episode_id']}")
        episode_sha256[str(row["episode_id"])] = observed
    identity = {
        "schema": manifest.get("schema_version"),
        "episode_sha256": episode_sha256,
        "split_seed": manifest.get("split_seed"),
        "train": train,
        "validation": validation,
        "instruction": manifest.get("instruction"),
    }
    if manifest.get("dataset_identity_sha256") != sha256_text(
        json.dumps(identity, sort_keys=True)
    ):
        raise ValueError("dataset identity is invalid")
    norm_path = manifest_path.parent / str(manifest["norm_stats"]["path"])
    if sha256_file(norm_path) != manifest["norm_stats"].get("sha256"):
        raise ValueError("dataset norm-statistics checksum mismatch")


def _legacy_cache(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != LEGACY_CACHE_SCHEMA:
        raise ValueError("source cache is not the reviewed legacy v1 cache")
    if manifest.get("verdict") != "REAL_CONDITION_CACHE_PASS":
        raise ValueError("source cache is incomplete")
    count = int(manifest.get("count", -1))
    expected = {
        "condition.npy": ((count, CONDITION_TOKENS, CONDITION_DIM), np.dtype("float32")),
        "proprio.npy": ((count, 8), np.dtype("float32")),
        "action.npy": ((count, 10, 7), np.dtype("float32")),
        "complete.npy": ((count,), np.dtype("uint8")),
    }
    for name, (shape, dtype) in expected.items():
        path = root / name
        spec = manifest.get("arrays", {}).get(name, {})
        if not path.is_file() or path.stat().st_size != int(spec.get("size_bytes", -1)):
            raise ValueError(f"legacy cache file size mismatch: {name}")
        array = np.load(path, mmap_mode="r")
        if tuple(array.shape) != shape or array.dtype != dtype:
            raise ValueError(f"legacy cache array contract mismatch: {name}")
        if sha256_file(path) != spec.get("sha256"):
            raise ValueError(f"legacy cache checksum mismatch: {name}")
    records_path = root / "records.json"
    if sha256_file(records_path) != manifest.get("records_sha256"):
        raise ValueError("legacy cache records checksum mismatch")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if len(records) != count:
        raise ValueError("legacy cache record count mismatch")
    dataset_path = Path(manifest["dataset_manifest"]).expanduser().resolve()
    dataset = _load_json(dataset_path)
    if dataset.get("schema_version") != LEGACY_DATASET_SCHEMA:
        raise ValueError("legacy cache does not reference the reviewed v2 dataset")
    if dataset.get("dataset_identity_sha256") != manifest.get("dataset_identity_sha256"):
        raise ValueError("legacy cache and dataset identities differ")
    _validate_dataset_integrity(dataset, dataset_path)
    legacy_identity = {
        "schema": LEGACY_CACHE_SCHEMA,
        "dataset_identity_sha256": manifest.get("dataset_identity_sha256"),
        "official_model_weights_sha256": manifest.get("official_base", {}).get(
            "model_weights_sha256"
        ),
        "records_sha256": manifest.get("records_sha256"),
        "preprocessing": dataset.get("image_contract", {}).get("model_preprocessing"),
    }
    if manifest.get("condition_cache_identity_sha256") != sha256_text(
        json.dumps(legacy_identity, sort_keys=True)
    ):
        raise ValueError("legacy cache identity is invalid")
    return manifest, dataset


def _equal_vlen(left: h5py.Dataset, right: h5py.Dataset) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(np.asarray(left[index]), np.asarray(right[index]))
        for index in range(len(left))
    )


def _episode_map(manifest: Mapping[str, Any], root: Path) -> dict[str, Path]:
    return {
        str(row["episode_id"]): (root / str(row["path"])).resolve()
        for row in manifest["episodes"]
    }


def _validate_dataset_equivalence(
    legacy_manifest: Mapping[str, Any],
    corrected_manifest: Mapping[str, Any],
    *,
    legacy_root: Path,
    corrected_root: Path,
) -> dict[str, Any]:
    if corrected_manifest.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("corrected dataset does not use the current schema")
    if corrected_manifest.get("verdict") != "REAL_DATASET_CONTRACT_PASS":
        raise ValueError("corrected dataset conversion contract did not pass")
    for key in (
        "dataset_id",
        "instruction",
        "target_hz",
        "sampling_mode",
        "action_horizon",
        "execution_horizon",
        "split_seed",
        "splits",
        "image_contract",
    ):
        if legacy_manifest.get(key) != corrected_manifest.get(key):
            raise ValueError(f"label-only migration changed dataset field: {key}")
    for key in ("representation", "gripper_max_opening_m"):
        if legacy_manifest.get("state_contract", {}).get(key) != corrected_manifest.get(
            "state_contract", {}
        ).get(key):
            raise ValueError(f"label-only migration changed state contract field: {key}")
    if corrected_manifest.get("state_contract", {}).get(
        "condition_updater_rotation_delta"
    ) != "current rotvec mapped to equivalent 2pi branch nearest previous rotvec":
        raise ValueError("corrected dataset lacks the real rotvec branch-alignment contract")
    legacy_paths = _episode_map(legacy_manifest, legacy_root)
    corrected_paths = _episode_map(corrected_manifest, corrected_root)
    if set(legacy_paths) != set(corrected_paths):
        raise ValueError("label-only migration changed episode IDs")

    changed_gripper_transitions = 0
    transition_count = 0
    for episode_id in sorted(legacy_paths):
        with h5py.File(legacy_paths[episode_id], "r") as old, h5py.File(
            corrected_paths[episode_id], "r"
        ) as new:
            for key in ("episode_id", "instruction", "source_frame_count", "target_hz", "sampling_mode"):
                if old.attrs.get(key) != new.attrs.get(key):
                    raise ValueError(f"episode {episode_id} changed attribute {key}")
            for key in ("timestamp_s", "source_index", "valid_transition", "state"):
                if not np.array_equal(np.asarray(old[key]), np.asarray(new[key])):
                    raise ValueError(f"episode {episode_id} changed condition/proprio input {key}")
            for key in ("base_rgb_jpeg", "wrist_rgb_jpeg"):
                if not _equal_vlen(old[key], new[key]):
                    raise ValueError(f"episode {episode_id} changed condition input {key}")
            old_action = np.asarray(old["step_action"])
            new_action = np.asarray(new["step_action"])
            old_raw = np.asarray(old["raw_normalized_step_action"])
            new_raw = np.asarray(new["raw_normalized_step_action"])
            if not np.array_equal(old_action[:, :6], new_action[:, :6]):
                raise ValueError(f"episode {episode_id} changed Cartesian action labels")
            if not np.array_equal(old_raw[:, :6], new_raw[:, :6]):
                raise ValueError(f"episode {episode_id} changed raw Cartesian action labels")
            command = np.asarray(new["gripper_command"], dtype=np.float32)
            expected = (1.0 - 2.0 * command[:-1]).astype(np.float32)
            if not np.allclose(new_action[:, 6], expected, rtol=0.0, atol=1e-7):
                raise ValueError(f"episode {episode_id} corrected gripper label is not command_t")
            if not np.allclose(new_raw[:, 6], expected, rtol=0.0, atol=1e-7):
                raise ValueError(f"episode {episode_id} corrected raw gripper label is not command_t")
            changed_gripper_transitions += int(
                np.count_nonzero(old_action[:, 6] != new_action[:, 6])
            )
            transition_count += int(new_action.shape[0])
    return {
        "episode_count": len(corrected_paths),
        "transition_count": transition_count,
        "changed_gripper_transition_count": changed_gripper_transitions,
        "condition_inputs_bitwise_equal": True,
        "proprio_inputs_bitwise_equal": True,
        "cartesian_action_labels_bitwise_equal": True,
        "corrected_gripper_labels_equal_command_t": True,
    }


def _write_corrected_arrays(
    building: Path,
    dataset: _AllSplits,
    records: list[dict[str, Any]],
) -> None:
    count = len(records)
    proprio = np.lib.format.open_memmap(
        building / "proprio.npy", mode="w+", dtype=np.float32, shape=(count, 8)
    )
    action = np.lib.format.open_memmap(
        building / "action.npy", mode="w+", dtype=np.float32, shape=(count, 10, 7)
    )
    complete = np.lib.format.open_memmap(
        building / "complete.npy", mode="w+", dtype=np.uint8, shape=(count,)
    )
    for row in records:
        index = int(row["cache_index"])
        split, local_index = dataset.lookup[index]
        episode_id, frame_index = dataset.datasets[split].samples[local_index]
        if (str(split), str(episode_id), int(frame_index)) != (
            str(row["split"]),
            str(row["episode_id"]),
            int(row["frame_index"]),
        ):
            raise ValueError(f"corrected cache record mapping changed at index {index}")
        handle = dataset.datasets[split].store.handle(str(episode_id))
        proprio[index] = np.asarray(handle["state"][frame_index], dtype=np.float32)
        action[index] = np.asarray(
            handle["step_action"][frame_index : frame_index + 10], dtype=np.float32
        )
        complete[index] = 1
    proprio.flush()
    action.flush()
    complete.flush()


def migrate(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.legacy_condition_cache).expanduser().resolve()
    corrected_manifest_path = Path(args.corrected_dataset_manifest).expanduser().resolve()
    corrected_root = corrected_manifest_path.parent
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"migration output already exists: {output}")
    legacy_cache, legacy_dataset = _legacy_cache(source_root)
    legacy_dataset_path = Path(legacy_cache["dataset_manifest"]).expanduser().resolve()
    corrected_dataset = _load_json(corrected_manifest_path)
    _validate_dataset_integrity(corrected_dataset, corrected_manifest_path)
    equivalence = _validate_dataset_equivalence(
        legacy_dataset,
        corrected_dataset,
        legacy_root=legacy_dataset_path.parent,
        corrected_root=corrected_root,
    )
    dataset = _AllSplits(corrected_manifest_path)
    records = dataset.records()
    legacy_records = json.loads((source_root / "records.json").read_text(encoding="utf-8"))
    if records != legacy_records:
        raise ValueError("label-only migration changed cache record order or membership")

    building = output.with_name(f".{output.name}.building-{os.getpid()}")
    building.mkdir(parents=True)
    try:
        condition_source = source_root / "condition.npy"
        condition_destination = building / "condition.npy"
        try:
            os.link(condition_source, condition_destination)
            storage_mode = "hardlink"
        except OSError as error:
            if error.errno != errno.EXDEV or not args.allow_condition_copy:
                raise RuntimeError(
                    "condition.npy must be migrated on the same filesystem for zero-copy reuse; "
                    "pass --allow-condition-copy only after approving an additional 9.1 GB copy"
                ) from error
            shutil.copy2(condition_source, condition_destination)
            storage_mode = "copy"
        atomic_write_json(building / "records.json", records)
        _write_corrected_arrays(building, dataset, records)
        array_specs = {
            name: {
                "sha256": (
                    legacy_cache["arrays"][name]["sha256"]
                    if name == "condition.npy"
                    else sha256_file(building / name)
                ),
                "size_bytes": (building / name).stat().st_size,
            }
            for name in ("condition.npy", "proprio.npy", "action.npy", "complete.npy")
        }
        if storage_mode == "copy" and sha256_file(condition_destination) != array_specs[
            "condition.npy"
        ]["sha256"]:
            raise ValueError("copied condition array differs from the verified legacy source")
        records_sha = sha256_file(building / "records.json")
        identity = {
            "schema": CACHE_SCHEMA,
            "dataset_identity_sha256": corrected_dataset["dataset_identity_sha256"],
            "official_model_weights_sha256": legacy_cache["official_base"][
                "model_weights_sha256"
            ],
            "records_sha256": records_sha,
            "array_sha256": {
                name: specification["sha256"]
                for name, specification in array_specs.items()
            },
            "preprocessing": corrected_dataset["image_contract"]["model_preprocessing"],
        }
        token_layout = canonical_condition_token_layout(legacy_cache["token_layout"])
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "verdict": "REAL_CONDITION_CACHE_PASS",
            "count": len(records),
            "shape": [len(records), CONDITION_TOKENS, CONDITION_DIM],
            "dtype": "float32",
            "dataset_manifest": str(corrected_manifest_path),
            "dataset_manifest_sha256": sha256_file(corrected_manifest_path),
            "dataset_identity_sha256": corrected_dataset["dataset_identity_sha256"],
            "official_base": legacy_cache["official_base"],
            "exact_loading": legacy_cache["exact_loading"],
            "records_sha256": records_sha,
            "condition_cache_identity_sha256": sha256_text(
                json.dumps(identity, sort_keys=True)
            ),
            "arrays": array_specs,
            "token_layout": token_layout,
            "migration": {
                "schema_version": MIGRATION_SCHEMA,
                "legacy_cache_manifest_sha256": sha256_file(source_root / "manifest.json"),
                "legacy_dataset_manifest_sha256": sha256_file(legacy_dataset_path),
                "legacy_condition_cache_identity_sha256": legacy_cache[
                    "condition_cache_identity_sha256"
                ],
                "condition_storage_mode": storage_mode,
                "condition_bytes_recomputed": 0,
                **equivalence,
            },
        }
        atomic_write_json(building / "manifest.json", manifest)
        os.replace(building, output)
        validate_real_condition_cache(output, verify_array_checksums=False)
        return manifest
    finally:
        if building.exists():
            shutil.rmtree(building)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-condition-cache", required=True)
    parser.add_argument("--corrected-dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-condition-copy", action="store_true")
    return parser


def main() -> int:
    result = migrate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "condition_cache_identity_sha256": result[
                    "condition_cache_identity_sha256"
                ],
                "migration": result["migration"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
