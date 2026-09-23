"""Versioned compressed plan-trace shards for Seer/LatentLoop rollouts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PLAN_TRACE_SCHEMA_VERSION = 1

REQUIRED_SCALAR_KEYS: tuple[str, ...] = (
    "row_id",
    "paired_group",
    "task_id",
    "episode_id",
    "timestep",
    "mode",
    "cache_age",
    "feature_source_step",
    "full_refresh_flag",
    "primary_raw_change_l1",
    "wrist_raw_change_l1",
    "proprio_delta_l2",
    "u_delta_norm",
)

REQUIRED_TENSOR_KEYS: tuple[str, ...] = (
    "raw_action_arm",
    "raw_gripper_logit",
    "raw_gripper_probability",
    "raw_gripper_thresholded",
    "post_ensemble_probability",
    "executed_action",
    "proprio_delta",
)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def validate_plan_trace_rows(
    scalar_rows: Sequence[Mapping[str, Any]], tensor_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Validate row count and required fields before writing a shard."""

    if not scalar_rows or len(scalar_rows) != len(tensor_rows):
        raise ValueError(
            "Plan trace requires non-empty scalar/tensor rows with equal length; "
            f"got {len(scalar_rows)} and {len(tensor_rows)}"
        )
    for index, row in enumerate(scalar_rows):
        missing = [key for key in REQUIRED_SCALAR_KEYS if key not in row]
        if missing:
            raise ValueError(f"Scalar trace row {index} is missing {missing}")
    for index, row in enumerate(tensor_rows):
        missing = [key for key in REQUIRED_TENSOR_KEYS if row.get(key) is None]
        if missing:
            raise ValueError(f"Tensor trace row {index} is missing {missing}")


def save_plan_trace_episode(
    output_dir: Path,
    episode_key: str,
    scalar_rows: Sequence[Mapping[str, Any]],
    tensor_rows: Sequence[Mapping[str, Any]],
    episode_metadata: Mapping[str, Any],
) -> dict[str, str]:
    """Write one episode as CSV + compressed NPZ + versioned JSON metadata."""

    validate_plan_trace_rows(scalar_rows, tensor_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / episode_key

    scalar_keys = sorted(set().union(*(row.keys() for row in scalar_rows)))
    csv_path = stem.with_suffix(".plan_trace.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in scalar_rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in scalar_keys})

    arrays: dict[str, np.ndarray] = {}
    tensor_keys = sorted(set().union(*(row.keys() for row in tensor_rows)))
    for key in tensor_keys:
        converted = [None if row.get(key) is None else _to_numpy(row[key]) for row in tensor_rows]
        template = next((value for value in converted if value is not None), None)
        if template is None:
            continue
        if any(value is not None and value.shape != template.shape for value in converted):
            shapes = [None if value is None else value.shape for value in converted]
            raise ValueError(f"Trace tensor {key} has inconsistent shapes: {shapes}")
        arrays[key] = np.stack(
            [
                np.full(template.shape, np.nan, dtype=np.float32)
                if value is None
                else value
                for value in converted
            ]
        )
        arrays[f"{key}__present"] = np.asarray(
            [value is not None for value in converted], dtype=np.uint8
        )
    npz_path = stem.with_suffix(".plan_trace.npz")
    np.savez_compressed(npz_path, **arrays)

    metadata_path = stem.with_suffix(".plan_trace.json")
    metadata = {
        "schema_name": "latentloop_plan_trace",
        "schema_version": PLAN_TRACE_SCHEMA_VERSION,
        "episode_key": episode_key,
        "num_steps": len(scalar_rows),
        "scalar_csv": csv_path.name,
        "tensor_npz": npz_path.name,
        "scalar_keys": scalar_keys,
        "tensor_keys": sorted(arrays),
        "episode": _json_safe(dict(episode_metadata)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "npz": str(npz_path), "json": str(metadata_path)}


def load_plan_trace_shard(
    metadata_path: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, np.ndarray]]:
    """Load and validate one plan-trace shard."""

    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_name") != "latentloop_plan_trace":
        raise ValueError(f"Not a LatentLoop plan trace: {metadata_path}")
    if int(metadata.get("schema_version", -1)) != PLAN_TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported plan trace schema {metadata.get('schema_version')} in {metadata_path}"
        )
    with (metadata_path.parent / metadata["scalar_csv"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    with np.load(metadata_path.parent / metadata["tensor_npz"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if len(rows) != int(metadata["num_steps"]):
        raise ValueError(f"Trace row count mismatch in {metadata_path}")
    return metadata, rows, arrays
