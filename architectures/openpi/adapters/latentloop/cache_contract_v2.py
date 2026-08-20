"""Fail-closed cache identity, tensor, and episode-split contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SPLIT_ROLES = (
    "train",
    "checkpoint_validation",
    "defect_fit",
    "defect_validity",
    "scheduler_calibration",
)
REQUIRED_RECORD_KEYS_V2 = frozenset(
    {
        "suite",
        "benchmark_task_index",
        "dataset_task_index",
        "canonical_task_name",
        "canonical_instruction",
        "episode_namespace",
        "episode_id",
        "environment_seed",
        "initial_state_identifier",
        "policy_query_index",
        "absolute_environment_step",
        "raw_images",
        "raw_image_identity",
        "preprocessed_images",
        "preprocessed_image_hash",
        "robot_state_raw",
        "robot_state_normalized",
        "prefix_embeddings",
        "pre_rope_keys",
        "values",
        "prefix_pad_mask",
        "prefix_attention_pattern",
        "prefix_position_ids",
        "action_noise",
        "action_noise_seed",
        "action_noise_hash",
        "teacher_action_chunk_normalized",
        "teacher_action_chunk_postprocessed",
        "executed_actions_postprocessed",
        "executed_action_length",
        "gripper_conversion",
        "next_query_observation",
        "source_lock_id",
        "source_hashes",
        "timing_ms",
    }
)


def canonical_instruction(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def canonical_payload_hash(payload: dict[str, Any], identity_key: str) -> str:
    body = {key: value for key, value in payload.items() if key != identity_key}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(torch.as_tensor(value).detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def array_hash(value: Any) -> str:
    return _array_hash(value)


def tree_hash(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_array_hash(values[key]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def tensor_contract_from_record(record: dict[str, Any]) -> dict[str, Any]:
    embeddings = torch.as_tensor(record["prefix_embeddings"])
    keys = torch.as_tensor(record["pre_rope_keys"])
    values = torch.as_tensor(record["values"])
    mask = torch.as_tensor(record["prefix_pad_mask"])
    attention = torch.as_tensor(record["prefix_attention_pattern"])
    positions = torch.as_tensor(record["prefix_position_ids"])
    noise = torch.as_tensor(record["action_noise"])
    teacher = torch.as_tensor(record["teacher_action_chunk_normalized"])
    postprocessed = torch.as_tensor(record["teacher_action_chunk_postprocessed"])
    executed = torch.as_tensor(record["executed_actions_postprocessed"])
    if embeddings.ndim != 2 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("cache KV must be dynamic [S,E] embeddings and aligned [L,Kh,S,Dh] K/V")
    layers, kv_heads, tokens, head_dim = keys.shape
    if embeddings.shape[0] != tokens:
        raise ValueError("prefix embeddings and KV token counts differ")
    for name, value in (("prefix mask", mask), ("attention pattern", attention), ("position IDs", positions)):
        if value.shape != (tokens,):
            raise ValueError(f"{name} must have dynamic shape [S]")
    if noise.ndim != 2 or teacher.shape != noise.shape:
        raise ValueError("flow noise and model-space action chunk must be aligned [H,D]")
    if postprocessed.ndim != 2 or postprocessed.shape != (noise.shape[0], 7):
        raise ValueError("postprocessed teacher chunk must be [H,7]")
    if executed.ndim != 2 or executed.shape[1] != 7:
        raise ValueError("actually executed actions must be [R_actual,7]")
    return {
        "layer_count": int(layers),
        "prefix_sequence_length": int(tokens),
        "prefix_embedding_dim": int(embeddings.shape[1]),
        "kv_head_count": int(kv_heads),
        "head_dim": int(head_dim),
        "kv_dtype": str(keys.dtype),
        "embedding_dtype": str(embeddings.dtype),
        "prefix_mask_dtype": str(mask.dtype),
        "attention_pattern_dtype": str(attention.dtype),
        "position_id_dtype": str(positions.dtype),
        "prefix_mask_shape": list(mask.shape),
        "attention_pattern_shape": list(attention.shape),
        "position_id_shape": list(positions.shape),
        "action_horizon_h": int(noise.shape[0]),
        "model_action_dim": int(noise.shape[1]),
        "environment_action_dim": 7,
    }


def _all_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, np.ndarray):
        yield torch.as_tensor(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _all_tensors(item)


def assert_finite(value: Any, label: str) -> None:
    for tensor in _all_tensors(value):
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{label} contains NaN or Inf")


def load_final_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("frozen") is not True:
        raise ValueError("final evaluation manifest must be frozen schema v2")
    if manifest.get("manifest_id") != canonical_payload_hash(manifest, "manifest_id"):
        raise ValueError("final evaluation manifest self-hash is absent or invalid")
    tasks = manifest.get("tasks", [])
    episodes = manifest.get("episodes", [])
    if len(tasks) != 40 or len(episodes) != 2000:
        raise ValueError("final evaluation manifest must contain 40 tasks and 2,000 episodes")
    by_suite = {suite: [row for row in tasks if row.get("suite") == suite] for suite in SUITES}
    if any(len(rows) != 10 for rows in by_suite.values()):
        raise ValueError("final evaluation task catalog must contain ten tasks per suite")
    task_keys = {(row["suite"], int(row["benchmark_task_index"])) for row in tasks}
    if len(task_keys) != 40:
        raise ValueError("final evaluation task identities are not unique")
    for suite in SUITES:
        indices = sorted(int(row["benchmark_task_index"]) for row in by_suite[suite])
        if indices != list(range(10)):
            raise ValueError(f"{suite} benchmark task indices must be exactly 0..9")
    for row in tasks:
        for field in ("dataset_task_index", "canonical_task_name", "canonical_instruction"):
            if row.get(field) in (None, ""):
                raise ValueError(f"final task catalog is missing {field}")
    episode_keys = {
        (row["suite"], int(row["benchmark_task_index"]), int(row["trial"])) for row in episodes
    }
    if len(episode_keys) != 2000:
        raise ValueError("final evaluation episode identities are not unique")
    if any(sum(1 for row in episodes if row["suite"] == suite) != 500 for suite in SUITES):
        raise ValueError("final evaluation manifest must contain 500 episodes per suite")
    for task_key in task_keys:
        trials = sorted(
            int(row["trial"])
            for row in episodes
            if (row["suite"], int(row["benchmark_task_index"])) == task_key
        )
        if trials != list(range(50)):
            raise ValueError(f"final task {task_key} must contain trials 0..49")
    return manifest


def resolve_task_identity(
    dataset_task_index: int,
    instruction: str,
    final_manifest: dict[str, Any],
) -> dict[str, Any]:
    normalized = canonical_instruction(instruction)
    matches = [
        row
        for row in final_manifest["tasks"]
        if int(row["dataset_task_index"]) == int(dataset_task_index)
        and canonical_instruction(row["canonical_instruction"]) == normalized
    ]
    if len(matches) != 1:
        raise ValueError(
            f"dataset task {dataset_task_index} / {instruction!r} does not map uniquely to the four-suite catalog"
        )
    return matches[0]


def _identity(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["suite"]),
        int(row["benchmark_task_index"]),
        str(row["episode_namespace"]),
        str(row["episode_id"]),
    )


def _physical_episode_identity(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["suite"]),
        int(row["benchmark_task_index"]),
        str(row["initial_state_identifier"]),
        str(row["environment_seed"]),
    )


def load_split_contract(
    path: str | Path,
    final_manifest: dict[str, Any],
) -> tuple[dict[tuple[str, int, str, str], dict[str, Any]], dict[str, Any]]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != 2 or contract.get("frozen") is not True:
        raise ValueError("episode split contract must be frozen schema v2")
    if contract.get("split_contract_id") != canonical_payload_hash(contract, "split_contract_id"):
        raise ValueError("episode split contract self-hash is absent or invalid")
    if contract.get("source_lock_id") != final_manifest.get("source_lock_id"):
        raise ValueError("split contract and final manifest use different source locks")
    if contract.get("final_manifest_id") != final_manifest.get("manifest_id"):
        raise ValueError("split contract does not name the supplied final evaluation manifest")
    assignments = contract.get("assignments", [])
    mapping: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    physical_cache_episodes: set[tuple[str, int, str, str]] = set()
    for row in assignments:
        if row.get("role") not in SPLIT_ROLES:
            raise ValueError(f"unknown cache split role: {row.get('role')}")
        if row.get("episode_namespace") != "teacher_demonstration":
            raise ValueError("cache split assignments must use teacher_demonstration namespace")
        identity = _identity(row)
        if identity in mapping:
            raise ValueError(f"episode appears in multiple cache roles: {identity}")
        mapping[identity] = row
        physical = _physical_episode_identity(row)
        if physical in physical_cache_episodes:
            raise ValueError(f"physical episode appears more than once in cache roles: {physical}")
        physical_cache_episodes.add(physical)
    final_physical = {_physical_episode_identity(row) for row in final_manifest["episodes"]}
    overlap = physical_cache_episodes & final_physical
    if overlap:
        raise ValueError(f"cache split physically overlaps final evaluation episodes: {sorted(overlap)[:3]}")
    required_roles = set(contract.get("required_cache_roles", ()))
    if required_roles != set(SPLIT_ROLES):
        raise ValueError("split contract must explicitly require all five non-final roles")
    observed_roles = {str(row["role"]) for row in assignments}
    if observed_roles != set(SPLIT_ROLES):
        raise ValueError("split contract must contain at least one episode in each required role")
    return mapping, contract


def validate_record_v2(
    record: dict[str, Any],
    *,
    expected_contract: dict[str, Any],
    expected_source_lock_id: str,
    expected_source_hashes: dict[str, str],
    task_catalog: dict[tuple[str, int], dict[str, Any]],
    execution_horizon: int,
) -> None:
    missing = REQUIRED_RECORD_KEYS_V2 - record.keys()
    if missing:
        raise ValueError(f"cache record is missing v2 keys: {sorted(missing)}")
    assert_finite(record, "cache record")
    observed_contract = tensor_contract_from_record(record)
    if observed_contract != expected_contract:
        raise ValueError("cache record dynamic tensor contract differs from manifest")
    task_key = (str(record["suite"]), int(record["benchmark_task_index"]))
    if task_key not in task_catalog:
        raise ValueError(f"cache record has unknown benchmark task identity: {task_key}")
    catalog = task_catalog[task_key]
    if canonical_instruction(record["canonical_instruction"]) != canonical_instruction(
        catalog["canonical_instruction"]
    ):
        raise ValueError("cache record instruction differs from frozen task catalog")
    if str(record["canonical_task_name"]) != str(catalog["canonical_task_name"]):
        raise ValueError("cache record task name differs from frozen task catalog")
    if int(record["dataset_task_index"]) != int(catalog["dataset_task_index"]):
        raise ValueError("cache record dataset task index differs from frozen task catalog")
    if record["episode_namespace"] != "teacher_demonstration":
        raise ValueError("teacher cache record has a non-training episode namespace")
    if record["environment_seed"] is None or not str(record["initial_state_identifier"]):
        raise ValueError("cache record lacks environment seed or initial-state identity")
    if int(record["policy_query_index"]) < 0 or int(record["absolute_environment_step"]) < 0:
        raise ValueError("cache query and environment-step indices must be nonnegative")
    if record["source_lock_id"] != expected_source_lock_id:
        raise ValueError("cache record source lock differs from manifest")
    required_hashes = {"source", "checkpoint", "norm_stats", "config", "preprocessing", "postprocessing"}
    if set(expected_source_hashes) != required_hashes:
        raise ValueError("expected source hash inventory is incomplete or ambiguous")
    if record["source_hashes"] != expected_source_hashes:
        raise ValueError("cache record source hashes differ from the frozen cache manifest")
    if int(record["executed_action_length"]) != len(record["executed_actions_postprocessed"]):
        raise ValueError("executed action length metadata is inconsistent")
    if not 1 <= int(record["executed_action_length"]) <= execution_horizon:
        raise ValueError("executed action length is outside the native R boundary")
    if record["action_noise_hash"] != _array_hash(record["action_noise"]):
        raise ValueError("query-keyed action noise hash mismatch")
    expected_positions = torch.cumsum(
        torch.as_tensor(record["prefix_pad_mask"], dtype=torch.long), dim=0
    ) - 1
    observed_positions = torch.as_tensor(record["prefix_position_ids"], dtype=torch.long)
    if not torch.equal(observed_positions, expected_positions):
        raise ValueError("prefix position IDs must equal cumsum(prefix_pad_mask)-1")
    if record["raw_image_identity"] != tree_hash(record["raw_images"]):
        raise ValueError("raw image identity hash mismatch")
    if record["preprocessed_image_hash"] != tree_hash(record["preprocessed_images"]):
        raise ValueError("preprocessed image hash mismatch")
    if record["gripper_conversion"] != "LiberoOutputs continuous source-correct 7D action; no binary target rewrite":
        raise ValueError("unknown gripper conversion contract")
