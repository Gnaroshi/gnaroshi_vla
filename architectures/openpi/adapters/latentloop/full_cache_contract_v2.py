"""Frozen full-cache episode/query inventory for pi0.5 LatentLoop.

The inventory is intentionally compact: every query is fixed by an episode
identity, a query count, native R-step progression, and the deterministic
query-noise rule.  Consumers reconstruct the complete query set rather than
trusting a generator-produced list.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .cache_contract_v2 import canonical_payload_hash
from .policy_io import policy_noise_seed


FULL_CACHE_INVENTORY_SCHEMA_VERSION = 2
FULL_CACHE_SCOPE = "full_teacher_cache_v2"
DEFAULT_FULL_CACHE_SHARDS = 4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_cache_episode_identity(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["suite"]),
        int(row["benchmark_task_index"]),
        str(row["episode_id"]),
    )


def expected_query_count(dataset_frame_start: int, dataset_frame_stop: int, execution_horizon: int) -> int:
    length = int(dataset_frame_stop) - int(dataset_frame_start)
    if length <= execution_horizon:
        return 0
    # Mirrors range(start, stop - R, R): every stored query has a complete
    # native R-action prefix, including the transition to its next observation.
    return (length - 1) // execution_horizon


def _shard_index(identity: tuple[str, int, str], num_shards: int) -> int:
    encoded = f"{identity[0]}:{identity[1]}:{identity[2]}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % num_shards


def expected_query_spec(
    episode: dict[str, Any],
    query_index: int,
    *,
    execution_horizon: int,
    noise_seed_base: int,
) -> dict[str, int]:
    query_index = int(query_index)
    count = int(episode["query_count"])
    if not 0 <= query_index < count:
        raise IndexError(f"query {query_index} is outside frozen count {count}")
    step = query_index * execution_horizon
    return {
        "policy_query_index": query_index,
        "absolute_environment_step": step,
        "dataset_frame_index": int(episode["dataset_frame_start"]) + step,
        "next_dataset_frame_index": int(episode["dataset_frame_start"]) + step + execution_horizon,
        "action_noise_seed": policy_noise_seed(
            noise_seed_base,
            str(episode["suite"]),
            int(episode["benchmark_task_index"]),
            int(episode["episode_id"]),
            query_index,
        ),
    }


def build_full_cache_inventory(
    *,
    source_lock_id: str,
    split_contract: dict[str, Any],
    final_manifest: dict[str, Any],
    split_contract_sha256: str,
    final_manifest_sha256: str,
    action_horizon: int = 10,
    execution_horizon: int = 5,
    flow_steps: int = 10,
    noise_seed_base: int = 20260820,
    num_shards: int = DEFAULT_FULL_CACHE_SHARDS,
) -> dict[str, Any]:
    if (action_horizon, execution_horizon, flow_steps) != (10, 5, 10):
        raise ValueError("full-cache protocol is pinned to H=10, R=5, and ten flow steps")
    if num_shards < 1:
        raise ValueError("full-cache shard count must be positive")
    if split_contract.get("source_lock_id") != source_lock_id:
        raise ValueError("split contract was frozen under another source lock")
    if final_manifest.get("source_lock_id") != source_lock_id:
        raise ValueError("final manifest was frozen under another source lock")
    if split_contract.get("final_manifest_id") != final_manifest.get("manifest_id"):
        raise ValueError("split contract does not name the supplied final manifest")

    episodes: list[dict[str, Any]] = []
    for assignment in split_contract.get("assignments", []):
        for key in ("dataset_frame_start", "dataset_frame_stop"):
            if assignment.get(key) is None:
                raise ValueError(f"split assignment lacks {key}")
        identity = full_cache_episode_identity(assignment)
        count = expected_query_count(
            int(assignment["dataset_frame_start"]),
            int(assignment["dataset_frame_stop"]),
            execution_horizon,
        )
        if count < 4:
            raise ValueError(f"episode {identity} cannot provide an anchor plus ages 1,2,3")
        episodes.append(
            {
                "suite": identity[0],
                "benchmark_task_index": identity[1],
                "episode_namespace": "teacher_demonstration",
                "episode_id": identity[2],
                "role": str(assignment["role"]),
                "environment_seed": assignment["environment_seed"],
                "initial_state_identifier": str(assignment["initial_state_identifier"]),
                "dataset_frame_start": int(assignment["dataset_frame_start"]),
                "dataset_frame_stop": int(assignment["dataset_frame_stop"]),
                "query_count": count,
                "shard_index": _shard_index(identity, num_shards),
            }
        )
    episodes.sort(key=lambda row: full_cache_episode_identity(row))
    identities = [full_cache_episode_identity(row) for row in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("full-cache inventory contains duplicate episode identities")

    role_counts = Counter(str(row["role"]) for row in episodes)
    suite_counts = Counter(str(row["suite"]) for row in episodes)
    shard_episode_counts = Counter(int(row["shard_index"]) for row in episodes)
    shard_query_counts = Counter()
    for row in episodes:
        shard_query_counts[int(row["shard_index"])] += int(row["query_count"])
    payload: dict[str, Any] = {
        "schema_version": FULL_CACHE_INVENTORY_SCHEMA_VERSION,
        "frozen": True,
        "FULL_CACHE_INVENTORY_V2_PASS": True,
        "markers": ["FULL_CACHE_INVENTORY_V2_PASS"],
        "cache_scope": FULL_CACHE_SCOPE,
        "source_lock_id": source_lock_id,
        "split_contract_id": split_contract["split_contract_id"],
        "split_contract_sha256": split_contract_sha256,
        "final_manifest_id": final_manifest["manifest_id"],
        "final_evaluation_manifest_sha256": final_manifest_sha256,
        "protocol": {
            "action_horizon_h": action_horizon,
            "execution_horizon_r": execution_horizon,
            "flow_steps": flow_steps,
            "noise_seed_base": noise_seed_base,
            "query_index_origin": 0,
            "environment_step_rule": "policy_query_index * execution_horizon_r",
            "dataset_frame_rule": "dataset_frame_start + absolute_environment_step",
            "noise_rule": "sha256(seed_base:suite:benchmark_task_index:episode_id:query_index)",
        },
        "num_shards": num_shards,
        "episodes": episodes,
        "statistics": {
            "episodes": len(episodes),
            "queries": sum(int(row["query_count"]) for row in episodes),
            "episodes_by_role": dict(sorted(role_counts.items())),
            "episodes_by_suite": dict(sorted(suite_counts.items())),
            "episodes_by_shard": {
                str(index): shard_episode_counts[index] for index in range(num_shards)
            },
            "queries_by_shard": {
                str(index): shard_query_counts[index] for index in range(num_shards)
            },
        },
    }
    payload["inventory_id"] = canonical_payload_hash(payload, "inventory_id")
    return payload


def load_full_cache_inventory(
    path: str | Path,
    *,
    source_lock_id: str,
    split_contract: dict[str, Any],
    final_manifest: dict[str, Any],
    split_contract_path: str | Path,
    final_manifest_path: str | Path,
) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FULL_CACHE_INVENTORY_SCHEMA_VERSION:
        raise ValueError("full-cache inventory has an unsupported schema")
    if payload.get("frozen") is not True or payload.get("FULL_CACHE_INVENTORY_V2_PASS") is not True:
        raise ValueError("full-cache inventory is not frozen and accepted")
    if payload.get("cache_scope") != FULL_CACHE_SCOPE:
        raise ValueError("full-cache inventory scope is invalid")
    if payload.get("inventory_id") != canonical_payload_hash(payload, "inventory_id"):
        raise ValueError("full-cache inventory self-hash is absent or invalid")
    protocol = payload.get("protocol", {})
    rebuilt = build_full_cache_inventory(
        source_lock_id=source_lock_id,
        split_contract=split_contract,
        final_manifest=final_manifest,
        split_contract_sha256=sha256_file(split_contract_path),
        final_manifest_sha256=sha256_file(final_manifest_path),
        action_horizon=int(protocol.get("action_horizon_h", -1)),
        execution_horizon=int(protocol.get("execution_horizon_r", -1)),
        flow_steps=int(protocol.get("flow_steps", -1)),
        noise_seed_base=int(protocol.get("noise_seed_base", -1)),
        num_shards=int(payload.get("num_shards", 0)),
    )
    if payload != rebuilt:
        raise ValueError("full-cache inventory differs from the deterministic split-derived inventory")
    return payload
