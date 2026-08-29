#!/usr/bin/env python3
"""Fail-closed validation for bounded and full schema-v2 teacher caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    array_hash,
    load_final_evaluation_manifest,
    load_split_contract,
    validate_record_v2,
)
from architectures.openpi.adapters.latentloop.full_cache_contract_v2 import (
    expected_query_spec,
    full_cache_episode_identity,
    load_full_cache_inventory,
    sha256_file,
)
from architectures.openpi.adapters.latentloop.policy_io import explicit_policy_noise
from source_lock_v2 import verify_lock


FULL_SHARD_SCOPE = "full_shard_v2"
FULL_MERGED_SCOPE = "full_merged_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_identity(entry: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(entry["suite"]),
        int(entry["benchmark_task_index"]),
        str(entry["episode_id"]),
    )


def _record_identity(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record["suite"]),
        int(record["benchmark_task_index"]),
        str(record["episode_id"]),
    )


def _same_array(left: Any, right: Any) -> bool:
    return array_hash(left) == array_hash(right)


def _validate_next_observation_link(current: dict[str, Any], following: dict[str, Any]) -> None:
    next_observation = current["next_query_observation"]
    expected_images = {
        "image": following["raw_images"]["base_0_rgb"],
        "wrist_image": following["raw_images"]["left_wrist_0_rgb"],
    }
    for key, expected in expected_images.items():
        if key not in next_observation or not _same_array(next_observation[key], expected):
            raise ValueError(f"next-query {key} does not match the following record")
    if not _same_array(next_observation.get("state"), following["robot_state_raw"]):
        raise ValueError("next-query state does not match the following record")
    if int(next_observation.get("frame_index", -1)) != int(
        following["absolute_environment_step"]
    ):
        raise ValueError("next-query frame index does not match the following record")


def _expected_source_hashes(lock_payload: dict[str, Any]) -> dict[str, str]:
    return {
        "source": lock_payload["ours_and_upstream_source"]["combined_sha256"],
        "checkpoint": lock_payload["checkpoint"]["model_sha256"],
        "norm_stats": lock_payload["normalization"]["sha256"],
        "config": lock_payload["checkpoint"]["config_sha256"],
        "preprocessing": lock_payload["preprocessing"]["combined_sha256"],
        "postprocessing": lock_payload["postprocessing"]["combined_sha256"],
    }


def validate_cache_v2(
    cache_root: str | Path,
    *,
    source_lock_path: str | Path,
    final_manifest_path: str | Path,
    split_contract_path: str | Path,
    full_cache_inventory_path: str | Path | None = None,
    verify_hashes: bool = True,
    require_full: bool = False,
) -> dict[str, object]:
    root = Path(cache_root).resolve()
    manifest_path = root / "pi05_latentloop_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = verify_lock(source_lock_path)
    final_manifest = load_final_evaluation_manifest(final_manifest_path)
    split_mapping, split_contract = load_split_contract(split_contract_path, final_manifest)
    lock_payload = json.loads(Path(source_lock_path).read_text(encoding="utf-8"))
    expected_source_hashes = _expected_source_hashes(lock_payload)
    errors: list[str] = []

    if manifest.get("schema_version") != 2 or manifest.get("complete") is not True:
        errors.append("cache manifest must be complete schema v2")
    metadata = manifest.get("metadata", {})
    scope = str(metadata.get("cache_scope", "bounded_smoke_v2"))
    if metadata.get("source_lock_id") != lock["source_lock_id"]:
        errors.append("cache manifest source lock is stale")
    if metadata.get("final_evaluation_manifest_sha256") != _sha256(Path(final_manifest_path)):
        errors.append("cache final-evaluation manifest hash mismatch")
    if metadata.get("split_contract_sha256") != _sha256(Path(split_contract_path)):
        errors.append("cache split-contract hash mismatch")
    manifest_body = {key: value for key, value in manifest.items() if key != "cache_manifest_id"}
    expected_manifest_id = hashlib.sha256(
        json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("cache_manifest_id") != expected_manifest_id:
        errors.append("cache manifest self-hash is absent or invalid")
    if metadata.get("source_hashes") != expected_source_hashes:
        errors.append("cache manifest source hashes differ from source lock v2")

    full_inventory: dict[str, Any] | None = None
    expected_inventory_episodes: dict[tuple[str, int, str], dict[str, Any]] = {}
    if full_cache_inventory_path is not None:
        full_inventory = load_full_cache_inventory(
            full_cache_inventory_path,
            source_lock_id=lock["source_lock_id"],
            split_contract=split_contract,
            final_manifest=final_manifest,
            split_contract_path=split_contract_path,
            final_manifest_path=final_manifest_path,
        )
        expected_inventory_episodes = {
            full_cache_episode_identity(row): row for row in full_inventory["episodes"]
        }
        if metadata.get("full_cache_inventory_id") != full_inventory["inventory_id"]:
            errors.append("cache manifest inventory ID mismatch")
        if metadata.get("full_cache_inventory_sha256") != sha256_file(full_cache_inventory_path):
            errors.append("cache manifest inventory hash mismatch")
        if int(metadata.get("num_shards", -1)) != int(full_inventory["num_shards"]):
            errors.append("cache manifest shard count differs from full-cache inventory")
    elif scope in {FULL_SHARD_SCOPE, FULL_MERGED_SCOPE} or require_full:
        errors.append("full-cache validation requires a frozen full-cache inventory")

    if full_inventory is not None and scope not in {FULL_SHARD_SCOPE, FULL_MERGED_SCOPE}:
        errors.append(f"inventory-backed cache has invalid scope {scope!r}")
    shard_index = metadata.get("shard_index")
    if scope == FULL_SHARD_SCOPE:
        if shard_index is None or not 0 <= int(shard_index) < int(metadata["num_shards"]):
            errors.append("full shard has an invalid shard index")
    if scope == FULL_MERGED_SCOPE and shard_index is not None:
        errors.append("merged full cache must not carry one shard index")

    expected_contract = metadata.get("tensor_contract")
    task_catalog = {
        (str(row["suite"]), int(row["benchmark_task_index"])): row
        for row in final_manifest["tasks"]
    }
    seen_episodes: set[tuple[str, int, str]] = set()
    seen_queries: set[tuple[str, int, str, int]] = set()
    records_count = 0
    for entry in manifest.get("episodes", []):
        try:
            entry_identity = _entry_identity(entry)
        except Exception as exc:
            errors.append(f"invalid cache manifest entry identity: {exc}")
            continue
        if int(entry.get("task_id", -1)) != entry_identity[1]:
            errors.append(f"entry task ID differs from benchmark task index: {entry_identity}")
        if entry_identity in seen_episodes:
            errors.append(f"duplicate episode identity: {entry_identity}")
        seen_episodes.add(entry_identity)
        contract_identity = (
            entry_identity[0],
            entry_identity[1],
            "teacher_demonstration",
            entry_identity[2],
        )
        assignment = split_mapping.get(contract_identity)
        if assignment is None or assignment["role"] != entry.get("split"):
            errors.append(f"episode split differs from frozen contract: {contract_identity}")

        inventory_episode = expected_inventory_episodes.get(entry_identity)
        if full_inventory is not None:
            if inventory_episode is None:
                errors.append(f"episode is absent from the full-cache inventory: {entry_identity}")
            elif entry.get("split") != inventory_episode["role"]:
                errors.append(f"episode role differs from the full-cache inventory: {entry_identity}")
            elif scope == FULL_SHARD_SCOPE and int(inventory_episode["shard_index"]) != int(shard_index):
                errors.append(f"episode was written to the wrong full-cache shard: {entry_identity}")

        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe episode shard path: {relative}")
            continue
        shard = root / relative
        if not shard.is_file():
            errors.append(f"missing episode shard: {shard}")
            continue
        if verify_hashes and _sha256(shard) != entry.get("sha256"):
            errors.append(f"episode shard hash mismatch: {shard}")
        rows = torch.load(shard, map_location="cpu", weights_only=False).get("records", [])
        if len(rows) != int(entry.get("records", -1)):
            errors.append(f"episode shard record count mismatch: {shard}")
        if inventory_episode is not None and len(rows) != int(inventory_episode["query_count"]):
            errors.append(f"episode query count differs from full-cache inventory: {entry_identity}")
        if rows and int(rows[0].get("policy_query_index", -1)) != 0:
            errors.append(f"episode does not begin at policy query q0: {shard}")

        for row_index, record in enumerate(rows):
            try:
                validate_record_v2(
                    record,
                    expected_contract=expected_contract,
                    expected_source_lock_id=lock["source_lock_id"],
                    expected_source_hashes=expected_source_hashes,
                    task_catalog=task_catalog,
                    execution_horizon=int(metadata["execution_horizon_r"]),
                )
                if _record_identity(record) != entry_identity:
                    raise ValueError("record identity differs from its manifest entry")
                if assignment is not None:
                    if record["environment_seed"] != assignment["environment_seed"]:
                        raise ValueError("record environment seed differs from split assignment")
                    if str(record["initial_state_identifier"]) != str(
                        assignment["initial_state_identifier"]
                    ):
                        raise ValueError("record initial-state identity differs from split assignment")
                query = int(record["policy_query_index"])
                step = int(record["absolute_environment_step"])
                if query != row_index:
                    raise ValueError("record query indices are not exactly contiguous from q0")
                if step != query * int(metadata["execution_horizon_r"]):
                    raise ValueError("record environment step does not advance by exact native R")
                if int(record["executed_action_length"]) != int(metadata["execution_horizon_r"]):
                    raise ValueError("teacher-cache record does not execute the exact native R prefix")
                if int(record["next_query_observation"].get("frame_index", -1)) != (
                    step + int(metadata["execution_horizon_r"])
                ):
                    raise ValueError("next-query observation is not exactly one native R step ahead")

                if inventory_episode is not None:
                    expected_query = expected_query_spec(
                        inventory_episode,
                        query,
                        execution_horizon=int(metadata["execution_horizon_r"]),
                        noise_seed_base=int(metadata["noise_seed_base"]),
                    )
                    if step != expected_query["absolute_environment_step"]:
                        raise ValueError("record step differs from frozen query inventory")
                    if int(record["action_noise_seed"]) != expected_query["action_noise_seed"]:
                        raise ValueError("record noise seed differs from frozen query inventory")
                regenerated = explicit_policy_noise(
                    tuple(torch.as_tensor(record["action_noise"]).shape),
                    seed=int(record["action_noise_seed"]),
                    device="cpu",
                )
                if not torch.equal(regenerated, torch.as_tensor(record["action_noise"]).cpu()):
                    raise ValueError("stored action noise does not equal deterministic seed regeneration")
                if row_index + 1 < len(rows):
                    _validate_next_observation_link(record, rows[row_index + 1])
            except Exception as exc:
                errors.append(f"{shard}: record {row_index}: {exc}")

            query_key = (*entry_identity, int(record.get("policy_query_index", -1)))
            if query_key in seen_queries:
                errors.append(f"duplicate query tuple: {query_key}")
            seen_queries.add(query_key)
        records_count += len(rows)

    roles = {entry.get("split") for entry in manifest.get("episodes", [])}
    if scope != FULL_SHARD_SCOPE:
        for role in split_contract.get("required_cache_roles", []):
            if role not in roles:
                errors.append(f"cache has no episode in required role {role}")

    if full_inventory is not None:
        expected_episodes = {
            identity
            for identity, row in expected_inventory_episodes.items()
            if scope == FULL_MERGED_SCOPE or int(row["shard_index"]) == int(shard_index)
        }
        expected_queries = {
            (*identity, query)
            for identity in expected_episodes
            for query in range(int(expected_inventory_episodes[identity]["query_count"]))
        }
        if seen_episodes != expected_episodes:
            missing = sorted(expected_episodes - seen_episodes)[:3]
            extra = sorted(seen_episodes - expected_episodes)[:3]
            errors.append(f"cache episode inventory mismatch; missing={missing}, extra={extra}")
        if seen_queries != expected_queries:
            missing = sorted(expected_queries - seen_queries)[:3]
            extra = sorted(seen_queries - expected_queries)[:3]
            errors.append(f"cache query inventory mismatch; missing={missing}, extra={extra}")

    full_pass = bool(
        not errors
        and verify_hashes
        and full_inventory is not None
        and scope == FULL_MERGED_SCOPE
    )
    if require_full and not full_pass:
        errors.append(
            "full-cache acceptance requires one merged inventory-exact cache with every shard hash verified"
        )
        full_pass = False
    schema_pass = not errors
    markers = ["CACHE_SCHEMA_V2_PASS"] if schema_pass else []
    if full_pass:
        markers.append("FULL_CACHE_SCHEMA_V2_PASS")
    return {
        "schema_version": 2,
        "CACHE_SCHEMA_V2_PASS": schema_pass,
        "FULL_CACHE_SCHEMA_V2_PASS": full_pass,
        "markers": markers,
        "source_lock_id": lock["source_lock_id"],
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_scope": scope,
        "full_cache_inventory_id": full_inventory.get("inventory_id") if full_inventory else None,
        "hashes_verified": verify_hashes,
        "episodes": len(seen_episodes),
        "records": records_count,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--full-cache-inventory")
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-shard-hashes", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = validate_cache_v2(
        args.cache,
        source_lock_path=args.source_lock,
        final_manifest_path=args.final_evaluation_manifest,
        split_contract_path=args.split_contract,
        full_cache_inventory_path=args.full_cache_inventory,
        verify_hashes=not args.skip_shard_hashes,
        require_full=args.require_full,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    required_marker = "FULL_CACHE_SCHEMA_V2_PASS" if args.require_full else "CACHE_SCHEMA_V2_PASS"
    if not payload[required_marker]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
