#!/usr/bin/env python3
"""Merge inventory-complete cache shards using hard links, then validate atomically."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from _common import require_run
from architectures.openpi.adapters.latentloop.cache_io import EpisodeCacheWriter, EpisodeIndexEntry
from architectures.openpi.adapters.latentloop.full_cache_contract_v2 import (
    load_full_cache_inventory,
    sha256_file,
)
from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    load_final_evaluation_manifest,
    load_split_contract,
)
from source_lock_v2 import verify_lock
from validate_pi05_cache_v2 import FULL_MERGED_SCOPE, FULL_SHARD_SCOPE, validate_cache_v2


INVARIANT_METADATA = (
    "source_lock_id",
    "source_hashes",
    "checkpoint",
    "final_evaluation_manifest_sha256",
    "split_contract_sha256",
    "full_cache_inventory_id",
    "full_cache_inventory_sha256",
    "tensor_contract",
    "action_horizon_h",
    "execution_horizon_r",
    "noise_seed_base",
    "num_shards",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--full-cache-inventory", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_FULL_CACHE_RUN")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse merged cache: {output}")

    lock = verify_lock(args.source_lock)
    final_manifest = load_final_evaluation_manifest(args.final_evaluation_manifest)
    _, split_contract = load_split_contract(args.split_contract, final_manifest)
    inventory = load_full_cache_inventory(
        args.full_cache_inventory,
        source_lock_id=lock["source_lock_id"],
        split_contract=split_contract,
        final_manifest=final_manifest,
        split_contract_path=args.split_contract,
        final_manifest_path=args.final_evaluation_manifest,
    )
    shard_roots = [Path(value).resolve() for value in args.shard]
    if len(shard_roots) != int(inventory["num_shards"]):
        raise ValueError("merge requires exactly every shard named by the full-cache inventory")

    manifests = []
    for root in shard_roots:
        status = validate_cache_v2(
            root,
            source_lock_path=args.source_lock,
            final_manifest_path=args.final_evaluation_manifest,
            split_contract_path=args.split_contract,
            full_cache_inventory_path=args.full_cache_inventory,
            verify_hashes=False,
        )
        if not status["CACHE_SCHEMA_V2_PASS"]:
            raise RuntimeError(f"full-cache shard failed structural validation: {root}: {status['errors']}")
        manifest = json.loads(
            (root / "pi05_latentloop_cache_manifest.json").read_text(encoding="utf-8")
        )
        if manifest["metadata"].get("cache_scope") != FULL_SHARD_SCOPE:
            raise RuntimeError(f"input is not a full-cache shard: {root}")
        manifests.append(manifest)

    indices = [int(item["metadata"]["shard_index"]) for item in manifests]
    if sorted(indices) != list(range(int(inventory["num_shards"]))):
        raise RuntimeError(f"full-cache shard indices are incomplete or duplicated: {indices}")
    reference = manifests[0]["metadata"]
    for manifest in manifests[1:]:
        for key in INVARIANT_METADATA:
            if manifest["metadata"].get(key) != reference.get(key):
                raise RuntimeError(f"cache shard metadata mismatch for {key}")

    staging = output.with_name(f".{output.name}.merge-tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale merge staging path exists: {staging}")
    metadata = dict(reference)
    metadata.update(
        {
            "cache_scope": FULL_MERGED_SCOPE,
            "shard_index": None,
            "merged_shards": [str(path) for path in shard_roots],
            "merged_shard_manifest_sha256": [
                sha256_file(root / "pi05_latentloop_cache_manifest.json") for root in shard_roots
            ],
            "expected_full_episodes": int(inventory["statistics"]["episodes"]),
            "expected_full_queries": int(inventory["statistics"]["queries"]),
        }
    )
    try:
        writer = EpisodeCacheWriter(staging, metadata)
        identities: set[tuple[str, int, str]] = set()
        for root, manifest in zip(shard_roots, manifests, strict=True):
            for entry in manifest["episodes"]:
                identity = (
                    str(entry["suite"]),
                    int(entry["benchmark_task_index"]),
                    str(entry["episode_id"]),
                )
                if identity in identities:
                    raise RuntimeError(f"duplicate episode across shards: {identity}")
                identities.add(identity)
                source = root / entry["path"]
                destination = staging / "episodes" / source.name
                try:
                    os.link(source, destination)
                except OSError as exc:
                    raise RuntimeError(
                        "full-cache merge requires same-filesystem hard links; refusing a copy or symlink fallback"
                    ) from exc
                writer.entries.append(
                    EpisodeIndexEntry(
                        suite=str(entry["suite"]),
                        task_id=int(entry["task_id"]),
                        benchmark_task_index=int(entry["benchmark_task_index"]),
                        episode_id=int(entry["episode_id"]),
                        split=str(entry["split"]),
                        records=int(entry["records"]),
                        path=str(Path("episodes") / source.name),
                        sha256=str(entry["sha256"]),
                    )
                )
        writer.entries.sort(key=lambda item: (item.suite, item.task_id, item.episode_id))
        writer.finalize(
            {
                "episodes": len(writer.entries),
                "records": sum(item.records for item in writer.entries),
                "storage": "same-filesystem hard links; no duplicated tensor payload",
            }
        )
        status = validate_cache_v2(
            staging,
            source_lock_path=args.source_lock,
            final_manifest_path=args.final_evaluation_manifest,
            split_contract_path=args.split_contract,
            full_cache_inventory_path=args.full_cache_inventory,
            verify_hashes=True,
            require_full=True,
        )
        if not status["FULL_CACHE_SCHEMA_V2_PASS"]:
            raise RuntimeError(f"merged full cache failed acceptance: {status['errors']}")
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output / "pi05_latentloop_cache_manifest.json")


if __name__ == "__main__":
    main()
