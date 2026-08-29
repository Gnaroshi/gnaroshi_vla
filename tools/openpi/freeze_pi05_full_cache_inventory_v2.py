#!/usr/bin/env python3
"""Freeze the complete teacher-cache episode/query inventory without GPU work."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    load_final_evaluation_manifest,
    load_split_contract,
)
from architectures.openpi.adapters.latentloop.full_cache_contract_v2 import (
    DEFAULT_FULL_CACHE_SHARDS,
    build_full_cache_inventory,
    sha256_file,
)
from source_lock_v2 import verify_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--num-shards", type=int, default=DEFAULT_FULL_CACHE_SHARDS)
    parser.add_argument("--noise-seed-base", type=int, default=20260820)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run or os.environ.get("OPENPI_LATENTLOOP_PREPARE_RUN") != "1":
        raise RuntimeError(
            "inventory freezing requires --run and OPENPI_LATENTLOOP_PREPARE_RUN=1"
        )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full-cache inventory: {output}")

    lock = verify_lock(args.source_lock)
    final_manifest = load_final_evaluation_manifest(args.final_evaluation_manifest)
    _, split_contract = load_split_contract(args.split_contract, final_manifest)
    payload = build_full_cache_inventory(
        source_lock_id=lock["source_lock_id"],
        split_contract=split_contract,
        final_manifest=final_manifest,
        split_contract_sha256=sha256_file(args.split_contract),
        final_manifest_sha256=sha256_file(args.final_evaluation_manifest),
        noise_seed_base=args.noise_seed_base,
        num_shards=args.num_shards,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["statistics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
