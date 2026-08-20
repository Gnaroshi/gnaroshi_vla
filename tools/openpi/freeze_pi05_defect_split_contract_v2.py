#!/usr/bin/env python3
"""Freeze four episode-disjoint defect/scheduler/final roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _common import require_source_lock_v2
from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    load_final_evaluation_manifest,
    load_split_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-split-contract", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite defect split contract: {output}")
    lock = require_source_lock_v2(args.source_lock)
    final_manifest = load_final_evaluation_manifest(args.final_evaluation_manifest)
    _, cache_contract = load_split_contract(args.cache_split_contract, final_manifest)
    if cache_contract["source_lock_id"] != lock["source_lock_id"]:
        raise RuntimeError("source mismatch: cache split contract is stale")
    roles = {
        role: [
            {
                "suite": row["suite"],
                "benchmark_task_index": row["benchmark_task_index"],
                "episode_namespace": row["episode_namespace"],
                "episode_id": row["episode_id"],
            }
            for row in cache_contract["assignments"]
            if row["role"] == role
        ]
        for role in ("defect_fit", "defect_validity", "scheduler_calibration")
    }
    roles["final_scientific_evaluation"] = [
        {
            "suite": row["suite"],
            "benchmark_task_index": row["benchmark_task_index"],
            "episode_namespace": row["episode_namespace"],
            "episode_id": str(row["trial"]),
        }
        for row in final_manifest["episodes"]
    ]
    payload = {
        "schema_version": 2,
        "frozen": True,
        "source_lock_id": lock["source_lock_id"],
        "cache_split_contract_id": cache_contract["split_contract_id"],
        "cache_split_contract": str(Path(args.cache_split_contract).resolve()),
        "cache_split_contract_sha256": hashlib.sha256(
            Path(args.cache_split_contract).read_bytes()
        ).hexdigest(),
        "final_manifest_id": final_manifest["manifest_id"],
        "final_evaluation_manifest": str(Path(args.final_evaluation_manifest).resolve()),
        "final_evaluation_manifest_sha256": hashlib.sha256(
            Path(args.final_evaluation_manifest).read_bytes()
        ).hexdigest(),
        "roles": roles,
    }
    payload["defect_split_contract_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
