#!/usr/bin/env python3
"""Fail-closed verifier for the frozen paired 2,000-episode protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    SUITES,
    load_final_evaluation_manifest,
)
from source_lock_v2 import verify_lock


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(
    manifest_path: str | Path,
    source_lock_path: str | Path,
    *,
    suite: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = load_final_evaluation_manifest(manifest_path)
    lock = verify_lock(source_lock_path)
    if manifest.get("source_lock_id") != lock["source_lock_id"]:
        raise RuntimeError("source mismatch: final evaluation manifest was frozen under another lock")
    protocol = manifest.get("protocol", {})
    required_protocol = {
        "action_horizon_h": 10,
        "execution_horizon_r": 5,
        "wait_steps": 10,
        "resize_size": 224,
        "renderer": "egl",
        "trials_per_task": 50,
        "noise_seed_base": 7,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"final manifest protocol {key}={protocol.get(key)!r}, expected {expected!r}")
    if protocol.get("policy_noise") != "explicit_query_keyed_sha256_v2":
        raise ValueError("final manifest must use explicit query-keyed policy noise")
    if suite is not None and suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}")
    episodes = manifest["episodes"]
    for row in episodes:
        if row.get("episode_namespace") != "final_scientific_evaluation":
            raise ValueError("final manifest episode namespace is not scientific-evaluation-only")
        required = (
            "suite",
            "benchmark_task_index",
            "trial",
            "environment_seed",
            "initial_state_identifier",
            "query_noise_key_prefix",
            "max_episode_steps",
        )
        missing = [name for name in required if row.get(name) in (None, "")]
        if missing:
            raise ValueError(f"final manifest episode is missing {missing}")
    selected = [row for row in episodes if suite is None or row["suite"] == suite]
    expected_rows = 2000 if suite is None else 500
    if len(selected) != expected_rows:
        raise ValueError(f"selected final manifest has {len(selected)} episodes, expected {expected_rows}")
    return {
        "FINAL_EVALUATION_MANIFEST_V2_PASS": True,
        "source_lock_id": lock["source_lock_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_id": manifest["manifest_id"],
        "suite": suite,
        "selected_episodes": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--suite", choices=SUITES)
    args = parser.parse_args()
    print(json.dumps(verify_manifest(args.manifest, args.source_lock, suite=args.suite), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
