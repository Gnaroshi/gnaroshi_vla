#!/usr/bin/env python3
"""Select a frozen V2 threshold tuple at effective K in [3.8,4.2]."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    document = json.loads(args.candidates.read_text(encoding="utf-8"))
    split_document = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    actual_hashes = {
        "source_lock_sha256": hashlib.sha256(args.source_lock.read_bytes()).hexdigest(),
        "v1_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
    }
    if document.get("split_role") != "scheduler_calibration" or document.get("uses_final_200_episodes", True):
        raise RuntimeError("scheduler thresholds must use the dedicated calibration split only")
    for field in ("source_lock_sha256", "v1_checkpoint_sha256", "split_manifest_sha256"):
        if document.get(field) != actual_hashes[field]:
            raise RuntimeError(f"scheduler calibration provenance mismatch: {field}")
    if document.get("defect_gate_verdict") != "DEFECT_SIGNAL_PASS":
        raise RuntimeError("scheduler calibration requires DEFECT_SIGNAL_PASS")
    rows = document["candidates"]
    locked_calibration_keys = set(split_document["splits"]["scheduler_calibration"]["episode_keys"])
    candidate_episode_keys = set(map(str, document.get("episode_keys", [])))
    if not candidate_episode_keys or not candidate_episode_keys <= locked_calibration_keys:
        raise RuntimeError("scheduler calibration uses keys outside the locked calibration split")
    normalized = []
    for row in rows:
        episodes = row.get("episode_summaries", [])
        if not episodes:
            raise RuntimeError("each candidate must contain complete ordered episode summaries")
        ordered_keys = [str(item.get("episode_key", "")) for item in episodes]
        if ordered_keys != list(map(str, document["episode_keys"])):
            raise RuntimeError("candidate episode summaries do not match the locked ordered calibration keys")
        if not all(bool(item.get("complete_episode", False)) for item in episodes):
            raise RuntimeError("scheduler calibration cannot use partial episode summaries")
        total_steps = sum(int(item["policy_steps"]) for item in episodes)
        full_calls = sum(int(item["full_calls"]) for item in episodes)
        if any(int(item.get("initial_mandatory_full_calls", 0)) != 1 for item in episodes):
            raise RuntimeError("every complete episode must count one mandatory initial Full Seer call")
        if total_steps <= 0 or full_calls <= 0:
            raise RuntimeError("invalid scheduler operation counts")
        computed = dict(row)
        computed["total_policy_steps"] = total_steps
        computed["total_full_calls"] = full_calls
        computed["effective_k"] = total_steps / full_calls
        normalized.append(computed)
    feasible = [row for row in normalized if 3.8 <= float(row["effective_k"]) <= 4.2]
    if not feasible:
        raise RuntimeError("no V2 scheduler candidate meets the predeclared K budget")
    selected = min(
        feasible,
        key=lambda row: (
            float(row["mean_reference_action_error"]),
            float(row["p95_reference_action_error"]),
            abs(float(row["effective_k"]) - 4.0),
            int(row["direct_reanchors"]),
        ),
    )
    payload = {
        "schema_version": 1,
        "status": "V2_THRESHOLD_LOCKED",
        "selection_uses_final_200_episodes": False,
        "target_effective_k": 4.0,
        "source_lock_sha256": actual_hashes["source_lock_sha256"],
        "v1_checkpoint_sha256": actual_hashes["v1_checkpoint_sha256"],
        "split_manifest_sha256": actual_hashes["split_manifest_sha256"],
        "calibration_episode_keys": document.get("episode_keys", []),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("V2_THRESHOLD_LOCKED")


if __name__ == "__main__":
    main()
