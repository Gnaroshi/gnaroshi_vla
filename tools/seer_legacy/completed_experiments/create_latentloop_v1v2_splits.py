#!/usr/bin/env python3
"""Create fixed episode-disjoint V1/V2 splits without final-eval leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-episode-keys", type=Path, required=True, help="JSON list from the locked dataset index")
    parser.add_argument("--final-episode-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    training = [str(value) for value in json.loads(args.training_episode_keys.read_text(encoding="utf-8"))]
    if len(training) != len(set(training)) or len(training) < 20:
        raise RuntimeError("training episode keys must be unique and nontrivial")
    with args.final_episode_manifest.open(newline="", encoding="utf-8") as handle:
        final = [f"eval:{row['task_id']}:{row['episode_id']}:{row['seed']}" for row in csv.DictReader(handle)]
    namespaced = [f"dataset:{value}" for value in training]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(namespaced))
    ordered = [namespaced[int(index)] for index in order]
    boundaries = [0, int(0.70 * len(ordered)), int(0.80 * len(ordered)), int(0.875 * len(ordered)), int(0.95 * len(ordered)), len(ordered)]
    roles = ("transition_train", "checkpoint_validation", "defect_fit", "defect_validation", "scheduler_calibration")
    splits = {role: ordered[boundaries[index] : boundaries[index + 1]] for index, role in enumerate(roles)}
    splits["final_evaluation"] = final
    sets = {name: set(values) for name, values in splits.items()}
    if any(sets[left] & sets[right] for index, left in enumerate(sets) for right in list(sets)[index + 1 :]):
        raise RuntimeError("generated split overlap")
    payload = {
        "schema_version": 1,
        "status": "V1_V2_SPLITS_LOCKED",
        "seed": args.seed,
        "training_episode_key_source_sha256": sha256(args.training_episode_keys),
        "final_episode_manifest_sha256": sha256(args.final_episode_manifest),
        "selection_uses_online_sr": False,
        "splits": {name: {"episode_keys": values, "count": len(values)} for name, values in splits.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["status"])


if __name__ == "__main__":
    main()
