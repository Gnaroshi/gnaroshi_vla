#!/usr/bin/env python3
"""Lock train-dataset episode identifiers from Seer's source-locked index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-info", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--dataset-name", default="libero_10_converted")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    actual_sha = sha256(args.data_info)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(
            f"source-locked data_info mismatch: {actual_sha} != {args.expected_sha256}"
        )
    rows = json.loads(args.data_info.read_text(encoding="utf-8"))
    keys: list[str] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            raise RuntimeError(f"invalid data_info row: {row!r}")
        episode_id, length = str(row[0]), int(row[1])
        if length <= 10:
            raise RuntimeError(f"episode is too short for the V1 window: {row!r}")
        keys.append(f"{args.dataset_name}:{episode_id}")
    if len(keys) != len(set(keys)) or len(keys) < 20:
        raise RuntimeError("locked dataset index has duplicate or insufficient episode keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(keys, indent=2) + "\n", encoding="utf-8")
    print(f"LOCKED_TRAINING_EPISODE_KEYS count={len(keys)} sha256={actual_sha}")


if __name__ == "__main__":
    main()
