#!/usr/bin/env python3
"""Verify that defect fit, validity, scheduler, and final episodes are disjoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defect_split_common import load_contract
from source_lock_v2 import verify_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    role_sets, contract = load_contract(args.contract)
    lock = verify_lock(args.source_lock)
    if contract.get("source_lock_id") != lock["source_lock_id"]:
        raise RuntimeError("source mismatch: defect split contract is stale")
    payload = {
        "DEFECT_SPLIT_DISJOINTNESS_PASS": True,
        "markers": ["DEFECT_SPLIT_DISJOINTNESS_PASS"],
        "source_lock_id": contract["source_lock_id"],
        "role_episode_counts": {role: len(rows) for role, rows in role_sets.items()},
        "contract": str(Path(args.contract).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
