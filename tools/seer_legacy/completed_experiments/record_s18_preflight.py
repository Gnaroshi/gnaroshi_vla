#!/usr/bin/env python3
"""Combine immutable source/runtime/four-GPU checks into one preflight record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--four-gpu-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))
    contract = json.loads(args.four_gpu_contract.read_text(encoding="utf-8"))
    if source.get("status") != "SOURCE_GATE_PASS":
        raise RuntimeError("source gate has not passed")
    if runtime.get("status") != "S18_RUNTIME_GATE_PASS":
        raise RuntimeError("runtime gate has not passed")
    if contract.get("world_size") != 4:
        raise RuntimeError("four-GPU contract is invalid")
    payload = {
        "schema_version": 1,
        "status": "S18_CANONICAL_PREFLIGHT_PASS",
        "source": source,
        "runtime": runtime,
        "four_gpu_contract": contract,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["status"])


if __name__ == "__main__":
    main()
