#!/usr/bin/env python3
"""Enforce one scientific row per exactly-four-GPU lane."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError(f"exactly four unique GPUs are required, got {devices}")
    allowed = {("0", "1", "2", "3"), ("4", "5", "6", "7")}
    if devices not in allowed:
        raise ValueError(f"GPU lane must be 0,1,2,3 or 4,5,6,7, got {devices}")
    return devices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--nproc-per-node", type=int, default=4)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    devices = parse_devices(args.cuda_visible_devices)
    if args.nproc_per_node != 4 or contract["world_size"] != 4:
        raise ValueError("nproc_per_node and contract world_size must both equal four")
    runtime_world = os.environ.get("WORLD_SIZE")
    if runtime_world is not None and int(runtime_world) != 4:
        raise ValueError(f"runtime WORLD_SIZE must equal four, got {runtime_world}")
    print(json.dumps({"status": "FOUR_GPU_GUARD_PASS", "devices": devices, "world_size": 4}))


if __name__ == "__main__":
    main()
