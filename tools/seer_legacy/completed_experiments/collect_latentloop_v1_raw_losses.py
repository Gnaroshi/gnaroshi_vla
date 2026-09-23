#!/usr/bin/env python3
"""Four-GPU reviewed entrypoint for V1 raw-loss calibration samples."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--adapter-init", type=Path, required=True)
    parser.add_argument("--vit-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--libero-path", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-gpu-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-microbatches", type=int, default=48)
    args = parser.parse_args()
    status = json.loads(args.integration_status.read_text(encoding="utf-8"))
    if status.get("status") != "V1_RUNTIME_INTEGRATION_PASS":
        raise RuntimeError(f"V1 runtime integration is not approved: {status.get('status')}")
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise RuntimeError("raw-loss collection requires exactly four distributed processes")
    if (args.seed, args.per_gpu_batch) != (42, 16):
        raise RuntimeError("raw-loss calibration seed/batch contract mismatch")
    if args.target_microbatches < 3 or args.target_microbatches % 3:
        raise RuntimeError("target microbatches must be a positive multiple of three")
    for name, path in (
        ("teacher", args.teacher),
        ("adapter initialization", args.adapter_init),
        ("ViT checkpoint", args.vit_checkpoint),
        ("dataset root", args.dataset_root),
        ("LIBERO source", args.libero_path),
        ("split manifest", args.split_manifest),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing locked {name}: {path}")
    module_name, separator, function_name = status.get("raw_loss_entrypoint", "").partition(":")
    if not separator:
        raise RuntimeError("reviewed raw_loss_entrypoint module:function is missing")
    getattr(importlib.import_module(module_name), function_name)(args)


if __name__ == "__main__":
    main()
