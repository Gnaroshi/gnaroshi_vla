#!/usr/bin/env python3
"""Fail-closed bridge to a reviewed V1 runtime integration entrypoint."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-status", type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=(20, 40), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--adapter-init", type=Path, required=True)
    parser.add_argument("--vit-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--libero-path", type=Path, required=True)
    parser.add_argument("--loss-weights", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-gpu-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    status = json.loads(args.integration_status.read_text(encoding="utf-8"))
    if status.get("status") != "V1_RUNTIME_INTEGRATION_PASS":
        raise RuntimeError(f"V1 runtime integration is not approved: {status.get('status')}")
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise RuntimeError("V1 training requires exactly four distributed processes")
    expected = (42, 16, 8, 1e-3, 1e-4, 0.05, "fp32", True)
    actual = (
        args.seed,
        args.per_gpu_batch,
        args.gradient_accumulation,
        args.learning_rate,
        args.weight_decay,
        args.warmup_fraction,
        args.precision,
        args.deterministic,
    )
    if actual != expected:
        raise RuntimeError(f"V1 training contract mismatch: {actual}")
    for name, path in (
        ("teacher", args.teacher),
        ("adapter initialization", args.adapter_init),
        ("ViT checkpoint", args.vit_checkpoint),
        ("dataset root", args.dataset_root),
        ("LIBERO source", args.libero_path),
        ("loss weights", args.loss_weights),
        ("split manifest", args.split_manifest),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing locked {name}: {path}")
    module_name, separator, function_name = status.get("training_entrypoint", "").partition(":")
    if not separator:
        raise RuntimeError("reviewed training_entrypoint module:function is missing")
    function = getattr(importlib.import_module(module_name), function_name)
    function(args)


if __name__ == "__main__":
    main()
