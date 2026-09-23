#!/usr/bin/env python3
"""Fail-closed bridge to reviewed V1/V2 online evaluation hooks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-status", type=Path, required=True)
    parser.add_argument("--mode", choices=("v1_fixed", "v2_dynamic", "random_matched"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--adapter-init", type=Path, required=True)
    parser.add_argument("--vit-checkpoint", type=Path, required=True)
    parser.add_argument("--libero-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--threshold-lock", type=Path)
    parser.add_argument("--matched-budget-reference", type=Path)
    parser.add_argument("--query-interval", type=int, default=4)
    parser.add_argument("--allow-conditional-interval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--action-prediction-horizon", type=int, default=3)
    parser.add_argument("--temporal-ensembling", action="store_true")
    parser.add_argument("--temporal-ensemble-temperature", type=float, default=0.01)
    parser.add_argument("--renderer", choices=("osmesa",), default="osmesa")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    status = json.loads(args.integration_status.read_text(encoding="utf-8"))
    expected = "V2_RUNTIME_INTEGRATION_PASS" if args.mode != "v1_fixed" else "V1_RUNTIME_INTEGRATION_PASS"
    if status.get("status") != expected:
        raise RuntimeError(f"{args.mode} runtime integration is not approved: {status.get('status')}")
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise RuntimeError("online evaluation requires exactly four distributed processes")
    for name, path in (
        ("candidate checkpoint", args.checkpoint),
        ("teacher checkpoint", args.teacher),
        ("V0 adapter initialization", args.adapter_init),
        ("ViT checkpoint", args.vit_checkpoint),
        ("LIBERO source", args.libero_path),
        ("episode manifest", args.episode_manifest),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing locked {name}: {path}")
    if (args.tasks, args.episodes_per_task, args.max_steps) != (10, 20, 600):
        raise RuntimeError("online evaluation must use 10 tasks x 20 episodes and 600 max steps")
    if args.control_hz != 20.0 or args.action_prediction_horizon != 3:
        raise RuntimeError("online evaluation control/action horizon differs from the canonical contract")
    if not args.temporal_ensembling or args.temporal_ensemble_temperature != 0.01:
        raise RuntimeError("temporal-ensemble temperature differs from the canonical contract")
    if args.seed != 42 or args.precision != "fp32" or not args.deterministic:
        raise RuntimeError("online evaluation seed/precision/deterministic contract mismatch")
    if args.mode == "v1_fixed":
        if args.query_interval not in (4, 8, 12):
            raise RuntimeError("V1 supports the primary K=4 row and conditional K=8/K=12 rows")
        if args.query_interval != 4 and not args.allow_conditional_interval:
            raise RuntimeError(
                "K=8/K=12 are conditional V1 rows; use the gated conditional wrapper"
            )
        if args.query_interval == 4 and args.allow_conditional_interval:
            raise RuntimeError("the primary V1 K=4 row must not use the conditional override")
    elif args.allow_conditional_interval:
        raise RuntimeError("conditional interval override is valid only for V1")
    if args.mode != "v1_fixed" and (args.threshold_lock is None or not args.threshold_lock.is_file()):
        raise RuntimeError("V2/random evaluation requires the frozen threshold lock")
    if args.mode == "random_matched":
        if args.matched_budget_reference is None or not args.matched_budget_reference.is_dir():
            raise RuntimeError("matched-random evaluation requires the completed V2 target-K budget reference")
    elif args.matched_budget_reference is not None:
        raise RuntimeError("matched-budget reference is valid only for random_matched mode")
    module_name, separator, function_name = status.get("evaluation_entrypoint", "").partition(":")
    if not separator:
        raise RuntimeError("reviewed evaluation_entrypoint module:function is missing")
    function = getattr(importlib.import_module(module_name), function_name)
    function(args)


if __name__ == "__main__":
    main()
