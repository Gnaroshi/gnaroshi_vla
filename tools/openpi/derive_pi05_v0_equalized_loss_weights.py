#!/usr/bin/env python3
"""Derive documented equal-contribution V0/V1 weights from raw calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def equalized_weights(metrics: dict[str, float]) -> dict[str, float]:
    means = {
        name: sum(float(metrics[f"age{age}_{name}"]) for age in (1, 2, 3)) / 3.0
        for name in ("state", "chunk", "executed", "gripper")
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in means.values()):
        raise ValueError("raw age-averaged losses must be finite and positive")
    anchor = means["state"]
    return {
        "state": 1.0,
        "chunk": anchor / means["chunk"],
        "executed": anchor / means["executed"],
        "gripper": anchor / means["gripper"],
        "composition": 0.0,
    }


def v1_equalized_weights(metrics: dict[str, float]) -> dict[str, float]:
    means = {
        name: (float(metrics[f"direct_{name}"]) + float(metrics[f"composed_{name}"])) / 2.0
        for name in ("state", "chunk", "executed", "gripper")
    }
    composition = float(metrics["composition"])
    if any(not math.isfinite(value) or value <= 0.0 for value in (*means.values(), composition)):
        raise ValueError("raw V1 direct/composed losses must be finite and positive")
    anchor = means["state"]
    return {
        "state": 1.0,
        "chunk": anchor / means["chunk"],
        "executed": anchor / means["executed"],
        "gripper": anchor / means["gripper"],
        "composition": anchor / composition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v0", "v1"), default="v0")
    parser.add_argument("--raw-loss-calibration", required=True)
    parser.add_argument(
        "--expected-action-execution-mode", choices=("A", "B"), required=True
    )
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    raw = json.loads(Path(args.raw_loss_calibration).read_text(encoding="utf-8"))
    marker = (
        "V0_RAW_LOSS_CALIBRATION_COMPLETE"
        if args.variant == "v0"
        else "V1_STREAMING_RAW_LOSS_CALIBRATION_COMPLETE"
    )
    if raw.get(marker) is not True or raw.get("variant") != args.variant:
        raise RuntimeError(f"raw {args.variant.upper()} loss calibration is incomplete")
    if raw.get("action_execution_mode", "A") != args.expected_action_execution_mode:
        raise RuntimeError("raw loss calibration used another action execution mode")
    weights = (
        equalized_weights(raw["raw_metrics"])
        if args.variant == "v0"
        else v1_equalized_weights(raw["raw_metrics"])
    )
    if args.shell:
        names = ["state", "chunk", "executed", "gripper"]
        if args.variant == "v1":
            names.append("composition")
        print(*(weights[name] for name in names))
    else:
        print(json.dumps(weights, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
