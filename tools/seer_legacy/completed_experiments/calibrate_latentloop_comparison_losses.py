#!/usr/bin/env python3
"""Derive pre-training loss weights from held-out raw teacher-tuple scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(payload: dict, name: str) -> float:
    value = float(payload["metrics"][name]["mean"])
    if not value > 0.0:
        raise ValueError(f"Raw metric must be positive: {name}={value}")
    return value


def calibrate(
    raw_metrics: Path,
    output: Path,
    *,
    mode: str,
    executed_ratio: float,
    regularization_ratio: float,
) -> dict:
    """Create a deterministic calibration artifact without test outcomes."""

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    source = json.loads(raw_metrics.read_text(encoding="utf-8"))
    if source.get("mode") != mode:
        raise ValueError(f"Mode mismatch: {source.get('mode')} versus {mode}")
    if source.get("adapter_checkpoint") is not None:
        raise ValueError("Raw calibration must use the pre-training initialized adapter")
    if mode == "action_correction":
        arm = _mean(source, "raw_arm_smooth_l1")
        gripper = _mean(source, "raw_gripper_bce")
        executed = _mean(source, "raw_executed_arm_smooth_l1")
        residual = _mean(source, "raw_residual_l2")
        weights = {
            "arm": 1.0,
            "gripper": arm / gripper,
            "executed_token": executed_ratio * arm / executed,
            "residual_regularization": regularization_ratio * arm / residual,
        }
        contributions = {
            "arm": weights["arm"] * arm,
            "gripper": weights["gripper"] * gripper,
            "executed_token": weights["executed_token"] * executed,
            "residual_regularization": weights["residual_regularization"] * residual,
        }
        rule = (
            "arm is the unit scale; gripper is equalized to arm; executed-token "
            f"contribution is {executed_ratio:g}x arm; residual regularization is "
            f"{regularization_ratio:g}x arm"
        )
        compatible = True
    else:
        raw = {
            "latent": _mean(source, "raw_latent_mse"),
            "action": _mean(source, "raw_action_l1"),
            "smooth": _mean(source, "raw_smooth_mse"),
        }
        weights = {"latent": 0.05, "action": 0.1, "smooth": 0.001}
        contributions = {name: weights[name] * value for name, value in raw.items()}
        positive = list(contributions.values())
        scale_ratio = max(positive) / min(positive)
        compatible = scale_ratio <= 100.0
        rule = (
            "canonical LatentLoop weights (0.05, 0.1, 0.001) are retained; "
            "calibration only checks that weighted raw terms are within two orders of magnitude"
        )
    result = {
        "schema_version": 1,
        "protocol": "latentloop_q1_q2_raw_loss_calibration_v1",
        "mode": mode,
        "source_raw_metrics": str(raw_metrics.resolve()),
        "source_lock": source.get("source_lock"),
        "source_lock_sha256": source.get("source_lock_sha256"),
        "teacher_sha256": source.get("teacher_sha256"),
        "canonical_adapter_sha256": source.get("canonical_adapter_sha256"),
        "uses_libero_test_success": False,
        "rule": rule,
        "weights": weights,
        "calibrated_initial_contributions": contributions,
        "pass": compatible,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["action_correction", "anchor_bridge"], required=True
    )
    parser.add_argument("--executed-contribution-ratio", type=float, default=0.5)
    parser.add_argument("--regularization-contribution-ratio", type=float, default=0.01)
    args = parser.parse_args()
    calibrate(
        args.raw_metrics.resolve(),
        args.output.resolve(),
        mode=args.mode,
        executed_ratio=args.executed_contribution_ratio,
        regularization_ratio=args.regularization_contribution_ratio,
    )


if __name__ == "__main__":
    main()
