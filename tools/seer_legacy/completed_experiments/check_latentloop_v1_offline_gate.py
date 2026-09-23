#!/usr/bin/env python3
"""Apply the frozen V1 offline gate before any online LIBERO row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.metrics.read_text(encoding="utf-8"))
    expected_provenance = {
        "source_lock_sha256": hashlib.sha256(args.source_lock.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        "v1_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    }
    scalar_names = (
        "direct_latent_mse", "composed_latent_mse", "direct_action_l1",
        "composed_action_l1", "hold_latent_mse", "hold_action_l1",
        "composition_defect", "no_composition_defect",
    )
    finite = all(math.isfinite(float(data[name])) for name in scalar_names)
    age = [float(value) for value in data["composed_action_l1_by_age"]]
    checks = {
        "finite_losses": finite,
        "finite_gripper_output": bool(data["finite_gripper_output"]),
        "direct_beats_hold_condition": float(data["direct_latent_mse"]) < float(data["hold_latent_mse"]),
        "composed_beats_hold_condition": float(data["composed_latent_mse"]) < float(data["hold_latent_mse"]),
        "direct_beats_hold_action": float(data["direct_action_l1"]) < float(data["hold_action_l1"]),
        "composed_beats_hold_action": float(data["composed_action_l1"]) < float(data["hold_action_l1"]),
        "age_2_3_not_catastrophic": len(age) == 3 and max(age[1:]) <= 2.0 * max(age[0], 1e-12),
        "composition_improves_control": float(data["composition_defect"]) < float(data["no_composition_defect"]),
        "parameter_ratio_cap": int(data["trainable_parameters"]) <= int(1.25 * 470146),
        "absolute_parameter_cap": int(data["trainable_parameters"]) <= 600000,
        "teacher_gradient_zero": float(data["max_teacher_gradient_abs"]) == 0.0,
        "action_generator_gradient_zero": float(data["max_action_generator_gradient_abs"]) == 0.0,
        "source_lock_match": data.get("source_lock_sha256") == expected_provenance["source_lock_sha256"],
        "split_manifest_match": data.get("split_manifest_sha256") == expected_provenance["split_manifest_sha256"],
        "selected_checkpoint_match": data.get("v1_checkpoint_sha256") == expected_provenance["v1_checkpoint_sha256"],
        "no_gripper_collapse": not bool(data["gripper_collapse"]),
        "validation_only": data.get("split_role") == "checkpoint_validation" and not data.get("uses_online_sr", True),
    }
    passed = all(checks.values())
    payload = {"schema_version": 1, "verdict": "V1_OFFLINE_PASS" if passed else "V1_OFFLINE_FAIL", "checks": checks, "expected_provenance": expected_provenance, "metrics": data}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["verdict"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
