#!/usr/bin/env python3
"""Apply frozen Seer latent-location and recurrence decision rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def apply(
    inputs_path: Path,
    output_path: Path,
    repo_root: Path,
    additional_evidence: Path | None = None,
) -> dict:
    """Read aggregate inputs, apply predeclared rules, and refuse overwrite."""

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    sys.path.insert(0, str(repo_root))
    from methods.latentloop_comparison.decisions import (  # noqa: PLC0415
        apply_comparison_decisions,
    )

    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    if additional_evidence is not None:
        evidence = json.loads(additional_evidence.read_text(encoding="utf-8"))
        allowed = {
            "latentloop_better_k8_stability",
            "latentloop_lower_gripper_failure_or_chatter",
            "latentloop_better_libero_plus",
            "latentloop_better_heldout_checkpoint",
            "latentloop_better_k8_than_nonrecurrent",
            "latentloop_better_heldout_than_nonrecurrent",
        }
        inputs.update(
            {key: bool(value) for key, value in evidence.items() if key in allowed}
        )
    result = apply_comparison_decisions(inputs)
    result.update(
        {
            "protocol": "seer_latentloop_q1_q2_predeclared_decision_v1",
            "decision_inputs": str(inputs_path.resolve()),
            "decision_inputs_schema_version": inputs.get("schema_version"),
            "additional_evidence": str(additional_evidence.resolve())
            if additional_evidence
            else None,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    reloaded = json.loads(output_path.read_text(encoding="utf-8"))
    if reloaded["combined_verdict"] != result["combined_verdict"]:
        raise RuntimeError("Decision serialization/reload validation failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-evidence", type=Path)
    args = parser.parse_args()
    apply(
        args.inputs.resolve(),
        args.output.resolve(),
        args.repo_root.resolve(),
        args.additional_evidence.resolve() if args.additional_evidence else None,
    )


if __name__ == "__main__":
    main()
