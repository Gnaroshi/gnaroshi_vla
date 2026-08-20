#!/usr/bin/env python3
"""Fail-closed stage dependency checker that never creates result roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from source_lock_v2 import verify_lock


STAGE_REQUIREMENTS = {
    "stage0_source": (),
    "stage1_real_parity": (),
    "stage1_episode_smoke": (
        "REAL_KV_ROUNDTRIP_PASS",
        "K1_ACTION_PARITY_PASS",
        "BASE_FREEZE_PASS",
    ),
    "stage2_cache_smoke": (
        "REAL_KV_ROUNDTRIP_PASS",
        "K1_ACTION_PARITY_PASS",
        "K1_EPISODE_PARITY_PASS",
        "BASE_FREEZE_PASS",
    ),
    "stage2_full_cache": (
        "REAL_KV_ROUNDTRIP_PASS",
        "K1_ACTION_PARITY_PASS",
        "K1_EPISODE_PARITY_PASS",
        "BASE_FREEZE_PASS",
        "FULL_CACHE_INVENTORY_V2_PASS",
    ),
    "paired_full_baseline": (
        "REAL_KV_ROUNDTRIP_PASS",
        "K1_ACTION_PARITY_PASS",
        "K1_EPISODE_PARITY_PASS",
        "BASE_FREEZE_PASS",
    ),
    "stage3_v0_raw_loss": ("FULL_CACHE_SCHEMA_V2_PASS",),
    "stage3_v0": ("FULL_CACHE_SCHEMA_V2_PASS", "V0_LOSS_WEIGHTS_APPROVED"),
    "stage3_v0_streaming_raw_loss": (
        "REAL_KV_ROUNDTRIP_PASS",
        "K1_ACTION_PARITY_PASS",
        "K1_EPISODE_PARITY_PASS",
        "BASE_FREEZE_PASS",
    ),
    "stage3_v0_streaming": (
        "REAL_KV_ROUNDTRIP_PASS",
        "K1_ACTION_PARITY_PASS",
        "K1_EPISODE_PARITY_PASS",
        "BASE_FREEZE_PASS",
        "V0_STREAMING_LOSS_WEIGHTS_APPROVED",
    ),
    "cache_artifact_current": ("FULL_CACHE_SCHEMA_V2_PASS",),
    "stage4_v0_offline": ("V0_TRAIN_COMPLETE",),
    "stage5_v0_paired_eval": ("V0_OFFLINE_GATE_PASS",),
    "stage6_v1": ("V0_PAIRED_ROW_PASS",),
    "stage7_v1_offline": ("V1_TRAIN_COMPLETE",),
    "stage8_v1_paired_eval": ("V1_OFFLINE_GATE_PASS",),
    "stage9_defect_fit": ("V1_PAIRED_ROW_PASS",),
    "stage10_defect_validity": ("DEFECT_FIT_PASS",),
    "stage11_scheduler_calibration": ("DEFECT_VALIDITY_PASS",),
    "stage12_v2_paired_eval": ("DYNAMIC_BUDGET_LOCK_PASS",),
}


def _markers(payload: dict[str, Any]) -> set[str]:
    markers = {str(value) for value in payload.get("markers", [])}
    markers.update(key for key, value in payload.items() if key.isupper() and value is True)
    return markers


def verify_stage(
    stage: str,
    source_lock: str | Path,
    artifacts: list[str | Path],
    *,
    output_candidate: str | Path | None = None,
) -> dict[str, Any]:
    if stage not in STAGE_REQUIREMENTS:
        raise ValueError(f"unknown stage {stage!r}")
    lock_result = verify_lock(source_lock)
    lock_id = lock_result["source_lock_id"]
    observed: set[str] = set()
    artifact_paths: list[str] = []
    for artifact in artifacts:
        path = Path(artifact).resolve()
        if not path.is_file():
            raise RuntimeError(f"missing evidence: required stage artifact does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source_lock_id") != lock_id:
            raise RuntimeError(f"source mismatch: stage artifact was not produced under current lock: {path}")
        observed.update(_markers(payload))
        artifact_paths.append(str(path))
    required = set(STAGE_REQUIREMENTS[stage])
    missing = sorted(required - observed)
    if missing:
        raise RuntimeError(f"missing evidence: stage {stage} requires markers {missing}")
    if output_candidate is not None and Path(output_candidate).expanduser().exists():
        raise FileExistsError(f"refusing existing output candidate before stage execution: {output_candidate}")
    return {
        "stage_gate_v2_pass": True,
        "stage": stage,
        "source_lock_id": lock_id,
        "required_markers": sorted(required),
        "observed_markers": sorted(observed),
        "artifacts": artifact_paths,
        "output_candidate": str(Path(output_candidate).expanduser().resolve()) if output_candidate else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGE_REQUIREMENTS), required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--output-candidate")
    args = parser.parse_args()
    result = verify_stage(
        args.stage,
        args.source_lock,
        args.artifact,
        output_candidate=args.output_candidate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
