#!/usr/bin/env python3
"""Fit monotonic defect-to-error calibration on the fit role only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from defect_split_common import load_contract, load_role_rows, sha256_file, verify_offline_summary
from methods.variable_time_latentloop.budget_calibration import MonotonicBinnedCalibrator
from pi05_stage_gate_v2 import verify_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-metrics", required=True)
    parser.add_argument("--offline-summary", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--previous-stage-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise RuntimeError("defect fitting is disabled unless --run is explicitly supplied")
    output = Path(args.output).resolve()
    lock = verify_stage(
        "stage9_defect_fit",
        args.source_lock,
        [args.previous_stage_gate],
        output_candidate=output,
    )
    _, split_payload = load_contract(args.split_contract)
    if split_payload.get("source_lock_id") != lock["source_lock_id"]:
        raise RuntimeError("defect split contract was frozen under another source lock")
    producer = verify_offline_summary(
        args.offline_summary,
        metrics_path=args.offline_metrics,
        contract_path=args.split_contract,
        role="defect_fit",
        source_lock_id=lock["source_lock_id"],
    )
    rows = load_role_rows(args.offline_metrics, args.split_contract, "defect_fit")
    defect = np.asarray([float(row["latent_defect"]) for row in rows])
    error = np.asarray([float(row["sequential_executed_mse"]) for row in rows])
    direct_error = np.asarray([float(row["direct_executed_mse"]) for row in rows])
    if len(rows) < 10 or not np.isfinite(defect).all() or not np.isfinite(error).all():
        raise ValueError("defect fit requires at least ten finite rows")
    calibrator = MonotonicBinnedCalibrator.fit(defect, error, bins=args.bins)
    direct_calibrator = MonotonicBinnedCalibrator.fit(defect, direct_error, bins=args.bins)
    payload = {
        "schema_version": 2,
        "DEFECT_FIT_PASS": True,
        "markers": ["DEFECT_FIT_PASS"],
        "source_lock_id": lock["source_lock_id"],
        "role": "defect_fit",
        "rows": len(rows),
        "calibrator": asdict(calibrator),
        "sequential_error_calibrator": asdict(calibrator),
        "direct_error_calibrator": asdict(direct_calibrator),
        "high_error_threshold_from_fit": float(np.quantile(error, 0.9)),
        "metrics_sha256": sha256_file(args.offline_metrics),
        "split_contract_sha256": sha256_file(args.split_contract),
        "defect_split_contract_id": split_payload["defect_split_contract_id"],
        "offline_summary": str(Path(args.offline_summary).resolve()),
        "offline_summary_sha256": sha256_file(args.offline_summary),
        "producer": {
            key: producer[key]
            for key in (
                "adapter_checkpoint",
                "adapter_checkpoint_sha256",
                "base_checkpoint",
                "base_checkpoint_sha256",
                "cache_manifest",
                "cache_manifest_id",
                "cache_manifest_sha256",
                "final_evaluation_manifest",
                "final_evaluation_manifest_sha256",
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
