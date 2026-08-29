#!/usr/bin/env python3
"""Evaluate a frozen defect calibrator on the validity role only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from defect_split_common import load_contract, load_role_rows, sha256_file, verify_offline_summary
from methods.variable_time_latentloop.budget_calibration import MonotonicBinnedCalibrator
from methods.variable_time_latentloop.defect import evaluate_defect_validity
from pi05_stage_gate_v2 import verify_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-metrics", required=True)
    parser.add_argument("--offline-summary", required=True)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-auroc", type=float, default=0.70)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise RuntimeError("defect validity evaluation is disabled unless --run is explicitly supplied")
    output = Path(args.output).resolve()
    lock = verify_stage(
        "stage10_defect_validity",
        args.source_lock,
        [args.fit],
        output_candidate=output,
    )
    fit = json.loads(Path(args.fit).read_text(encoding="utf-8"))
    _, split_payload = load_contract(args.split_contract)
    if split_payload.get("source_lock_id") != lock["source_lock_id"]:
        raise RuntimeError("defect split contract was frozen under another source lock")
    if (
        fit.get("DEFECT_FIT_PASS") is not True
        or fit.get("source_lock_id") != lock["source_lock_id"]
        or fit.get("role") != "defect_fit"
        or fit.get("split_contract_sha256") != sha256_file(args.split_contract)
        or fit.get("defect_split_contract_id") != split_payload["defect_split_contract_id"]
    ):
        raise RuntimeError("defect fit artifact is missing or stale")
    producer = verify_offline_summary(
        args.offline_summary,
        metrics_path=args.offline_metrics,
        contract_path=args.split_contract,
        role="defect_validity",
        source_lock_id=lock["source_lock_id"],
    )
    producer_keys = (
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
    if fit.get("producer") != {key: producer[key] for key in producer_keys}:
        raise RuntimeError("defect fit and validity metrics came from different producers")
    rows = load_role_rows(args.offline_metrics, args.split_contract, "defect_validity")
    defect = np.asarray([float(row["latent_defect"]) for row in rows])
    error = np.asarray([float(row["sequential_executed_mse"]) for row in rows])
    direct_error = np.asarray([float(row["direct_executed_mse"]) for row in rows])
    baselines = {
        "age_only": np.asarray([float(row["delta_q"]) for row in rows]),
        "previous_action_translation_magnitude": np.asarray(
            [float(row["executed_action_magnitude"]) for row in rows]
        ),
        "observation_change_norm": np.asarray([float(row["observation_change_norm"]) for row in rows]),
    }
    result = evaluate_defect_validity(
        defect,
        error,
        baselines,
        minimum_auroc=args.minimum_auroc,
        high_error_threshold=float(fit["high_error_threshold_from_fit"]),
    )
    calibrator = MonotonicBinnedCalibrator.from_dict(fit["calibrator"])
    predicted = calibrator.predict(defect)
    direct_calibrator = MonotonicBinnedCalibrator.from_dict(fit["direct_error_calibrator"])
    predicted_direct = direct_calibrator.predict(defect)
    payload = {
        "schema_version": 2,
        "DEFECT_VALIDITY_PASS": bool(result.passed),
        "markers": ["DEFECT_VALIDITY_PASS"] if result.passed else [],
        "source_lock_id": lock["source_lock_id"],
        "role": "defect_validity",
        "rows": len(rows),
        **result.to_dict(),
        "calibration_mae": float(np.mean(np.abs(predicted - error))),
        "calibration_rmse": float(np.sqrt(np.mean((predicted - error) ** 2))),
        "direct_calibration_mae": float(np.mean(np.abs(predicted_direct - direct_error))),
        "direct_calibration_rmse": float(
            np.sqrt(np.mean((predicted_direct - direct_error) ** 2))
        ),
        "fit_sha256": sha256_file(args.fit),
        "metrics_sha256": sha256_file(args.offline_metrics),
        "split_contract_sha256": sha256_file(args.split_contract),
        "defect_split_contract_id": split_payload["defect_split_contract_id"],
        "offline_summary": str(Path(args.offline_summary).resolve()),
        "offline_summary_sha256": sha256_file(args.offline_summary),
        "fit_role": fit["role"],
        "producer": {key: producer[key] for key in producer_keys},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    if not payload["DEFECT_VALIDITY_PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
