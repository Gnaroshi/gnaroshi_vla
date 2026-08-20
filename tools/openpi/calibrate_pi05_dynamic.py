#!/usr/bin/env python3
"""Freeze V2 thresholds on the disjoint cache calibration split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _common import require_gate
from methods.variable_time_latentloop.budget_calibration import BudgetCalibrator


def main() -> None:
    raise RuntimeError(
        "DISABLED_BUDGET_CALIBRATION_V1: use simulate_dynamic_budget_v2.py on the disjoint "
        "scheduler-calibration split"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-metrics", required=True)
    parser.add_argument("--defect-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-k-q", type=float, default=4.0)
    args = parser.parse_args()
    require_gate(args.defect_gate, "defect_gate_pass")
    if args.target_k_q != 4.0:
        raise ValueError("the predeclared first V2 target is K_q=4")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = list(csv.DictReader(Path(args.offline_metrics).open(encoding="utf-8")))
    defect = np.asarray([float(row["latent_defect"]) for row in rows])
    sequential = np.asarray([float(row["sequential_executed_mse"]) for row in rows])
    direct = np.asarray([float(row["direct_executed_mse"]) for row in rows])
    calibration = BudgetCalibrator(target_full_prefix_ratio=1.0 / args.target_k_q).fit(
        defect, sequential, direct
    )
    observed_k_q = 1.0 / calibration.validation_full_prefix_ratio
    payload = {
        "schema_version": 1,
        "target_k_q": args.target_k_q,
        "target_k_a": 5 * args.target_k_q,
        "target_full_prefix_ratio": 1.0 / args.target_k_q,
        "calibration": calibration.to_dict(),
        "calibration_rows": len(rows),
        "actual_mean_k_q": observed_k_q,
        "actual_full_prefix_ratio": calibration.validation_full_prefix_ratio,
        "direct_reanchor_ratio": calibration.validation_direct_ratio,
        "dynamic_calibration_pass": bool(3.8 <= observed_k_q <= 4.2),
        "thresholds_frozen": True,
        "source_metrics": str(Path(args.offline_metrics).resolve()),
        "source_defect_gate": str(Path(args.defect_gate).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
