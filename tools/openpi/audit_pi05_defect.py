#!/usr/bin/env python3
"""Apply the predeclared offline defect-validity gate on calibration rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from methods.variable_time_latentloop.budget_calibration import MonotonicBinnedCalibrator
from methods.variable_time_latentloop.defect import evaluate_defect_validity


def main() -> None:
    raise RuntimeError(
        "DISABLED_LABEL_LEAKING_DEFECT_V1: use defect_fit.py and defect_validate.py with a frozen "
        "episode-disjoint defect_data_split_contract.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-auroc", type=float, default=0.70)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = list(csv.DictReader(Path(args.offline_metrics).open(encoding="utf-8")))
    defect = np.asarray([float(row["latent_defect"]) for row in rows])
    error = np.asarray([float(row["sequential_executed_mse"]) for row in rows])
    baselines = {
        "age_only": np.asarray([float(row["delta_q"]) for row in rows]),
        "previous_action_translation_magnitude": np.asarray(
            [float(row["executed_action_magnitude"]) for row in rows]
        ),
        "observation_change_norm": np.asarray(
            [float(row["observation_change_norm"]) for row in rows]
        ),
    }
    result = evaluate_defect_validity(
        defect,
        error,
        baselines,
        minimum_auroc=args.minimum_auroc,
    )
    calibrator = MonotonicBinnedCalibrator.fit(defect, error)
    predicted = calibrator.predict(defect)
    payload = {
        **result.to_dict(),
        "rows": len(rows),
        "calibration_mae": float(np.mean(np.abs(predicted - error))),
        "calibration_rmse": float(np.sqrt(np.mean((predicted - error) ** 2))),
        "defect_gate_pass": result.passed,
        "dynamic_online_evaluation_enabled": result.passed,
        "source_metrics": str(Path(args.offline_metrics).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
