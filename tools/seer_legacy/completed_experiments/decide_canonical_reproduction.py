#!/usr/bin/env python3
"""Apply the frozen canonical teacher33 reproduction decision rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    baseline = analysis["baseline"]
    v0 = analysis["v0"]
    b_success = int(baseline["successes"])
    v_success = int(v0["successes"])
    b_total = int(baseline["episodes"])
    v_total = int(v0["episodes"])
    if (b_total, v_total) != (200, 200):
        raise ValueError("canonical decision requires exactly 200 episodes per row")
    b_sr = b_success / b_total
    v_sr = v_success / v_total
    reduction = float(v0["full_query_reduction"])
    paired_ci_low_pp = float(analysis["paired_task_hierarchical_bootstrap_ci_pp"]["low"])
    tasks_regressing = int(analysis["task_regression_count"])
    if b_success == 166 and v_success == 182:
        verdict = "EXACT_CANONICAL_REPRODUCTION"
    elif (
        0.80 <= b_sr <= 0.86
        and 0.88 <= v_sr <= 0.94
        and (v_sr - b_sr) >= 0.04
        and paired_ci_low_pp > -1.0
        and 0.74 <= reduction <= 0.76
        and tasks_regressing <= 2
    ):
        verdict = "ACCEPTABLE_CANONICAL_REPRODUCTION"
    else:
        verdict = "REPRODUCTION_FAILED"
    payload = {
        "schema_version": 1,
        "verdict": verdict,
        "v1_training_enabled": verdict in {
            "EXACT_CANONICAL_REPRODUCTION", "ACCEPTABLE_CANONICAL_REPRODUCTION"
        },
        "full_k1": {"successes": b_success, "episodes": b_total, "sr": b_sr},
        "v0_k4": {
            "successes": v_success,
            "episodes": v_total,
            "sr": v_sr,
            "full_query_reduction": reduction,
        },
        "paired_ci_low_pp": paired_ci_low_pp,
        "tasks_regressing_over_20pp": tasks_regressing,
        "canonical_episode_outcomes_identical": bool(analysis.get("canonical_episode_outcomes_identical", False)),
        "analysis_path": str(args.analysis.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(verdict)


if __name__ == "__main__":
    main()
