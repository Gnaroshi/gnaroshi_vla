#!/usr/bin/env python3
"""Build frozen-rule booleans from LatentLoop plan-continuation analyses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PRIMARY_OVERLAP_METRICS = (
    "translation_l1",
    "translation_l2",
    "rotation_l1",
    "rotation_l2",
)
ASSOCIATION_DRIVERS = (
    "primary_raw_change_l1",
    "wrist_raw_change_l1",
    "proprio_delta_l2",
    "u_delta_norm",
)
ASSOCIATION_RESPONSES = (
    "correction_translation_l2",
    "correction_rotation_l2",
    "correction_arm_l2",
    "correction_gripper_probability_abs",
)


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _strictly_better_overlap(comparisons: dict[str, Any]) -> tuple[bool, bool]:
    intervals = [comparisons[metric] for metric in PRIMARY_OVERLAP_METRICS]
    at_least_one_better = any(
        _finite(interval.get("ci_high")) and float(interval["ci_high"]) < 0.0
        for interval in intervals
    )
    any_reversal = any(
        _finite(interval.get("ci_low")) and float(interval["ci_low"]) > 0.0
        for interval in intervals
    )
    return at_least_one_better, any_reversal


def _sr_strictly_better(interval: dict[str, Any]) -> bool:
    return _finite(interval.get("ci_low")) and float(interval["ci_low"]) > 0.0


def _sr_indistinguishable(interval: dict[str, Any]) -> bool:
    return (
        _finite(interval.get("ci_low"))
        and _finite(interval.get("ci_high"))
        and float(interval["ci_low"]) <= 0.0 <= float(interval["ci_high"])
    )


def _sr_strictly_worse(interval: dict[str, Any]) -> bool:
    """Return whether dense LatentLoop is clearly worse than the comparator."""

    return _finite(interval.get("ci_high")) and float(interval["ci_high"]) < 0.0


def _matched_baseline_not_explanation(
    sr_interval: dict[str, Any], overlap: dict[str, Any]
) -> bool:
    if _sr_strictly_better(sr_interval):
        return True
    better, reversal = _strictly_better_overlap(overlap)
    return _sr_indistinguishable(sr_interval) and better and not reversal


def _matched_baseline_clearly_exceeds(
    sr_interval: dict[str, Any], overlap: dict[str, Any]
) -> bool:
    """Apply a conservative refutation test without inventing equivalence bounds.

    A confidence interval containing zero is inconclusive, not proof that two
    methods are equivalent. The comparator therefore refutes the mechanism
    only when it has clearly higher SR, or when SR is indistinguishable and it
    is strictly better on every predeclared primary overlap metric.
    """

    if _sr_strictly_worse(sr_interval):
        return True
    if not _sr_indistinguishable(sr_interval):
        return False
    intervals = [overlap[metric] for metric in PRIMARY_OVERLAP_METRICS]
    return bool(intervals) and all(
        _finite(interval.get("ci_low")) and float(interval["ci_low"]) > 0.0
        for interval in intervals
    )


def build_evidence(
    alignment: dict[str, Any], overlap: dict[str, Any], feedback: dict[str, Any]
) -> dict[str, Any]:
    """Apply only predeclared interval directions; do not fit thresholds."""

    dense = str(overlap["dense_row"])
    baseline = str(overlap["baseline_row"])
    dense_vs_k1 = overlap["paired_differences_vs_baseline"][dense]
    overlap_better, overlap_reversal = _strictly_better_overlap(dense_vs_k1)

    controls = ("k4_no_observation", "k4_time_shifted_observation")
    dense_correlations = feedback["rows"][dense]["correlations"]
    association_witnesses: list[dict[str, Any]] = []
    association_reversals: list[dict[str, Any]] = []
    for driver in ASSOCIATION_DRIVERS:
        for response in ASSOCIATION_RESPONSES:
            dense_interval = dense_correlations.get(driver, {}).get(response, {})
            control_intervals = [
                feedback["rows"].get(control, {})
                .get("correlations", {})
                .get(driver, {})
                .get(response, {})
                for control in controls
            ]
            values = [
                dense_interval.get("ci_low"),
                dense_interval.get("ci_high"),
                *(interval.get("ci_low") for interval in control_intervals),
                *(interval.get("ci_high") for interval in control_intervals),
            ]
            if not all(_finite(value) for value in values):
                continue
            dense_low = float(dense_interval["ci_low"])
            dense_high = float(dense_interval["ci_high"])
            control_high = max(float(item["ci_high"]) for item in control_intervals)
            control_low = min(float(item["ci_low"]) for item in control_intervals)
            if dense_low > 0.0 and dense_low > control_high:
                association_witnesses.append({"driver": driver, "response": response})
            if control_low > dense_high:
                association_reversals.append({"driver": driver, "response": response})

    sr_pairs = overlap["dense_pairwise_sr_differences"]
    overlap_pairs = overlap["dense_pairwise_overlap_differences"]
    no_observation = "k4_no_observation"
    replay = "k4_predicted_horizon_replay"
    action_baseline = "k4_action_space_correction"
    anchor_baseline = "k4_anchor_to_current"

    phase_a_required_rows = (no_observation, replay)
    missing_phase_a_rows = [
        row for row in phase_a_required_rows if row not in sr_pairs
    ]
    if missing_phase_a_rows:
        raise KeyError(
            "Phase A analysis is missing required intervention rows: "
            f"{missing_phase_a_rows}"
        )

    evidence = {
        "token_time_alignment_verified": bool(alignment.get("verified", False)),
        "k4_dense_reduces_primary_overlap_without_other_primary_degradation": (
            overlap_better and not overlap_reversal
        ),
        "dense_feedback_association_is_positive_and_stronger_than_controls": (
            bool(association_witnesses) and not association_reversals
        ),
        "k4_dense_sr_exceeds_no_observation_and_replay": (
            _sr_strictly_better(sr_pairs[no_observation])
            and _sr_strictly_better(sr_pairs[replay])
        ),
        "token_time_alignment_failed": not bool(alignment.get("verified", False)),
        "k4_dense_does_not_reduce_overlap_in_any_primary_metric": not overlap_better,
        "dense_corrections_are_not_aligned_with_current_observations": not bool(
            association_witnesses
        ),
        "mechanism_fails_fixed_checkpoint_36_38_robustness": False,
        "robustness_evaluated": False,
        "diagnostics": {
            "dense_row": dense,
            "baseline_row": baseline,
            "primary_overlap_witness": overlap_better,
            "primary_overlap_reversal": overlap_reversal,
            "association_witnesses": association_witnesses,
            "association_reversals": association_reversals,
            "matched_baseline_rows_available": [
                row
                for row in (action_baseline, anchor_baseline)
                if row in sr_pairs and row in overlap_pairs
            ],
            "matched_baseline_rows_missing": [
                row
                for row in (action_baseline, anchor_baseline)
                if row not in sr_pairs or row not in overlap_pairs
            ],
            "indistinguishable_is_not_equivalence": True,
            "robustness_refutation_must_be_updated_after_fixed_36_38_runs": True,
        },
    }

    if action_baseline in sr_pairs and action_baseline in overlap_pairs:
        evidence["matched_action_space_correction_does_not_fully_explain_result"] = (
            _matched_baseline_not_explanation(
                sr_pairs[action_baseline], overlap_pairs[action_baseline]
            )
        )
        evidence["matched_action_space_baseline_clearly_exceeds_latentloop"] = (
            _matched_baseline_clearly_exceeds(
                sr_pairs[action_baseline], overlap_pairs[action_baseline]
            )
        )
    if anchor_baseline in sr_pairs and anchor_baseline in overlap_pairs:
        evidence["nonrecurrent_anchor_baseline_does_not_match_or_exceed_latentloop"] = (
            _matched_baseline_not_explanation(
                sr_pairs[anchor_baseline], overlap_pairs[anchor_baseline]
            )
        )
        evidence["nonrecurrent_anchor_baseline_clearly_exceeds_latentloop"] = (
            _matched_baseline_clearly_exceeds(
                sr_pairs[anchor_baseline], overlap_pairs[anchor_baseline]
            )
        )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--consistency-summary", type=Path, required=True)
    parser.add_argument("--feedback-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    overlap = json.loads(args.consistency_summary.read_text(encoding="utf-8"))
    feedback = json.loads(args.feedback_summary.read_text(encoding="utf-8"))
    evidence = build_evidence(alignment, overlap, feedback)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
