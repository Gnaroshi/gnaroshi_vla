"""Predeclared stop and scientific gates for exact-q2 regeneration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


OFFLINE_THRESHOLDS = {
    "paired_ci_level": 0.95,
    "old_reference_noninferiority_fraction": 0.05,
    "p99_multiplier_over_better_reference": 1.25,
    "maximum_q2_to_q1_mean_error_ratio": 2.0,
}


def online_native_r5_enabled(gate: Mapping[str, Any] | None) -> bool:
    """Keep the 200-episode online matrix default-off until the full offline gate passes."""

    if not gate:
        return False
    return bool(gate.get("EXACT_Q2_OFFLINE_PASS")) and gate.get("verdict") in {
        "RECURRENT_EXACT_Q2_PASS",
        "DIRECT_EXACT_Q2_PASS",
        "BOTH_PASS",
    }


def evaluate_short_budget_gate(
    candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Decide whether each 37,500-step candidate may continue to full budget."""

    results: dict[str, Any] = {}
    required_steps = {12_500, 25_000, 37_500}
    for candidate in ("recurrent_exact_q2", "direct_exact_q2"):
        rows = sorted(candidate_rows.get(candidate, ()), key=lambda row: int(row["step"]))
        if not rows:
            results[candidate] = {"pass": False, "checks": {"validation_present": False}}
            continue
        prefix = [float(row["metrics"]["q2_prefix_l1"]["mean"]) for row in rows]
        hold = float(rows[-1]["references"]["hold_stale_condition"]["q2_prefix_l1"]["mean"])
        observed_steps = {int(row["step"]) for row in rows}
        monotonic = len(prefix) >= 3 and all(
            right <= left for left, right in zip(prefix, prefix[1:])
        )
        final = rows[-1]
        checks = {
            "validation_present": True,
            "required_short_checkpoints_present": required_steps.issubset(observed_steps),
            "final_step_is_37500": int(final["step"]) == 37_500,
            "below_hold_or_monotonic_toward_it": prefix[-1] < hold or monotonic,
            "gripper_noncollapsed": bool(final["metrics"]["gripper_noncollapsed"]),
            "finite": bool(final["metrics"]["finite"]),
            "p99_not_exploding": float(final["metrics"]["q2_prefix_l1"]["p99"])
            <= OFFLINE_THRESHOLDS["p99_multiplier_over_better_reference"]
            * float(final["references"]["hold_stale_condition"]["q2_prefix_l1"]["p99"]),
            "no_obvious_validation_divergence": len(prefix) < 2 or prefix[-1] <= max(prefix[:-1]),
        }
        results[candidate] = {
            "pass": all(checks.values()),
            "checks": checks,
            "prefix_means_by_step": [
                {"step": int(row["step"]), "mean": value}
                for row, value in zip(rows, prefix)
            ],
            "hold_mean": hold,
        }
    passing = [name for name, row in results.items() if row["pass"]]
    if not passing:
        verdict = "EARLY_STOP_BOTH"
        stop_rule = "STOP_SIMVLA_CONDITION_REGENERATION"
    else:
        verdict = "SHORT_BUDGET_PASS"
        stop_rule = None
    return {
        "schema_version": "simvla_exact_q2_short_gate_v1",
        "verdict": verdict,
        "candidates_allowed_full_training": passing,
        "candidates": results,
        "stop_rule": stop_rule,
        "thresholds": dict(OFFLINE_THRESHOLDS),
        "required_validation_steps": sorted(required_steps),
    }


def _candidate_offline_checks(row: Mapping[str, Any]) -> dict[str, bool]:
    old_mean = float(row["references"]["old_observation_only"]["q2_prefix_l1"]["mean"])
    hold_p99 = float(row["references"]["hold_stale_condition"]["q2_prefix_l1"]["p99"])
    old_p99 = float(row["references"]["old_observation_only"]["q2_prefix_l1"]["p99"])
    return {
        "cache_integrity": bool(row["prerequisites"]["cache_integrity"]),
        "same_noise": bool(row["prerequisites"]["same_noise"]),
        "selected_by_validation_only": bool(row["prerequisites"]["selected_by_validation_only"]),
        "better_than_hold_paired95": float(row["paired_ci95"]["candidate_minus_hold"][1]) < 0.0,
        "noninferior_to_old_observation_paired95": float(
            row["paired_ci95"]["candidate_minus_old_observation"][1]
        )
        < OFFLINE_THRESHOLDS["old_reference_noninferiority_fraction"] * old_mean,
        "p99_tail_bounded": float(row["metrics"]["q2_prefix_l1"]["p99"])
        <= OFFLINE_THRESHOLDS["p99_multiplier_over_better_reference"]
        * min(hold_p99, old_p99),
        "gripper_finite_noncollapsed": bool(row["metrics"]["gripper_noncollapsed"])
        and bool(row["metrics"]["finite"]),
        "q2_no_uncontrolled_q1_explosion": float(row["metrics"]["q2_prefix_l1"]["mean"])
        <= OFFLINE_THRESHOLDS["maximum_q2_to_q1_mean_error_ratio"]
        * max(float(row["metrics"]["q1_prefix_l1"]["mean"]), 1e-12),
    }


def evaluate_exact_q2_offline_gate(
    candidate_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the immutable paired-validation gate to both exact-q2 candidates."""

    results: dict[str, Any] = {}
    for candidate in ("recurrent_exact_q2", "direct_exact_q2"):
        row = candidate_rows.get(candidate)
        if row is None:
            results[candidate] = {"pass": False, "checks": {"result_present": False}}
            continue
        checks = {"result_present": True, **_candidate_offline_checks(row)}
        results[candidate] = {"pass": all(checks.values()), "checks": checks}
    recurrent = results["recurrent_exact_q2"]["pass"]
    direct = results["direct_exact_q2"]["pass"]
    if recurrent and direct:
        verdict = "BOTH_PASS"
    elif recurrent:
        verdict = "RECURRENT_EXACT_Q2_PASS"
    elif direct:
        verdict = "DIRECT_EXACT_Q2_PASS"
    elif all(candidate in candidate_rows for candidate in results):
        verdict = "BOTH_FAIL"
    else:
        verdict = "INCONCLUSIVE"
    stop = verdict in {"BOTH_FAIL", "INCONCLUSIVE"}
    return {
        "schema_version": "simvla_exact_q2_offline_gate_v1",
        "verdict": verdict,
        "EXACT_Q2_OFFLINE_PASS": verdict in {
            "BOTH_PASS",
            "RECURRENT_EXACT_Q2_PASS",
            "DIRECT_EXACT_Q2_PASS",
        },
        "ONLINE_NATIVE_R5_ALLOWED": not stop,
        "stop_rule": "STOP_SIMVLA_CONDITION_REGENERATION" if stop else None,
        "candidates": results,
        "thresholds": dict(OFFLINE_THRESHOLDS),
    }
