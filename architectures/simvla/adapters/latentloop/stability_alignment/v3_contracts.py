"""Contracts for recurrence-focused stability alignment V3."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


V3_SCHEMA = "simvla_condition_stability_alignment_v3"
V3_LOSS_SCHEMA = "simvla_stability_v3_gradient_weights_v3"
V3_HARD_POOL_SCHEMA = "simvla_stability_v3_hard_pool_v1"
V3_CHECKPOINT_SCHEMA = "simvla_stability_alignment_checkpoint_v3"

V3_LOSS_NAMES = (
    "recurrence_gain",
    "exact_condition_reference",
    "frozen_teacher_preservation",
    "frozen_ng3_execution",
    "gripper_transition",
    "rotating_full_nfe_execution",
    "hard_sequence_execution",
)

# The exact-reference and frozen-teacher terms split the requested combined 20%
# equally. They are also reported as one combined gradient in conflict audits.
V3_GRADIENT_TARGETS = {
    "recurrence_gain": 0.35,
    "exact_condition_reference": 0.10,
    "frozen_teacher_preservation": 0.10,
    "frozen_ng3_execution": 0.25,
    "gripper_transition": 0.10,
    "rotating_full_nfe_execution": 0.05,
    "hard_sequence_execution": 0.05,
}

V3_STAGE_ORDER = (
    "V0_AUDIT",
    "V1_CHECKPOINT_SWEEP",
    "V2_V2_GRADIENT_AUDIT",
    "V3_HARD_POOL_AND_GAMMA",
    "V4_V3_GRADIENT_CALIBRATION",
    "V5_R50_500_BOUNDED_PILOT",
    "V6_R50_500_SAFETY_GATE",
    "V7_R50_2K",
    "V8_R50_2K_GATE",
    "V9_R150_2K_CONDITIONAL",
    "V10_R50_5K",
    "V11_R50_5K_GATE",
    "V12_R50_10K_CONDITIONAL",
    "V13_FINAL_OFFLINE_GATE",
    "V14_EXPORT_GATE_PASSING_ONLY",
)

V3_GAMMA_QUANTILE = 0.50
V3_AGE_WEIGHTS = {2: 1.0, 3: 2.0}


@dataclass(frozen=True)
class V3GateResult:
    verdict: str
    passed: bool
    checks: dict[str, bool]
    measurements: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_positive(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) > 1e-12


def true_count_reduction(parent_count: int, candidate_count: int) -> float | None:
    """Return a true relative reduction, or None for a zero parent count."""

    parent = int(parent_count)
    candidate = int(candidate_count)
    if parent < 0 or candidate < 0:
        raise ValueError("sequence counts must be non-negative")
    if parent == 0:
        return None
    return (parent - candidate) / parent


def sequence_reduction_passes(
    parent_count: int,
    candidate_count: int,
    *,
    minimum_reduction: float,
) -> bool:
    reduction = true_count_reduction(parent_count, candidate_count)
    if reduction is None:
        return int(candidate_count) == 0
    return reduction >= float(minimum_reduction)


def evaluate_v3_scientific_gate(metrics: Mapping[str, Any]) -> V3GateResult:
    sequence_reduction = true_count_reduction(
        int(metrics["parent_age3_mismatch_sequences"]),
        int(metrics["candidate_age3_mismatch_sequences"]),
    )
    checks = {
        "age2_recurrence_mean_improved_20pct": float(
            metrics["age2_recurrence_improvement"]
        )
        >= 0.20,
        "age3_recurrence_mean_improved_30pct": float(
            metrics["age3_recurrence_improvement"]
        )
        >= 0.30,
        "age3_first_r_p95_improved": float(metrics["age3_first_r_p95_ratio"])
        < 1.0,
        "age1_first_r_fidelity_within_5pct": float(
            metrics["age1_first_r_ratio_to_parent"]
        )
        <= 1.05,
        "exact_ng3_fidelity_within_5pct": float(
            metrics["exact_ng3_ratio_to_parent"]
        )
        <= 1.05,
        "age3_mismatch_sequences_reduced_25pct": sequence_reduction_passes(
            int(metrics["parent_age3_mismatch_sequences"]),
            int(metrics["candidate_age3_mismatch_sequences"]),
            minimum_reduction=0.25,
        ),
        "switch_mismatch_sequences_not_increased": int(
            metrics["candidate_age3_switch_mismatch_sequences"]
        )
        <= int(metrics["parent_age3_switch_mismatch_sequences"]),
        "no_p99_collapse": float(metrics["age3_first_r_p99_ratio"]) <= 1.25,
        "original_simvla_frozen": bool(metrics["original_simvla_frozen"]),
    }
    passed = all(checks.values())
    measurements = dict(metrics)
    measurements["age3_mismatch_sequence_true_relative_reduction"] = sequence_reduction
    return V3GateResult(
        verdict="STABILITY_V3_GATE_PASS" if passed else "STABILITY_V3_GATE_FAIL",
        passed=passed,
        checks=checks,
        measurements=measurements,
    )


def evaluate_v3_stage_gate(
    metrics: Mapping[str, Any],
    *,
    optimizer_step: int,
    previous_metrics: Mapping[str, Any] | None = None,
) -> V3GateResult:
    """Apply bounded continuation gates before the strict 10K scientific gate."""

    step = int(optimizer_step)
    if step not in {500, 2_000, 5_000, 10_000}:
        raise ValueError("V3 stage gate is defined only at 500, 2K, 5K, and 10K")
    if step == 10_000:
        return evaluate_v3_scientific_gate(metrics)

    parent_mismatch = int(metrics["parent_age3_mismatch_sequences"])
    candidate_mismatch = int(metrics["candidate_age3_mismatch_sequences"])
    parent_switch = int(metrics["parent_age3_switch_mismatch_sequences"])
    candidate_switch = int(metrics["candidate_age3_switch_mismatch_sequences"])
    minimum_age2 = 0.0 if step in {500, 2_000} else 0.10
    minimum_age3 = 0.0 if step in {500, 2_000} else 0.15
    checks = {
        "age2_recurrence_not_regressing": float(
            metrics["age2_recurrence_improvement"]
        )
        >= minimum_age2,
        "age3_recurrence_not_regressing": float(
            metrics["age3_recurrence_improvement"]
        )
        >= minimum_age3,
        "age3_first_r_p95_not_worse": float(metrics["age3_first_r_p95_ratio"])
        <= 1.0,
        "age1_first_r_fidelity_within_5pct": float(
            metrics["age1_first_r_ratio_to_parent"]
        )
        <= 1.05,
        "exact_ng3_fidelity_within_5pct": float(
            metrics["exact_ng3_ratio_to_parent"]
        )
        <= 1.05,
        "age3_mismatch_sequences_not_increased": candidate_mismatch
        <= parent_mismatch,
        "switch_mismatch_sequences_not_increased": candidate_switch
        <= parent_switch,
        "no_p99_collapse": float(metrics["age3_first_r_p99_ratio"]) <= 1.25,
        "original_simvla_frozen": bool(metrics["original_simvla_frozen"]),
    }
    if step == 5_000:
        if previous_metrics is None:
            raise ValueError("5K continuation requires the immutable 2K metrics")
        checks.update(
            {
                "age2_continued_improvement": float(
                    metrics["age2_recurrence_improvement"]
                )
                > float(previous_metrics["age2_recurrence_improvement"]),
                "age3_continued_improvement": float(
                    metrics["age3_recurrence_improvement"]
                )
                > float(previous_metrics["age3_recurrence_improvement"]),
                "tail_continued_improvement": float(
                    metrics["age3_first_r_p95_ratio"]
                )
                < float(previous_metrics["age3_first_r_p95_ratio"]),
            }
        )
    passed = all(checks.values())
    step_label = "500" if step == 500 else f"{step // 1000}K"
    return V3GateResult(
        verdict=(
            f"STABILITY_V3_{step_label}_CONTINUE"
            if passed
            else f"STABILITY_V3_{step_label}_STOP"
        ),
        passed=passed,
        checks=checks,
        measurements=dict(metrics),
    )


def select_v2_checkpoint(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select only on checkpoint-validation using the declared priority order."""

    candidates = [
        dict(row)
        for row in rows
        if str(row.get("split")) == "checkpoint_validation"
        and float(row["age1_first_r_ratio_to_parent"]) <= 1.05
    ]
    if not candidates:
        return {
            "verdict": "NO_V2_CHECKPOINT_PRESERVES_AGE1",
            "selected_step": None,
            "selected_checkpoint": None,
        }
    selected = min(
        candidates,
        key=lambda row: (
            int(row["candidate_age3_mismatch_sequences"]),
            float(row["candidate_age3_first_r_p95"]),
            float(row["candidate_age3_recurrence_excess_mean"]),
            int(row["optimizer_step"]),
        ),
    )
    return {
        "verdict": "V2_VALIDATION_CHECKPOINT_SELECTED",
        "selected_step": int(selected["optimizer_step"]),
        "selected_checkpoint": str(selected["checkpoint"]),
        "selection_key": {
            "candidate_age3_mismatch_sequences": int(
                selected["candidate_age3_mismatch_sequences"]
            ),
            "candidate_age3_first_r_p95": float(
                selected["candidate_age3_first_r_p95"]
            ),
            "candidate_age3_recurrence_excess_mean": float(
                selected["candidate_age3_recurrence_excess_mean"]
            ),
            "age1_first_r_ratio_to_parent": float(
                selected["age1_first_r_ratio_to_parent"]
            ),
        },
    }


def calibrate_v3_gradient_weights(
    gradient_norms: Mapping[str, float],
    *,
    recurrence_cosine_with_ng3: float | None = None,
    recurrence_cosine_with_reference_group: float | None = None,
    recurrence_cosines_with_dominant_losses: Mapping[str, float] | None = None,
    weighted_total_cosines_with_protected_losses: Mapping[str, float] | None = None,
    gradient_budget: float = 1.0,
) -> dict[str, Any]:
    if set(gradient_norms) != set(V3_LOSS_NAMES):
        raise ValueError("V3 gradient norms changed loss names")
    if not math.isclose(sum(V3_GRADIENT_TARGETS.values()), 1.0, abs_tol=1e-12):
        raise AssertionError("V3 contribution targets must sum to one")
    invalid = {
        name: float(gradient_norms[name])
        for name in V3_LOSS_NAMES
        if not _finite_positive(float(gradient_norms[name]))
    }
    if invalid:
        raise ValueError(f"V3 calibration needs nonzero finite gradients: {invalid}")
    if recurrence_cosines_with_dominant_losses is None:
        if (
            recurrence_cosine_with_ng3 is None
            or recurrence_cosine_with_reference_group is None
        ):
            raise ValueError("recurrence conflict audit is incomplete")
        conflicts = {
            "recurrence_vs_frozen_ng3": float(recurrence_cosine_with_ng3),
            "recurrence_vs_exact_teacher_group": float(
                recurrence_cosine_with_reference_group
            ),
        }
    else:
        conflicts = {
            str(name): float(value)
            for name, value in recurrence_cosines_with_dominant_losses.items()
        }
        if not conflicts:
            raise ValueError("recurrence conflict audit has no dominant losses")
    nonfinite_conflicts = {
        name: value for name, value in conflicts.items() if not math.isfinite(value)
    }
    if nonfinite_conflicts:
        raise ValueError(
            f"recurrence conflict audit has non-finite cosine: {nonfinite_conflicts}"
        )
    pairwise_warnings = {
        name: value for name, value in conflicts.items() if value < -0.30
    }
    raw_weights = {
        name: V3_GRADIENT_TARGETS[name] / float(gradient_norms[name])
        for name in V3_LOSS_NAMES
    }
    weighted_norms = {
        name: raw_weights[name] * float(gradient_norms[name])
        for name in V3_LOSS_NAMES
    }
    total = sum(weighted_norms.values())
    scale = min(1.0, float(gradient_budget) / max(total, 1e-12))
    weights = {name: value * scale for name, value in raw_weights.items()}
    scaled = {
        name: weights[name] * float(gradient_norms[name])
        for name in V3_LOSS_NAMES
    }
    scaled_total = sum(scaled.values())
    shares = {name: value / scaled_total for name, value in scaled.items()}
    total_cosines = {
        str(name): float(value)
        for name, value in (weighted_total_cosines_with_protected_losses or {}).items()
    }
    nonfinite_total_cosines = {
        name: value for name, value in total_cosines.items() if not math.isfinite(value)
    }
    if nonfinite_total_cosines:
        raise ValueError(
            "weighted-total gradient audit has non-finite cosine: "
            f"{nonfinite_total_cosines}"
        )
    total_alignment_warnings = {
        name: value for name, value in total_cosines.items() if value <= 0.0
    }
    checks = {
        "recurrence_share_at_least_25pct": shares["recurrence_gain"] >= 0.25,
        "no_nonprimary_share_above_30pct": all(
            value <= 0.30
            for name, value in shares.items()
            if name != "recurrence_gain"
        ),
        "weighted_total_alignment_audit_complete": bool(total_cosines),
    }
    approved = all(checks.values())
    verdict = "STABILITY_V3_WEIGHTS_BLOCKED"
    if approved:
        verdict = (
            "STABILITY_V3_BOUNDED_PILOT_APPROVED_WITH_ALIGNMENT_WARNING"
            if pairwise_warnings or total_alignment_warnings
            else "STABILITY_V3_WEIGHTS_APPROVED"
        )
    return {
        "schema_version": V3_LOSS_SCHEMA,
        "verdict": verdict,
        "approved": approved,
        "approved_for_bounded_pilot": approved,
        "approval_scope": "500_step_then_2k_gated_pilot",
        "checks": checks,
        "conflicts": conflicts,
        "severe_conflicts": pairwise_warnings,
        "pairwise_conflict_warnings": pairwise_warnings,
        "pairwise_warning_threshold": -0.30,
        "pairwise_threshold_is_heuristic": True,
        "weighted_total_cosines_with_protected_losses": total_cosines,
        "weighted_total_alignment_warnings": total_alignment_warnings,
        "alignment_warnings_are_decided_by_500_step_gate": True,
        "gradient_norms": {name: float(gradient_norms[name]) for name in V3_LOSS_NAMES},
        "contribution_targets": dict(V3_GRADIENT_TARGETS),
        "weights": weights,
        "weighted_gradient_norms": scaled,
        "weighted_gradient_shares": shares,
        "gradient_budget": float(gradient_budget),
        "global_weight_scale": float(scale),
        "weights_frozen_before_optimizer_step_zero": True,
        "parent_preservation_contribution": 0.0,
    }


def evaluate_v3_moving_window(
    weighted_gradient_norm_rows: Sequence[Mapping[str, float]],
    *,
    clipping_flags: Sequence[bool],
) -> dict[str, Any]:
    if not weighted_gradient_norm_rows:
        raise ValueError("V3 moving-window audit has no gradient rows")
    if any(set(row) != set(V3_LOSS_NAMES) for row in weighted_gradient_norm_rows):
        raise ValueError("V3 moving-window loss names changed")
    means = {
        name: sum(float(row[name]) for row in weighted_gradient_norm_rows)
        / len(weighted_gradient_norm_rows)
        for name in V3_LOSS_NAMES
    }
    total = sum(means.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("V3 moving-window gradient norm is not positive and finite")
    shares = {name: value / total for name, value in means.items()}
    clipping_fraction = (
        sum(bool(value) for value in clipping_flags) / len(clipping_flags)
        if clipping_flags
        else 0.0
    )
    checks = {
        "recurrence_share_at_least_25pct": shares["recurrence_gain"] >= 0.25,
        "no_nonprimary_share_above_30pct": all(
            value <= 0.30
            for name, value in shares.items()
            if name != "recurrence_gain"
        ),
        "gradient_clipping_fraction_below_25pct": clipping_fraction < 0.25,
    }
    target_balance_passed = (
        checks["recurrence_share_at_least_25pct"]
        and checks["no_nonprimary_share_above_30pct"]
    )
    hard_safety_passed = checks["gradient_clipping_fraction_below_25pct"]
    passed = target_balance_passed and hard_safety_passed
    return {
        "verdict": (
            "STABILITY_V3_MOVING_WINDOW_PASS"
            if passed
            else "STABILITY_V3_MOVING_WINDOW_FAIL"
        ),
        "passed": passed,
        "target_balance_passed": target_balance_passed,
        "hard_safety_passed": hard_safety_passed,
        "target_balance_is_diagnostic": True,
        "continuation_requires_offline_multimetric_gate": True,
        "checks": checks,
        "weighted_gradient_norm_means": means,
        "weighted_gradient_shares": shares,
        "gradient_rows": len(weighted_gradient_norm_rows),
        "clipping_observations": len(clipping_flags),
        "clipping_fraction": clipping_fraction,
    }


def v3_stage_graph() -> dict[str, Any]:
    return {
        "schema_version": "simvla_stability_v3_stage_graph_v2",
        "stage_order": list(V3_STAGE_ORDER),
        "training": {
            "R50": [
                "0_to_500",
                "safety_gate",
                "500_to_2k",
                "gate",
                "2k_to_5k",
                "gate",
                "5k_to_10k_conditional",
            ],
            "R150": "conditional two-GPU control only when no experiment is displaced",
            "scheduler_horizon": 30_000,
            "automatic_30k_continuation": False,
        },
        "calibration": {
            "pairwise_cosine_below_minus_0_30": "warning_only",
            "nonpositive_weighted_total_alignment": "warning_only",
            "approval": "finite calibrated gradients and declared contribution shares",
            "decision": "500-step multi-metric safety gate",
            "scope": "bounded 500-step pilot followed by measured gates",
        },
        "moving_window": {
            "gradient_share_targets": "diagnostic_warning",
            "clipping_fraction": "hard_numerical_safety_check",
            "no_mid_segment_abort_for_share_target_miss": True,
            "continuation": "offline_multi_metric_gate_and_numerical_safety",
        },
        "online": {
            "kc3": "only after corrected offline gate pass",
            "kc4": "conditional after kc3",
            "kc8": "blocked until kc4 passes",
        },
    }
