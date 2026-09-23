"""Frozen decision rules for the first R=1, K_F=4, K_G=2 diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class HybridGateInputs:
    """All inputs used by the predeclared K4 decision."""

    k1_parity_pass: bool
    offline_gate_pass: bool
    hybrid_minus_action_ci95_pp: tuple[float, float]
    hybrid_minus_condition_pp: float
    hybrid_action_transformer_calls: int
    condition_action_transformer_calls: int
    hybrid_amortized_policy_ms: float
    condition_amortized_policy_ms: float
    action_amortized_policy_ms: float
    hybrid_gripper_reversals: float
    action_gripper_reversals: float
    hybrid_translation_second_difference: float
    action_translation_second_difference: float
    hybrid_rotation_second_difference: float
    action_rotation_second_difference: float
    catastrophic_task_regressions_gt20pp: int
    regeneration_recovers_condition_path: bool


FROZEN_THRESHOLDS = {
    "noninferiority_margin_pp": 3.0,
    "minimum_action_transformer_call_reduction": 0.40,
    "minimum_latency_reduction": 0.15,
    "catastrophic_task_regression_pp": 20.0,
    "maximum_catastrophic_tasks": 1,
}


def evaluate_hybrid_gate(inputs: HybridGateInputs) -> dict[str, Any]:
    """Apply the immutable decision hierarchy without result-dependent tuning."""

    lower, upper = inputs.hybrid_minus_action_ci95_pp
    transformer_reduction = 1.0 - inputs.hybrid_action_transformer_calls / max(
        inputs.condition_action_transformer_calls, 1
    )
    latency_reduction = 1.0 - inputs.hybrid_amortized_policy_ms / max(
        inputs.condition_amortized_policy_ms, 1e-12
    )
    stability_not_worse = all(
        (
            inputs.hybrid_gripper_reversals <= inputs.action_gripper_reversals,
            inputs.hybrid_translation_second_difference
            <= inputs.action_translation_second_difference,
            inputs.hybrid_rotation_second_difference
            <= inputs.action_rotation_second_difference,
        )
    )
    clear_stability_benefit = any(
        (
            inputs.hybrid_gripper_reversals < inputs.action_gripper_reversals,
            inputs.hybrid_translation_second_difference
            < inputs.action_translation_second_difference,
            inputs.hybrid_rotation_second_difference
            < inputs.action_rotation_second_difference,
        )
    )
    checks = {
        "k1_parity": bool(inputs.k1_parity_pass),
        "hybrid_noninferior_to_action_3pp": lower > -FROZEN_THRESHOLDS["noninferiority_margin_pp"],
        "action_transformer_calls_reduced_at_least_40pct": transformer_reduction
        >= FROZEN_THRESHOLDS["minimum_action_transformer_call_reduction"],
        "latency_reduced_at_least_15pct": latency_reduction
        >= FROZEN_THRESHOLDS["minimum_latency_reduction"],
        "action_stability_not_worse": stability_not_worse,
        "catastrophic_regression_on_at_most_one_task": inputs.catastrophic_task_regressions_gt20pp
        <= FROZEN_THRESHOLDS["maximum_catastrophic_tasks"],
    }
    prerequisites = {
        "offline_gate": bool(inputs.offline_gate_pass),
        "scientific_matrix_complete": True,
    }
    if all(checks.values()) and all(prerequisites.values()):
        verdict = "HYBRID_K4_SUPPORTED"
    else:
        action_noninferior = upper < FROZEN_THRESHOLDS["noninferiority_margin_pp"]
        action_faster = inputs.action_amortized_policy_ms < inputs.hybrid_amortized_policy_ms
        condition_preferred = (
            inputs.hybrid_minus_condition_pp
            < -FROZEN_THRESHOLDS["noninferiority_margin_pp"]
            and not inputs.regeneration_recovers_condition_path
        )
        if all(prerequisites.values()) and action_noninferior and action_faster and not clear_stability_benefit:
            verdict = "PURE_ACTION_PREFERRED"
        elif all(prerequisites.values()) and condition_preferred:
            verdict = "PURE_CONDITION_PREFERRED"
        else:
            verdict = "HYBRID_INCONCLUSIVE"
    return {
        "verdict": verdict,
        "inputs": asdict(inputs),
        "frozen_thresholds": dict(FROZEN_THRESHOLDS),
        "derived": {
            "action_transformer_call_reduction": transformer_reduction,
            "latency_reduction": latency_reduction,
            "clear_stability_benefit": clear_stability_benefit,
        },
        "checks": checks,
        "prerequisites": prerequisites,
        "k8_diagnostic_allowed": verdict == "HYBRID_K4_SUPPORTED"
        or (verdict == "HYBRID_INCONCLUSIVE" and clear_stability_benefit),
        "r5_k_gt_1_allowed": False,
    }


@dataclass(frozen=True)
class RegenerationCandidateGateInputs:
    """Frozen exact-age R5 gate inputs for one Level-1 candidate."""

    name: str
    finite: bool
    mean_prefix_l1: float
    candidate_minus_hold_prefix_l1_ci95: tuple[float, float]
    candidate_minus_old_observation_prefix_l1_ci95: tuple[float, float]
    gripper_noncollapsed: bool
    prefix_l1_p99: float
    old_observation_prefix_l1_p99: float
    level1_provenance_reset: bool
    latency_ms_mean: float


R5_REGENERATION_THRESHOLDS = {
    "paired_ci_level": 0.95,
    "noninferiority_margin_prefix_l1": 0.0,
    "catastrophic_tail_reference": "old_observation_only_prefix_l1_p99",
}


def evaluate_regeneration_candidate_gate(
    inputs: RegenerationCandidateGateInputs,
) -> dict[str, Any]:
    """Apply the predeclared same-noise, first-executed-prefix R5 gate."""

    hold_upper = float(inputs.candidate_minus_hold_prefix_l1_ci95[1])
    old_upper = float(inputs.candidate_minus_old_observation_prefix_l1_ci95[1])
    checks = {
        "finite": bool(inputs.finite),
        "executed_prefix_better_than_hold_paired95": hold_upper < 0.0,
        "executed_prefix_no_worse_than_old_observation_paired95": old_upper <= 0.0,
        "gripper_noncollapsed": bool(inputs.gripper_noncollapsed),
        "p99_no_worse_than_old_observation": float(inputs.prefix_l1_p99)
        <= float(inputs.old_observation_prefix_l1_p99),
        "level1_provenance_reset": bool(inputs.level1_provenance_reset),
    }
    return {
        "name": inputs.name,
        "pass": all(checks.values()),
        "checks": checks,
        "inputs": asdict(inputs),
        "thresholds": dict(R5_REGENERATION_THRESHOLDS),
    }


def evaluate_r5_regeneration_gate(
    candidates: Iterable[RegenerationCandidateGateInputs],
    *,
    k1_parity_pass: bool,
    exact_age2_pairs_present: bool,
    cache_continuity_pass: bool,
    same_noise_teacher_reload_pass: bool,
) -> dict[str, Any]:
    """Choose the lowest-prefix-error passing candidate without post-hoc tuning."""

    candidate_results = [evaluate_regeneration_candidate_gate(item) for item in candidates]
    prerequisites = {
        "k1_parity": bool(k1_parity_pass),
        "exact_age2_pairs_present": bool(exact_age2_pairs_present),
        "cache_continuity": bool(cache_continuity_pass),
        "same_noise_teacher_reload": bool(same_noise_teacher_reload_pass),
    }
    passing = [item for item in candidate_results if item["pass"]]
    selected = min(
        passing,
        key=lambda item: (
            float(item["inputs"]["mean_prefix_l1"]),
            float(item["inputs"]["latency_ms_mean"]),
            str(item["name"]),
        ),
        default=None,
    )
    passed = all(prerequisites.values()) and selected is not None
    return {
        "R5_REGENERATION_GATE_PASS": passed,
        "ONLINE_R5_GATE_PASS": passed,
        "selected_candidate": selected["name"] if passed else None,
        "selection_rule": "lowest mean executed-prefix L1, then latency, among passing candidates",
        "prerequisites": prerequisites,
        "candidates": {item["name"]: item for item in candidate_results},
        "thresholds": dict(R5_REGENERATION_THRESHOLDS),
    }


@dataclass(frozen=True)
class NativeHybridDecisionInputs:
    """Inputs for the post-online native-horizon scientific decision."""

    k1_parity_pass: bool
    offline_r5_gate_pass: bool
    hybrid_minus_action_ci95_pp: tuple[float, float]
    hybrid_materially_better_after_exhaustion: bool
    hybrid_full_vlm_calls: int
    k1_full_vlm_calls: int
    hybrid_action_transformer_decodes: int
    condition_action_transformer_decodes: int
    hybrid_gripper_reversals: float
    better_endpoint_gripper_reversals: float
    pure_action_as_successful_or_better: bool
    pure_action_faster: bool
    pure_action_long_gap_failure: bool
    regeneration_recovers_long_gap: bool
    scientific_matrix_complete: bool
    hybrid_improves_success_compute_tradeoff: bool


def evaluate_native_hybrid_decision(inputs: NativeHybridDecisionInputs) -> dict[str, Any]:
    """Apply the immutable native H/R verdict hierarchy from the study protocol."""

    gripper_limit = 1.2 * max(float(inputs.better_endpoint_gripper_reversals), 1e-12)
    supported_checks = {
        "k1_parity": bool(inputs.k1_parity_pass),
        "offline_r5_gate": bool(inputs.offline_r5_gate_pass),
        "hybrid_noninferior_to_action_3pp": float(inputs.hybrid_minus_action_ci95_pp[0]) > -3.0,
        "late_outcome_improvement_after_exhaustion": bool(
            inputs.hybrid_materially_better_after_exhaustion
        ),
        "fewer_full_vlm_calls_than_k1": inputs.hybrid_full_vlm_calls < inputs.k1_full_vlm_calls,
        "fewer_action_decodes_than_condition_endpoint": (
            inputs.hybrid_action_transformer_decodes
            < inputs.condition_action_transformer_decodes
        ),
        "gripper_reversals_within_20pct_of_better_endpoint": (
            float(inputs.hybrid_gripper_reversals) <= gripper_limit
        ),
        "scientific_matrix_complete": bool(inputs.scientific_matrix_complete),
    }
    if all(supported_checks.values()):
        verdict = "NATIVE_HYBRID_SUPPORTED"
    elif (
        inputs.pure_action_as_successful_or_better
        and inputs.pure_action_faster
        and not inputs.pure_action_long_gap_failure
    ):
        verdict = "PURE_ACTION_PREFERRED"
    elif inputs.pure_action_long_gap_failure and (
        not inputs.offline_r5_gate_pass or not inputs.regeneration_recovers_long_gap
    ):
        verdict = "REGENERATION_MODEL_INADEQUATE"
    elif inputs.scientific_matrix_complete and not inputs.hybrid_improves_success_compute_tradeoff:
        verdict = "NATIVE_HYBRID_NOT_SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "inputs": asdict(inputs),
        "supported_checks": supported_checks,
        "frozen_thresholds": {
            "success_noninferiority_margin_pp": 3.0,
            "maximum_gripper_reversal_ratio": 1.2,
        },
    }
