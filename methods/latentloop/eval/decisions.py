"""Frozen go/no-go and cross-architecture scientific decisions."""

from __future__ import annotations

from typing import Any, Mapping


def _number(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    return None if value is None else float(value)


def apply_predeclared_decisions(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the immutable gates defined before Chunk-aware experiments."""

    k1_pass = (
        bool(summary.get("k1_exact_action_chunk_equality", False))
        and bool(summary.get("k1_identical_paired_outcomes", False))
        and int(summary.get("k1_updater_calls", -1)) == 0
        and int(summary.get("k1_observation_encoder_calls", -1)) == 0
        and int(summary.get("k1_action_encoder_calls", -1)) == 0
    )
    offline_pass = (
        bool(summary.get("offline_chunk_aware_beats_hold_prefix_error", False))
        and bool(summary.get("offline_chunk_aware_beats_old_observation_only_prefix_error", False))
    )
    r1_ci = _number(summary, "r1_k4_paired_ci_lower_pp")
    r1_no_obs_ci = _number(summary, "r1_k4_vs_no_observation_ci_lower_pp")
    r1_reduction = _number(summary, "r1_k4_full_condition_reduction")
    r1_k4_pass = bool(
        bool(summary.get("r1_confirmation_complete", False))
        and r1_ci is not None
        and r1_ci > -3.0
        and r1_no_obs_ci is not None
        and r1_no_obs_ci > 0.0
        and r1_reduction is not None
        and abs(r1_reduction - 0.75) <= 0.03
        and not bool(summary.get("r1_k4_worse_than_both_matched_baselines", True))
    )
    r5_ci = _number(summary, "r5_k2_paired_ci_lower_pp")
    r5_reduction = _number(summary, "r5_k2_full_condition_reduction")
    r5_k2_pass = bool(
        r5_ci is not None
        and r5_ci > -3.0
        and r5_reduction is not None
        and abs(r5_reduction - 0.50) <= 0.03
        and bool(summary.get("r5_k2_beats_hold", False))
        and bool(summary.get("r5_k2_beats_no_observation", False))
        and bool(summary.get("r5_k2_improves_old_observation_only", False))
    )
    r5_k3_pass = bool(summary.get("r5_k3_margin_pass", False))
    proceed_r5_k3 = r5_k2_pass
    proceed_r5_k4 = r5_k2_pass and r5_k3_pass and bool(
        summary.get("r5_recursive_errors_bounded", False)
    ) and not bool(summary.get("r5_catastrophic_tail_failure", True))

    cross_architecture = "INSUFFICIENT_EVIDENCE"
    if bool(summary.get("simvla_r1_credible", False)) or bool(summary.get("simvla_r5_credible", False)):
        if bool(summary.get("recurrent_beats_nonrecurrent_both_architectures", False)) and bool(
            summary.get("recurrent_beats_action_correction_both_architectures", False)
        ):
            cross_architecture = "LATENT_LOCATION_SUPPORTED"
        elif bool(summary.get("nonrecurrent_matches_or_beats_recurrent", False)):
            cross_architecture = "RECURRENCE_NOT_NEEDED"
        elif bool(summary.get("action_correction_matches_or_beats_recurrent", False)):
            cross_architecture = "ACTION_CORRECTION_SUFFICIENT"
        elif bool(summary.get("simvla_r1_credible", False)) and not bool(
            summary.get("simvla_r5_credible", False)
        ):
            cross_architecture = "ENVSTEP_ONLY"
        elif bool(summary.get("simvla_r5_chunk_aware_supported", False)):
            cross_architecture = "CHUNK_AWARE_SUPPORTED"
    elif bool(summary.get("simvla_screening_complete", False)):
        cross_architecture = "SIMVLA_NOT_SUPPORTED"
    return {
        "K1_PARITY_PASS": k1_pass,
        "OFFLINE_PREFIX_GATE_PASS": offline_pass,
        "R1_K4_PASS": r1_k4_pass,
        "R5_K2_PASS": r5_k2_pass,
        "PROCEED_TO_R5_K3": proceed_r5_k3,
        "PROCEED_TO_R5_K4": proceed_r5_k4,
        "scientific_verdict": cross_architecture,
    }
