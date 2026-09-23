"""Frozen scientific verdict rules for the Seer comparison."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


NONINFERIORITY_MARGIN = 0.03
COMPUTE_TOLERANCE = 0.10


def _number(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping or mapping[key] is None:
        raise KeyError(f"Missing decision input: {key}")
    return float(mapping[key])


def _not_worse(candidate: float, reference: float) -> bool:
    return candidate <= reference * (1.0 + COMPUTE_TOLERANCE)


def latent_location_verdict(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared latent-vs-action correction rule."""

    latent_sr = _number(inputs, "latentloop_sr")
    action_sr = _number(inputs, "action_correction_sr")
    latent_minus_action_low = _number(inputs, "latent_minus_action_ci_low")
    action_minus_latent_low = _number(inputs, "action_minus_latent_ci_low")
    latent_cost_ok = _not_worse(
        _number(inputs, "latentloop_parameters"),
        _number(inputs, "action_correction_parameters"),
    ) and _not_worse(
        _number(inputs, "latentloop_skip_p50_ms"),
        _number(inputs, "action_correction_skip_p50_ms"),
    )
    additional_advantage = any(
        bool(inputs.get(key, False))
        for key in (
            "latentloop_better_k8_stability",
            "latentloop_lower_gripper_failure_or_chatter",
            "latentloop_better_libero_plus",
            "latentloop_better_heldout_checkpoint",
        )
    )
    supported = (
        latent_sr - action_sr >= NONINFERIORITY_MARGIN
        and latent_minus_action_low > 0.0
        and latent_cost_ok
        and additional_advantage
    )
    action_cost_ok = _not_worse(
        _number(inputs, "action_correction_parameters"),
        _number(inputs, "latentloop_parameters"),
    ) and _not_worse(
        _number(inputs, "action_correction_skip_p50_ms"),
        _number(inputs, "latentloop_skip_p50_ms"),
    )
    sufficient = (
        action_minus_latent_low > -NONINFERIORITY_MARGIN and action_cost_ok
    )
    if supported:
        verdict = "LATENT_LOCATION_SUPPORTED"
    elif sufficient:
        verdict = "ACTION_CORRECTION_SUFFICIENT"
    else:
        verdict = "LATENT_LOCATION_INCONCLUSIVE"
    return {
        "verdict": verdict,
        "point_difference": latent_sr - action_sr,
        "latent_minus_action_ci_low": latent_minus_action_low,
        "action_minus_latent_ci_low": action_minus_latent_low,
        "latent_cost_ok": latent_cost_ok,
        "action_cost_ok": action_cost_ok,
        "additional_advantage": additional_advantage,
    }


def recurrence_verdict(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared recurrent-vs-nonrecurrent rule."""

    latent_sr = _number(inputs, "latentloop_sr")
    nonrecurrent_sr = _number(inputs, "nonrecurrent_sr")
    latent_minus_nr_low = _number(inputs, "latent_minus_nonrecurrent_ci_low")
    nr_minus_latent_low = _number(inputs, "nonrecurrent_minus_latent_ci_low")
    latent_cost_ok = _not_worse(
        _number(inputs, "latentloop_parameters"),
        _number(inputs, "nonrecurrent_parameters"),
    ) and _not_worse(
        _number(inputs, "latentloop_skip_p50_ms"),
        _number(inputs, "nonrecurrent_skip_p50_ms"),
    )
    robustness = bool(inputs.get("latentloop_better_k8_than_nonrecurrent", False)) or bool(
        inputs.get("latentloop_better_heldout_than_nonrecurrent", False)
    )
    supported = (
        latent_sr - nonrecurrent_sr >= NONINFERIORITY_MARGIN
        and latent_minus_nr_low > 0.0
        and latent_cost_ok
        and robustness
    )
    nonrecurrent_cost_ok = _not_worse(
        _number(inputs, "nonrecurrent_parameters"),
        _number(inputs, "latentloop_parameters"),
    ) and _not_worse(
        _number(inputs, "nonrecurrent_skip_p50_ms"),
        _number(inputs, "latentloop_skip_p50_ms"),
    )
    not_needed = (
        nr_minus_latent_low > -NONINFERIORITY_MARGIN and nonrecurrent_cost_ok
    )
    if supported:
        verdict = "RECURRENCE_SUPPORTED"
    elif not_needed:
        verdict = "RECURRENCE_NOT_NEEDED"
    else:
        verdict = "RECURRENCE_INCONCLUSIVE"
    return {
        "verdict": verdict,
        "point_difference": latent_sr - nonrecurrent_sr,
        "latent_minus_nonrecurrent_ci_low": latent_minus_nr_low,
        "nonrecurrent_minus_latent_ci_low": nr_minus_latent_low,
        "latent_cost_ok": latent_cost_ok,
        "nonrecurrent_cost_ok": nonrecurrent_cost_ok,
        "robustness_direction_confirmed": robustness,
    }


def combined_verdict(
    latent_location: Mapping[str, Any], recurrence: Mapping[str, Any]
) -> str:
    """Combine the two independent scientific questions."""

    location = str(latent_location["verdict"])
    recurrent = str(recurrence["verdict"])
    if location == "LATENT_LOCATION_SUPPORTED" and recurrent == "RECURRENCE_SUPPORTED":
        return "LATENT_DYNAMICS_JUSTIFIED"
    if location == "ACTION_CORRECTION_SUFFICIENT" or recurrent == "RECURRENCE_NOT_NEEDED":
        return "NO_LATENT_ADVANTAGE"
    return "MIXED_EVIDENCE"


def apply_comparison_decisions(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return both primary verdicts and the combined method verdict."""

    location = latent_location_verdict(inputs)
    recurrence = recurrence_verdict(inputs)
    return {
        "schema_version": 1,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "compute_tolerance": COMPUTE_TOLERANCE,
        "latent_location": location,
        "recurrence": recurrence,
        "combined_verdict": combined_verdict(location, recurrence),
    }
