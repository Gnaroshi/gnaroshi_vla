"""Frozen scientific decision rule for the regeneration intervention."""

from __future__ import annotations

from typing import Mapping


DECISION_RULE = {
    "noninferiority_margin_pp": 3.0,
    "minimum_action_head_reduction_fraction": 0.50,
    "maximum_gripper_reversal_increase_fraction": 0.20,
    "maximum_tasks_regressing_over_20pp": 1,
    "verdicts": [
        "HORIZON_REGENERATION_INTERVENTION_SUPPORTED",
        "REGENERATION_RECOVERS_ACTION_CORRECTION_ONLY",
        "HORIZON_REGENERATION_NOT_SUPPORTED",
        "INCONCLUSIVE",
    ],
}


def evaluate_decision(values: Mapping[str, object]) -> str:
    """Apply the immutable, predeclared rule to already-computed statistics."""

    parity = bool(values.get("k1_and_endpoint_parity_pass", False))
    hybrid_vs_action_lower = float(values.get("hybrid_vs_action_ci_lower_pp", float("-inf")))
    hybrid_vs_latent_lower = float(values.get("hybrid_vs_latentloop_ci_lower_pp", float("-inf")))
    action_head_reduction = float(values.get("action_head_reduction_fraction", float("-inf")))
    reversal_increase = float(values.get("gripper_reversal_increase_fraction", float("inf")))
    regressed_tasks = int(values.get("tasks_regressing_over_20pp", 999))
    hybrid_vs_action_delta = float(values.get("hybrid_vs_action_delta_pp", float("-inf")))

    supported = (
        parity
        and hybrid_vs_action_lower > 0.0
        and hybrid_vs_latent_lower > -DECISION_RULE["noninferiority_margin_pp"]
        and action_head_reduction >= DECISION_RULE["minimum_action_head_reduction_fraction"]
        and reversal_increase <= DECISION_RULE["maximum_gripper_reversal_increase_fraction"]
        and regressed_tasks <= DECISION_RULE["maximum_tasks_regressing_over_20pp"]
    )
    if supported:
        return "HORIZON_REGENERATION_INTERVENTION_SUPPORTED"
    if parity and hybrid_vs_action_lower > 0.0 and hybrid_vs_latent_lower <= -3.0:
        return "REGENERATION_RECOVERS_ACTION_CORRECTION_ONLY"
    if hybrid_vs_action_delta <= 0.0 or hybrid_vs_action_lower <= 0.0:
        return "HORIZON_REGENERATION_NOT_SUPPORTED"
    return "INCONCLUSIVE"
