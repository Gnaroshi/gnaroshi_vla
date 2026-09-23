"""Frozen decision labels and deterministic verdict logic."""

from __future__ import annotations

from enum import Enum


class JointVerdict(str, Enum):
    SUPPORTED = "JOINT_HIERARCHY_SUPPORTED"
    CAPACITY_ONLY = "CAPACITY_ONLY"
    RECOVERS_NOT_PARETO = "SURROGATE_RECOVERS_BUT_NOT_PARETO"
    NOT_SUPPORTED = "JOINT_HIERARCHY_NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"


def apply_joint_decision_rule(metrics: dict[str, float | int | bool]) -> JointVerdict:
    """Apply the immutable rule without filling missing evidence by assumption."""

    required = {
        "k1_parity",
        "joint_vs_canonical_ci_low_pp",
        "joint_vs_recursive_ci_low_pp",
        "action_head_reduction_fraction",
        "parameter_ratio",
        "latency_reduction_fraction",
        "supported_advantage_at_latency_overhead",
        "latency_overhead_fraction",
        "gripper_short_reversal_ratio",
        "joint_better_pareto_than_wide",
        "tasks_regressed_over_20pp",
        "joint_materially_improves_naive",
        "offline_gates_pass",
        "wide_matches_or_exceeds_joint",
        "recovers_recursive_collapse",
    }
    if not required.issubset(metrics):
        return JointVerdict.INCONCLUSIVE
    supported = all(
        [
            bool(metrics["k1_parity"]),
            float(metrics["joint_vs_canonical_ci_low_pp"]) > -3.0,
            float(metrics["joint_vs_recursive_ci_low_pp"]) > 0.0,
            float(metrics["action_head_reduction_fraction"]) >= 0.5,
            float(metrics["parameter_ratio"]) <= 1.25,
            (
                float(metrics["latency_reduction_fraction"]) >= 0.05
                or (
                    bool(metrics["supported_advantage_at_latency_overhead"])
                    and float(metrics["latency_overhead_fraction"]) <= 0.05
                )
            ),
            float(metrics["gripper_short_reversal_ratio"]) <= 1.20,
            bool(metrics["joint_better_pareto_than_wide"]),
            int(metrics["tasks_regressed_over_20pp"]) <= 1,
            bool(metrics["offline_gates_pass"]),
        ]
    )
    if supported:
        return JointVerdict.SUPPORTED
    if bool(metrics["wide_matches_or_exceeds_joint"]):
        return JointVerdict.CAPACITY_ONLY
    if bool(metrics["recovers_recursive_collapse"]) and not bool(
        metrics["joint_better_pareto_than_wide"]
    ):
        return JointVerdict.RECOVERS_NOT_PARETO
    if not bool(metrics["offline_gates_pass"]) or not bool(
        metrics["joint_materially_improves_naive"]
    ):
        return JointVerdict.NOT_SUPPORTED
    return JointVerdict.INCONCLUSIVE
