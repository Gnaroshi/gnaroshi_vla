"""Predeclared scientific decisions for the LatentLoop segment grid."""

from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


SCIENTIFIC_VERDICTS = (
    "COMMITMENT_FEEDBACK_SUPPORTED",
    "PARETO_IMPROVEMENT_ONLY",
    "COMMITMENT_FEEDBACK_NOT_SUPPORTED",
    "COMMITMENT_FEEDBACK_INCONCLUSIVE",
)

PI05_VERDICTS = (
    "PI05_PORT_JUSTIFIED",
    "PI05_PORT_DIAGNOSTIC_ONLY",
    "PI05_PORT_NOT_JUSTIFIED",
)


def _row_index(
    summary: Mapping[str, object],
) -> Dict[Tuple[int, int, str, str], Mapping[str, object]]:
    index: Dict[Tuple[int, int, str, str], Mapping[str, object]] = {}
    for row in summary.get("rows", []):
        key = (
            int(row["checkpoint_id"]),
            int(row["segment_length"]),
            str(row.get("feedback_schedule", "not_applicable")),
            str(row.get("baseline_kind", "dense_latentloop")),
        )
        index[key] = row
    return index


def _find_row(
    index: Mapping[Tuple[int, int, str, str], Mapping[str, object]],
    checkpoint_id: int,
    segment_length: int,
    *,
    schedule: Optional[str] = None,
    kind: Optional[str] = None,
) -> Optional[Mapping[str, object]]:
    for (ckpt, length, row_schedule, row_kind), row in index.items():
        if ckpt != checkpoint_id or length != segment_length:
            continue
        if schedule is not None and row_schedule != schedule:
            continue
        if kind is not None and row_kind != kind:
            continue
        return row
    return None


def _success_rate(row: Optional[Mapping[str, object]]) -> Optional[float]:
    return None if row is None else float(row["success_rate"])


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def apply_commitment_feedback_decision(summary: Mapping[str, object]) -> Dict[str, object]:
    """Apply the predeclared held-out checkpoint decision rules.

    The function expects rows for checkpoints 36 and 38. It never uses
    checkpoint 33 to tune a threshold.
    """

    index = _row_index(summary)
    checkpoints = (36, 38)
    parity = summary.get("k1_parity", {})
    parity_passes = all(
        bool(parity.get(str(checkpoint_id), {}).get("pass", False))
        for checkpoint_id in checkpoints
    )

    l1_rows = {
        checkpoint_id: _find_row(
            index, checkpoint_id, 1, kind="full_replanning"
        )
        for checkpoint_id in checkpoints
    }
    dense_l4 = {
        checkpoint_id: _find_row(
            index,
            checkpoint_id,
            4,
            schedule="dense",
            kind="dense_latentloop",
        )
        for checkpoint_id in checkpoints
    }
    none_l4 = {
        checkpoint_id: _find_row(
            index,
            checkpoint_id,
            4,
            schedule="none",
            kind="no_observation_latent_dynamics",
        )
        for checkpoint_id in checkpoints
    }

    complete_core = all(
        row is not None
        for rows in (l1_rows, dense_l4, none_l4)
        for row in rows.values()
    )
    dense_minus_l1 = {
        checkpoint_id: (
            None
            if dense_l4[checkpoint_id] is None or l1_rows[checkpoint_id] is None
            else float(_success_rate(dense_l4[checkpoint_id]))
            - float(_success_rate(l1_rows[checkpoint_id]))
        )
        for checkpoint_id in checkpoints
    }
    dense_minus_none = {
        checkpoint_id: (
            None
            if dense_l4[checkpoint_id] is None or none_l4[checkpoint_id] is None
            else float(_success_rate(dense_l4[checkpoint_id]))
            - float(_success_rate(none_l4[checkpoint_id]))
        )
        for checkpoint_id in checkpoints
    }
    mean_dense_l4_minus_l1 = _mean(dense_minus_l1.values())
    mean_dense_l4_minus_none = _mean(dense_minus_none.values())

    noninferior_l4 = complete_core and all(
        dense_minus_l1[checkpoint_id] is not None
        and float(dense_minus_l1[checkpoint_id]) >= -0.01
        for checkpoint_id in checkpoints
    )
    l4_improvement = (
        mean_dense_l4_minus_l1 is not None
        and mean_dense_l4_minus_l1 >= 0.02
    )
    feedback_benefit = (
        mean_dense_l4_minus_none is not None
        and mean_dense_l4_minus_none >= 0.03
    )
    feedback_direction_consistent = complete_core and all(
        dense_minus_none[checkpoint_id] is not None
        and float(dense_minus_none[checkpoint_id]) > 0.0
        for checkpoint_id in checkpoints
    )

    hold_checks: Dict[str, object] = {}
    hold_passes: List[bool] = []
    for kind in ("hold_latent", "hold_action"):
        hold_rows = [
            _find_row(index, checkpoint_id, 4, kind=kind)
            for checkpoint_id in checkpoints
        ]
        if any(row is not None for row in hold_rows):
            dense_mean = _mean(_success_rate(dense_l4[c]) for c in checkpoints)
            hold_mean = _mean(_success_rate(row) for row in hold_rows)
            passed = (
                dense_mean is not None
                and hold_mean is not None
                and dense_mean > hold_mean
            )
            hold_checks[kind] = {
                "executed": True,
                "dense_mean": dense_mean,
                "baseline_mean": hold_mean,
                "pass": passed,
            }
            hold_passes.append(passed)
        else:
            hold_checks[kind] = {"executed": False, "pass": True}
    hold_baselines_pass = all(hold_passes) if hold_passes else True

    dense_means_by_l: Dict[int, Optional[float]] = {}
    for length in (4, 8, 10):
        dense_means_by_l[length] = _mean(
            _success_rate(
                _find_row(
                    index,
                    checkpoint_id,
                    length,
                    schedule="dense",
                    kind="dense_latentloop",
                )
            )
            for checkpoint_id in checkpoints
        )
    moderate = [
        value for value in (dense_means_by_l[4], dense_means_by_l[8])
        if value is not None
    ]
    l10 = dense_means_by_l[10]
    non_monotonic = bool(
        moderate
        and l10 is not None
        and max(moderate) >= l10
        and dense_means_by_l[4] is not None
        and l10 <= float(dense_means_by_l[4]) + 0.01
    )

    support_criteria = {
        "checkpoint_36_38_k1_parity": parity_passes,
        "dense_l4_not_worse_than_l1_by_more_than_1pp": noninferior_l4,
        "mean_dense_l4_minus_l1_at_least_2pp": l4_improvement,
        "mean_dense_l4_minus_none_at_least_3pp": feedback_benefit,
        "dense_l4_beats_executed_hold_baselines": hold_baselines_pass,
        "non_monotonic_segment_curve": non_monotonic,
        "positive_feedback_direction_on_both_checkpoints": feedback_direction_consistent,
    }
    support = complete_core and all(support_criteria.values())

    opposing_feedback = (
        complete_core
        and float(dense_minus_none[36]) * float(dense_minus_none[38]) < 0.0
    )
    repeatedly_worse = complete_core and all(
        float(dense_minus_l1[checkpoint_id]) < 0.0
        for checkpoint_id in checkpoints
    )
    none_matches_or_exceeds = complete_core and (
        mean_dense_l4_minus_none is not None
        and mean_dense_l4_minus_none <= 0.0
    )
    hold_matches_or_exceeds = any(
        bool(item.get("executed")) and not bool(item.get("pass"))
        for item in hold_checks.values()
    )
    checkpoint33_optimum_does_not_transfer = complete_core and all(
        float(dense_minus_l1[checkpoint_id]) <= 0.0
        for checkpoint_id in checkpoints
    )
    not_supported_conditions = {
        "dense_l4_repeatedly_worse_than_l1": repeatedly_worse,
        "none_matches_or_exceeds_dense": none_matches_or_exceeds,
        "hold_matches_or_exceeds_dense": hold_matches_or_exceeds,
        "opposing_feedback_effects": opposing_feedback,
        "checkpoint33_l4_optimum_does_not_transfer": checkpoint33_optimum_does_not_transfer,
    }
    not_supported = complete_core and any(not_supported_conditions.values())

    pareto_by_checkpoint: Dict[str, object] = {}
    for checkpoint_id in checkpoints:
        baseline_sr = _success_rate(l1_rows[checkpoint_id])
        qualifying = []
        if baseline_sr is not None:
            for length in (4, 8, 10):
                row = _find_row(
                    index,
                    checkpoint_id,
                    length,
                    schedule="dense",
                    kind="dense_latentloop",
                )
                if row is None:
                    continue
                reduction = float(row.get("full_query_reduction_ratio", 0.0))
                if float(_success_rate(row)) >= baseline_sr - 0.02 and reduction >= 0.5:
                    qualifying.append(length)
        pareto_by_checkpoint[str(checkpoint_id)] = {
            "pass": bool(qualifying),
            "qualifying_segment_lengths": qualifying,
        }
    pareto = all(bool(item["pass"]) for item in pareto_by_checkpoint.values())

    if support:
        verdict = "COMMITMENT_FEEDBACK_SUPPORTED"
    elif not_supported:
        verdict = "COMMITMENT_FEEDBACK_NOT_SUPPORTED"
    elif pareto:
        verdict = "PARETO_IMPROVEMENT_ONLY"
    else:
        verdict = "COMMITMENT_FEEDBACK_INCONCLUSIVE"

    return {
        "verdict": verdict,
        "complete_core_rows": complete_core,
        "support_criteria": support_criteria,
        "not_supported_conditions": not_supported_conditions,
        "pareto_criteria": pareto_by_checkpoint,
        "dense_l4_minus_l1": {str(k): v for k, v in dense_minus_l1.items()},
        "dense_l4_minus_none": {str(k): v for k, v in dense_minus_none.items()},
        "mean_dense_l4_minus_l1": mean_dense_l4_minus_l1,
        "mean_dense_l4_minus_none": mean_dense_l4_minus_none,
        "dense_mean_by_segment_length": {
            str(k): v for k, v in dense_means_by_l.items()
        },
        "hold_checks": hold_checks,
        "rule_scope": "predeclared screening rules, not universal statistical laws",
    }


def apply_pi05_port_readiness(
    scientific_decision: Mapping[str, object],
) -> Dict[str, object]:
    """Map the Seer scientific verdict to the predeclared pi0.5 port verdict."""

    scientific_verdict = str(scientific_decision.get("verdict", ""))
    if scientific_verdict == "COMMITMENT_FEEDBACK_SUPPORTED":
        verdict = "PI05_PORT_JUSTIFIED"
    elif scientific_verdict == "PARETO_IMPROVEMENT_ONLY":
        verdict = "PI05_PORT_DIAGNOSTIC_ONLY"
    elif scientific_verdict == "COMMITMENT_FEEDBACK_NOT_SUPPORTED":
        verdict = "PI05_PORT_NOT_JUSTIFIED"
    else:
        verdict = "PI05_PORT_NOT_JUSTIFIED"
    return {
        "verdict": verdict,
        "scientific_verdict": scientific_verdict,
        "universal_mechanism_claim_allowed": verdict == "PI05_PORT_JUSTIFIED",
        "note": (
            "Only a supported result justifies a full port, and only a Pareto-only "
            "result justifies a diagnostic port. Inconclusive or unsupported "
            "evidence does not justify the port."
        ),
    }
