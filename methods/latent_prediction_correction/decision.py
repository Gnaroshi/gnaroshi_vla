"""Predeclared scientific decision rule for held-out latent-filter results."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def _row(
    rows: Sequence[Mapping[str, Any]],
    checkpoint: int,
    mode: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if int(row["checkpoint_id"]) == int(checkpoint) and row["mode"] == mode
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for checkpoint={checkpoint}, mode={mode}; "
            f"found {len(matches)}"
        )
    return matches[0]


def apply_decision_rule(
    rows: Sequence[Mapping[str, Any]],
    endpoint_parity: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply immutable thresholds to completed checkpoint-36/38 summaries."""
    checkpoints = [int(value) for value in rule["heldout_checkpoints"]]
    raw_rows = [_row(rows, checkpoint, "raw_full") for checkpoint in checkpoints]
    fixed_rows = [
        _row(rows, checkpoint, "fixed_filter") for checkpoint in checkpoints
    ]
    ema_rows = [
        _row(rows, checkpoint, "full_latent_ema") for checkpoint in checkpoints
    ]

    endpoint_requirements = rule.get(
        "endpoint_requirements",
        {"alpha_1_exact": True},
    )
    endpoint_checks = {
        str(key): bool(endpoint_parity.get(key, False)) == bool(expected)
        for key, expected in endpoint_requirements.items()
    }
    endpoint_pass = all(endpoint_checks.values())
    deltas = [
        100.0 * (float(fixed["success_rate"]) - float(raw["success_rate"]))
        for fixed, raw in zip(fixed_rows, raw_rows)
    ]
    net_flips = [
        int(fixed.get("paired_net_flip_vs_raw", -10**9))
        for fixed in fixed_rows
    ]
    mean_delta = sum(deltas) / len(deltas)
    mean_fixed = (
        100.0
        * sum(float(row["success_rate"]) for row in fixed_rows)
        / len(fixed_rows)
    )
    mean_ema = (
        100.0
        * sum(float(row["success_rate"]) for row in ema_rows)
        / len(ema_rows)
    )
    fixed_minus_ema = mean_fixed - mean_ema

    continuity_advantages = []
    no_chatter_increase = []
    for fixed, ema in zip(fixed_rows, ema_rows):
        trans_reduction = 1.0 - float(
            fixed["translation_second_difference_p95"]
        ) / max(float(ema["translation_second_difference_p95"]), 1e-12)
        rot_reduction = 1.0 - float(
            fixed["rotation_second_difference_p95"]
        ) / max(float(ema["rotation_second_difference_p95"]), 1e-12)
        continuity_advantages.append(
            trans_reduction
            >= float(rule["ema_exclusion"]["continuity_reduction_fraction"])
            and rot_reduction
            >= float(rule["ema_exclusion"]["continuity_reduction_fraction"])
        )
        no_chatter_increase.append(
            float(fixed["gripper_reverse_within_5_per_100_steps"])
            <= float(ema["gripper_reverse_within_5_per_100_steps"])
        )

    ema_excluded = (
        fixed_minus_ema
        >= float(rule["ema_exclusion"]["minimum_sr_advantage_pp"])
        or (
            fixed_minus_ema
            >= -float(rule["ema_exclusion"]["sr_equivalence_margin_pp"])
            and all(continuity_advantages)
            and all(no_chatter_increase)
        )
    )
    fixed_confirmatory = (
        all(value >= 0 for value in net_flips)
        and mean_delta
        >= float(rule["fixed_vs_raw"]["minimum_mean_sr_gain_pp"])
        and all(
            delta
            >= -float(rule["fixed_vs_raw"]["maximum_checkpoint_drop_pp"])
            for delta in deltas
        )
    )

    if not endpoint_pass or any(
        delta < -float(rule["fixed_vs_raw"]["maximum_checkpoint_drop_pp"])
        for delta in deltas
    ):
        verdict = "LATENT_FILTER_NOT_CONFIRMED"
    elif not ema_excluded and mean_ema >= mean_fixed:
        verdict = "LATENT_FILTER_NOT_CONFIRMED"
    elif endpoint_pass and fixed_confirmatory and ema_excluded:
        verdict = "LATENT_FILTER_CONFIRMED"
    else:
        verdict = "LATENT_FILTER_INCONCLUSIVE"

    return {
        "verdict": verdict,
        "endpoint_parity_pass": endpoint_pass,
        "endpoint_checks": endpoint_checks,
        "heldout_checkpoints": checkpoints,
        "fixed_minus_raw_pp_by_checkpoint": dict(
            zip(map(str, checkpoints), deltas)
        ),
        "paired_net_flip_by_checkpoint": dict(
            zip(map(str, checkpoints), net_flips)
        ),
        "mean_fixed_minus_raw_pp": mean_delta,
        "mean_fixed_minus_ema_pp": fixed_minus_ema,
        "ema_excluded": ema_excluded,
        "fixed_confirmatory_criteria_pass": fixed_confirmatory,
        "thresholds_are_predeclared_screening_rules": True,
    }
