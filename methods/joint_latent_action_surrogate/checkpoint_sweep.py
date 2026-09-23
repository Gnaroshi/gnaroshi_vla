"""Frozen eligibility, ranking, and utility rules for the Stage-A sweep."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


EXACT_ENSEMBLE_METRIC_UNAVAILABLE = "EXACT_ENSEMBLE_METRIC_UNAVAILABLE"
EXACT_ENSEMBLE_METRIC_AVAILABLE = "EXACT_ENSEMBLE_METRIC_AVAILABLE"


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric {label}={value!r}")
    return result


def build_fidelity_eligibility(
    *,
    exact_first_token_mean_by_age: Mapping[int, float],
    hold_first_token_mean_by_age: Mapping[int, float],
    exact_arm_first_token_p95_by_age: Mapping[int, float],
    hold_arm_first_token_p95_by_age: Mapping[int, float],
    gripper_finite_by_age: Mapping[int, bool],
    gripper_noncollapsed_by_age: Mapping[int, bool],
    identity_checks: Mapping[str, bool],
    ensemble_status: str,
    exact_ensemble_mean: float | None = None,
    hold_ensemble_mean: float | None = None,
) -> dict[str, Any]:
    """Apply the predeclared canonical-action fidelity gate at ages one and two."""

    required_ages = (1, 2)
    numeric_maps = {
        "exact_first_token_mean": exact_first_token_mean_by_age,
        "hold_first_token_mean": hold_first_token_mean_by_age,
        "exact_arm_first_token_p95": exact_arm_first_token_p95_by_age,
        "hold_arm_first_token_p95": hold_arm_first_token_p95_by_age,
    }
    normalized: dict[str, dict[int, float]] = {}
    for name, values in numeric_maps.items():
        if set(values) != set(required_ages):
            raise ValueError(f"{name} must contain exactly regeneration ages 1 and 2")
        normalized[name] = {
            age: _finite(values[age], label=f"{name}[{age}]")
            for age in required_ages
        }
    for name, values in (
        ("gripper_finite_by_age", gripper_finite_by_age),
        ("gripper_noncollapsed_by_age", gripper_noncollapsed_by_age),
    ):
        if set(values) != set(required_ages):
            raise ValueError(f"{name} must contain exactly regeneration ages 1 and 2")

    exact_mean = sum(normalized["exact_first_token_mean"].values()) / 2.0
    hold_mean = sum(normalized["hold_first_token_mean"].values()) / 2.0
    checks: dict[str, bool] = {
        "overall_exact_first_token_better_than_hold": exact_mean < hold_mean,
        "age1_exact_first_token_better_than_hold": (
            normalized["exact_first_token_mean"][1]
            < normalized["hold_first_token_mean"][1]
        ),
        "age2_exact_first_token_better_than_hold": (
            normalized["exact_first_token_mean"][2]
            < normalized["hold_first_token_mean"][2]
        ),
        "age1_exact_arm_p95_within_hold_plus_5pct": (
            normalized["exact_arm_first_token_p95"][1]
            <= 1.05 * normalized["hold_arm_first_token_p95"][1]
        ),
        "age2_exact_arm_p95_within_hold_plus_5pct": (
            normalized["exact_arm_first_token_p95"][2]
            <= 1.05 * normalized["hold_arm_first_token_p95"][2]
        ),
        "age1_gripper_probability_finite": bool(gripper_finite_by_age[1]),
        "age2_gripper_probability_finite": bool(gripper_finite_by_age[2]),
        "age1_gripper_probability_noncollapsed": bool(
            gripper_noncollapsed_by_age[1]
        ),
        "age2_gripper_probability_noncollapsed": bool(
            gripper_noncollapsed_by_age[2]
        ),
    }
    checks.update({str(name): bool(value) for name, value in identity_checks.items()})

    if ensemble_status == EXACT_ENSEMBLE_METRIC_AVAILABLE:
        if exact_ensemble_mean is None or hold_ensemble_mean is None:
            raise ValueError("Available ensemble metrics require candidate and hold means")
        checks["exact_ensemble_better_than_hold"] = _finite(
            exact_ensemble_mean, label="exact_ensemble_mean"
        ) < _finite(hold_ensemble_mean, label="hold_ensemble_mean")
        ensemble_applicable = True
    elif ensemble_status == EXACT_ENSEMBLE_METRIC_UNAVAILABLE:
        ensemble_applicable = False
    else:
        raise ValueError(f"Unknown ensemble status: {ensemble_status}")

    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "ensemble_check_applicable": ensemble_applicable,
        "ensemble_status": ensemble_status,
        "summary": {
            "exact_first_token_l1_mean_age_average": exact_mean,
            "hold_first_token_l1_mean_age_average": hold_mean,
            "exact_arm_first_token_l1_p95_age_average": sum(
                normalized["exact_arm_first_token_p95"].values()
            )
            / 2.0,
            "hold_arm_first_token_l1_p95_age_average": sum(
                normalized["hold_arm_first_token_p95"].values()
            )
            / 2.0,
        },
    }


def rank_checkpoint_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank eligible checkpoints by the frozen canonical-action hierarchy."""

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    ensemble_statuses: set[str] = set()
    for source in rows:
        row = dict(source)
        checkpoint_id = str(row["checkpoint_id"])
        if checkpoint_id in seen_ids:
            raise ValueError(f"Duplicate checkpoint_id: {checkpoint_id}")
        seen_ids.add(checkpoint_id)
        row["fidelity_eligible"] = bool(row["fidelity_eligible"])
        for key in (
            "exact_first_token_l1_mean_age_average",
            "exact_arm_first_token_l1_p95_age_average",
            "exact_full_horizon_l1_mean_age_average",
        ):
            row[key] = _finite(row[key], label=f"{checkpoint_id}.{key}")
        status = str(row["exact_ensemble_status"])
        if status not in {
            EXACT_ENSEMBLE_METRIC_AVAILABLE,
            EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
        }:
            raise ValueError(f"Unknown ensemble status for {checkpoint_id}: {status}")
        ensemble_statuses.add(status)
        if status == EXACT_ENSEMBLE_METRIC_AVAILABLE:
            row["exact_ensemble_executed_action_l1_mean"] = _finite(
                row["exact_ensemble_executed_action_l1_mean"],
                label=f"{checkpoint_id}.exact_ensemble_executed_action_l1_mean",
            )
        normalized.append(row)

    if not normalized:
        raise ValueError("Checkpoint sweep requires at least one candidate")
    if len(ensemble_statuses) != 1:
        raise ValueError("All checkpoint rows must share one ensemble availability state")
    ensemble_available = (
        next(iter(ensemble_statuses)) == EXACT_ENSEMBLE_METRIC_AVAILABLE
    )
    normalized.sort(
        key=lambda row: (
            0 if row["fidelity_eligible"] else 1,
            (
                float(row["exact_ensemble_executed_action_l1_mean"])
                if ensemble_available
                else 0.0
            ),
            float(row["exact_first_token_l1_mean_age_average"]),
            float(row["exact_arm_first_token_l1_p95_age_average"]),
            float(row["exact_full_horizon_l1_mean_age_average"]),
            int(row["global_microbatches"]),
        )
    )
    for rank, row in enumerate(normalized, start=1):
        row["rank"] = rank
    return normalized


def select_eligible_checkpoint(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return deterministic ranking and its best fidelity-eligible checkpoint."""

    ranked = rank_checkpoint_rows(rows)
    selected = next((row for row in ranked if row["fidelity_eligible"]), None)
    return ranked, selected


def utility_verdict(
    selected: dict[str, Any] | None,
    *,
    surrogate_incremental_latency_ms: float | None,
    exact_action_head_latency_ms: float | None,
    additional_observation_encoder_calls: int,
    projected_hybrid_latency_delta_ms: float | None,
) -> str:
    """Map frozen fidelity and latency evidence to the one-shot Seer verdict."""

    if selected is None:
        return "STOP_SEER_SURROGATE"
    if surrogate_incremental_latency_ms is None or exact_action_head_latency_ms is None:
        return "INCONCLUSIVE"
    surrogate_ms = _finite(
        surrogate_incremental_latency_ms, label="surrogate_incremental_latency_ms"
    )
    exact_ms = _finite(exact_action_head_latency_ms, label="exact_action_head_latency_ms")
    projected_delta = _finite(
        projected_hybrid_latency_delta_ms, label="projected_hybrid_latency_delta_ms"
    )
    if (
        surrogate_ms < exact_ms
        and int(additional_observation_encoder_calls) == 0
        and projected_delta < 0.0
    ):
        return "SEER_DEPLOYMENT_CANDIDATE"
    return "TRANSFER_ONLY_CANDIDATE"
