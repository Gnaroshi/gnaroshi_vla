#!/usr/bin/env python3
"""Analyze versioned LatentLoop raw-horizon traces without loading a policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from architectures.seer.adapters.latentloop_plan_continuation.token_alignment import (
    assert_canonical_seer_alignment,
    verified_overlap_pairs,
)
from architectures.seer.adapters.latentloop_plan_continuation.trace_adapter import (
    load_plan_trace_shard,
)
from methods.latentloop_plan_continuation.feedback_metrics import (
    binned_summary,
    clustered_spearman_interval,
    paired_clustered_mean_difference_interval,
)
from methods.latentloop_plan_continuation.overlap_metrics import (
    aligned_overlap_components,
    summarize_metric_rows,
    summarize_values,
)


OVERLAP_METRICS = (
    "translation_l1",
    "translation_l2",
    "rotation_l1",
    "rotation_l2",
    "gripper_logit_abs",
    "gripper_probability_abs",
    "gripper_threshold_disagreement",
    "all_token_l1",
    "all_token_l2",
)
DRIVERS = (
    "primary_raw_change_l1",
    "wrist_raw_change_l1",
    "primary_preprocessed_change_l1",
    "wrist_preprocessed_change_l1",
    "proprio_delta_l2",
    "u_delta_norm",
    "cache_age",
)
RESPONSES = (
    "correction_translation_l2",
    "correction_rotation_l2",
    "correction_gripper_logit_abs",
    "correction_gripper_probability_abs",
    "correction_arm_l2",
)
PROPRIO_DRIVERS = tuple(f"proprio_delta_dim{index}" for index in range(8))
SIGNED_RESPONSES = (
    *(f"correction_arm_dim{index}" for index in range(6)),
    "correction_gripper_logit",
    "correction_gripper_probability",
)
DIRECTION_METRICS = (
    "correction_proprio_cosine",
    "correction_proprio_dot",
    "correction_gripper_proprio_sign_agreement",
)

# These 16 associations were frozen in build_plan_continuation_evidence.py
# before the locked Phase A results were analyzed. Confidence intervals are
# restricted to this family; the remaining cross-product is descriptive.
PRIMARY_ASSOCIATION_DRIVERS = (
    "primary_raw_change_l1",
    "wrist_raw_change_l1",
    "proprio_delta_l2",
    "u_delta_norm",
)
PRIMARY_ASSOCIATION_RESPONSES = (
    "correction_translation_l2",
    "correction_rotation_l2",
    "correction_arm_l2",
    "correction_gripper_probability_abs",
)
PRIMARY_ASSOCIATION_PAIRS = frozenset(
    (driver, response)
    for driver in PRIMARY_ASSOCIATION_DRIVERS
    for response in PRIMARY_ASSOCIATION_RESPONSES
)


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_rows(specifications: Iterable[str]) -> dict[str, Path]:
    rows: dict[str, Path] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Expected ROW=PATH, got {specification!r}")
        row_id, raw_path = specification.split("=", 1)
        if not row_id or row_id in rows:
            raise ValueError(f"Invalid or duplicate row id: {row_id!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        rows[row_id] = path
    if not rows:
        raise ValueError("At least one --row ROW=PATH is required")
    return rows


def _load_row(row_id: str, root: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    metadata_paths = sorted(root.rglob("*.plan_trace.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No plan trace shards under {root}")
    for metadata_path in metadata_paths:
        metadata, scalars, arrays = load_plan_trace_shard(metadata_path)
        steps: list[dict[str, Any]] = []
        for index, scalar in enumerate(scalars):
            steps.append(
                {
                    "scalar": scalar,
                    "arm": arrays["raw_action_arm"][index],
                    "gripper_logit": arrays["raw_gripper_logit"][index],
                    "gripper_probability": arrays["raw_gripper_probability"][index],
                    "gripper_thresholded": arrays["raw_gripper_thresholded"][index],
                    "executed_action": arrays["executed_action"][index],
                    "proprio_delta": arrays["proprio_delta"][index],
                }
            )
        steps.sort(key=lambda item: int(item["scalar"]["timestep"]))
        episode = dict(metadata.get("episode", {}))
        episodes.append(
            {
                "row_id": row_id,
                "task_id": int(episode.get("task_id", scalars[0]["task_id"])),
                "episode_id": int(episode.get("episode_id", scalars[0]["episode_id"])),
                "success": int(episode.get("success", scalars[0].get("episode_success", 0))),
                "steps": steps,
                "source": str(metadata_path),
            }
        )
    return episodes


def _annotate_outcome_groups(
    episodes: list[dict[str, Any]], baseline_row: str
) -> None:
    baseline = {
        (episode["task_id"], episode["episode_id"]): int(episode["success"])
        for episode in episodes
        if episode["row_id"] == baseline_row
    }
    labels = {
        (0, 1): "fail_to_success",
        (1, 0): "success_to_fail",
        (1, 1): "both_success",
        (0, 0): "both_fail",
    }
    for episode in episodes:
        if episode["row_id"] == baseline_row:
            episode["outcome_group"] = "baseline_reference"
            continue
        key = (episode["task_id"], episode["episode_id"])
        episode["outcome_group"] = (
            "unpaired"
            if key not in baseline
            else labels[(baseline[key], int(episode["success"]))]
        )


def _analyze_episode(episode: dict[str, Any], pairs: list[tuple[int, int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overlap_rows: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []
    steps = episode["steps"]
    for previous, current in zip(steps, steps[1:]):
        previous_t = int(previous["scalar"]["timestep"])
        current_t = int(current["scalar"]["timestep"])
        if current_t != previous_t + 1:
            continue
        previous_horizon = np.concatenate(
            [previous["arm"], previous["gripper_probability"]], axis=-1
        )
        current_horizon = np.concatenate(
            [current["arm"], current["gripper_probability"]], axis=-1
        )
        components = aligned_overlap_components(
            previous_horizon,
            current_horizon,
            pairs,
            previous_gripper_logit=previous["gripper_logit"],
            current_gripper_logit=current["gripper_logit"],
        )
        base = {
            "row_id": episode["row_id"],
            "task_id": episode["task_id"],
            "episode_id": episode["episode_id"],
            "success": episode["success"],
            "previous_timestep": previous_t,
            "timestep": current_t,
            "cache_age": int(_float(current["scalar"], "cache_age", 0.0)),
            "mode": current["scalar"].get("mode", ""),
            "feedback_source": current["scalar"].get("feedback_source", ""),
            "outcome_group": episode.get("outcome_group", ""),
            "feature_source_step": int(
                _float(current["scalar"], "feature_source_step", -1.0)
            ),
            "simulator_signals_json": current["scalar"].get(
                "simulator_signals_json", "{}"
            ),
        }
        for component in components:
            overlap_rows.append({**base, **component})

        previous_token, current_token = pairs[0]
        arm_delta = current["arm"][current_token] - previous["arm"][previous_token]
        grip_logit_delta = float(
            current["gripper_logit"][current_token, 0]
            - previous["gripper_logit"][previous_token, 0]
        )
        grip_probability_delta = float(
            current["gripper_probability"][current_token, 0]
            - previous["gripper_probability"][previous_token, 0]
        )
        proprio_delta = np.asarray(current["proprio_delta"], dtype=np.float64).reshape(-1)
        proprio_arm = proprio_delta[:6]
        direction_denominator = float(
            np.linalg.norm(arm_delta) * np.linalg.norm(proprio_arm)
        )
        correction_proprio_cosine = (
            float(np.dot(arm_delta, proprio_arm) / direction_denominator)
            if direction_denominator > 0.0
            else float("nan")
        )
        previous_gripper_open = bool(previous["gripper_probability"][previous_token, 0] > 0.5)
        current_gripper_open = bool(current["gripper_probability"][current_token, 0] > 0.5)
        proprio_gripper_delta = float(proprio_delta[-1]) if proprio_delta.size > 6 else float("nan")
        gripper_sign_agreement = (
            float(np.sign(grip_probability_delta) == np.sign(proprio_gripper_delta))
            if grip_probability_delta != 0.0
            and np.isfinite(proprio_gripper_delta)
            and proprio_gripper_delta != 0.0
            else float("nan")
        )
        feedback = {
            **base,
            "previous_token": previous_token,
            "current_token": current_token,
            "correction_translation_l2": float(np.linalg.norm(arm_delta[:3])),
            "correction_rotation_l2": float(np.linalg.norm(arm_delta[3:6])),
            "correction_arm_l2": float(np.linalg.norm(arm_delta)),
            "correction_gripper_logit": grip_logit_delta,
            "correction_gripper_logit_abs": abs(grip_logit_delta),
            "correction_gripper_probability": grip_probability_delta,
            "correction_gripper_probability_abs": abs(grip_probability_delta),
            "correction_proprio_cosine": correction_proprio_cosine,
            "correction_proprio_dot": float(np.dot(arm_delta, proprio_arm)),
            "correction_gripper_proprio_sign_agreement": gripper_sign_agreement,
            "gripper_threshold_event": int(previous_gripper_open != current_gripper_open),
        }
        for key in DRIVERS:
            feedback[key] = _float(current["scalar"], key)
        for dimension, value in enumerate(arm_delta):
            feedback[f"correction_arm_dim{dimension}"] = float(value)
        for dimension, value in enumerate(proprio_delta):
            feedback[f"proprio_delta_dim{dimension}"] = float(value)
        feedback_rows.append(feedback)
    return overlap_rows, feedback_rows


def _episode_means(rows: list[dict[str, Any]], metric: str, row_id: str) -> dict[tuple[object, object], float]:
    grouped: dict[tuple[object, object], list[float]] = defaultdict(list)
    for row in rows:
        if row["row_id"] == row_id and np.isfinite(float(row[metric])):
            grouped[(row["task_id"], row["episode_id"])].append(float(row[metric]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _feedback_association_population(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Select actual lightweight-update transitions for feedback inference.

    K4 rows contain periodic ``mode=full`` refresh transitions where the
    LatentLoop updater is not called and ``u_delta`` is absent. Including those
    transitions would mix Full Seer replanning with lightweight feedback. K1 has
    no intermediate transitions, so it remains an explicitly labelled full-only
    reference row.
    """

    intermediate = [row for row in rows if str(row.get("mode", "")) != "full"]
    if intermediate:
        return intermediate, "intermediate_non_full_transitions"
    return list(rows), "full_only_reference_transitions"


def _outcome_flips(episodes: list[dict[str, Any]], baseline_row: str) -> dict[str, Any]:
    outcomes = {
        (episode["row_id"], episode["task_id"], episode["episode_id"]): episode["success"]
        for episode in episodes
    }
    baseline = {
        (task, episode): success
        for (row, task, episode), success in outcomes.items()
        if row == baseline_row
    }
    result: dict[str, Any] = {}
    for row_id in sorted({episode["row_id"] for episode in episodes}):
        if row_id == baseline_row:
            continue
        current = {
            (task, episode): success
            for (row, task, episode), success in outcomes.items()
            if row == row_id
        }
        common = sorted(set(baseline) & set(current))
        counts = {key: 0 for key in ("fail_to_success", "success_to_fail", "both_success", "both_fail")}
        for key in common:
            pair = baseline[key], current[key]
            label = {
                (0, 1): "fail_to_success",
                (1, 0): "success_to_fail",
                (1, 1): "both_success",
                (0, 0): "both_fail",
            }[pair]
            counts[label] += 1
        result[row_id] = {"paired_episodes": len(common), **counts}
    return result


def _episode_successes(
    episodes: list[dict[str, Any]], row_id: str
) -> dict[tuple[object, object], float]:
    return {
        (episode["task_id"], episode["episode_id"]): float(episode["success"])
        for episode in episodes
        if episode["row_id"] == row_id
    }


def analyze(
    rows: dict[str, Path],
    output_dir: Path,
    baseline_row: str,
    dense_row: str,
    iterations: int,
) -> None:
    """Run all requested offline analyses and write deterministic artifacts."""

    assert_canonical_seer_alignment(3)
    pairs = verified_overlap_pairs(3)
    print(f"[ANALYSIS] loading {len(rows)} trace rows", flush=True)
    episodes = [episode for row_id, root in rows.items() for episode in _load_row(row_id, root)]
    print(f"[ANALYSIS] loaded {len(episodes)} episode traces", flush=True)
    _annotate_outcome_groups(episodes, baseline_row)
    overlap_rows: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_overlap, episode_feedback = _analyze_episode(episode, pairs)
        overlap_rows.extend(episode_overlap)
        feedback_rows.extend(episode_feedback)

    analyzer_path = Path(__file__).resolve()
    feedback_metrics_path = (
        analyzer_path.parents[2]
        / "methods"
        / "latentloop_plan_continuation"
        / "feedback_metrics.py"
    )
    analysis_provenance = {
        "analyzer_path": str(analyzer_path),
        "analyzer_sha256": _sha256(analyzer_path),
        "feedback_metrics_path": str(feedback_metrics_path),
        "feedback_metrics_sha256": _sha256(feedback_metrics_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_provenance.json").write_text(
        json.dumps(analysis_provenance, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "cross_query_consistency.csv", overlap_rows)
    _write_csv(output_dir / "feedback_correction.csv", feedback_rows)
    print(
        f"[ANALYSIS] wrote derived records: overlap={len(overlap_rows)}, "
        f"feedback={len(feedback_rows)}",
        flush=True,
    )

    overlap_summary: dict[str, Any] = {
        "schema_version": 1,
        "verified_overlap_pairs": pairs,
        "baseline_row": baseline_row,
        "dense_row": dense_row,
        "analysis_provenance": analysis_provenance,
        "rows": {},
        "paired_differences_vs_baseline": {},
        "dense_pairwise_overlap_differences": {},
        "dense_pairwise_sr_differences": {},
        "paired_outcome_flips": _outcome_flips(episodes, baseline_row),
    }
    for row_index, row_id in enumerate(rows, start=1):
        print(
            f"[ANALYSIS] overlap summaries {row_index}/{len(rows)}: {row_id}",
            flush=True,
        )
        selected = [row for row in overlap_rows if row["row_id"] == row_id]
        by_age: dict[str, Any] = {}
        for age in sorted({int(row["cache_age"]) for row in selected}):
            age_rows = [row for row in selected if int(row["cache_age"]) == age]
            by_age[str(age)] = summarize_metric_rows(age_rows, OVERLAP_METRICS)
        per_task = {}
        for task in sorted({int(row["task_id"]) for row in selected}):
            task_rows = [row for row in selected if int(row["task_id"]) == task]
            per_task[str(task)] = summarize_metric_rows(task_rows, OVERLAP_METRICS)
        per_episode_means = []
        for task_id, episode_id in sorted(
            {(row["task_id"], row["episode_id"]) for row in selected}
        ):
            episode_rows = [
                row
                for row in selected
                if row["task_id"] == task_id and row["episode_id"] == episode_id
            ]
            per_episode_means.append(
                {
                    "task_id": int(task_id),
                    "episode_id": int(episode_id),
                    "outcome_group": episode_rows[0].get("outcome_group", ""),
                    **{
                        metric: float(np.mean([float(row[metric]) for row in episode_rows]))
                        for metric in OVERLAP_METRICS
                    },
                }
            )
        by_outcome_group = {}
        for group in sorted({str(row.get("outcome_group", "")) for row in selected}):
            group_rows = [row for row in selected if row.get("outcome_group", "") == group]
            by_outcome_group[group] = summarize_metric_rows(group_rows, OVERLAP_METRICS)
        overlap_summary["rows"][row_id] = {
            "episodes": len(
                {(row["task_id"], row["episode_id"]) for row in selected}
            ),
            "all": summarize_metric_rows(selected, OVERLAP_METRICS),
            "by_cache_age": by_age,
            "by_task": per_task,
            "by_outcome_group": by_outcome_group,
            "per_episode_means": per_episode_means,
        }
        if row_id != baseline_row:
            comparisons = {}
            for metric in OVERLAP_METRICS:
                comparisons[metric] = paired_clustered_mean_difference_interval(
                    _episode_means(overlap_rows, metric, row_id),
                    _episode_means(overlap_rows, metric, baseline_row),
                    iterations=iterations,
                )
            overlap_summary["paired_differences_vs_baseline"][row_id] = comparisons

    dense_success = _episode_successes(episodes, dense_row)
    for row_index, row_id in enumerate(rows, start=1):
        print(
            f"[ANALYSIS] dense pairwise summaries {row_index}/{len(rows)}: {row_id}",
            flush=True,
        )
        if row_id == dense_row:
            continue
        overlap_summary["dense_pairwise_sr_differences"][row_id] = (
            paired_clustered_mean_difference_interval(
                dense_success,
                _episode_successes(episodes, row_id),
                iterations=iterations,
            )
        )
        overlap_summary["dense_pairwise_overlap_differences"][row_id] = {
            metric: paired_clustered_mean_difference_interval(
                _episode_means(overlap_rows, metric, dense_row),
                _episode_means(overlap_rows, metric, row_id),
                iterations=iterations,
            )
            for metric in OVERLAP_METRICS
        }

    feedback_summary: dict[str, Any] = {
        "schema_version": 3,
        "baseline_row": baseline_row,
        "analysis_provenance": analysis_provenance,
        "inference": {
            "point_estimand": "timestep_level_spearman_rho",
            "bootstrap_method": "hierarchical_task_episode_fixed_rank",
            "rank_reference": "full_valid_sample",
            "association_population": (
                "intermediate non-full transitions when available; "
                "full-only fallback for K1 reference"
            ),
            "bootstrap_iterations": int(iterations),
            "primary_association_pairs": [
                {"driver": driver, "response": response}
                for driver, response in sorted(PRIMARY_ASSOCIATION_PAIRS)
            ],
            "secondary_associations": "descriptive_only_without_confidence_intervals",
        },
        "rows": {},
    }
    for row_index, row_id in enumerate(rows, start=1):
        print(
            f"[ANALYSIS] feedback summaries {row_index}/{len(rows)}: {row_id}",
            flush=True,
        )
        all_transitions = [row for row in feedback_rows if row["row_id"] == row_id]
        selected, population = _feedback_association_population(all_transitions)
        correlations = {}
        bins = {}
        available_drivers = tuple(DRIVERS) + tuple(
            driver
            for driver in PROPRIO_DRIVERS
            if any(driver in row for row in selected)
        )
        available_responses = tuple(RESPONSES) + SIGNED_RESPONSES
        for driver in available_drivers:
            correlations[driver] = {}
            bins[driver] = {}
            for response in available_responses:
                is_primary = (driver, response) in PRIMARY_ASSOCIATION_PAIRS
                correlation = clustered_spearman_interval(
                    selected,
                    driver,
                    response,
                    iterations=iterations if is_primary else 0,
                )
                correlation["inference_scope"] = (
                    "predeclared_primary" if is_primary else "descriptive_only"
                )
                correlations[driver][response] = correlation
                bins[driver][response] = binned_summary(
                    (float(row[driver]) for row in selected),
                    (float(row[response]) for row in selected),
                )
        feedback_summary["rows"][row_id] = {
            "episodes": len(
                {(row["task_id"], row["episode_id"]) for row in all_transitions}
            ),
            "all_transition_records": len(all_transitions),
            "association_records": len(selected),
            "association_population": population,
            "correlations": correlations,
            "binned_analysis": bins,
            "response_summary": {
                metric: summarize_values(float(row[metric]) for row in selected)
                for metric in (*RESPONSES, *DIRECTION_METRICS, "gripper_threshold_event")
            },
            "by_cache_age": {
                str(age): {
                    metric: summarize_values(
                        float(row[metric])
                        for row in selected
                        if int(row["cache_age"]) == age
                    )
                    for metric in (*RESPONSES, *DIRECTION_METRICS)
                }
                for age in sorted({int(row["cache_age"]) for row in selected})
            },
        }

    (output_dir / "cross_query_consistency_summary.json").write_text(
        json.dumps(_json_safe(overlap_summary), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "feedback_correction_summary.json").write_text(
        json.dumps(_json_safe(feedback_summary), indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Cross-query plan-consistency analysis",
        "",
        f"- Verified overlap pairs: `{pairs}`",
        f"- Baseline row: `{baseline_row}`",
        f"- Episode traces: `{len(episodes)}`",
        f"- Overlap records: `{len(overlap_rows)}`",
        "",
        "Lower overlap error is descriptive only. It is not interpreted as better unless paired success and intervention evidence agree.",
        "",
        "See `cross_query_consistency_summary.json` for p50/p90/p95/p99, task/cache-age summaries, clustered paired intervals, and outcome flips.",
    ]
    (output_dir / "cross_query_consistency_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    feedback_report = [
        "# Feedback-correction analysis",
        "",
        "The primary correction is the verified same-execution-time pair `current token 0 - previous token 1`.",
        "",
        "K4 feedback inference uses only non-full intermediate transitions where the lightweight updater is actually called. K1 is retained as a labelled full-only reference.",
        "",
        "The Spearman point estimate uses all valid timestep records. Confidence intervals use a fixed-rank hierarchical bootstrap that resamples tasks and then episodes within tasks.",
        "",
        "Inferential intervals are computed only for the 16 driver-response pairs frozen in the decision rule. All other correlations and quantile-bin tables are descriptive.",
        "",
        "No causal mechanism claim is made by this script; use the predeclared decision rule after all interventions and matched baselines finish.",
    ]
    (output_dir / "feedback_correction_report.md").write_text(
        "\n".join(feedback_report) + "\n", encoding="utf-8"
    )
    print(f"[ANALYSIS][DONE] {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", action="append", default=[], metavar="ROW=PATH")
    parser.add_argument("--baseline-row", required=True)
    parser.add_argument("--dense-row", default="k4_dense_latentloop")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    rows = _parse_rows(args.row)
    if args.baseline_row not in rows:
        raise ValueError("--baseline-row must name one of the supplied rows")
    if args.dense_row not in rows:
        raise ValueError("--dense-row must name one of the supplied rows")
    analyze(
        rows,
        args.output_dir,
        args.baseline_row,
        args.dense_row,
        args.bootstrap_iterations,
    )


if __name__ == "__main__":
    main()
