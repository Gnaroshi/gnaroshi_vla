#!/usr/bin/env python3
"""Aggregate source-locked Seer Q1/Q2 evaluations without selecting on test SR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_NAMES = (
    "full_seer_k1",
    "canonical_latentloop_k4",
    "matched_action_correction_k4",
    "nonrecurrent_latent_k4",
    "no_observation_latentloop_k4",
    "predicted_horizon_replay_k4",
)
NEW_BASELINES = (
    "matched_action_correction_k4",
    "nonrecurrent_latent_k4",
)
SCREENING_MARGIN = 0.06
MATERIAL_COMPUTE_IMPROVEMENT = 0.10


@dataclass(frozen=True)
class RowData:
    """Parsed episode and step evidence for one evaluator row."""

    name: str
    root: Path
    episode_path: Path
    step_log_dir: Path
    episode_rows: list[dict[str, str]]
    step_rows: list[dict[str, str]]
    outcomes: dict[tuple[int, int, int], int]
    steps: dict[tuple[int, int, int], float]
    summary: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {name} below {root}, found {len(matches)}: {matches[:5]}"
        )
    return matches[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in {None, ""}:
        return float(default)
    return float(raw)


def _key(row: dict[str, str]) -> tuple[int, int, int]:
    return int(row["task_id"]), int(row["episode_id"]), int(row["seed"])


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    quantiles = np.quantile(array, [0.50, 0.90, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(array.max()),
    }


def _hierarchical_interval(
    left: dict[tuple[int, int, int], int],
    right: dict[tuple[int, int, int], int],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int | str]:
    if set(left) != set(right):
        raise RuntimeError("Paired bootstrap requires identical episode keys")
    by_task: dict[int, list[tuple[int, int]]] = {}
    for key in sorted(left):
        by_task.setdefault(key[0], []).append((left[key], right[key]))
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        differences: list[int] = []
        for task_id in rng.choice(tasks, size=len(tasks), replace=True):
            pairs = by_task[int(task_id)]
            draw = rng.integers(0, len(pairs), size=len(pairs))
            differences.extend(pairs[item][0] - pairs[item][1] for item in draw)
        samples[index] = np.mean(differences, dtype=np.float64)
    observed = np.mean([left[key] - right[key] for key in sorted(left)])
    return {
        "difference": float(observed),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "iterations": int(iterations),
        "seed": int(seed),
        "paired_episodes": len(left),
        "tasks": len(tasks),
        "method": "paired_task_hierarchical_bootstrap",
    }


def _paired_counts(
    left: dict[tuple[int, int, int], int],
    right: dict[tuple[int, int, int], int],
) -> dict[str, int]:
    if set(left) != set(right):
        raise RuntimeError("Paired transitions require identical episode keys")
    result = {
        "both_success": 0,
        "both_failure": 0,
        "right_fail_to_left_success": 0,
        "right_success_to_left_fail": 0,
    }
    for key in sorted(left):
        pair = left[key], right[key]
        if pair == (1, 1):
            result["both_success"] += 1
        elif pair == (0, 0):
            result["both_failure"] += 1
        elif pair == (1, 0):
            result["right_fail_to_left_success"] += 1
        else:
            result["right_success_to_left_fail"] += 1
    return result


def _task_rates(rows: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[int, list[int]] = {}
    names: dict[int, str] = {}
    for row in rows:
        task_id = int(row["task_id"])
        grouped.setdefault(task_id, []).append(int(float(row["success"])))
        names[task_id] = row["task_name"]
    return {
        str(task_id): {
            "task_name": names[task_id],
            "successes": int(sum(values)),
            "episodes": len(values),
            "success_rate": float(np.mean(values)),
        }
        for task_id, values in sorted(grouped.items())
    }


def _step_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    policy = [_float(row, "protocol_policy_ms") for row in rows]
    skipped = [
        _float(row, "protocol_policy_ms")
        for row in rows
        if row.get("mode") != "full"
    ]
    full = [
        _float(row, "full_forward_ms")
        for row in rows
        if int(float(row.get("full_forward_called", "0") or 0)) == 1
    ]
    fast = [
        _float(row, "fast_encoder_ms")
        for row in rows
        if int(float(row.get("fast_encoder_called", "0") or 0)) == 1
    ]
    update = [
        _float(row, "node_update_ms")
        for row in rows
        if int(float(row.get("lrnode_update_called", "0") or 0)) == 1
    ]
    action_head = [
        _float(row, "action_head_ms")
        for row in rows
        if int(float(row.get("action_head_called", "0") or 0)) == 1
    ]
    full_calls = sum(
        int(float(row.get("full_forward_called", "0") or 0)) for row in rows
    )

    trajectories: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    for row in rows:
        action = np.asarray([_float(row, f"action_{index}") for index in range(7)])
        episode_key = int(row["task_id"]), int(row["episode_id"])
        trajectories.setdefault(episode_key, []).append(
            (int(row["timestep"]), action)
        )
    translation_second: list[float] = []
    rotation_second: list[float] = []
    gripper_switches = 0
    action_steps = 0
    for trajectory in trajectories.values():
        actions = np.stack([item[1] for item in sorted(trajectory)])
        action_steps += len(actions)
        if len(actions) >= 3:
            second = actions[2:, :6] - 2.0 * actions[1:-1, :6] + actions[:-2, :6]
            translation_second.extend(np.linalg.norm(second[:, :3], axis=-1).tolist())
            rotation_second.extend(np.linalg.norm(second[:, 3:6], axis=-1).tolist())
        if len(actions) >= 2:
            gripper_switches += int(np.count_nonzero(actions[1:, 6] != actions[:-1, 6]))
    return {
        "policy_latency_ms": _distribution(policy),
        "skipped_step_latency_ms": _distribution(skipped),
        "full_forward_latency_ms": _distribution(full),
        "fast_encoder_latency_ms": _distribution(fast),
        "updater_latency_ms": _distribution(update),
        "skip_action_head_latency_ms": _distribution(action_head),
        "full_calls": int(full_calls),
        "policy_steps": len(rows),
        "full_call_reduction": 1.0 - full_calls / float(len(rows)),
        "translation_second_difference": _distribution(translation_second),
        "rotation_second_difference": _distribution(rotation_second),
        "gripper_switches_recomputed": gripper_switches,
        "gripper_switches_per_100_steps_recomputed": (
            100.0 * gripper_switches / float(max(1, action_steps))
        ),
    }


def _episode_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    successes = [int(float(row["success"])) for row in rows]
    steps = [_float(row, "num_steps") for row in rows]
    success_steps = [
        _float(row, "num_steps")
        for row in rows
        if int(float(row["success"])) == 1
    ]
    total_steps = max(1.0, sum(steps))
    return {
        "episodes": len(rows),
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "task_success_rates": _task_rates(rows),
        "environment_steps": _distribution(steps),
        "successful_completion_steps": _distribution(success_steps),
        "gripper_switches": int(sum(_float(row, "gripper_switch_count") for row in rows)),
        "gripper_switches_per_100_steps": 100.0
        * sum(_float(row, "gripper_switch_count") for row in rows)
        / total_steps,
        "gripper_reversals_within_1_per_100_steps": 100.0
        * sum(_float(row, "gripper_reverse_within_1_count") for row in rows)
        / total_steps,
        "gripper_reversals_within_2_per_100_steps": 100.0
        * sum(_float(row, "gripper_reverse_within_2_count") for row in rows)
        / total_steps,
        "gripper_reversals_within_5_per_100_steps": 100.0
        * sum(_float(row, "gripper_reverse_within_5_count") for row in rows)
        / total_steps,
    }


def _load_row(name: str, root: Path) -> RowData:
    episode_path = _find_one(root, "eval_episode_metrics.csv")
    analysis = episode_path.parent
    step_dir = analysis / "eval_step_logs"
    if not step_dir.is_dir():
        raise FileNotFoundError(f"Step logs are required for exact latency: {step_dir}")
    episode_rows = _read_csv(episode_path)
    step_paths = sorted(step_dir.glob("*.csv"))
    if not episode_rows or not step_paths:
        raise RuntimeError(f"Incomplete evaluator artifacts for {name}: {root}")
    step_rows = [row for path in step_paths for row in _read_csv(path)]
    outcomes = {_key(row): int(float(row["success"])) for row in episode_rows}
    steps = {_key(row): _float(row, "num_steps") for row in episode_rows}
    if len(outcomes) != len(episode_rows):
        raise RuntimeError(f"Duplicate episode keys in {episode_path}")
    summary = {
        **_episode_metrics(episode_rows),
        **_step_metrics(step_rows),
        "episode_metrics_path": str(episode_path.resolve()),
        "episode_metrics_sha256": _sha256(episode_path),
        "step_log_file_count": len(step_paths),
    }
    return RowData(
        name,
        root,
        episode_path,
        step_dir,
        episode_rows,
        step_rows,
        outcomes,
        steps,
        summary,
    )


def _selection_parameters(path: Path) -> int:
    selection = json.loads(path.read_text(encoding="utf-8"))
    selected = Path(selection["candidates"][0]["validation_artifact"])
    for candidate in selection["candidates"]:
        if candidate["checkpoint"] == selection["selected_checkpoint"]:
            selected = Path(candidate["validation_artifact"])
            break
    validation = json.loads(selected.read_text(encoding="utf-8"))
    return int(validation["adapter_parameter_report"]["baseline_total_parameters"])


def _common_success(left: RowData, right: RowData) -> dict[str, Any]:
    keys = [
        key
        for key in sorted(left.outcomes)
        if left.outcomes[key] and right.outcomes[key]
    ]
    differences = [left.steps[key] - right.steps[key] for key in keys]
    return {
        "episodes": len(keys),
        "left_mean_steps": float(np.mean([left.steps[key] for key in keys]))
        if keys
        else float("nan"),
        "right_mean_steps": float(np.mean([right.steps[key] for key in keys]))
        if keys
        else float("nan"),
        "left_minus_right_mean_steps": float(np.mean(differences))
        if differences
        else float("nan"),
    }


def _pair(
    left: RowData,
    right: RowData,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "left": left.name,
        "right": right.name,
        "success_rate": _hierarchical_interval(
            left.outcomes, right.outcomes, iterations=iterations, seed=seed
        ),
        "paired_outcomes": _paired_counts(left.outcomes, right.outcomes),
        "common_success_completion_steps": _common_success(left, right),
    }


def _validate_manifests(
    rows: dict[str, RowData], source_lock: dict[str, Any], stage: str
) -> list[str]:
    ordered = {name: sorted(row.outcomes) for name, row in rows.items()}
    reference = next(iter(ordered.values()))
    for name, keys in ordered.items():
        if keys != reference:
            raise RuntimeError(f"Episode manifest mismatch for row {name}")
    expected = 100 if stage == "screening" else 200 if stage == "confirmation" else None
    if expected is not None and len(reference) != expected:
        raise RuntimeError(f"{stage} requires {expected} episodes/row, got {len(reference)}")
    locked_keys = set(source_lock["evaluator"]["episode_keys"])
    current_keys = {f"{task}:{episode}:{seed}" for task, episode, seed in reference}
    if not current_keys.issubset(locked_keys):
        raise RuntimeError("Evaluation contains episode keys outside the source lock")
    if stage == "confirmation" and current_keys != locked_keys:
        raise RuntimeError("Confirmation manifest is not the exact locked 200 episodes")
    return [f"{task}:{episode}:{seed}" for task, episode, seed in reference]


def _screening_gate(
    summaries: dict[str, dict[str, Any]], parameters: dict[str, int]
) -> dict[str, Any]:
    full = summaries["full_seer_k1"]
    latent = summaries["canonical_latentloop_k4"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "predeclared_100_episode_screening_v1",
        "success_rate_margin": SCREENING_MARGIN,
        "material_compute_improvement": MATERIAL_COMPUTE_IMPROVEMENT,
        "uses_test_success_for_checkpoint_selection": False,
        "rows": {},
    }
    for name in NEW_BASELINES:
        row = summaries[name]
        within_margin = row["success_rate"] >= latent["success_rate"] - SCREENING_MARGIN
        at_least_k1 = row["success_rate"] >= full["success_rate"]
        lower_parameters = parameters[name] <= (
            1.0 - MATERIAL_COMPUTE_IMPROVEMENT
        ) * parameters["canonical_latentloop_k4"]
        candidate_skip = row["skipped_step_latency_ms"]["p50"]
        latent_skip = latent["skipped_step_latency_ms"]["p50"]
        lower_latency = (
            math.isfinite(float(candidate_skip))
            and float(candidate_skip)
            <= (1.0 - MATERIAL_COMPUTE_IMPROVEMENT) * float(latent_skip)
        )
        material_tradeoff = lower_parameters or lower_latency
        result["rows"][name] = {
            "proceed_to_confirmation": bool(
                within_margin or at_least_k1 or material_tradeoff
            ),
            "within_6pp_of_latentloop": bool(within_margin),
            "at_least_full_k1": bool(at_least_k1),
            "materially_lower_parameters": bool(lower_parameters),
            "materially_lower_skipped_p50_latency": bool(lower_latency),
            "screening_success_rate": row["success_rate"],
        }
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Aggregate rows and persist all paired evidence with overwrite refusal."""

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    source_lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    if source_lock.get("status") != "PASS":
        raise RuntimeError("Source lock is not PASS")
    row_paths: dict[str, Path] = {}
    for specification in args.row:
        name, separator, raw_path = specification.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Expected --row NAME=PATH, got {specification!r}")
        if name in row_paths:
            raise ValueError(f"Duplicate row name: {name}")
        row_paths[name] = Path(raw_path).resolve()
    if args.stage in {"screening", "confirmation"}:
        missing_fixed = {
            "full_seer_k1",
            "canonical_latentloop_k4",
            "no_observation_latentloop_k4",
            "predicted_horizon_replay_k4",
        } - set(row_paths)
        if missing_fixed:
            raise ValueError(f"Missing fixed comparison rows: {sorted(missing_fixed)}")
    rows = {name: _load_row(name, path) for name, path in row_paths.items()}
    episode_keys = _validate_manifests(rows, source_lock, args.stage)
    parameters = {
        "full_seer_k1": 0,
        "canonical_latentloop_k4": int(args.latentloop_parameters),
        "no_observation_latentloop_k4": int(args.latentloop_parameters),
        "predicted_horizon_replay_k4": 0,
    }
    if args.action_selection:
        parameters["matched_action_correction_k4"] = _selection_parameters(
            args.action_selection
        )
    if args.nonrecurrent_selection:
        parameters["nonrecurrent_latent_k4"] = _selection_parameters(
            args.nonrecurrent_selection
        )
    summaries = {name: dict(row.summary) for name, row in rows.items()}
    for name, summary in summaries.items():
        summary["trainable_parameters"] = parameters.get(name)

    pairs: dict[str, Any] = {}
    if "canonical_latentloop_k4" in rows:
        for name in (
            "full_seer_k1",
            "matched_action_correction_k4",
            "nonrecurrent_latent_k4",
            "no_observation_latentloop_k4",
            "predicted_horizon_replay_k4",
        ):
            if name in rows:
                pairs[f"canonical_minus_{name}"] = _pair(
                    rows["canonical_latentloop_k4"],
                    rows[name],
                    iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed,
                )
    if args.stage == "diagnostic":
        for k_value in (2, 8):
            canonical_name = f"canonical_latentloop_k{k_value}"
            for candidate_name in (
                f"matched_action_correction_k{k_value}",
                f"nonrecurrent_latent_k{k_value}",
            ):
                if canonical_name in rows and candidate_name in rows:
                    pairs[f"{canonical_name}_minus_{candidate_name}"] = _pair(
                        rows[canonical_name],
                        rows[candidate_name],
                        iterations=args.bootstrap_iterations,
                        seed=args.bootstrap_seed,
                    )

    result = {
        "schema_version": 1,
        "protocol": "seer_latentloop_q1_q2_online_aggregation_v1",
        "stage": args.stage,
        "source_lock": str(args.source_lock.resolve()),
        "source_lock_sha256": _sha256(args.source_lock),
        "episode_keys": episode_keys,
        "rows": summaries,
        "paired_comparisons": pairs,
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "comparison_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "row",
            "episodes",
            "successes",
            "success_rate",
            "trainable_parameters",
            "full_call_reduction",
            "policy_p50_ms",
            "policy_p95_ms",
            "policy_p99_ms",
            "skipped_p50_ms",
            "translation_second_difference_mean",
            "rotation_second_difference_mean",
            "gripper_switches_per_100_steps",
            "average_environment_steps",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in summaries.items():
            writer.writerow(
                {
                    "row": name,
                    "episodes": row["episodes"],
                    "successes": row["successes"],
                    "success_rate": row["success_rate"],
                    "trainable_parameters": row["trainable_parameters"],
                    "full_call_reduction": row["full_call_reduction"],
                    "policy_p50_ms": row["policy_latency_ms"]["p50"],
                    "policy_p95_ms": row["policy_latency_ms"]["p95"],
                    "policy_p99_ms": row["policy_latency_ms"]["p99"],
                    "skipped_p50_ms": row["skipped_step_latency_ms"]["p50"],
                    "translation_second_difference_mean": row[
                        "translation_second_difference"
                    ]["mean"],
                    "rotation_second_difference_mean": row[
                        "rotation_second_difference"
                    ]["mean"],
                    "gripper_switches_per_100_steps": row[
                        "gripper_switches_per_100_steps"
                    ],
                    "average_environment_steps": row["environment_steps"]["mean"],
                }
            )

    if args.stage == "screening":
        if not all(name in summaries for name in PRIMARY_NAMES):
            raise RuntimeError("Screening aggregation requires all six primary rows")
        gate = _screening_gate(summaries, parameters)
        gate["source_lock_sha256"] = result["source_lock_sha256"]
        (args.output_dir / "screening_gate.json").write_text(
            json.dumps(gate, indent=2) + "\n", encoding="utf-8"
        )

    if args.stage == "confirmation" and all(
        name in rows for name in NEW_BASELINES
    ):
        action_pair = pairs["canonical_minus_matched_action_correction_k4"]
        nonrecurrent_pair = pairs["canonical_minus_nonrecurrent_latent_k4"]
        evidence = (
            json.loads(args.additional_evidence.read_text(encoding="utf-8"))
            if args.additional_evidence
            else {}
        )
        canonical_chatter = summaries["canonical_latentloop_k4"]
        action_chatter = summaries["matched_action_correction_k4"]
        lower_chatter = (
            canonical_chatter["success_rate"] >= action_chatter["success_rate"]
            and canonical_chatter["gripper_switches_per_100_steps"]
            <= 0.9 * action_chatter["gripper_switches_per_100_steps"]
            and canonical_chatter["gripper_reversals_within_5_per_100_steps"]
            <= 0.9 * action_chatter["gripper_reversals_within_5_per_100_steps"]
        )
        decision_inputs = {
            "schema_version": 1,
            "source_aggregation": str(
                (args.output_dir / "comparison_summary.json").resolve()
            ),
            "latentloop_sr": summaries["canonical_latentloop_k4"]["success_rate"],
            "action_correction_sr": summaries["matched_action_correction_k4"][
                "success_rate"
            ],
            "nonrecurrent_sr": summaries["nonrecurrent_latent_k4"]["success_rate"],
            "latent_minus_action_ci_low": action_pair["success_rate"]["ci_low"],
            "action_minus_latent_ci_low": -action_pair["success_rate"]["ci_high"],
            "latent_minus_nonrecurrent_ci_low": nonrecurrent_pair["success_rate"][
                "ci_low"
            ],
            "nonrecurrent_minus_latent_ci_low": -nonrecurrent_pair[
                "success_rate"
            ]["ci_high"],
            "latentloop_parameters": parameters["canonical_latentloop_k4"],
            "action_correction_parameters": parameters[
                "matched_action_correction_k4"
            ],
            "nonrecurrent_parameters": parameters["nonrecurrent_latent_k4"],
            "latentloop_skip_p50_ms": summaries["canonical_latentloop_k4"][
                "skipped_step_latency_ms"
            ]["p50"],
            "action_correction_skip_p50_ms": summaries[
                "matched_action_correction_k4"
            ]["skipped_step_latency_ms"]["p50"],
            "nonrecurrent_skip_p50_ms": summaries["nonrecurrent_latent_k4"][
                "skipped_step_latency_ms"
            ]["p50"],
            "latentloop_better_k8_stability": bool(
                evidence.get("latentloop_better_k8_stability", False)
            ),
            "latentloop_lower_gripper_failure_or_chatter": bool(
                evidence.get(
                    "latentloop_lower_gripper_failure_or_chatter", lower_chatter
                )
            ),
            "latentloop_better_libero_plus": bool(
                evidence.get("latentloop_better_libero_plus", False)
            ),
            "latentloop_better_heldout_checkpoint": bool(
                evidence.get("latentloop_better_heldout_checkpoint", False)
            ),
            "latentloop_better_k8_than_nonrecurrent": bool(
                evidence.get("latentloop_better_k8_than_nonrecurrent", False)
            ),
            "latentloop_better_heldout_than_nonrecurrent": bool(
                evidence.get("latentloop_better_heldout_than_nonrecurrent", False)
            ),
        }
        (args.output_dir / "decision_inputs.json").write_text(
            json.dumps(decision_inputs, indent=2) + "\n", encoding="utf-8"
        )

    if args.stage == "diagnostic":
        expected = {
            f"{method}_k{k_value}"
            for method in (
                "canonical_latentloop",
                "matched_action_correction",
                "nonrecurrent_latent",
            )
            for k_value in (2, 8)
        }
        if set(rows) != expected:
            raise RuntimeError(
                f"Diagnostic aggregation requires rows {sorted(expected)}, got {sorted(rows)}"
            )
        primary = (
            json.loads(args.primary_summary.read_text(encoding="utf-8"))
            if args.primary_summary
            else None
        )
        action_k8 = pairs[
            "canonical_latentloop_k8_minus_matched_action_correction_k8"
        ]["success_rate"]
        nonrecurrent_k8 = pairs[
            "canonical_latentloop_k8_minus_nonrecurrent_latent_k8"
        ]["success_rate"]
        action_stable = action_k8["difference"] > 0.0 and action_k8["ci_low"] > 0.0
        nonrecurrent_stable = (
            nonrecurrent_k8["difference"] > 0.0 and nonrecurrent_k8["ci_low"] > 0.0
        )
        if primary is not None:
            primary_rows = primary["rows"]
            latent_k4 = primary_rows["canonical_latentloop_k4"]["success_rate"]
            action_k4 = primary_rows["matched_action_correction_k4"]["success_rate"]
            nonrecurrent_k4 = primary_rows["nonrecurrent_latent_k4"]["success_rate"]
            latent_k8 = summaries["canonical_latentloop_k8"]["success_rate"]
            action_k8_sr = summaries["matched_action_correction_k8"]["success_rate"]
            nonrecurrent_k8_sr = summaries["nonrecurrent_latent_k8"]["success_rate"]
            action_stable = action_stable and (
                latent_k4 - latent_k8 <= action_k4 - action_k8_sr
            )
            nonrecurrent_stable = nonrecurrent_stable and (
                latent_k4 - latent_k8 <= nonrecurrent_k4 - nonrecurrent_k8_sr
            )
        evidence = {
            "schema_version": 1,
            "protocol": "predeclared_k8_additional_evidence_v1",
            "requires_positive_k8_point_difference": True,
            "requires_positive_k8_ci_lower_bound": True,
            "requires_no_larger_k4_to_k8_degradation_when_primary_summary_is_supplied": bool(primary),
            "latentloop_better_k8_stability": bool(action_stable),
            "latentloop_better_k8_than_nonrecurrent": bool(nonrecurrent_stable),
            "latentloop_better_libero_plus": False,
            "latentloop_better_heldout_checkpoint": False,
            "latentloop_better_heldout_than_nonrecurrent": False,
        }
        (args.output_dir / "additional_evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )

    report = [
        "# Seer LatentLoop Q1/Q2 aggregate",
        "",
        f"- Stage: `{args.stage}`",
        f"- Source lock: `{result['source_lock_sha256']}`",
        f"- Shared episode keys: `{len(episode_keys)}`",
        "",
        "| Row | SR | Parameters | Full-call reduction | Policy p50/p95/p99 (ms) | Skip p50 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summaries.items():
        policy = row["policy_latency_ms"]
        report.append(
            f"| {name} | {row['successes']}/{row['episodes']} "
            f"({100.0 * row['success_rate']:.1f}%) | {row['trainable_parameters']} | "
            f"{100.0 * row['full_call_reduction']:.3f}% | "
            f"{policy['p50']:.3f}/{policy['p95']:.3f}/{policy['p99']:.3f} | "
            f"{row['skipped_step_latency_ms']['p50']:.3f} |"
        )
    report.extend(
        [
            "",
            "Second differences are recomputed from the postprocessed executed `action_0..6` step logs. "
            "Checkpoint selection remains validation-only; this online aggregate is used only for scientific comparison.",
        ]
    )
    (args.output_dir / "aggregate_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screening", "confirmation", "diagnostic"], required=True)
    parser.add_argument("--row", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--action-selection", type=Path)
    parser.add_argument("--nonrecurrent-selection", type=Path)
    parser.add_argument("--additional-evidence", type=Path)
    parser.add_argument("--primary-summary", type=Path)
    parser.add_argument("--latentloop-parameters", type=int, default=470146)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap-iterations must be positive")
    aggregate(args)


if __name__ == "__main__":
    main()
