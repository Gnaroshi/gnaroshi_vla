#!/usr/bin/env python3
"""Aggregate every-step latent-filter rollouts without loading model weights."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "architectures" / "seer" / "upstream"
for search_path in (REPO_ROOT, UPSTREAM):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from methods.latent_prediction_correction.decision import apply_decision_rule  # noqa: E402
from methods.latent_prediction_correction.metrics import (  # noqa: E402
    checkpoint_hierarchical_paired_bootstrap_ci,
    hierarchical_paired_bootstrap_ci,
    paired_outcome_counts,
    wilson_interval,
)
from utils.lrnode_mechanism_utils import load_trace_shard  # noqa: E402


PRIMARY_MODES = (
    "raw_full",
    "recurrent_prior",
    "fixed_filter",
    "full_latent_ema",
)


@dataclass(frozen=True)
class RowArtifacts:
    """Resolved artifacts for one predeclared evaluation row."""

    root: Path
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    episodes: Sequence[Mapping[str, str]]
    trace_metadata: Sequence[Path]

    @property
    def label(self) -> str:
        return str(self.manifest["row_label"])

    @property
    def mode(self) -> str:
        return str(self.manifest["mode"])

    @property
    def checkpoint(self) -> int:
        return int(self.manifest["checkpoint_id"])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in materialized))) if materialized else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(materialized)


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_exactly_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern!r} under {root}, found {len(matches)}")
    return matches[0]


def discover_rows(result_roots: Sequence[Path]) -> list[RowArtifacts]:
    """Discover row manifests and their canonical evaluator artifacts."""
    rows = []
    seen_roots = set()
    for result_root in result_roots:
        result_root = result_root.resolve()
        manifests = sorted(result_root.glob("**/latent_filter_row.json"))
        if not manifests:
            raise ValueError(f"No latent_filter_row.json found under {result_root}")
        for manifest_path in manifests:
            row_root = manifest_path.parent.resolve()
            if row_root in seen_roots:
                continue
            seen_roots.add(row_root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary_path = _find_exactly_one(row_root, "**/analysis/eval_summary.json")
            episode_path = _find_exactly_one(
                row_root,
                "**/analysis/eval_episode_metrics.csv",
            )
            rows.append(
                RowArtifacts(
                    root=row_root,
                    manifest=manifest,
                    summary=json.loads(summary_path.read_text(encoding="utf-8")),
                    episodes=_read_csv(episode_path),
                    trace_metadata=sorted(
                        row_root.glob("**/analysis/mechanism_trace/*.json")
                    ),
                )
            )
    return sorted(rows, key=lambda row: (row.checkpoint, row.label))


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def _mean_episode_metric(row: RowArtifacts, key: str) -> float:
    values = [_float(episode, key, math.nan) for episode in row.episodes]
    values = [value for value in values if not math.isnan(value)]
    return float(np.mean(values)) if values else math.nan


def _outcome_map(row: RowArtifacts) -> Dict[tuple[int, int], int]:
    return {
        (int(item["task_id"]), int(item["episode_id"])): int(item["success"])
        for item in row.episodes
    }


def _episode_metadata_map(row: RowArtifacts) -> Dict[tuple[int, int], Dict[str, int]]:
    return {
        (int(item["task_id"]), int(item["episode_id"])): {
            "success": int(item["success"]),
            "num_steps": int(item["num_steps"]),
        }
        for item in row.episodes
    }


def _trace_map(row: RowArtifacts) -> Dict[tuple[int, int], Dict[str, Any]]:
    traces = {}
    for metadata_path in row.trace_metadata:
        metadata, scalar_rows, tensors = load_trace_shard(metadata_path)
        episode = metadata["episode"]
        key = (int(episode["task_id"]), int(episode["episode_id"]))
        if key in traces:
            raise ValueError(f"Duplicate trace key {key} in {row.root}")
        traces[key] = {
            "metadata": metadata,
            "scalars": scalar_rows,
            "tensors": tensors,
        }
    return traces


def _present_tensor(tensors: Mapping[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    if key not in tensors:
        return None
    values = np.asarray(tensors[key])
    present = tensors.get(f"{key}__present")
    if present is None:
        return values
    mask = np.asarray(present, dtype=bool)
    return values[mask]


def _paired_present_tensors(
    tensors: Mapping[str, np.ndarray],
    left_key: str,
    right_key: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return two trace tensors aligned by their shared timestep presence."""
    if left_key not in tensors or right_key not in tensors:
        return None, None
    left = np.asarray(tensors[left_key])
    right = np.asarray(tensors[right_key])
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"Trace timestep mismatch for {left_key}/{right_key}: "
            f"{left.shape[0]} vs {right.shape[0]}"
        )
    left_present = np.asarray(
        tensors.get(f"{left_key}__present", np.ones(left.shape[0], dtype=bool)),
        dtype=bool,
    )
    right_present = np.asarray(
        tensors.get(f"{right_key}__present", np.ones(right.shape[0], dtype=bool)),
        dtype=bool,
    )
    if left_present.shape != (left.shape[0],):
        raise ValueError(
            f"Invalid presence mask for {left_key}: {left_present.shape}"
        )
    if right_present.shape != (right.shape[0],):
        raise ValueError(
            f"Invalid presence mask for {right_key}: {right_present.shape}"
        )
    shared = left_present & right_present
    return left[shared], right[shared]


def _raw_action_tokens(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    while array.ndim > 3 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3 or array.shape[-1] != 7:
        raise ValueError(f"Expected raw action tokens [T,H,7], got {array.shape}")
    return array


def _append_pair_action_metrics(
    accumulator: Dict[str, list[float]],
    prefix: str,
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
) -> None:
    if left is None or right is None:
        return
    left_tokens = _raw_action_tokens(left)
    right_tokens = _raw_action_tokens(right)
    if left_tokens.shape != right_tokens.shape:
        raise ValueError(
            f"Action diagnostic shape mismatch: {left_tokens.shape} vs {right_tokens.shape}"
        )
    delta = left_tokens - right_tokens
    accumulator[f"{prefix}_first_translation_l2"].extend(
        np.linalg.norm(delta[:, 0, :3], axis=-1).tolist()
    )
    accumulator[f"{prefix}_first_rotation_l2"].extend(
        np.linalg.norm(delta[:, 0, 3:6], axis=-1).tolist()
    )
    accumulator[f"{prefix}_first_gripper_probability_abs"].extend(
        np.abs(delta[:, 0, 6]).tolist()
    )
    accumulator[f"{prefix}_all_token_translation_l2"].extend(
        np.linalg.norm(delta[..., :3], axis=-1).reshape(-1).tolist()
    )
    accumulator[f"{prefix}_all_token_rotation_l2"].extend(
        np.linalg.norm(delta[..., 3:6], axis=-1).reshape(-1).tolist()
    )
    accumulator[f"{prefix}_all_token_gripper_probability_abs"].extend(
        np.abs(delta[..., 6]).reshape(-1).tolist()
    )


def _trace_metrics(row: RowArtifacts) -> Dict[str, float]:
    translation_second = []
    rotation_second = []
    total_steps = 0
    switch_count = 0
    reverse_counts = {1: 0, 2: 0, 5: 0}
    action_pairs: Dict[str, list[float]] = defaultdict(list)
    timing_samples: Dict[str, list[float]] = defaultdict(list)

    for trace in _trace_map(row).values():
        tensors = trace["tensors"]
        final_actions = _present_tensor(tensors, "a_executed_final")
        if final_actions is None:
            continue
        actions = np.asarray(final_actions, dtype=np.float64).reshape(-1, 7)
        total_steps += len(actions)
        if len(actions) >= 3:
            second = actions[2:, :6] - 2.0 * actions[1:-1, :6] + actions[:-2, :6]
            translation_second.extend(np.linalg.norm(second[:, :3], axis=-1).tolist())
            rotation_second.extend(np.linalg.norm(second[:, 3:6], axis=-1).tolist())
        states = actions[:, 6] > 0.0
        switches = np.flatnonzero(states[1:] != states[:-1]) + 1
        switch_count += len(switches)
        for index, switch in enumerate(switches[:-1]):
            distance = int(switches[index + 1] - switch)
            for horizon in reverse_counts:
                reverse_counts[horizon] += int(distance <= horizon)

        filtered, full_for_filter = _paired_present_tensors(
            tensors,
            "a_filter_raw",
            "a_full_raw",
        )
        prior, full_for_prior = _paired_present_tensors(
            tensors,
            "a_prior_raw",
            "a_full_raw",
        )
        _append_pair_action_metrics(
            action_pairs,
            "filter_vs_full",
            filtered,
            full_for_filter,
        )
        _append_pair_action_metrics(
            action_pairs,
            "prior_vs_full",
            prior,
            full_for_prior,
        )

        for scalar in trace["scalars"]:
            for timing_key in (
                "protocol_policy_ms",
                "causal_executed_policy_ms",
                "diagnostic_full_forward_ms",
                "filter_diagnostic_ms",
                "total_diagnostic_only_ms",
            ):
                if scalar.get(timing_key, "") not in ("", None):
                    timing_samples[timing_key].append(float(scalar[timing_key]))
            for prefix, left_key, right_key in (
                (
                    "filter_vs_full",
                    "filter_raw_first_token_gripper_logit",
                    "full_raw_first_token_gripper_logit",
                ),
                (
                    "prior_vs_full",
                    "prior_raw_first_token_gripper_logit",
                    "full_raw_first_token_gripper_logit",
                ),
            ):
                if scalar.get(left_key, "") not in ("", None) and scalar.get(
                    right_key,
                    "",
                ) not in ("", None):
                    action_pairs[f"{prefix}_first_gripper_logit_abs"].append(
                        abs(float(scalar[left_key]) - float(scalar[right_key]))
                    )

    def summarize(values: Sequence[float], statistic: str) -> float:
        if not values:
            return math.nan
        if statistic == "mean":
            return float(np.mean(values))
        return float(np.percentile(values, 95))

    metrics = {
        "translation_second_difference_mean": summarize(translation_second, "mean"),
        "translation_second_difference_p95": summarize(translation_second, "p95"),
        "rotation_second_difference_mean": summarize(rotation_second, "mean"),
        "rotation_second_difference_p95": summarize(rotation_second, "p95"),
        "gripper_switches_per_100_steps": 100.0 * switch_count / max(1, total_steps),
        **{
            f"gripper_reverse_within_{horizon}_per_100_steps": (
                100.0 * count / max(1, total_steps)
            )
            for horizon, count in reverse_counts.items()
        },
        "trace_episode_count": len(row.trace_metadata),
        "trace_policy_step_count": total_steps,
    }
    for key, values in action_pairs.items():
        metrics[f"{key}_mean"] = summarize(values, "mean")
        metrics[f"{key}_p95"] = summarize(values, "p95")
    for key, values in timing_samples.items():
        metrics[f"{key}_mean"] = summarize(values, "mean")
        metrics[f"{key}_total_sec"] = float(np.sum(values)) / 1000.0
    return metrics


def _row_main_metrics(row: RowArtifacts) -> Dict[str, Any]:
    successes = sum(int(item["success"]) for item in row.episodes)
    total = len(row.episodes)
    low, high = wilson_interval(successes, total)
    lrnode = row.summary.get("lrnode", {})
    query = row.summary.get("query_reduction", {})
    trace = _trace_metrics(row)
    output: Dict[str, Any] = {
        "scope": row.manifest.get("scope", "unspecified"),
        "row_label": row.label,
        "checkpoint_id": row.checkpoint,
        "adapter_id": row.manifest.get("adapter_id", ""),
        "mode": row.mode,
        "alpha": float(row.manifest.get("alpha", 0.5)),
        "beta": float(row.manifest.get("beta", 0.5)),
        "diagnostics": int(row.manifest.get("diagnostics", 0)),
        "successes": successes,
        "num_episodes": total,
        "success_rate": successes / total if total else math.nan,
        "wilson_low": low,
        "wilson_high": high,
        "full_seer_calls": int(query.get("num_full_forward_calls", 0)),
        "policy_steps": int(query.get("num_env_steps", 0)),
        "lrnode_update_calls": int(query.get("num_lrnode_update_calls", 0)),
        "filter_prior_calls": int(
            lrnode.get("every_step_filter_prior_calls", 0)
        ),
        "filter_fusion_calls": int(
            lrnode.get("every_step_filter_fusion_calls", 0)
        ),
        "filter_action_head_calls": int(
            row.summary.get("num_filter_action_head_calls", 0)
        ),
        "full_forward_calls_per_policy_step": float(
            query.get("full_forward_calls_per_policy_step", 0.0)
        ),
        "query_reduction_claim_allowed": int(
            bool(query.get("query_reduction_claim_allowed", True))
        ),
        "full_query_reduction_ratio": float(
            query.get("full_query_reduction_ratio", 0.0)
        ),
        "avg_full_seer_latency_ms": 1000.0
        * float(lrnode.get("avg_full_forward_latency_sec", 0.0)),
        "avg_full_action_decoder_latency_ms": 1000.0
        * float(lrnode.get("avg_full_action_head_latency_sec", 0.0)),
        "avg_full_non_action_decoder_latency_ms": 1000.0
        * float(lrnode.get("avg_full_non_action_head_latency_sec", 0.0)),
        "avg_lrnode_prior_latency_ms": 1000.0
        * float(lrnode.get("avg_every_step_filter_prior_latency_sec", 0.0)),
        "avg_fusion_latency_ms": 1000.0
        * float(lrnode.get("avg_every_step_filter_fusion_latency_sec", 0.0)),
        "avg_action_decoder_latency_ms": 1000.0
        * float(lrnode.get("avg_every_step_filter_action_head_latency_sec", 0.0)),
        "avg_protocol_policy_latency_ms": 1000.0
        * float(lrnode.get("avg_policy_step_latency_sec", 0.0)),
        "avg_logging_diagnostic_latency_ms": 1000.0
        * float(lrnode.get("avg_every_step_filter_diagnostic_latency_sec", 0.0)),
        "diagnostic_action_head_calls": int(
            lrnode.get("every_step_filter_diagnostic_action_head_calls", 0)
        ),
        "diagnostic_rng_failures": int(
            lrnode.get("every_step_filter_rng_failures", 0)
        ),
        **trace,
    }
    output["avg_causal_executed_policy_latency_ms"] = float(
        trace.get(
            "causal_executed_policy_ms_mean",
            output["avg_protocol_policy_latency_ms"],
        )
    )
    output["avg_diagnostic_full_forward_latency_ms"] = float(
        trace.get("diagnostic_full_forward_ms_mean", 0.0)
    )
    output["avg_total_diagnostic_only_latency_ms"] = float(
        trace.get(
            "total_diagnostic_only_ms_mean",
            output["avg_logging_diagnostic_latency_ms"],
        )
    )
    # Backward-readable names point to the causally executed branch and the
    # complete diagnostic-only branch, respectively.
    output["avg_executed_policy_latency_ms"] = output[
        "avg_causal_executed_policy_latency_ms"
    ]
    output["avg_diagnostic_latency_ms"] = output[
        "avg_total_diagnostic_only_latency_ms"
    ]
    output.update(
        {
            "total_full_seer_latency_sec": (
                output["avg_full_seer_latency_ms"]
                * output["full_seer_calls"] / 1000.0
            ),
            "total_full_action_decoder_latency_sec": (
                output["avg_full_action_decoder_latency_ms"]
                * output["full_seer_calls"] / 1000.0
            ),
            "total_lrnode_prior_latency_sec": (
                output["avg_lrnode_prior_latency_ms"]
                * output["filter_prior_calls"] / 1000.0
            ),
            "total_fusion_latency_sec": (
                output["avg_fusion_latency_ms"]
                * output["policy_steps"] / 1000.0
            ),
            "total_action_decoder_latency_sec": (
                output["avg_action_decoder_latency_ms"]
                * output["filter_action_head_calls"] / 1000.0
            ),
            "total_executed_policy_latency_sec": (
                output["avg_executed_policy_latency_ms"]
                * output["policy_steps"] / 1000.0
            ),
            "total_protocol_policy_latency_sec": (
                output["avg_protocol_policy_latency_ms"]
                * output["policy_steps"] / 1000.0
            ),
            "total_diagnostic_latency_sec": (
                output["avg_diagnostic_latency_ms"]
                * output["policy_steps"] / 1000.0
            ),
        }
    )
    latent_keys = (
        "avg_latent_prior_vs_full_l2",
        "avg_latent_prior_vs_full_cosine",
        "avg_latent_filter_vs_full_l2",
        "avg_latent_correction_l2",
        "recurrent_prior_path_length",
        "full_latent_path_length",
        "filtered_latent_path_length",
        "latent_prior_second_difference_mean",
        "latent_full_second_difference_mean",
        "latent_filter_second_difference_mean",
    )
    for key in latent_keys:
        output[key] = _mean_episode_metric(row, key)
    for token_index in range(3):
        for name in (
            "full_norm",
            "prior_norm",
            "filter_norm",
            "prior_vs_full_l2",
            "filter_vs_full_l2",
        ):
            key = f"avg_latent_token{token_index}_{name}"
            output[key] = _mean_episode_metric(row, key)
    if row.mode not in {"recurrent_prior", "fixed_filter"}:
        for key in list(output):
            if (
                "latent_prior" in key
                or "prior_path" in key
                or ("latent_token" in key and "_prior" in key)
            ):
                output[key] = math.nan
    if row.mode != "fixed_filter":
        output["avg_latent_correction_l2"] = math.nan
    return output


def _per_task_rows(rows: Sequence[RowArtifacts]) -> list[Dict[str, Any]]:
    output = []
    for row in rows:
        grouped: Dict[int, list[Mapping[str, str]]] = defaultdict(list)
        for episode in row.episodes:
            grouped[int(episode["task_id"])].append(episode)
        for task_id, episodes in sorted(grouped.items()):
            successes = sum(int(item["success"]) for item in episodes)
            low, high = wilson_interval(successes, len(episodes))
            output.append(
                {
                    "scope": row.manifest.get("scope", "unspecified"),
                    "checkpoint_id": row.checkpoint,
                    "row_label": row.label,
                    "mode": row.mode,
                    "task_id": task_id,
                    "task_name": episodes[0].get("task_name", ""),
                    "successes": successes,
                    "num_episodes": len(episodes),
                    "success_rate": successes / len(episodes),
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    return output


def _paired_rows(
    rows: Sequence[RowArtifacts],
    main_rows: list[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, int]]]:
    output = []
    all_fixed_pairs: list[Dict[str, int]] = []
    main_index = {
        (int(item["checkpoint_id"]), str(item["row_label"])): item
        for item in main_rows
    }
    by_checkpoint: Dict[int, list[RowArtifacts]] = defaultdict(list)
    for row in rows:
        if row.manifest.get("scope") == "primary_heldout":
            by_checkpoint[row.checkpoint].append(row)
    for checkpoint, checkpoint_rows in sorted(by_checkpoint.items()):
        raw_matches = [row for row in checkpoint_rows if row.mode == "raw_full"]
        if len(raw_matches) != 1:
            continue
        raw = raw_matches[0]
        raw_outcomes = _outcome_map(raw)
        for candidate in checkpoint_rows:
            if candidate is raw or candidate.mode not in PRIMARY_MODES:
                continue
            candidate_outcomes = _outcome_map(candidate)
            counts = paired_outcome_counts(raw_outcomes, candidate_outcomes)
            common = sorted(set(raw_outcomes) & set(candidate_outcomes))
            pairs = [
                {
                    "checkpoint_id": checkpoint,
                    "task_id": key[0],
                    "episode_id": key[1],
                    "baseline_success": raw_outcomes[key],
                    "candidate_success": candidate_outcomes[key],
                }
                for key in common
            ]
            ci_low, ci_high = hierarchical_paired_bootstrap_ci(pairs)
            pair_row = {
                "checkpoint_id": checkpoint,
                "baseline_row": raw.label,
                "candidate_row": candidate.label,
                "candidate_mode": candidate.mode,
                **counts,
                "paired_sr_difference": (
                    np.mean(
                        [
                            pair["candidate_success"] - pair["baseline_success"]
                            for pair in pairs
                        ]
                    )
                    if pairs else math.nan
                ),
                "paired_bootstrap_low": ci_low,
                "paired_bootstrap_high": ci_high,
                "bootstrap_hierarchy": "task_then_episode",
            }
            output.append(pair_row)
            main = main_index[(checkpoint, candidate.label)]
            main["paired_net_flip_vs_raw"] = counts["net_flip"]
            main["paired_fail_to_success_vs_raw"] = counts["fail_to_success"]
            main["paired_success_to_fail_vs_raw"] = counts["success_to_fail"]
            main["paired_bootstrap_low_vs_raw"] = ci_low
            main["paired_bootstrap_high_vs_raw"] = ci_high
            if candidate.mode == "fixed_filter":
                all_fixed_pairs.extend(pairs)
    if all_fixed_pairs:
        low, high = checkpoint_hierarchical_paired_bootstrap_ci(all_fixed_pairs)
        output.append(
            {
                "checkpoint_id": "heldout_36_38",
                "baseline_row": "raw_full",
                "candidate_row": "fixed_filter_alpha0.50",
                "candidate_mode": "fixed_filter",
                "paired": len(all_fixed_pairs),
                "paired_sr_difference": float(
                    np.mean(
                        [
                            row["candidate_success"] - row["baseline_success"]
                            for row in all_fixed_pairs
                        ]
                    )
                ),
                "paired_bootstrap_low": low,
                "paired_bootstrap_high": high,
                "bootstrap_hierarchy": "checkpoint_then_task_then_episode",
            }
        )
    return output, all_fixed_pairs


def _tensor_equal(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    key: str,
) -> bool:
    left_value = _present_tensor(left, key)
    right_value = _present_tensor(right, key)
    if left_value is None or right_value is None:
        return False
    return left_value.shape == right_value.shape and np.array_equal(
        left_value,
        right_value,
        equal_nan=True,
    )


def _compare_endpoint_rows(
    left: RowArtifacts,
    right: RowArtifacts,
    tensor_keys: Sequence[str],
) -> Dict[str, Any]:
    left_meta = _episode_metadata_map(left)
    right_meta = _episode_metadata_map(right)
    left_traces = _trace_map(left)
    right_traces = _trace_map(right)
    keys_match = set(left_meta) == set(right_meta) == set(left_traces) == set(right_traces)
    detail = []
    for key in sorted(set(left_traces) | set(right_traces)):
        if key not in left_traces or key not in right_traces:
            detail.append({"episode_key": list(key), "missing_trace": True})
            continue
        tensor_equal = {
            tensor_key: _tensor_equal(
                left_traces[key]["tensors"],
                right_traces[key]["tensors"],
                tensor_key,
            )
            for tensor_key in tensor_keys
        }
        detail.append(
            {
                "episode_key": list(key),
                "metadata_equal": left_meta.get(key) == right_meta.get(key),
                "tensor_equal": tensor_equal,
            }
        )
    exact = keys_match and all(
        item.get("metadata_equal", False)
        and all(item.get("tensor_equal", {}).values())
        for item in detail
    )
    return {
        "left": left.label,
        "right": right.label,
        "keys_match": keys_match,
        "exact": exact,
        "episodes": detail,
    }


def endpoint_parity(rows: Sequence[RowArtifacts]) -> Dict[str, Any]:
    """Apply the predeclared exact smoke checks to trace artifacts."""
    by_label = {row.label: row for row in rows}
    required = {
        "canonical_k1",
        "raw_full",
        "raw_full_diagnostics_on",
        "fixed_filter_alpha1",
        "recurrent_prior",
        "fixed_filter_alpha0",
    }
    missing = sorted(required - set(by_label))
    if missing:
        raise ValueError(f"Endpoint smoke rows are missing: {missing}")
    comparisons = {
        "canonical_vs_raw": _compare_endpoint_rows(
            by_label["canonical_k1"],
            by_label["raw_full"],
            ("a_executed_final", "a_executed_raw"),
        ),
        "raw_vs_alpha1": _compare_endpoint_rows(
            by_label["raw_full"],
            by_label["fixed_filter_alpha1"],
            ("a_executed_final", "a_executed_raw"),
        ),
        "recurrent_vs_alpha0": _compare_endpoint_rows(
            by_label["recurrent_prior"],
            by_label["fixed_filter_alpha0"],
            ("a_executed_final", "a_executed_raw", "z_executed"),
        ),
        "diagnostics_off_vs_on": _compare_endpoint_rows(
            by_label["raw_full"],
            by_label["raw_full_diagnostics_on"],
            ("a_executed_final", "a_executed_raw"),
        ),
    }

    def counters(label: str) -> Dict[str, int]:
        summary = by_label[label].summary
        query = summary.get("query_reduction", {})
        return {
            "policy_steps": int(query.get("num_env_steps", 0)),
            "full_calls": int(query.get("num_full_forward_calls", 0)),
            "updater_calls": int(query.get("num_lrnode_update_calls", 0)),
            "rng_failures": int(
                summary.get("lrnode", {}).get("every_step_filter_rng_failures", 0)
            ),
        }

    counter_rows = {label: counters(label) for label in sorted(required)}
    full_endpoint_invariants = all(
        counter_rows[label]["full_calls"] == counter_rows[label]["policy_steps"]
        and counter_rows[label]["updater_calls"] == 0
        for label in ("raw_full", "fixed_filter_alpha1")
    )
    diagnostics_rng_pass = all(
        counter_rows[label]["rng_failures"] == 0
        for label in ("raw_full", "raw_full_diagnostics_on")
    )
    alpha_1_exact = (
        comparisons["canonical_vs_raw"]["exact"]
        and comparisons["raw_vs_alpha1"]["exact"]
        and full_endpoint_invariants
    )
    alpha_0_exact = comparisons["recurrent_vs_alpha0"]["exact"]
    diagnostics_invariant = (
        comparisons["diagnostics_off_vs_on"]["exact"] and diagnostics_rng_pass
    )
    return {
        "schema_version": 1,
        "alpha_1_exact": alpha_1_exact,
        "alpha_0_exact": alpha_0_exact,
        "raw_full_exact": comparisons["canonical_vs_raw"]["exact"],
        "diagnostics_invariant": diagnostics_invariant,
        "full_endpoint_invariants": full_endpoint_invariants,
        "diagnostics_rng_pass": diagnostics_rng_pass,
        "all_pass": alpha_1_exact and alpha_0_exact and diagnostics_invariant,
        "counters": counter_rows,
        "comparisons": comparisons,
    }


def _write_endpoint_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    _write_json(output_dir / "endpoint_parity.json", payload)
    lines = [
        "# Endpoint parity report",
        "",
        f"- Overall: **{'PASS' if payload['all_pass'] else 'FAIL'}**",
        f"- canonical K1 = raw_full: `{payload['raw_full_exact']}`",
        f"- alpha=1 = raw_full: `{payload['alpha_1_exact']}`",
        f"- alpha=0 = recurrent_prior: `{payload['alpha_0_exact']}`",
        f"- diagnostics off = on: `{payload['diagnostics_invariant']}`",
        "",
        "A FAIL is a hard stop: do not launch checkpoint-36/38 primary rows.",
    ]
    (output_dir / "endpoint_parity_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _report_sections(rows: Sequence[RowArtifacts]) -> Dict[str, list[str]]:
    sections: Dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sections[str(row.manifest.get("scope", "unspecified"))].append(row.label)
    return {key: sorted(values) for key, values in sections.items()}


def aggregate(
    rows: Sequence[RowArtifacts],
    output_dir: Path,
    endpoint: Optional[Mapping[str, Any]],
    decision_rule: Optional[Mapping[str, Any]],
    require_primary_complete: bool,
) -> None:
    """Write all offline aggregate tables and the predeclared decision."""
    primary = [row for row in rows if row.manifest.get("scope") == "primary_heldout"]
    expected = {(checkpoint, mode) for checkpoint in (36, 38) for mode in PRIMARY_MODES}
    observed = {(row.checkpoint, row.mode) for row in primary}
    if require_primary_complete and observed != expected:
        raise ValueError(
            f"Incomplete primary design: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    main_rows = [_row_main_metrics(row) for row in rows]
    paired_rows, _ = _paired_rows(rows, main_rows)
    per_task = _per_task_rows(rows)
    continuity_keys = (
        "scope",
        "checkpoint_id",
        "row_label",
        "mode",
        "translation_second_difference_mean",
        "translation_second_difference_p95",
        "rotation_second_difference_mean",
        "rotation_second_difference_p95",
        "gripper_switches_per_100_steps",
        "gripper_reverse_within_1_per_100_steps",
        "gripper_reverse_within_2_per_100_steps",
        "gripper_reverse_within_5_per_100_steps",
    )
    latency_keys = (
        "scope",
        "checkpoint_id",
        "row_label",
        "mode",
        "full_seer_calls",
        "policy_steps",
        "lrnode_update_calls",
        "filter_prior_calls",
        "filter_fusion_calls",
        "filter_action_head_calls",
        "avg_full_seer_latency_ms",
        "avg_full_action_decoder_latency_ms",
        "avg_full_non_action_decoder_latency_ms",
        "avg_lrnode_prior_latency_ms",
        "avg_fusion_latency_ms",
        "avg_action_decoder_latency_ms",
        "avg_protocol_policy_latency_ms",
        "avg_causal_executed_policy_latency_ms",
        "avg_diagnostic_full_forward_latency_ms",
        "avg_logging_diagnostic_latency_ms",
        "avg_total_diagnostic_only_latency_ms",
        "avg_executed_policy_latency_ms",
        "avg_diagnostic_latency_ms",
        "total_full_seer_latency_sec",
        "total_full_action_decoder_latency_sec",
        "total_lrnode_prior_latency_sec",
        "total_fusion_latency_sec",
        "total_action_decoder_latency_sec",
        "total_executed_policy_latency_sec",
        "total_protocol_policy_latency_sec",
        "total_diagnostic_latency_sec",
        "diagnostic_action_head_calls",
        "query_reduction_claim_allowed",
        "full_query_reduction_ratio",
    )
    _write_csv(output_dir / "latent_filter_main_table.csv", main_rows)
    _write_csv(output_dir / "latent_filter_per_task.csv", per_task)
    _write_csv(output_dir / "latent_filter_paired_flips.csv", paired_rows)
    _write_csv(
        output_dir / "latent_filter_continuity.csv",
        [{key: row.get(key, "") for key in continuity_keys} for row in main_rows],
    )
    _write_csv(
        output_dir / "latent_filter_latency.csv",
        [{key: row.get(key, "") for key in latency_keys} for row in main_rows],
    )

    decision = None
    if decision_rule is not None:
        if endpoint is None:
            raise ValueError("Decision application requires endpoint parity results")
        primary_decision_rows = [
            row for row in main_rows if row.get("scope") == "primary_heldout"
        ]
        decision = apply_decision_rule(
            primary_decision_rows,
            endpoint,
            decision_rule,
        )
        _write_json(output_dir / "latent_filter_decision.json", decision)

    sections = _report_sections(rows)
    lines = [
        "# Final latent-filter result report",
        "",
        "This report is generated only from completed rollout artifacts. It does not "
        "pool checkpoint 36 and 38 episodes as one independent Bernoulli sample.",
        "",
        "## Result scopes",
        "",
    ]
    for scope, labels in sorted(sections.items()):
        lines.append(f"- `{scope}`: {', '.join(labels)}")
    lines.extend(
        [
            "",
            "## Held-out analysis",
            "",
            "Per-checkpoint Wilson intervals and task-then-episode paired bootstrap "
            "intervals are in the CSV tables. The combined fixed-vs-raw interval "
            "uses checkpoint-then-task-then-episode hierarchical resampling.",
            "",
            "## Compute interpretation",
            "",
            "Every primary row executes full Seer at every policy step. The LR-NODE "
            "prior, fusion, extra decoder, and diagnostics are reported separately. "
            "No query-reduction benefit is claimed for this experiment.",
            "",
            "## Decision",
            "",
            (
                f"Predeclared verdict: **{decision['verdict']}**."
                if decision is not None
                else "Decision not applied in this aggregation run."
            ),
            "",
            "Checkpoint 33 screening is exploratory; checkpoint 36/38 rows are the "
            "checkpoint-held-out confirmation. Independent-state repeats and optional "
            "alpha curves remain separate secondary analyses.",
        ]
    )
    (output_dir / "final_latent_filter_result_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse offline artifact roots and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", action="append", default=[], type=Path)
    parser.add_argument("--endpoint-smoke-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--decision-rule", type=Path)
    parser.add_argument("--require-primary-complete", action="store_true")
    parser.add_argument("--endpoint-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run endpoint validation and/or result aggregation."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = None
    if args.endpoint_smoke_root is not None:
        smoke_rows = discover_rows([args.endpoint_smoke_root])
        endpoint = endpoint_parity(smoke_rows)
        _write_endpoint_report(args.output_dir, endpoint)
        if args.endpoint_only:
            if not endpoint["all_pass"]:
                raise SystemExit(2)
            return
    if not args.result_root:
        raise ValueError("At least one --result-root is required unless --endpoint-only")
    rows = discover_rows(args.result_root)
    rule = (
        json.loads(args.decision_rule.read_text(encoding="utf-8"))
        if args.decision_rule is not None else None
    )
    aggregate(
        rows=rows,
        output_dir=args.output_dir,
        endpoint=endpoint,
        decision_rule=rule,
        require_primary_complete=args.require_primary_complete,
    )


if __name__ == "__main__":
    main()
