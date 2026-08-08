"""Revalidate K=1 bypass parity without comparing different rollout states."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .source_lock import require_empty_output, sha256_file


BASELINE_ROW = "full_k1"
ADAPTER_ROW = "adapter_loaded_full_k1"
REQUIRED_FILES = (
    "online_summary.json",
    "eval_config.json",
    "episode_metrics.csv",
    "query_trace.jsonl",
    "source_lock.json",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0"}:
        raise ValueError(f"invalid boolean value in episode CSV: {value!r}")
    return normalized in {"true", "1"}


def _episode_outcomes(path: Path) -> dict[str, dict[tuple[int, int], bool]]:
    outcomes: dict[str, dict[tuple[int, int], bool]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            row = record["row"]
            key = (int(record["task_id"]), int(record["episode"]))
            if key in outcomes[row]:
                raise ValueError(f"duplicate episode row {row} key {key}")
            outcomes[row][key] = _as_bool(record["success"])
    return dict(outcomes)


def _query_traces(path: Path) -> dict[str, dict[tuple[int, int, int], dict[str, Any]]]:
    traces: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            row = str(record["row"])
            key = (
                int(record["task_id"]),
                int(record["episode"]),
                int(record["policy_query_index"]),
            )
            if key in traces[row]:
                raise ValueError(f"duplicate query row {row} key {key} at line {line_number}")
            traces[row][key] = record
    return dict(traces)


def analyze_k1_parity(
    *,
    raw_summary: dict[str, Any],
    outcomes: dict[str, dict[tuple[int, int], bool]],
    traces: dict[str, dict[tuple[int, int, int], dict[str, Any]]],
) -> dict[str, Any]:
    """Check exact actions only where condition and action noise are identical."""

    if raw_summary.get("matrix") != "k1_parity":
        raise ValueError("raw summary is not a k1_parity run")
    if set(raw_summary.get("rows", {})) != {BASELINE_ROW, ADAPTER_ROW}:
        raise ValueError("raw summary does not contain the two required K1 rows")
    if set(outcomes) != {BASELINE_ROW, ADAPTER_ROW}:
        raise ValueError("episode CSV does not contain the two required K1 rows")
    if set(traces) != {BASELINE_ROW, ADAPTER_ROW}:
        raise ValueError("query trace does not contain the two required K1 rows")

    baseline_outcomes = outcomes[BASELINE_ROW]
    adapter_outcomes = outcomes[ADAPTER_ROW]
    baseline = traces[BASELINE_ROW]
    adapter = traces[ADAPTER_ROW]
    common_keys = sorted(set(baseline) & set(adapter))
    missing_keys = sorted(set(baseline) ^ set(adapter))
    outcome_keys = sorted(set(baseline_outcomes) | set(adapter_outcomes))

    matched_input_keys: list[tuple[int, int, int]] = []
    action_mismatch_on_matched_input: list[tuple[int, int, int]] = []
    noise_mismatch_keys: list[tuple[int, int, int]] = []
    condition_divergence_keys: list[tuple[int, int, int]] = []
    raw_action_mismatch_keys: list[tuple[int, int, int]] = []
    first_divergence_by_episode: dict[str, int] = {}

    common_by_episode: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for key in common_keys:
        common_by_episode[key[:2]].append(key)
        left = baseline[key]
        right = adapter[key]
        noise_equal = (
            left.get("action_noise_seed") == right.get("action_noise_seed")
            and left.get("action_noise_hash") == right.get("action_noise_hash")
        )
        condition_equal = left.get("condition_hash") == right.get("condition_hash")
        action_equal = left.get("action_chunk_hash") == right.get("action_chunk_hash")
        if not noise_equal:
            noise_mismatch_keys.append(key)
        if not condition_equal:
            condition_divergence_keys.append(key)
        if not action_equal:
            raw_action_mismatch_keys.append(key)
        if noise_equal and condition_equal:
            matched_input_keys.append(key)
            if not action_equal:
                action_mismatch_on_matched_input.append(key)

    pre_divergence_action_mismatches: list[tuple[int, int, int]] = []
    for episode_key, keys in common_by_episode.items():
        ordered = sorted(keys, key=lambda item: item[2])
        divergence = next(
            (
                key[2]
                for key in ordered
                if baseline[key].get("condition_hash") != adapter[key].get("condition_hash")
            ),
            None,
        )
        if divergence is not None:
            first_divergence_by_episode[f"task{episode_key[0]}_episode{episode_key[1]}"] = divergence
        for key in ordered:
            if divergence is not None and key[2] >= divergence:
                break
            if baseline[key].get("action_chunk_hash") != adapter[key].get("action_chunk_hash"):
                pre_divergence_action_mismatches.append(key)

    initial_query_failures: list[tuple[int, int]] = []
    for task_id, episode in outcome_keys:
        key = (task_id, episode, 0)
        if key not in baseline or key not in adapter:
            initial_query_failures.append((task_id, episode))
            continue
        left = baseline[key]
        right = adapter[key]
        if not (
            left.get("condition_hash") == right.get("condition_hash")
            and left.get("action_noise_seed") == right.get("action_noise_seed")
            and left.get("action_noise_hash") == right.get("action_noise_hash")
            and left.get("action_chunk_hash") == right.get("action_chunk_hash")
        ):
            initial_query_failures.append((task_id, episode))

    adapter_counters = raw_summary["rows"][ADAPTER_ROW].get("counters", {})
    updater_calls = int(adapter_counters.get("num_condition_updater_calls", 0))
    observation_encoder_calls = int(adapter_counters.get("num_observation_encoder_calls", 0))
    action_encoder_calls = int(adapter_counters.get("num_executed_action_encoder_calls", 0))
    identical_outcomes = baseline_outcomes == adapter_outcomes
    exact_on_matched_inputs = bool(matched_input_keys) and not action_mismatch_on_matched_input
    first_query_exact = bool(outcome_keys) and not initial_query_failures
    pass_value = bool(
        exact_on_matched_inputs
        and first_query_exact
        and not noise_mismatch_keys
        and not pre_divergence_action_mismatches
        and identical_outcomes
        and updater_calls == 0
        and observation_encoder_calls == 0
        and action_encoder_calls == 0
    )

    raw_parity = raw_summary.get("k1_parity") or {}
    return {
        "K1_PARITY_PASS": pass_value,
        "equality_scope": "identical condition hash and identical action-noise seed/hash",
        "exact_action_chunk_equality": exact_on_matched_inputs,
        "identical_paired_outcomes": identical_outcomes,
        "updater_calls": updater_calls,
        "observation_encoder_calls": observation_encoder_calls,
        "action_encoder_calls": action_encoder_calls,
        "paired_action_chunks": len(matched_input_keys),
        "missing_chunk_keys": len(missing_keys),
        "max_abs_action_chunk_diff": 0.0 if exact_on_matched_inputs else None,
        "common_query_keys": len(common_keys),
        "matched_input_query_keys": len(matched_input_keys),
        "condition_diverged_query_keys": len(condition_divergence_keys),
        "episodes_with_condition_divergence": len(first_divergence_by_episode),
        "first_condition_divergence_by_episode": first_divergence_by_episode,
        "raw_action_hash_mismatch_query_keys": len(raw_action_mismatch_keys),
        "action_hash_mismatch_on_matched_input_keys": len(action_mismatch_on_matched_input),
        "action_hash_mismatch_before_first_condition_divergence": len(
            pre_divergence_action_mismatches
        ),
        "action_noise_mismatch_query_keys": len(noise_mismatch_keys),
        "initial_query_exact_episodes": len(outcome_keys) - len(initial_query_failures),
        "initial_query_total_episodes": len(outcome_keys),
        "initial_query_failure_keys": [list(key) for key in initial_query_failures],
        "raw_missing_query_keys": len(missing_keys),
        "raw_unfiltered_max_abs_action_chunk_diff": raw_parity.get(
            "max_abs_action_chunk_diff"
        ),
        "raw_unfiltered_exact_action_chunk_equality": raw_parity.get(
            "exact_action_chunk_equality"
        ),
        "interpretation": (
            "Independent MuJoCo rollouts can diverge after an earlier state difference. "
            "Actions at the same query index are therefore comparable only while the "
            "condition and seeded flow-noise inputs are identical."
        ),
    }


def revalidate(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Write an immutable corrected K1 decision next to the untouched raw run."""

    source = Path(input_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete raw K1 output {source}: missing {missing}")
    output = require_empty_output(output_dir)
    raw_summary = _read_json(source / "online_summary.json")
    config = _read_json(source / "eval_config.json")
    outcomes = _episode_outcomes(source / "episode_metrics.csv")
    traces = _query_traces(source / "query_trace.jsonl")
    parity = analyze_k1_parity(
        raw_summary=raw_summary,
        outcomes=outcomes,
        traces=traces,
    )
    result = {
        **raw_summary,
        "k1_parity": parity,
        "raw_k1_parity": raw_summary.get("k1_parity"),
        "revalidation": {
            "method": "matched_condition_and_seeded_noise_hash_v1",
            "source_directory": str(source),
            "execution_horizon": config.get("execution_horizon"),
            "raw_artifacts": {
                name: {
                    "path": str(source / name),
                    "sha256": sha256_file(source / name),
                    "size_bytes": (source / name).stat().st_size,
                }
                for name in REQUIRED_FILES
            },
            "raw_artifacts_unchanged": True,
        },
    }
    _write_json(output / "online_summary.json", result)
    _write_json(output / "k1_parity_revalidated.json", parity)
    _write_json(output / "revalidation_provenance.json", result["revalidation"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = revalidate(args.input, args.output)
    print(json.dumps(result["k1_parity"], indent=2, sort_keys=True))
    return 0 if result["k1_parity"]["K1_PARITY_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
