"""Aggregate the fixed SimVLA K_C x N_G 2x2 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_GENERATION_SOURCE_SHA256,
    atomic_write_json,
    exact_mcnemar,
    load_json,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import (
    COMBINED_ROW,
    CONDITION_ROW,
    FROZEN_CONDITION_SOURCE_SHA256,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    COUPLED_ROW,
)


BASELINE_ROW = "full_nfe10"
GENERATION_ROW = "generation_ng3"
ROWS = (BASELINE_ROW, CONDITION_ROW, GENERATION_ROW, COMBINED_ROW)
AGGREGATABLE_ROWS = (*ROWS, COUPLED_ROW)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _int(row: Mapping[str, Any], name: str, default: int = 0) -> int:
    value = row.get(name)
    if value in {None, ""}:
        return int(default)
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no"}:
        return 0
    return int(float(text))


def _float(row: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name)
    return float(default) if value in {None, ""} else float(value)


def _key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (_int(row, "task_id"), _int(row, "trial_id"))


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]], value_name: str, weight_name: str
) -> float:
    values = np.asarray([_float(row, value_name) for row in rows], dtype=np.float64)
    weights = np.asarray([_int(row, weight_name) for row in rows], dtype=np.float64)
    return float(np.average(values, weights=weights)) if weights.sum() else 0.0


def _expected_counts(row_name: str, queries: int) -> dict[str, int]:
    if row_name == BASELINE_ROW:
        return {"vlm": queries, "condition": 0, "transformer": 10 * queries, "generation": 0}
    if row_name == CONDITION_ROW:
        return {
            "vlm": (queries + 1) // 2,
            "condition": queries // 2,
            "transformer": 10 * queries,
            "generation": 0,
        }
    if row_name == GENERATION_ROW:
        return {"vlm": queries, "condition": 0, "transformer": 3 * queries, "generation": 7 * queries}
    if row_name in {COMBINED_ROW, COUPLED_ROW}:
        return {
            "vlm": (queries + 1) // 2,
            "condition": queries // 2,
            "transformer": 3 * queries,
            "generation": 7 * queries,
        }
    raise ValueError(f"unknown row: {row_name}")


def _validate_episode_table(row_name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [_key(row) for row in rows]
    expected_keys = {(task, trial) for task in range(10) for trial in range(50)}
    if len(rows) != 500 or len(set(keys)) != 500 or set(keys) != expected_keys:
        raise RuntimeError(f"{row_name} is not the exact LIBERO-Long 10x50 episode set")
    for row in rows:
        queries = _int(row, "num_policy_queries")
        expected = _expected_counts(row_name, queries)
        observed = {
            "vlm": _int(row, "num_full_vlm_calls"),
            "condition": _int(row, "num_condition_updater_calls"),
            "transformer": _int(row, "num_full_action_transformer_evaluations"),
            "generation": _int(row, "num_generation_loop_updates"),
        }
        if observed != expected:
            raise RuntimeError(
                f"counter mismatch row={row_name} episode={_key(row)} "
                f"observed={observed} expected={expected}"
            )
        if _int(row, "num_integration_updates") != expected["transformer"] + expected["generation"]:
            raise RuntimeError(f"integration counter mismatch: {row_name} {_key(row)}")


def _summarize(row_name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _validate_episode_table(row_name, rows)
    episodes = len(rows)
    successes = sum(_int(row, "success") for row in rows)
    actions = sum(_int(row, "episode_length") for row in rows)
    queries = sum(_int(row, "num_policy_queries") for row in rows)
    vlm_calls = sum(_int(row, "num_full_vlm_calls") for row in rows)
    condition_calls = sum(_int(row, "num_condition_updater_calls") for row in rows)
    transformer_calls = sum(
        _int(row, "num_full_action_transformer_evaluations") for row in rows
    )
    generation_updates = sum(
        _int(row, "num_generation_loop_updates") for row in rows
    )
    policy_seconds = sum(_float(row, "policy_wall_time_seconds") for row in rows)
    per_task = {}
    for task_id in range(10):
        task_rows = [row for row in rows if _int(row, "task_id") == task_id]
        task_successes = sum(_int(row, "success") for row in task_rows)
        per_task[str(task_id)] = {
            "episodes": 50,
            "successes": task_successes,
            "success_rate": task_successes / 50.0,
        }
    return {
        "row": row_name,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "per_task": per_task,
        "executed_actions": actions,
        "policy_queries": queries,
        "full_vlm_calls": vlm_calls,
        "condition_updater_calls": condition_calls,
        "full_action_transformer_evaluations": transformer_calls,
        "generation_loop_updates": generation_updates,
        "integration_updates": transformer_calls + generation_updates,
        "effective_k_c": queries / max(1, vlm_calls),
        "full_action_transformer_calls_per_query": transformer_calls / queries,
        "generation_loop_updates_per_query": generation_updates / queries,
        "latency_per_policy_query_ms": _weighted_mean(
            rows, "latency_per_policy_query_ms", "num_policy_queries"
        ),
        "latency_per_executed_action_ms": policy_seconds * 1000.0 / actions,
        "model_vlm_encoder_per_call_ms": _weighted_mean(
            rows, "model_vlm_encoder_per_query_ms", "num_full_vlm_calls"
        ),
        "model_condition_updater_per_call_ms": _weighted_mean(
            rows, "model_condition_updater_per_update_ms", "num_condition_updater_calls"
        ),
        "model_action_generation_per_query_ms": _weighted_mean(
            rows, "model_action_generation_per_query_ms", "num_policy_queries"
        ),
        "policy_wall_time_seconds": policy_seconds,
    }


def aggregate_row(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    shard = Path(args.shard).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    shard_summary = load_json(shard / "shard_summary.json")
    if shard_summary.get("verdict") != "FIXED_2X2_SHARD_PASS":
        raise RuntimeError("fixed 2x2 shard gate did not pass")
    if shard_summary.get("row") != args.row:
        raise RuntimeError("fixed 2x2 shard row mismatch")
    if shard_summary.get("manifest_sha256") != args.expected_manifest_sha256:
        raise RuntimeError("fixed 2x2 shard manifest mismatch")
    rows = _read_csv(shard / "episode_metrics.csv")
    summary = {
        "verdict": "FIXED_2X2_ROW_PASS",
        "classification": shard_summary["classification"],
        "inference_seed": shard_summary["inference_seed"],
        "manifest_sha256": args.expected_manifest_sha256,
        "source_combined_sha256": shard_summary.get(
            "source_combined_sha256",
            {
                "condition": FROZEN_CONDITION_SOURCE_SHA256,
                "generation": FROZEN_GENERATION_SOURCE_SHA256,
            },
        ),
        "generation_checkpoint_sha256": shard_summary.get(
            "generation_checkpoint_sha256"
        ),
        "paper_runtime_match": bool(shard_summary["paper_runtime_match"]),
        **_summarize(args.row, rows),
    }
    output.mkdir(parents=True)
    _write_csv(output / "episode_metrics.csv", rows)
    shutil.copy2(shard / "action_chunks.npz", output / "action_chunks.npz")
    atomic_write_json(output / "row_summary.json", summary)
    return summary


def _paired(
    reference: Mapping[tuple[int, int], Mapping[str, Any]],
    candidate: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise RuntimeError("paired episode IDs differ")
    reference_only = candidate_only = both_success = both_fail = 0
    for key in sorted(reference):
        left = bool(_int(reference[key], "success"))
        right = bool(_int(candidate[key], "success"))
        if left and right:
            both_success += 1
        elif left:
            reference_only += 1
        elif right:
            candidate_only += 1
        else:
            both_fail += 1
    return {
        "episodes": len(reference),
        "both_success": both_success,
        "both_fail": both_fail,
        "reference_only": reference_only,
        "candidate_only": candidate_only,
        "exact_mcnemar_p": exact_mcnemar(reference_only, candidate_only),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    roots = {
        BASELINE_ROW: Path(args.baseline).expanduser().resolve(),
        CONDITION_ROW: Path(args.condition).expanduser().resolve(),
        GENERATION_ROW: Path(args.generation).expanduser().resolve(),
        COMBINED_ROW: Path(args.combined).expanduser().resolve(),
    }
    summaries = {name: load_json(root / "row_summary.json") for name, root in roots.items()}
    accepted = {
        BASELINE_ROW: {"GENERATION_CONTROL_ROW_PASS", "FIXED_2X2_ROW_PASS"},
        GENERATION_ROW: {"GENERATION_CONTROL_ROW_PASS", "FIXED_2X2_ROW_PASS"},
        CONDITION_ROW: {"FIXED_2X2_ROW_PASS"},
        COMBINED_ROW: {"FIXED_2X2_ROW_PASS"},
    }
    for name, summary in summaries.items():
        if summary.get("verdict") not in accepted[name] or summary.get("row") != name:
            raise RuntimeError(f"row gate mismatch: {name}")
    manifest_hashes = {summary.get("manifest_sha256") for summary in summaries.values()}
    if len(manifest_hashes) != 1:
        raise RuntimeError("2x2 rows do not share one immutable manifest")
    classifications = {summary.get("classification") for summary in summaries.values()}
    if len(classifications) != 1:
        raise RuntimeError("2x2 rows do not share one runtime classification")
    inference_seeds = {summary.get("inference_seed") for summary in summaries.values()}
    if len(inference_seeds) != 1:
        raise RuntimeError("2x2 rows do not share one inference seed")
    rows = {name: _read_csv(root / "episode_metrics.csv") for name, root in roots.items()}
    for name, table in rows.items():
        _validate_episode_table(name, table)
    by_key = {name: {_key(row): row for row in table} for name, table in rows.items()}
    if len({frozenset(table) for table in by_key.values()}) != 1:
        raise RuntimeError("2x2 episode IDs differ")
    metrics = {name: _summarize(name, table) for name, table in rows.items()}
    a = metrics[BASELINE_ROW]
    b = metrics[CONDITION_ROW]
    c = metrics[GENERATION_ROW]
    d = metrics[COMBINED_ROW]
    effects_pp = {
        "condition_at_ng10": 100.0 * (b["success_rate"] - a["success_rate"]),
        "generation_at_kc1": 100.0 * (c["success_rate"] - a["success_rate"]),
        "generation_at_kc2": 100.0 * (d["success_rate"] - b["success_rate"]),
        "condition_at_ng3": 100.0 * (d["success_rate"] - c["success_rate"]),
    }
    effects_pp["interaction"] = (
        effects_pp["condition_at_ng3"] - effects_pp["condition_at_ng10"]
    )
    latency_effects = {
        "condition_at_ng10_ms": b["latency_per_policy_query_ms"] - a["latency_per_policy_query_ms"],
        "generation_at_kc1_ms": c["latency_per_policy_query_ms"] - a["latency_per_policy_query_ms"],
        "generation_at_kc2_ms": d["latency_per_policy_query_ms"] - b["latency_per_policy_query_ms"],
        "condition_at_ng3_ms": d["latency_per_policy_query_ms"] - c["latency_per_policy_query_ms"],
    }
    latency_effects["interaction_ms"] = (
        latency_effects["condition_at_ng3_ms"] - latency_effects["condition_at_ng10_ms"]
    )
    paired = {
        "condition_at_ng10": _paired(by_key[BASELINE_ROW], by_key[CONDITION_ROW]),
        "generation_at_kc1": _paired(by_key[BASELINE_ROW], by_key[GENERATION_ROW]),
        "generation_at_kc2": _paired(by_key[CONDITION_ROW], by_key[COMBINED_ROW]),
        "condition_at_ng3": _paired(by_key[GENERATION_ROW], by_key[COMBINED_ROW]),
        "baseline_vs_combined": _paired(by_key[BASELINE_ROW], by_key[COMBINED_ROW]),
    }
    comparison_rows = []
    for key in sorted(by_key[BASELINE_ROW]):
        comparison_rows.append(
            {
                "task_id": key[0],
                "trial_id": key[1],
                **{
                    f"{name}_success": _int(by_key[name][key], "success")
                    for name in ROWS
                },
            }
        )
    result = {
        "verdict": "FIXED_2X2_DIAGNOSTIC_COMPLETE",
        "paper_table_eligible": False,
        "interpretation": (
            f"Predeclared fixed {next(iter(inference_seeds))} mechanism diagnostic. "
            "K_C=2 was selected after "
            "the strict K_C=4 offline gate failed; promote only after independent confirmation."
        ),
        "classification": next(iter(classifications)),
        "inference_seed": next(iter(inference_seeds)),
        "manifest_sha256": next(iter(manifest_hashes)),
        "rows": metrics,
        "success_rate_effects_percentage_points": effects_pp,
        "latency_effects_per_policy_query": latency_effects,
        "paired_outcomes": paired,
        "full_vlm_call_reduction": {
            "condition_at_ng10": 1.0 - b["full_vlm_calls"] / a["full_vlm_calls"],
            "condition_at_ng3": 1.0 - d["full_vlm_calls"] / c["full_vlm_calls"],
        },
    }
    output.mkdir(parents=True)
    atomic_write_json(output / "fixed_2x2_summary.json", result)
    _write_csv(output / "paired_episode_outcomes.csv", comparison_rows)
    return result


def compare_coupling(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    roots = {
        COMBINED_ROW: Path(args.uncoupled).expanduser().resolve(),
        COUPLED_ROW: Path(args.coupled).expanduser().resolve(),
    }
    summaries = {name: load_json(root / "row_summary.json") for name, root in roots.items()}
    for name, summary in summaries.items():
        if summary.get("verdict") != "FIXED_2X2_ROW_PASS" or summary.get("row") != name:
            raise RuntimeError(f"coupling comparison row gate mismatch: {name}")
    if summaries[COMBINED_ROW].get("manifest_sha256") != summaries[COUPLED_ROW].get(
        "manifest_sha256"
    ):
        raise RuntimeError("uncoupled and coupled rows use different manifests")
    rows = {name: _read_csv(root / "episode_metrics.csv") for name, root in roots.items()}
    for name, table in rows.items():
        _validate_episode_table(name, table)
    by_key = {name: {_key(row): row for row in table} for name, table in rows.items()}
    uncoupled = _summarize(COMBINED_ROW, rows[COMBINED_ROW])
    coupled = _summarize(COUPLED_ROW, rows[COUPLED_ROW])
    result = {
        "verdict": "COUPLED_VS_UNCOUPLED_COMPARISON_COMPLETE",
        "paper_table_eligible": False,
        "classification": "fixed-seed paired mechanism screening",
        "manifest_sha256": summaries[COMBINED_ROW]["manifest_sha256"],
        "rows": {COMBINED_ROW: uncoupled, COUPLED_ROW: coupled},
        "success_rate_delta_percentage_points": 100.0
        * (coupled["success_rate"] - uncoupled["success_rate"]),
        "latency_per_policy_query_delta_ms": coupled["latency_per_policy_query_ms"]
        - uncoupled["latency_per_policy_query_ms"],
        "paired_outcomes": _paired(by_key[COMBINED_ROW], by_key[COUPLED_ROW]),
        "same_compute_schedule": {
            "full_vlm_calls": uncoupled["full_vlm_calls"] == coupled["full_vlm_calls"],
            "condition_updater_calls": uncoupled["condition_updater_calls"]
            == coupled["condition_updater_calls"],
            "full_action_transformer_evaluations": uncoupled[
                "full_action_transformer_evaluations"
            ]
            == coupled["full_action_transformer_evaluations"],
            "generation_loop_updates": uncoupled["generation_loop_updates"]
            == coupled["generation_loop_updates"],
        },
    }
    output.mkdir(parents=True)
    atomic_write_json(output / "coupled_vs_uncoupled_summary.json", result)
    comparison_rows = []
    for key in sorted(by_key[COMBINED_ROW]):
        comparison_rows.append(
            {
                "task_id": key[0],
                "trial_id": key[1],
                "uncoupled_success": _int(by_key[COMBINED_ROW][key], "success"),
                "coupled_success": _int(by_key[COUPLED_ROW][key], "success"),
                "uncoupled_episode_length": _int(
                    by_key[COMBINED_ROW][key], "episode_length"
                ),
                "coupled_episode_length": _int(
                    by_key[COUPLED_ROW][key], "episode_length"
                ),
            }
        )
    _write_csv(output / "paired_episode_outcomes.csv", comparison_rows)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    row = subparsers.add_parser("aggregate-row")
    row.add_argument("--row", choices=AGGREGATABLE_ROWS, required=True)
    row.add_argument("--output", required=True)
    row.add_argument("--shard", required=True)
    row.add_argument("--expected-manifest-sha256", required=True)
    row.set_defaults(handler=aggregate_row)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--condition", required=True)
    comparison.add_argument("--generation", required=True)
    comparison.add_argument("--combined", required=True)
    comparison.set_defaults(handler=compare)
    coupling = subparsers.add_parser("compare-coupling")
    coupling.add_argument("--output", required=True)
    coupling.add_argument("--uncoupled", required=True)
    coupling.add_argument("--coupled", required=True)
    coupling.set_defaults(handler=compare_coupling)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
