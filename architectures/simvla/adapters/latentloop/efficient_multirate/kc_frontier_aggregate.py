"""Aggregate the paired SimVLA K_C={3,4} efficiency frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate import (
    BASELINE_ROW,
    GENERATION_ROW,
    _int,
    _key,
    _paired,
    _read_csv,
    _summarize,
    _validate_episode_table,
    _write_csv,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    atomic_write_json,
    load_json,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    FRONTIER_ROWS,
    condition_row_name,
    row_spec,
)


def _load_row(root: Path, expected_row: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary = load_json(root / "row_summary.json")
    accepted = {"FIXED_2X2_ROW_PASS", "GENERATION_CONTROL_ROW_PASS", "KC_FRONTIER_ROW_PASS"}
    if summary.get("verdict") not in accepted or summary.get("row") != expected_row:
        raise RuntimeError(f"row gate mismatch for {expected_row}: {root}")
    rows = _read_csv(root / "episode_metrics.csv")
    _validate_episode_table(expected_row, rows)
    return summary, rows


def _effect(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    reference_rows: dict[tuple[int, int], dict[str, str]],
    candidate_rows: dict[tuple[int, int], dict[str, str]],
) -> dict[str, Any]:
    return {
        "success_rate_delta_percentage_points": 100.0
        * (candidate["success_rate"] - reference["success_rate"]),
        "latency_per_policy_query_reduction_fraction": 1.0
        - candidate["latency_per_policy_query_ms"]
        / reference["latency_per_policy_query_ms"],
        "latency_per_executed_action_reduction_fraction": 1.0
        - candidate["latency_per_executed_action_ms"]
        / reference["latency_per_executed_action_ms"],
        "executed_action_speedup": reference["latency_per_executed_action_ms"]
        / candidate["latency_per_executed_action_ms"],
        "full_vlm_call_reduction_fraction": 1.0
        - candidate["full_vlm_calls"] / reference["full_vlm_calls"],
        "full_action_transformer_call_reduction_fraction": 1.0
        - candidate["full_action_transformer_evaluations"]
        / reference["full_action_transformer_evaluations"],
        "paired_outcomes": _paired(reference_rows, candidate_rows),
    }


def _observed_pareto(rows: dict[str, dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    for name, candidate in rows.items():
        dominated = False
        for other_name, other in rows.items():
            if other_name == name:
                continue
            no_worse = (
                other["success_rate"] >= candidate["success_rate"]
                and other["latency_per_executed_action_ms"]
                <= candidate["latency_per_executed_action_ms"]
            )
            strictly_better = (
                other["success_rate"] > candidate["success_rate"]
                or other["latency_per_executed_action_ms"]
                < candidate["latency_per_executed_action_ms"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            selected.append(name)
    return selected


def compare_frontier(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    roots = {
        BASELINE_ROW: Path(args.baseline).expanduser().resolve(),
        GENERATION_ROW: Path(args.generation).expanduser().resolve(),
        condition_row_name(3, 10): Path(args.kc3_ng10).expanduser().resolve(),
        condition_row_name(4, 10): Path(args.kc4_ng10).expanduser().resolve(),
        condition_row_name(3, 3): Path(args.kc3_ng3).expanduser().resolve(),
        condition_row_name(4, 3): Path(args.kc4_ng3).expanduser().resolve(),
    }
    loaded = {name: _load_row(root, name) for name, root in roots.items()}
    source_summaries = {name: value[0] for name, value in loaded.items()}
    tables = {name: value[1] for name, value in loaded.items()}
    manifest_hashes = {value.get("manifest_sha256") for value in source_summaries.values()}
    inference_seeds = {value.get("inference_seed") for value in source_summaries.values()}
    classifications = {value.get("classification") for value in source_summaries.values()}
    if len(manifest_hashes) != 1 or len(inference_seeds) != 1 or len(classifications) != 1:
        raise RuntimeError("frontier rows do not share manifest, inference seed, and runtime axis")
    keyed = {name: {_key(row): row for row in table} for name, table in tables.items()}
    if len({frozenset(value) for value in keyed.values()}) != 1:
        raise RuntimeError("frontier rows do not share the exact 500 paired episodes")
    metrics = {name: _summarize(name, table) for name, table in tables.items()}
    baseline = metrics[BASELINE_ROW]
    generation = metrics[GENERATION_ROW]
    effects: dict[str, Any] = {}
    for row in FRONTIER_ROWS:
        spec = row_spec(row)
        matched_name = BASELINE_ROW if spec.n_g == 10 else GENERATION_ROW
        effects[row] = {
            "matched_reference": matched_name,
            "vs_matched_reference": _effect(
                metrics[matched_name], metrics[row], keyed[matched_name], keyed[row]
            ),
            "vs_full_baseline": _effect(
                baseline, metrics[row], keyed[BASELINE_ROW], keyed[row]
            ),
            "nominal_full_vlm_call_reduction_fraction": 1.0 - 1.0 / spec.k_c,
            "nominal_full_action_transformer_call_reduction_fraction": (
                1.0 - spec.n_g / 10.0
            ),
        }
    comparison_rows = []
    for key in sorted(keyed[BASELINE_ROW]):
        comparison_rows.append(
            {
                "task_id": key[0],
                "trial_id": key[1],
                **{
                    f"{name}_success": _int(keyed[name][key], "success")
                    for name in roots
                },
            }
        )
    result = {
        "verdict": "KC_EFFICIENCY_FRONTIER_COMPLETE",
        "paper_table_eligible": False,
        "classification": next(iter(classifications)),
        "inference_seed": next(iter(inference_seeds)),
        "manifest_sha256": next(iter(manifest_hashes)),
        "episodes_per_row": 500,
        "rows": metrics,
        "effects": effects,
        "observed_success_latency_pareto_rows": _observed_pareto(metrics),
        "interpretation": (
            "Same-host fixed-seed EGL efficiency frontier. Treat latency as host-local and "
            "success differences as paired diagnostic evidence until independent seeds/hosts confirm."
        ),
    }
    output.mkdir(parents=True)
    atomic_write_json(output / "kc_efficiency_frontier_summary.json", result)
    _write_csv(output / "paired_episode_outcomes.csv", comparison_rows)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--kc3-ng10", required=True)
    parser.add_argument("--kc4-ng10", required=True)
    parser.add_argument("--kc3-ng3", required=True)
    parser.add_argument("--kc4-ng3", required=True)
    return parser


def main() -> int:
    result = compare_frontier(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
