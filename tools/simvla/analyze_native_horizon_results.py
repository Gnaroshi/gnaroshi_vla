"""Aggregate a completed native-R5 matrix under its frozen scientific rule."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.hierarchical_correction.decisions import (  # noqa: E402
    NativeHybridDecisionInputs,
    evaluate_native_hybrid_decision,
)
from methods.hierarchical_correction.metrics import paired_outcome_summary  # noqa: E402


FULL = "native_full_simvla_k1"
ACTION = "native_action_correction_kf4"
CONDITION = "native_condition_regeneration_kf4"
HYBRID = "native_horizon_hybrid_kf4_kg2"
HOLD = "native_stale_action_chunk_kf4"
REQUIRED_ROWS = {FULL, ACTION, CONDITION, HYBRID, HOLD}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _gripper_reversal_mean(summary: dict[str, Any], row: str) -> float:
    value = summary["rows"][row]["action_diagnostics"]["gripper_reversals"]["mean"]
    if value is None:
        raise RuntimeError(f"missing gripper reversal mean for {row}")
    return float(value)


def _counter(summary: dict[str, Any], row: str, name: str) -> int:
    return int(summary["rows"][row]["counters"].get(name, 0))


def _policy_ms(summary: dict[str, Any], row: str) -> float:
    return float(summary["rows"][row]["amortized_policy_ms_per_environment_action"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    summary_path = Path(args.online_summary).expanduser().resolve()
    gate_path = Path(args.r5_gate).expanduser().resolve()
    summary = _load_json(summary_path)
    gate = _load_json(gate_path)

    if summary.get("matrix") != "native_r5":
        raise RuntimeError("online summary is not the native_r5 matrix")
    if not gate.get("ONLINE_R5_GATE_PASS", False):
        raise RuntimeError("aggregation blocked: exact-age R5 offline gate did not pass")
    if summary.get("source_signature") != gate.get("source_signature"):
        raise RuntimeError("online and offline source signatures differ")
    rows = set(summary.get("rows", {}))
    if not REQUIRED_ROWS.issubset(rows):
        raise RuntimeError(f"native matrix rows missing: {sorted(REQUIRED_ROWS - rows)}")
    matrix_complete = (
        summary.get("suite") == "libero_10"
        and sorted(summary.get("task_ids", [])) == list(range(10))
        and int(summary.get("episodes_per_row", 0)) == 200
        and all(int(summary["rows"][name].get("episodes", 0)) == 200 for name in rows)
    )

    episode_path = Path(summary["episode_metrics_csv"])
    if not episode_path.is_absolute():
        episode_path = summary_path.parent / episode_path
    episodes: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    with episode_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episodes[row["row"]][(int(row["task_id"]), int(row["episode"]))] = row

    exhaustion_query = int(gate["first_provenance_exhaustion_query"])
    late_action: dict[tuple[int, int], bool] = {}
    late_hybrid: dict[tuple[int, int], bool] = {}
    for key in sorted(set(episodes[ACTION]) & set(episodes[HYBRID])):
        action_row = episodes[ACTION][key]
        hybrid_row = episodes[HYBRID][key]
        if min(int(action_row["policy_queries"]), int(hybrid_row["policy_queries"])) > exhaustion_query:
            late_action[key] = action_row["success"].lower() in {"true", "1"}
            late_hybrid[key] = hybrid_row["success"].lower() in {"true", "1"}
    late = paired_outcome_summary(late_action, late_hybrid, seed=args.bootstrap_seed)
    late_ci = late["task_hierarchical_paired_ci95_pp"]
    materially_better_late = (
        late["pairs"] >= args.minimum_late_pairs
        and late["candidate_minus_baseline_pp"] is not None
        and float(late["candidate_minus_baseline_pp"]) >= args.minimum_late_improvement_pp
        and late_ci[0] is not None
        and float(late_ci[0]) > 0.0
    )

    paired = summary["paired"]["hybrid_minus_action_correction"]
    hybrid_sr = float(summary["rows"][HYBRID]["success_rate"])
    action_sr = float(summary["rows"][ACTION]["success_rate"])
    hybrid_ms = _policy_ms(summary, HYBRID)
    action_ms = _policy_ms(summary, ACTION)
    condition_ms = _policy_ms(summary, CONDITION)
    gripper_hybrid = _gripper_reversal_mean(summary, HYBRID)
    gripper_better_endpoint = min(
        _gripper_reversal_mean(summary, ACTION),
        _gripper_reversal_mean(summary, CONDITION),
    )
    pure_action_as_successful_or_better = action_sr >= hybrid_sr
    pure_action_faster = action_ms <= hybrid_ms
    pure_action_long_gap_failure = bool(
        late["candidate_minus_baseline_pp"] is not None
        and float(late["candidate_minus_baseline_pp"]) > 0.0
        and late_ci[0] is not None
        and float(late_ci[0]) > 0.0
    )
    success_noninferior = float(paired["task_hierarchical_paired_ci95_pp"][0]) > -3.0
    compute_better = (
        _counter(summary, HYBRID, "num_full_vlm_calls")
        < _counter(summary, FULL, "num_full_vlm_calls")
        and _counter(summary, HYBRID, "num_action_transformer_decodes")
        < _counter(summary, CONDITION, "num_action_transformer_decodes")
    )
    hybrid_tradeoff = success_noninferior and compute_better and hybrid_ms < condition_ms

    inputs = NativeHybridDecisionInputs(
        k1_parity_pass=bool(gate.get("prerequisites", {}).get("k1_parity", False)),
        offline_r5_gate_pass=bool(gate.get("R5_REGENERATION_GATE_PASS", False)),
        hybrid_minus_action_ci95_pp=tuple(
            float(value) for value in paired["task_hierarchical_paired_ci95_pp"]
        ),
        hybrid_materially_better_after_exhaustion=materially_better_late,
        hybrid_full_vlm_calls=_counter(summary, HYBRID, "num_full_vlm_calls"),
        k1_full_vlm_calls=_counter(summary, FULL, "num_full_vlm_calls"),
        hybrid_action_transformer_decodes=_counter(
            summary, HYBRID, "num_action_transformer_decodes"
        ),
        condition_action_transformer_decodes=_counter(
            summary, CONDITION, "num_action_transformer_decodes"
        ),
        hybrid_gripper_reversals=gripper_hybrid,
        better_endpoint_gripper_reversals=gripper_better_endpoint,
        pure_action_as_successful_or_better=pure_action_as_successful_or_better,
        pure_action_faster=pure_action_faster,
        pure_action_long_gap_failure=pure_action_long_gap_failure,
        regeneration_recovers_long_gap=materially_better_late,
        scientific_matrix_complete=matrix_complete,
        hybrid_improves_success_compute_tradeoff=hybrid_tradeoff,
    )
    decision = evaluate_native_hybrid_decision(inputs)
    decision.update(
        {
            "schema_version": "simvla_native_horizon_decision_v1",
            "online_summary": str(summary_path),
            "r5_offline_gate": str(gate_path),
            "source_signature": summary["source_signature"],
            "late_episode_rule": {
                "selection": f"both rows have policy_queries > {exhaustion_query}",
                "minimum_pairs": args.minimum_late_pairs,
                "material_improvement_pp": args.minimum_late_improvement_pp,
                "paired_ci_lower_must_exceed_pp": 0.0,
            },
            "late_paired_outcomes": late,
            "latency_ms_per_environment_action": {
                "hybrid": hybrid_ms,
                "action": action_ms,
                "condition": condition_ms,
            },
        }
    )
    (output / "native_horizon_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "native_horizon_result_summary.md").write_text(
        "\n".join(
            (
                "# Native SimVLA horizon-regeneration result",
                "",
                f"Verdict: **`{decision['verdict']}`**.",
                "",
                f"- Hybrid success: {100.0 * hybrid_sr:.1f}%.",
                f"- Pure action success: {100.0 * action_sr:.1f}%.",
                f"- Hybrid minus action paired CI95: {paired['task_hierarchical_paired_ci95_pp']} pp.",
                f"- Late paired episodes: {late['pairs']}.",
                f"- Late hybrid minus action: {late['candidate_minus_baseline_pp']} pp; CI95 {late_ci}.",
                f"- Exact-age gate: `{gate['ONLINE_R5_GATE_PASS']}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-summary", required=True)
    parser.add_argument("--r5-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--minimum-late-pairs", type=int, default=50)
    parser.add_argument("--minimum-late-improvement-pp", type=float, default=3.0)
    decision = run(parser.parse_args())
    print(json.dumps({"verdict": decision["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
