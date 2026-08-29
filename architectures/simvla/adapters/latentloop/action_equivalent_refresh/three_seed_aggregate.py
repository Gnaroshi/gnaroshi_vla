"""Aggregate the fixed three-inference-seed selective-refresh evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    atomic_write_json,
    exact_mcnemar,
    load_json,
)


FINAL_VERDICT = "ACTION_EQUIVALENT_REFRESH_THREE_INFERENCE_SEED_COMPLETE"
EXPECTED_MANIFESTS = {
    "seed01": "d1d9bf5a0ff6b20c235eb92dae80189ed3ebdc9eb1591a51fd0d8d572521e74a",
    "seed02": "9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48",
    "seed03": "25c3741fd73034cff2d83640dccb675a9fc526c2dc4b406490209e53fd76c61d",
}
CONTROL_LABELS = ("full_nfe10", "generation_ng3", "periodic_kc3_ng3")


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values))


def _sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episodes = sum(int(row["episodes"]) for row in rows)
    successes = sum(int(row["successes"]) for row in rows)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": float(successes / episodes),
        "seed_mean_success_rate": _mean(
            [float(row["success_rate"]) for row in rows]
        ),
        "seed_sample_std_success_rate": _sample_std(
            [float(row["success_rate"]) for row in rows]
        ),
        "latency_per_executed_action_ms_seed_mean": _mean(
            [float(row["latency_per_executed_action_ms"]) for row in rows]
        ),
        "latency_per_executed_action_ms_seed_sample_std": _sample_std(
            [float(row["latency_per_executed_action_ms"]) for row in rows]
        ),
        "observed_exact_fraction_seed_mean": _mean(
            [float(row["observed_exact_fraction"]) for row in rows]
        ),
        "effective_k_c_seed_mean": _mean(
            [float(row["effective_k_c"]) for row in rows]
        ),
        "total_policy_queries": sum(int(row["total_policy_queries"]) for row in rows),
        "total_full_vlm_calls": sum(int(row["total_full_vlm_calls"]) for row in rows),
        "total_full_action_transformer_evaluations": sum(
            int(row["total_full_action_transformer_evaluations"]) for row in rows
        ),
        "total_generation_loop_updates": sum(
            int(row["total_generation_loop_updates"]) for row in rows
        ),
    }


def _relative_reduction(candidate: float, control: float) -> float:
    return float(1.0 - candidate / control) if control else 0.0


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "seed01": Path(args.seed01).expanduser().resolve(),
        "seed02": Path(args.seed02).expanduser().resolve(),
        "seed03": Path(args.seed03).expanduser().resolve(),
    }
    payloads: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    per_seed: dict[str, Any] = {}
    for seed, path in paths.items():
        payload = load_json(path)
        checks = {
            "verdict": payload.get("verdict")
            == "ACTION_EQUIVALENT_REFRESH_ONLINE_COMPLETE",
            "classification": payload.get("classification")
            == "RB2_HOST_LOCAL_EGL_LONG500",
            "manifest": payload.get("manifest_sha256")
            == EXPECTED_MANIFESTS[seed],
            "candidate_episodes": int(payload.get("candidate", {}).get("episodes", -1))
            == 500,
            "candidate_controls": all(
                label in payload.get("controls", {}) for label in CONTROL_LABELS
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"invalid {seed} summary {path}: {checks}")
        source_hashes[seed] = str(payload["source_combined_sha256"])
        payloads[seed] = payload
        per_seed[seed] = {
            "manifest_sha256": payload["manifest_sha256"],
            "candidate": payload["candidate"],
            "controls": {
                label: payload["controls"][label]
                for label in CONTROL_LABELS
            },
            "paired_outcomes": {
                label: payload["paired_outcomes"][label]
                for label in CONTROL_LABELS
            },
            "efficiency": {
                label: payload["efficiency"][label]
                for label in CONTROL_LABELS
            },
        }
    aggregate_rows = {
        "action_equivalent_refresh_ng3": _aggregate_rows(
            [payloads[seed]["candidate"] for seed in paths]
        )
    }
    for label in CONTROL_LABELS:
        aggregate_rows[label] = _aggregate_rows(
            [payloads[seed]["controls"][label] for seed in paths]
        )

    paired: dict[str, Any] = {}
    efficiency: dict[str, Any] = {}
    candidate = aggregate_rows["action_equivalent_refresh_ng3"]
    for label in CONTROL_LABELS:
        outcomes = [payloads[seed]["paired_outcomes"][label] for seed in paths]
        candidate_only = sum(int(row["candidate_only_success"]) for row in outcomes)
        control_only = sum(int(row["control_only_success"]) for row in outcomes)
        paired[label] = {
            "both_success": sum(int(row["both_success"]) for row in outcomes),
            "candidate_only_success": candidate_only,
            "control_only_success": control_only,
            "both_failure": sum(int(row["both_failure"]) for row in outcomes),
            "candidate_minus_control_successes": candidate_only - control_only,
            "mcnemar_exact_two_sided_p": exact_mcnemar(
                control_only, candidate_only
            ),
        }
        control = aggregate_rows[label]
        efficiency[label] = {
            "success_rate_delta": float(
                candidate["success_rate"] - control["success_rate"]
            ),
            "full_vlm_call_reduction": _relative_reduction(
                float(candidate["total_full_vlm_calls"]),
                float(control["total_full_vlm_calls"]),
            ),
            "full_action_transformer_evaluation_reduction": _relative_reduction(
                float(candidate["total_full_action_transformer_evaluations"]),
                float(control["total_full_action_transformer_evaluations"]),
            ),
            "latency_per_action_reduction": _relative_reduction(
                float(candidate["latency_per_executed_action_ms_seed_mean"]),
                float(control["latency_per_executed_action_ms_seed_mean"]),
            ),
        }

    result = {
        "verdict": FINAL_VERDICT,
        "classification": "RB2_HOST_LOCAL_EGL_LONG500_THREE_INFERENCE_SEEDS",
        "suite": "libero_10",
        "seeds": 3,
        "episodes_per_method_total": 1500,
        "episodes_per_row_per_seed": 500,
        "source_combined_sha256_by_seed": source_hashes,
        "important_scope": (
            "Three deterministic inference/action-noise seeds for fixed trained "
            "checkpoints; these are not three independently trained checkpoints."
        ),
        "per_seed": per_seed,
        "aggregate": aggregate_rows,
        "paired_outcomes": paired,
        "efficiency": efficiency,
        "automatic_quality_gate": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "three_inference_seed_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed01", required=True)
    parser.add_argument("--seed02", required=True)
    parser.add_argument("--seed03", required=True)
    return parser


def main() -> int:
    result = aggregate(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
