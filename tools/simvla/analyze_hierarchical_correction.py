#!/usr/bin/env python3
"""Source-lock and aggregate Hierarchical Latent-Action Correction diagnostics."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.hierarchical_correction.simvla_hybrid_policy import (  # noqa: E402
    hybrid_parameter_audit,
)
from architectures.simvla.adapters.latentloop.checkpoint import (  # noqa: E402
    freeze_module,
    load_adapter_checkpoint,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
    resolve_huggingface_checkpoint,
    sha256_file,
)
from methods.hierarchical_correction.decisions import (  # noqa: E402
    HybridGateInputs,
    evaluate_hybrid_gate,
)
from methods.hierarchical_correction.metrics import (  # noqa: E402
    correction_residuals_by_age,
    paired_outcome_summary,
    trace_metrics_by_age,
)
from methods.hierarchical_correction.provenance import (  # noqa: E402
    experiment_source_signature,
    hierarchical_source_manifest,
)
from methods.latentloop.eval import distribution_summary  # noqa: E402
from methods.latentloop.training.query_cache_dataset import load_manifest  # noqa: E402


PRIMARY_ROW = "hierarchical_hybrid_r1_kf4_kg2"
CONDITION_ROW = "chunk_aware_latentloop_k4"
ACTION_ROW = "action_chunk_correction_k4"


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return json.loads(resolved.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_lock(args: argparse.Namespace) -> dict[str, Any]:
    output = require_empty_output(args.output)
    condition_path = Path(args.condition_checkpoint).expanduser().resolve()
    correction_path = Path(args.action_correction_checkpoint).expanduser().resolve()
    cache = Path(args.cache).expanduser().resolve()
    k1_summary = Path(args.k1_summary).expanduser().resolve()
    for path in (condition_path, correction_path, cache / "manifest.json", k1_summary):
        if not path.exists():
            raise FileNotFoundError(path)
    condition, condition_payload = load_adapter_checkpoint(condition_path, device="cpu")
    correction, correction_payload = load_adapter_checkpoint(correction_path, device="cpu")
    freeze_module(condition)
    freeze_module(correction)
    manifest = load_manifest(cache)
    k1 = _read_json(k1_summary)
    source = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    source["hierarchical_checkpoints"] = {
        "condition": {"path": str(condition_path), "sha256": sha256_file(condition_path)},
        "action_correction": {
            "path": str(correction_path),
            "sha256": sha256_file(correction_path),
        },
    }
    source["processor_checkpoint"] = resolve_huggingface_checkpoint(
        args.smolvlm_model_path
    )
    source["hierarchical_implementation"] = hierarchical_source_manifest(ROOT)
    source_signature = experiment_source_signature(source)
    audit = {
        "method": "Hierarchical Latent-Action Correction",
        "protocol": "A_R1",
        "fixed_schedule": {"execution_horizon": 1, "K_F": 4, "K_G": 2},
        "checkpoint": args.checkpoint,
        "condition_checkpoint": {
            "path": str(condition_path),
            "sha256": sha256_file(condition_path),
            "variant": condition.variant,
            "step": int(condition_payload.get("step", -1)),
        },
        "action_correction_checkpoint": {
            "path": str(correction_path),
            "sha256": sha256_file(correction_path),
            "variant": correction.variant,
            "step": int(correction_payload.get("step", -1)),
        },
        "parameter_audit": hybrid_parameter_audit(condition, correction),
        "cache": {
            "path": str(cache),
            "manifest_sha256": sha256_file(cache / "manifest.json"),
            "schema_version": manifest.get("schema_version"),
            "execution_horizon": manifest.get("execution_horizon"),
            "records": manifest.get("total_records"),
        },
        "k1_summary": {
            "path": str(k1_summary),
            "sha256": sha256_file(k1_summary),
            "K1_PARITY_PASS": bool(k1.get("k1_parity", {}).get("K1_PARITY_PASS", False)),
        },
        "separate_encoder_weights_preserved": True,
        "source_signature": source_signature,
    }
    checks = {
        "condition_variant_exact": condition.variant == "chunk_aware_latentloop",
        "action_variant_exact": correction.variant == "action_chunk_correction",
        "condition_step_150000": int(condition_payload.get("step", -1)) == 150_000,
        "action_step_150000": int(correction_payload.get("step", -1)) == 150_000,
        "r1_cache": int(manifest.get("execution_horizon", -1)) == 1,
        "k1_parity_preexisting_pass": audit["k1_summary"]["K1_PARITY_PASS"],
        "base_checkpoint_cached": bool(source["checkpoint"].get("revision"))
        and Path(str(source["checkpoint"].get("snapshot_path"))).is_dir(),
        "processor_checkpoint_cached": bool(
            source["processor_checkpoint"].get("revision")
        )
        and Path(str(source["processor_checkpoint"].get("snapshot_path"))).is_dir(),
    }
    checks["implementation_manifest_complete"] = not source[
        "hierarchical_implementation"
    ]["missing"]
    result = {
        "SOURCE_LOCK_PASS": all(checks.values()),
        "checks": checks,
        "audit": audit,
        "source_signature": source_signature,
    }
    _write_json(output / "environment_source_lock.json", source)
    _write_json(output / "source_lock_audit.json", result)
    return result


def _metric_mean(row: dict[str, Any], name: str) -> float:
    value = row["action_diagnostics"][name]["mean"]
    if value is None:
        raise ValueError(f"missing action diagnostic mean: {name}")
    return float(value)


def _merge_scientific(args: argparse.Namespace) -> dict[str, Any]:
    """Merge disjoint task shards from the fixed 10x20 scientific matrix."""

    shard_dirs = [Path(path).expanduser().resolve() for path in args.shard]
    if len(shard_dirs) < 2:
        raise ValueError("merge-scientific requires at least two --shard directories")
    summaries = [_read_json(path / "online_summary.json") for path in shard_dirs]
    sources = [_read_json(path / "source_lock.json") for path in shard_dirs]
    signatures = [experiment_source_signature(source) for source in sources]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("scientific shards have different source-lock signatures")
    if any(summary.get("matrix") != "scientific_r1_k4" for summary in summaries):
        raise ValueError("all shards must be scientific_r1_k4")
    task_sets = [set(int(task) for task in summary["task_ids"]) for summary in summaries]
    observed: set[int] = set()
    for tasks in task_sets:
        if observed & tasks:
            raise ValueError(f"overlapping task shards: {sorted(observed & tasks)}")
        observed.update(tasks)
    if observed != set(range(10)):
        raise ValueError(f"scientific shards must cover task IDs 0..9 exactly, got {sorted(observed)}")
    row_names = set(summaries[0]["rows"])
    if any(set(summary["rows"]) != row_names for summary in summaries[1:]):
        raise ValueError("scientific shards have different row sets")
    required_rows = {"full_k1", CONDITION_ROW, ACTION_ROW, PRIMARY_ROW}
    if not required_rows.issubset(row_names):
        raise ValueError("scientific shards lack a required row")

    output = require_empty_output(args.output)
    episode_rows: list[dict[str, str]] = []
    outcomes: dict[str, dict[tuple[int, int], bool]] = {
        row: {} for row in row_names
    }
    for shard in shard_dirs:
        with (shard / "episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["task_id"]), int(row["episode"]))
                name = row["row"]
                if key in outcomes[name]:
                    raise ValueError(f"duplicate episode key for {name}: {key}")
                outcomes[name][key] = row["success"].lower() == "true"
                episode_rows.append(dict(row))
    for name, values in outcomes.items():
        if len(values) != 200:
            raise ValueError(f"row {name} has {len(values)} episodes, expected 200")

    counters = {row: collections.Counter() for row in row_names}
    latencies: dict[str, dict[str, list[float]]] = {
        row: collections.defaultdict(list) for row in row_names
    }
    diagnostics: dict[str, dict[str, list[float]]] = {
        row: collections.defaultdict(list) for row in row_names
    }
    tracking: dict[str, dict[int, dict[str, list[float]]]] = {
        row: collections.defaultdict(lambda: collections.defaultdict(list)) for row in row_names
    }
    residual_records: dict[str, list[dict[str, Any]]] = {row: [] for row in row_names}
    drift_records: dict[str, list[dict[str, Any]]] = {row: [] for row in row_names}
    for shard, summary in zip(shard_dirs, summaries):
        samples = torch.load(shard / "metric_samples.pt", map_location="cpu", weights_only=False)
        if samples.get("schema_version") != "simvla_hierarchical_metric_samples_v2":
            raise ValueError(f"unsupported metric sample schema in {shard}")
        for row in row_names:
            counters[row].update(summary["rows"][row]["counters"])
            for name, values in samples["latencies"][row].items():
                latencies[row][name].extend(float(value) for value in values)
            for name, values in samples["action_diagnostics"][row].items():
                diagnostics[row][name].extend(float(value) for value in values)
            for age, metrics in samples["condition_action_tracking_by_query_age"][row].items():
                for name, values in metrics.items():
                    tracking[row][int(age)][name].extend(float(value) for value in values)
            residual_records[row].extend(samples["correction_residual_records"][row])
            drift_records[row].extend(samples["condition_drift_records"][row])

    merged_rows: dict[str, Any] = {}
    for row in sorted(row_names):
        values = outcomes[row]
        env_actions = int(counters[row]["num_env_steps"])
        merged_rows[row] = {
            "successes": sum(values.values()),
            "episodes": len(values),
            "success_rate": sum(values.values()) / len(values),
            "task_wise_success": {
                str(task): sum(
                    success for (candidate_task, _), success in values.items() if candidate_task == task
                )
                / 20.0
                for task in range(10)
            },
            "counters": dict(counters[row]),
            "latency_ms": {
                name: distribution_summary(samples) for name, samples in latencies[row].items()
            },
            "amortized_policy_ms_per_environment_action": sum(
                latencies[row].get("policy_total_ms", [])
            )
            / max(env_actions, 1),
            "action_diagnostics": {
                name: distribution_summary(samples) for name, samples in diagnostics[row].items()
            },
            "correction_residual_by_query_age": correction_residuals_by_age(
                residual_records[row]
            ),
            "condition_cache_drift_by_query_age": trace_metrics_by_age(
                drift_records[row], field="condition_cache_drift"
            ),
            "condition_action_tracking_by_query_age": {
                str(age): {
                    name: distribution_summary(samples) for name, samples in metrics.items()
                }
                for age, metrics in sorted(tracking[row].items())
            },
            "teacher_tracking_enabled": True,
            "teacher_tracking_excluded_from_operational_latency": True,
        }
    paired = {
        f"{row}_minus_full_k1": paired_outcome_summary(
            outcomes["full_k1"], outcomes[row], seed=args.bootstrap_seed
        )
        for row in sorted(row_names - {"full_k1"})
    }
    paired["hybrid_minus_action_correction"] = paired_outcome_summary(
        outcomes[ACTION_ROW], outcomes[PRIMARY_ROW], seed=args.bootstrap_seed
    )
    paired["hybrid_minus_pure_condition"] = paired_outcome_summary(
        outcomes[CONDITION_ROW], outcomes[PRIMARY_ROW], seed=args.bootstrap_seed
    )
    episode_csv = output / "episode_metrics.csv"
    episode_rows.sort(key=lambda row: (row["row"], int(row["task_id"]), int(row["episode"])))
    with episode_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    merged_samples = {
        "schema_version": "simvla_hierarchical_metric_samples_v2",
        "latencies": {row: dict(values) for row, values in latencies.items()},
        "action_diagnostics": {row: dict(values) for row, values in diagnostics.items()},
        "condition_action_tracking_by_query_age": {
            row: {age: dict(values) for age, values in ages.items()} for row, ages in tracking.items()
        },
        "correction_residual_records": residual_records,
        "condition_drift_records": drift_records,
    }
    torch.save(merged_samples, output / "metric_samples.pt")
    shard_manifest = {
        "schema_version": "simvla_hierarchical_scientific_shards_v1",
        "source_signature": signatures[0],
        "shards": [
            {
                "path": str(path),
                "task_ids": summary["task_ids"],
                "online_summary_sha256": sha256_file(path / "online_summary.json"),
                "episode_metrics_sha256": sha256_file(path / "episode_metrics.csv"),
                "query_trace": summary["query_trace_jsonl"],
            }
            for path, summary in zip(shard_dirs, summaries)
        ],
    }
    _write_json(output / "shard_manifest.json", shard_manifest)
    _write_json(
        output / "source_lock.json",
        {"source_signature": signatures[0], "shard_source_locks": [str(path / "source_lock.json") for path in shard_dirs]},
    )
    result = {
        "matrix": "scientific_r1_k4",
        "suite": summaries[0]["suite"],
        "task_ids": list(range(10)),
        "episodes_per_row": 200,
        "rows": merged_rows,
        "paired": paired,
        "parameter_audit": summaries[0]["parameter_audit"],
        "source_signature": signatures[0],
        "episode_metrics_csv": str(episode_csv),
        "query_trace_jsonl": None,
        "query_trace_shards": [summary["query_trace_jsonl"] for summary in summaries],
        "metric_samples_pt": str(output / "metric_samples.pt"),
    }
    _write_json(output / "online_summary.json", result)
    return result


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = require_empty_output(args.output)
    source = _read_json(args.source_lock_audit)
    parity = _read_json(args.parity_summary)
    smoke = _read_json(args.smoke_summary)
    offline = _read_json(args.offline_summary)
    scientific = _read_json(args.scientific_summary)
    required_rows = {"full_k1", CONDITION_ROW, ACTION_ROW, PRIMARY_ROW}
    observed_rows = set(scientific.get("rows", {}))
    signatures = [
        source.get("source_signature"),
        parity.get("source_signature"),
        smoke.get("source_signature"),
        offline.get("source_signature"),
        scientific.get("source_signature"),
    ]
    preflight = {
        "source_lock_pass": bool(source.get("SOURCE_LOCK_PASS", False)),
        "endpoint_parity_pass": bool(parity.get("ENDPOINT_PARITY_PASS", False)),
        "smoke_invariants_pass": bool(smoke.get("SMOKE_INVARIANTS_PASS", False)),
        "offline_gate_pass": bool(offline.get("ONLINE_EVALUATION_GATE_PASS", False)),
        "matrix_name_exact": scientific.get("matrix") == "scientific_r1_k4",
        "episodes_per_row_200": int(scientific.get("episodes_per_row", -1)) == 200,
        "required_rows_present": required_rows.issubset(observed_rows),
        "source_signatures_match": bool(signatures[0])
        and all(signature == signatures[0] for signature in signatures[1:]),
    }
    if not all(preflight.values()):
        result = {
            "verdict": "HYBRID_INCONCLUSIVE",
            "preflight": preflight,
            "reason": "aggregation inputs are incomplete or a prerequisite gate failed",
            "k8_diagnostic_allowed": False,
            "r5_k_gt_1_allowed": False,
        }
        _write_json(output / "hybrid_gate_decision.json", result)
        _write_json(output / "aggregate_summary.json", result)
        return result

    rows = scientific["rows"]
    hybrid = rows[PRIMARY_ROW]
    condition = rows[CONDITION_ROW]
    action = rows[ACTION_ROW]
    hybrid_minus_action = scientific["paired"]["hybrid_minus_action_correction"]
    hybrid_minus_condition = scientific["paired"]["hybrid_minus_pure_condition"]
    task_regressions = {
        task: 100.0 * (
            float(hybrid["task_wise_success"][task])
            - float(action["task_wise_success"][task])
        )
        for task in sorted(set(hybrid["task_wise_success"]) & set(action["task_wise_success"]))
    }
    catastrophic = sum(value < -20.0 for value in task_regressions.values())
    ci = hybrid_minus_action["task_hierarchical_paired_ci95_pp"]
    gate_inputs = HybridGateInputs(
        k1_parity_pass=bool(parity["ENDPOINT_PARITY_PASS"]),
        offline_gate_pass=bool(offline["ONLINE_EVALUATION_GATE_PASS"]),
        hybrid_minus_action_ci95_pp=(float(ci[0]), float(ci[1])),
        hybrid_minus_condition_pp=float(hybrid_minus_condition["candidate_minus_baseline_pp"]),
        hybrid_action_transformer_calls=int(hybrid["counters"]["num_action_transformer_decodes"]),
        condition_action_transformer_calls=int(condition["counters"]["num_action_transformer_decodes"]),
        hybrid_amortized_policy_ms=float(hybrid["amortized_policy_ms_per_environment_action"]),
        condition_amortized_policy_ms=float(condition["amortized_policy_ms_per_environment_action"]),
        action_amortized_policy_ms=float(action["amortized_policy_ms_per_environment_action"]),
        hybrid_gripper_reversals=_metric_mean(hybrid, "gripper_reversals"),
        action_gripper_reversals=_metric_mean(action, "gripper_reversals"),
        hybrid_translation_second_difference=_metric_mean(hybrid, "translation_second_difference"),
        action_translation_second_difference=_metric_mean(action, "translation_second_difference"),
        hybrid_rotation_second_difference=_metric_mean(hybrid, "rotation_second_difference"),
        action_rotation_second_difference=_metric_mean(action, "rotation_second_difference"),
        catastrophic_task_regressions_gt20pp=catastrophic,
        regeneration_recovers_condition_path=bool(
            offline.get("regeneration_recovers_condition_path", False)
        ),
    )
    decision = evaluate_hybrid_gate(gate_inputs)
    decision["source_signature"] = signatures[0]
    decision["preflight"] = preflight
    decision["task_regressions_hybrid_minus_action_pp"] = task_regressions
    decision["scientific_success_rates"] = {
        name: float(rows[name]["success_rate"]) for name in sorted(required_rows)
    }
    decision["source_paths"] = {
        "source_lock_audit": str(Path(args.source_lock_audit).resolve()),
        "parity_summary": str(Path(args.parity_summary).resolve()),
        "smoke_summary": str(Path(args.smoke_summary).resolve()),
        "offline_summary": str(Path(args.offline_summary).resolve()),
        "scientific_summary": str(Path(args.scientific_summary).resolve()),
    }
    _write_json(output / "hybrid_gate_decision.json", decision)
    aggregate = {
        "method": "Hierarchical Latent-Action Correction",
        "protocol": "A_R1",
        "fixed_row": PRIMARY_ROW,
        "preflight": preflight,
        "decision": decision,
        "rows": rows,
        "paired": scientific["paired"],
        "offline_gate": offline,
    }
    _write_json(output / "aggregate_summary.json", aggregate)
    report = [
        "# Hierarchical Latent-Action Correction verdict",
        "",
        f"- Verdict: `{decision['verdict']}`",
        f"- Hybrid SR: `{100.0 * hybrid['success_rate']:.2f}%`",
        f"- Pure condition SR: `{100.0 * condition['success_rate']:.2f}%`",
        f"- Pure action SR: `{100.0 * action['success_rate']:.2f}%`",
        f"- Hybrid minus action paired CI95: `{ci}` pp",
        f"- K8 diagnostic allowed: `{decision['k8_diagnostic_allowed']}`",
        "- R5 K>1 allowed: `False`",
        "",
        "The verdict uses the thresholds stored in `hybrid_gate_decision.json`; no threshold is inferred from these results.",
    ]
    (output / "final_hybrid_gate_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return aggregate


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    """Print a concise, dependency-free view of one staged result."""

    payload = _read_json(args.input)
    if args.kind == "source-lock":
        return {
            "SOURCE_LOCK_PASS": payload.get("SOURCE_LOCK_PASS"),
            "checks": payload.get("checks"),
            "parameters": payload.get("audit", {}).get("parameter_audit"),
            "source_signature": payload.get("source_signature"),
        }
    if args.kind == "parity":
        return payload
    if args.kind == "smoke":
        return {
            key: payload.get(key)
            for key in (
                "SMOKE_INVARIANTS_PASS",
                "failures",
                "episodes_checked",
                "query_records_checked",
                "source_signature",
            )
        }
    if args.kind == "offline":
        return {
            key: payload.get(key)
            for key in (
                "ONLINE_EVALUATION_GATE_PASS",
                "finite_failures",
                "hybrid_gripper_noncollapsed",
                "regenerated_first_action_no_worse_than_pure_condition",
                "action_correction_reset_checks",
                "action_correction_resets_after_regeneration",
                "regeneration_recovers_condition_path",
                "source_signature",
            )
        }
    if args.kind == "scientific":
        return {
            "matrix": payload.get("matrix"),
            "task_ids": payload.get("task_ids"),
            "episodes_per_row": payload.get("episodes_per_row"),
            "success_rates": {
                name: row.get("success_rate")
                for name, row in payload.get("rows", {}).items()
            },
            "paired": payload.get("paired"),
            "source_signature": payload.get("source_signature"),
        }
    if args.kind == "verdict":
        decision = payload.get("decision", payload)
        return {
            key: decision.get(key)
            for key in (
                "verdict",
                "checks",
                "prerequisites",
                "preflight",
                "derived",
                "k8_diagnostic_allowed",
                "r5_k_gt_1_allowed",
                "source_signature",
            )
        }
    raise ValueError(f"unsupported inspect kind: {args.kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source-lock")
    source.add_argument("--output", required=True)
    source.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    source.add_argument(
        "--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct"
    )
    source.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    source.add_argument("--condition-checkpoint", required=True)
    source.add_argument("--action-correction-checkpoint", required=True)
    source.add_argument("--cache", required=True)
    source.add_argument("--k1-summary", required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--source-lock-audit", required=True)
    aggregate.add_argument("--parity-summary", required=True)
    aggregate.add_argument("--smoke-summary", required=True)
    aggregate.add_argument("--offline-summary", required=True)
    aggregate.add_argument("--scientific-summary", required=True)
    merge = subparsers.add_parser("merge-scientific")
    merge.add_argument("--output", required=True)
    merge.add_argument("--shard", action="append", required=True)
    merge.add_argument("--bootstrap-seed", type=int, default=20260814)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument(
        "--kind",
        choices=("source-lock", "parity", "smoke", "offline", "scientific", "verdict"),
        required=True,
    )
    inspect.add_argument("--input", required=True)
    args = parser.parse_args()
    if args.command == "source-lock":
        result = _source_lock(args)
    elif args.command == "merge-scientific":
        result = _merge_scientific(args)
    elif args.command == "inspect":
        result = _inspect(args)
    else:
        result = _aggregate(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
