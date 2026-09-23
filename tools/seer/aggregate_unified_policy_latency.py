#!/usr/bin/env python3
"""Aggregate four independent RTX 3090 Seer latency replicates."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


EXPECTED_REPLICATE_COUNT = 4
PRIMARY_METHODS = (
    "seer_full_k1",
    "latentloop_k2",
    "latentloop_k3",
    "latentloop_k4",
    "latentloop_k5",
    "latentloop_k6",
    "latentloop_k7",
    "latentloop_k8",
    "latent_bridge_large_k4_compiled",
    "vla_cache_reuse",
)
CONTROL_METHODS = (
    "latent_bridge_large_k4_eager",
    "vla_cache_indexed_full",
)
COMPONENT_METHODS = (
    "component_shared_action_head",
    "component_latentloop_skip_age1",
    "component_latent_bridge_skip_eager",
    "component_latent_bridge_skip_compiled",
)


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("latency samples must be finite and non-empty")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _same(replicates: list[dict], path: tuple[str, ...], label: str):
    values = []
    for payload in replicates:
        value = payload
        for key in path:
            value = value[key]
        values.append(value)
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError(f"replicate contract differs for {label}: {values}")
    return values[0]


def _method_label(name: str) -> str:
    labels = {
        "seer_full_k1": "Seer",
        "latent_bridge_large_k4_compiled": "Latent Bridge Large (compiled)",
        "latent_bridge_large_k4_eager": "Latent Bridge Large (eager control)",
        "vla_cache_reuse": "VLA-Cache (reuse)",
        "vla_cache_indexed_full": "VLA-Cache (indexed-full control)",
        "component_shared_action_head": "Shared action head",
        "component_latentloop_skip_age1": "LatentLoop skip + shared head",
        "component_latent_bridge_skip_eager": "Latent Bridge skip + shared head (eager)",
        "component_latent_bridge_skip_compiled": "Latent Bridge skip + shared head (compiled)",
    }
    if name.startswith("latentloop_k"):
        return f"LatentLoop (K={name.removeprefix('latentloop_k')})"
    return labels.get(name, name)


def _k_value(name: str) -> int | str:
    if name == "seer_full_k1":
        return 1
    if name.startswith("latentloop_k"):
        return int(name.removeprefix("latentloop_k"))
    if "_k4_" in name:
        return 4
    return "-"


def _format(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _table(method_names: tuple[str, ...], aggregate: dict) -> list[str]:
    lines = [
        "| Method | K | Full-query fraction | Wall ms/action | CUDA ms/action | Speedup | Reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in method_names:
        row = aggregate[name]
        lines.append(
            "| "
            + " | ".join(
                (
                    _method_label(name),
                    str(_k_value(name)),
                    _format(row["full_forward_fraction"], 3),
                    f"{_format(row['wall_ms_per_action']['replicate_mean'])} $\\pm$ "
                    f"{_format(row['wall_ms_per_action']['replicate_std'])}",
                    f"{_format(row['cuda_event_ms_per_action']['replicate_mean'])} $\\pm$ "
                    f"{_format(row['cuda_event_ms_per_action']['replicate_std'])}",
                    f"{_format(row['speedup_vs_paired_seer']['mean'])}$\\times$",
                    f"{_format(row['latency_reduction_vs_paired_seer_percent']['mean'], 1)}\\%",
                )
            )
            + " |"
        )
    return lines


def aggregate(inputs: list[Path]) -> dict:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if any(payload.get("status") != "PASS" for payload in payloads):
        raise RuntimeError("all replicate files must have status=PASS")
    by_id = {payload["replicate_id"]: payload for payload in payloads}
    if len(by_id) != EXPECTED_REPLICATE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_REPLICATE_COUNT} unique replicates, got {tuple(by_id)}"
        )
    invalid_ids = sorted(name for name in by_id if re.fullmatch(r"gpu[0-9]+", name) is None)
    if invalid_ids:
        raise RuntimeError(f"replicate IDs must have the form gpu<physical-index>: {invalid_ids}")
    replicate_ids = tuple(sorted(by_id, key=lambda name: int(name.removeprefix("gpu"))))
    payloads = [by_id[name] for name in replicate_ids]

    method_names = tuple(payloads[0]["results"])
    expected_methods = set(PRIMARY_METHODS + CONTROL_METHODS + COMPONENT_METHODS)
    if set(method_names) != expected_methods:
        raise RuntimeError(
            f"unexpected method set; missing={sorted(expected_methods - set(method_names))}, "
            f"extra={sorted(set(method_names) - expected_methods)}"
        )
    for payload in payloads[1:]:
        if set(payload["results"]) != set(method_names):
            raise RuntimeError("method set differs across replicates")

    hardware_names = [payload["hardware"]["name"] for payload in payloads]
    if any("RTX 3090" not in name for name in hardware_names):
        raise RuntimeError(f"all replicates must run on RTX 3090: {hardware_names}")
    _same(payloads, ("timing_contract",), "timing contract")
    _same(payloads, ("software",), "software stack")
    _same(payloads, ("assets", "sha256"), "asset hashes")
    _same(payloads, ("source", "file_sha256"), "source hashes")
    _same(payloads, ("benchmark", "measured_cycles"), "measured cycles")
    _same(payloads, ("benchmark", "warmup_cycles_excluded"), "warmup cycles")
    _same(payloads, ("input_audit",), "cached input contract")

    baseline_means = [
        payload["results"]["seer_full_k1"]["wall_ms_per_action"]["mean"]
        for payload in payloads
    ]
    aggregate_methods = {}
    for name in method_names:
        entries = [payload["results"][name] for payload in payloads]
        wall_means = [entry["wall_ms_per_action"]["mean"] for entry in entries]
        cuda_means = [entry["cuda_event_ms_per_action"]["mean"] for entry in entries]
        wall_raw = [
            sample
            for entry in entries
            for sample in entry["raw_wall_ms_per_action"]
        ]
        cuda_raw = [
            sample
            for entry in entries
            for sample in entry["raw_cuda_event_ms_per_action"]
        ]
        speedups = [base / method for base, method in zip(baseline_means, wall_means)]
        reductions = [(1.0 - method / base) * 100.0 for base, method in zip(baseline_means, wall_means)]
        wall_pooled = _summary(wall_raw)
        cuda_pooled = _summary(cuda_raw)
        aggregate_methods[name] = {
            "label": _method_label(name),
            "group": entries[0]["group"],
            "k": _k_value(name),
            "actions_per_measured_call": entries[0]["actions_per_measured_call"],
            "full_forwards_per_measured_call": entries[0]["full_forwards_per_measured_call"],
            "lightweight_updates_per_measured_call": entries[0]["lightweight_updates_per_measured_call"],
            "full_forward_fraction": entries[0]["full_forward_fraction"],
            "wall_ms_per_action": {
                "replicate_means": wall_means,
                "replicate_mean": float(np.mean(wall_means)),
                "replicate_std": float(np.std(wall_means, ddof=1)),
                "pooled": wall_pooled,
            },
            "cuda_event_ms_per_action": {
                "replicate_means": cuda_means,
                "replicate_mean": float(np.mean(cuda_means)),
                "replicate_std": float(np.std(cuda_means, ddof=1)),
                "pooled": cuda_pooled,
            },
            "speedup_vs_paired_seer": {
                "replicate_values": speedups,
                "mean": float(np.mean(speedups)),
                "std": float(np.std(speedups, ddof=1)),
            },
            "latency_reduction_vs_paired_seer_percent": {
                "replicate_values": reductions,
                "mean": float(np.mean(reductions)),
                "std": float(np.std(reductions, ddof=1)),
            },
            "throughput_upper_bound_hz": 1000.0 / float(np.mean(wall_means)),
        }

    baseline = aggregate_methods["seer_full_k1"]["wall_ms_per_action"]
    baseline_cv = baseline["replicate_std"] / baseline["replicate_mean"]
    latentloop_means = [
        aggregate_methods[f"latentloop_k{k}"]["wall_ms_per_action"]["replicate_mean"]
        for k in range(2, 9)
    ]
    monotonic_violations = [
        {"lower_k": k, "higher_k": k + 1, "lower_ms": left, "higher_ms": right}
        for k, (left, right) in enumerate(zip(latentloop_means, latentloop_means[1:]), start=2)
        if right > left
    ]

    full_ms = aggregate_methods["seer_full_k1"]["wall_ms_per_action"]["replicate_mean"]
    ll_skip_ms = aggregate_methods["component_latentloop_skip_age1"]["wall_ms_per_action"]["replicate_mean"]
    bridge_skip_ms = aggregate_methods["component_latent_bridge_skip_compiled"]["wall_ms_per_action"]["replicate_mean"]
    formula_checks = {}
    for k in range(2, 9):
        predicted = (full_ms + (k - 1) * ll_skip_ms) / k
        observed = aggregate_methods[f"latentloop_k{k}"]["wall_ms_per_action"]["replicate_mean"]
        formula_checks[f"latentloop_k{k}"] = {
            "component_formula_ms": predicted,
            "observed_schedule_ms": observed,
            "relative_difference_percent": (observed / predicted - 1.0) * 100.0,
        }
    bridge_predicted = (full_ms + 3 * bridge_skip_ms) / 4
    bridge_observed = aggregate_methods["latent_bridge_large_k4_compiled"]["wall_ms_per_action"]["replicate_mean"]
    formula_checks["latent_bridge_large_k4_compiled"] = {
        "component_formula_ms": bridge_predicted,
        "observed_schedule_ms": bridge_observed,
        "relative_difference_percent": (bridge_observed / bridge_predicted - 1.0) * 100.0,
    }

    warnings = []
    if baseline_cv > 0.05:
        warnings.append(f"Seer baseline replicate CV exceeds 5%: {baseline_cv * 100:.2f}%")
    if monotonic_violations:
        warnings.append("Measured LatentLoop K curve is not strictly monotonic; inspect raw replicates.")
    return {
        "status": "PASS_WITH_WARNINGS" if warnings else "PASS",
        "scope": "Seer-only unified pure method latency",
        "replicate_ids": list(replicate_ids),
        "replicate_files": [str(path.resolve()) for path in inputs],
        "hardware_names": hardware_names,
        "timing_contract": payloads[0]["timing_contract"],
        "software": payloads[0]["software"],
        "assets": payloads[0]["assets"],
        "source": payloads[0]["source"],
        "benchmark": payloads[0]["benchmark"],
        "method_audit": payloads[0]["method_audit"],
        "vla_cache_runtime_validation": {
            payload["replicate_id"]: payload["vla_cache_runtime_validation"]
            for payload in payloads
        },
        "aggregate": aggregate_methods,
        "diagnostics": {
            "seer_baseline_replicate_cv": baseline_cv,
            "latentloop_monotonic_violations": monotonic_violations,
            "component_formula_checks": formula_checks,
            "warnings": warnings,
        },
    }


def write_outputs(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "unified_policy_latency.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    fieldnames = (
        "method_id", "method", "group", "k", "full_forward_fraction",
        "wall_ms_per_action_mean", "wall_ms_per_action_std_across_gpus",
        "wall_ms_per_action_p50_pooled", "wall_ms_per_action_p95_pooled",
        "cuda_ms_per_action_mean", "cuda_ms_per_action_std_across_gpus",
        "speedup_vs_paired_seer", "latency_reduction_percent",
        "throughput_upper_bound_hz",
    )
    with (output_dir / "unified_policy_latency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in payload["aggregate"].items():
            writer.writerow({
                "method_id": name,
                "method": row["label"],
                "group": row["group"],
                "k": row["k"],
                "full_forward_fraction": row["full_forward_fraction"],
                "wall_ms_per_action_mean": row["wall_ms_per_action"]["replicate_mean"],
                "wall_ms_per_action_std_across_gpus": row["wall_ms_per_action"]["replicate_std"],
                "wall_ms_per_action_p50_pooled": row["wall_ms_per_action"]["pooled"]["p50"],
                "wall_ms_per_action_p95_pooled": row["wall_ms_per_action"]["pooled"]["p95"],
                "cuda_ms_per_action_mean": row["cuda_event_ms_per_action"]["replicate_mean"],
                "cuda_ms_per_action_std_across_gpus": row["cuda_event_ms_per_action"]["replicate_std"],
                "speedup_vs_paired_seer": row["speedup_vs_paired_seer"]["mean"],
                "latency_reduction_percent": row["latency_reduction_vs_paired_seer_percent"]["mean"],
                "throughput_upper_bound_hz": row["throughput_upper_bound_hz"],
            })

    lines = [
        "# Unified Seer pure method latency",
        "",
        "All values below come from one benchmark implementation, one Seer instance per GPU, "
        "the same cached input windows, and a paired Seer denominator on that GPU. Values are "
        "the mean and standard deviation across four independent RTX 3090 GPUs.",
        "",
        "## Paper-facing methods",
        "",
        *_table(PRIMARY_METHODS, payload["aggregate"]),
        "",
        "## Controls",
        "",
        *_table(CONTROL_METHODS, payload["aggregate"]),
        "",
        "## Components",
        "",
        *_table(COMPONENT_METHODS, payload["aggregate"]),
        "",
        "## Timing boundary",
        "",
        f"Primary metric: {payload['timing_contract']['primary_metric']}.",
        "Included: " + "; ".join(payload["timing_contract"]["included"]) + ".",
        "Excluded: " + "; ".join(payload["timing_contract"]["excluded"]) + ".",
        "",
        "## Diagnostics",
        "",
        f"- Seer baseline CV across GPUs: {payload['diagnostics']['seer_baseline_replicate_cv'] * 100:.2f}%",
        f"- LatentLoop monotonic violations: {len(payload['diagnostics']['latentloop_monotonic_violations'])}",
    ]
    for warning in payload["diagnostics"]["warnings"]:
        lines.append(f"- WARNING: {warning}")
    (output_dir / "unified_policy_latency.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    inputs = [Path(value) for value in args.input]
    if len(inputs) != EXPECTED_REPLICATE_COUNT:
        raise ValueError(f"expected four --input arguments, got {len(inputs)}")
    payload = aggregate(inputs)
    output_dir = Path(args.output_dir)
    write_outputs(payload, output_dir)
    (output_dir.parent / "campaign_complete.txt").write_text(
        "status=" + payload["status"] + "\n", encoding="utf-8"
    )
    print(f"[AGGREGATE][{payload['status']}] {output_dir / 'unified_policy_latency.md'}")


if __name__ == "__main__":
    main()
