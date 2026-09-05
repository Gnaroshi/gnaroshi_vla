"""Aggregate repaired VLA-Cache results against existing native paper results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .eval import _sha256, implementation_identity


def summarize(args):
    root = Path(args.eval_root).expanduser().resolve()
    reference_path = Path(args.reference_summary).expanduser().resolve()
    reference = json.loads(reference_path.read_text())
    sources = {item["row"]: item for item in reference["row_summaries"]}
    baseline = sources["full_nfe10"]
    if baseline["episodes"] != 1500 or baseline["seeds"] != 3:
        raise RuntimeError("reference baseline must contain three complete 500-episode seeds")
    baseline_latency = float(baseline["seed_mean_latency_per_action_ms"])
    rows = []
    for seed in ("seed01", "seed02", "seed03"):
        path = root / "vla_cache" / seed / "summary.json"
        item = json.loads(path.read_text())
        if item.get("verdict") != "SIMVLA_VLA_CACHE_LIBERO_EVAL_COMPLETE" or item.get("episodes") != 500:
            raise RuntimeError(f"incomplete 500-episode row: {seed}")
        if item.get("implementation_identity") != implementation_identity():
            raise RuntimeError(f"different source version: {seed}")
        if item.get("manifest_declared_sha256") != reference["manifest_sha256"][seed]:
            raise RuntimeError(f"different paired manifest: {seed}")
        rows.append({"seed": seed, "episodes": item["episodes"], "successes": item["successes"],
                     "success_rate_percent": 100 * item["success_rate"],
                     "latency_ms_per_action": item["latency_per_executed_action_ms"],
                     "text_token_layer_reduction_percent": 100 * item["text_token_layer_reduction"],
                     "actual_kv_reuse_queries": item["actual_kv_reuse_queries"],
                     "peak_cuda_memory_gib": item["peak_cuda_memory_gib"]})
    mean_latency = float(np.mean([row["latency_ms_per_action"] for row in rows]))
    result = {
        "verdict": "SIMVLA_VLA_CACHE_LIBERO_THREE_SEED_COMPLETE",
        "implementation_identity": implementation_identity(),
        "vla_cache_three_seed": {"episodes_total": 1500, "per_seed": rows,
            "success_rate_percent_mean": float(np.mean([row["success_rate_percent"] for row in rows])),
            "latency_ms_per_action_mean": mean_latency,
            "historical_speed_ratio_native_over_cache": baseline_latency / mean_latency},
        "existing_native_baseline": baseline,
        "existing_ours": {key: sources[key] for key in ("generation_ng3", "condition_kc2_ng3", "condition_kc2_ng3_coupled") if key in sources},
        "reference_summary": str(reference_path), "reference_summary_sha256": _sha256(reference_path),
        "comparison_axis": {"suite": "libero_10", "episodes_per_seed": 500, "seeds": 3,
            "gpu": "RTX 5090 (rb2)", "H": 10, "R": 5, "flow_steps": 10,
            "baseline_rerun": False, "same_declared_episode_manifests": True,
            "latency_note": "Historical end-to-end timings, not a simultaneous paired timing trial; do not use the old eager cache-off denominator. A ratio below one means slower than native."},
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "vla_cache_per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison_summary.md").write_text(
        "# SimVLA VLA-Cache LIBERO-Long\n\n"
        "500 episodes per seed, 3 seeds, RTX 5090, H=10, R=5, flow=10.\n\n"
        f"VLA-Cache: {result['vla_cache_three_seed']['success_rate_percent_mean']:.2f}%, {mean_latency:.3f} ms/action.\n\n"
        f"Existing native baseline: {100 * baseline['seed_mean_success_rate']:.2f}%, {baseline_latency:.3f} ms/action.\n\n"
        "Baseline/Ours were not rerun. Timing comparison uses historical measurements; no inflated cache-off baseline is used.\n")
    return result


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--eval-root", required=True)
    value.add_argument("--reference-summary", required=True)
    value.add_argument("--output", required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(summarize(parser().parse_args()), indent=2, sort_keys=True))
