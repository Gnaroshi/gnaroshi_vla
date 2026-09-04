"""Aggregate matched-control and three-seed VLA-Cache LIBERO results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(args: argparse.Namespace) -> dict:
    root = Path(args.eval_root).expanduser().resolve()
    control = _load(root / "matched_full_control" / "seed01" / "summary.json")
    cache = [
        _load(root / "vla_cache" / f"seed{index:02d}" / "summary.json")
        for index in (1, 2, 3)
    ]
    expected_verdict = "SIMVLA_VLA_CACHE_LIBERO_EVAL_COMPLETE"
    if control.get("verdict") != expected_verdict or any(
        item.get("verdict") != expected_verdict for item in cache
    ):
        raise RuntimeError("one or more evaluation rows are incomplete")
    if control.get("episodes") != 500 or any(item.get("episodes") != 500 for item in cache):
        raise RuntimeError("paper comparison requires 500 episodes per evaluation seed")
    control_latency = float(control["latency_per_executed_action_ms"])
    rows = []
    for seed, item in enumerate(cache, start=1):
        rows.append(
            {
                "method": "VLA-Cache",
                "seed": f"seed{seed:02d}",
                "episodes": item["episodes"],
                "success_rate_percent": 100.0 * float(item["success_rate"]),
                "latency_ms_per_action": item["latency_per_executed_action_ms"],
                "speedup_vs_matched_full": control_latency / float(item["latency_per_executed_action_ms"]),
                "text_token_layer_reduction_percent": 100.0 * float(item["text_token_layer_reduction"]),
                "actual_kv_reuse_queries": item["actual_kv_reuse_queries"],
                "peak_cuda_memory_gib": item["peak_cuda_memory_gib"],
            }
        )
    result = {
        "verdict": "SIMVLA_VLA_CACHE_LIBERO_THREE_SEED_COMPLETE",
        "matched_full_control": control,
        "vla_cache_three_seed": {
            "episodes_total": sum(int(item["episodes"]) for item in cache),
            "success_rate_percent_mean": float(np.mean([100 * item["success_rate"] for item in cache])),
            "success_rate_percent_min": float(np.min([100 * item["success_rate"] for item in cache])),
            "success_rate_percent_max": float(np.max([100 * item["success_rate"] for item in cache])),
            "latency_ms_per_action_mean": float(np.mean([item["latency_per_executed_action_ms"] for item in cache])),
            "speedup_vs_matched_full_mean": float(np.mean([item["speedup_vs_matched_full"] for item in rows])),
            "text_token_layer_reduction_percent_mean": float(np.mean([100 * item["text_token_layer_reduction"] for item in cache])),
            "per_seed": rows,
        },
        "comparison_axis": {
            "suite": "libero_10",
            "episodes_per_seed": 500,
            "gpu": "RTX 5090 (rb2)",
            "action_horizon": 10,
            "execution_horizon": 5,
            "flow_steps": 10,
            "paired_with_existing_simvla_manifests": True,
            "existing_baseline_and_ours_result_root": str(Path(args.reference_root).expanduser().resolve()),
            "baseline_rerun": False,
            "matched_full_control_purpose": (
                "isolate the eager-attention backend required by VLA-Cache; "
                "it is not a replacement for the existing SimVLA baseline"
            ),
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "vla_cache_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# SimVLA VLA-Cache LIBERO-Long comparison",
        "",
        "- VLA-Cache is evaluated on the same three 500-episode manifests used by the SimVLA paper rows.",
        "- Existing SimVLA baseline and Ours rows are referenced, not rerun.",
        "- The cache-off row is a matched eager-attention backend control, not another official baseline.",
        f"- VLA-Cache SR: {result['vla_cache_three_seed']['success_rate_percent_mean']:.2f}% (three-seed mean)",
        f"- VLA-Cache latency: {result['vla_cache_three_seed']['latency_ms_per_action_mean']:.3f} ms/action",
        f"- Speedup vs matched cache-off control: {result['vla_cache_three_seed']['speedup_vs_matched_full_mean']:.3f}x",
        f"- Decoder token-layer reduction: {result['vla_cache_three_seed']['text_token_layer_reduction_percent_mean']:.2f}%",
        "",
        f"Existing baseline/Ours root: `{result['comparison_axis']['existing_baseline_and_ours_result_root']}`",
    ]
    (output / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--eval-root", required=True)
    value.add_argument("--reference-root", required=True)
    value.add_argument("--output", required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(summarize(parser().parse_args()), indent=2, sort_keys=True))
