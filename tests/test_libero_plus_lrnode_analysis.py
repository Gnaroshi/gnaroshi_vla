import csv
import json
from pathlib import Path

from tools.seer.analyze_libero_plus_lrnode import analyze, paired_outcomes


def test_paired_outcomes_counts_flips():
    left = {"name": "baseline", "tasks": [
        {"task_id": "0", "task_name": "a", "result": "0"},
        {"task_id": "1", "task_name": "b", "result": "1"},
        {"task_id": "2", "task_name": "c", "result": "1"},
    ]}
    right = {"name": "ours", "tasks": [
        {"task_id": "0", "task_name": "a", "result": "1"},
        {"task_id": "1", "task_name": "b", "result": "0"},
        {"task_id": "2", "task_name": "c", "result": "1"},
    ]}
    result = paired_outcomes(left, right)
    assert result["fail_to_success"] == 1
    assert result["success_to_fail"] == 1
    assert result["net_flip"] == 0
    assert result["success_rate_delta"] == 0.0
    assert not result["exact_outcome_parity"]


def _write_row(root: Path, name: str, outcomes: list[int], full: int, lrnode: int) -> None:
    analysis_dir = root / name / "analysis"
    analysis_dir.mkdir(parents=True)
    summary = {
        "success_rate": sum(value == 1 for value in outcomes) / len(outcomes),
        "libero_plus": {"evaluated_task_count": len(outcomes)},
        "query_reduction": {
            "num_full_forward_calls": full,
            "num_lrnode_update_calls": lrnode,
            "full_query_reduction_ratio": lrnode / (full + lrnode) if full + lrnode else 0.0,
            "effective_query_interval": (full + lrnode) / full if full else 0.0,
        },
        "lrnode": {
            "avg_full_forward_latency_sec": 0.1,
            "avg_lrnode_latency_sec": 0.01,
            "avg_policy_step_latency_sec": 0.05,
        },
        "renderer_backend": {"effective_backend": "osmesa"},
    }
    (analysis_dir / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (analysis_dir / "libero_plus_task_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "task_name", "result"])
        writer.writeheader()
        for task_id, outcome in enumerate(outcomes):
            writer.writerow({"task_id": task_id, "task_name": f"task_{task_id}", "result": outcome})
    with (analysis_dir / "libero_plus_category_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "avg_success"])
        writer.writeheader()
        writer.writerow({"category": "Total", "avg_success": summary["success_rate"]})


def test_analyze_three_rows(tmp_path):
    _write_row(tmp_path, "baseline_k1", [1, 0], 20, 0)
    _write_row(tmp_path, "ours_k1", [1, 0], 20, 0)
    _write_row(tmp_path, "ours_k4", [1, 1], 5, 15)
    payload = analyze(tmp_path)
    assert payload["baseline_vs_ours_k1"]["exact_outcome_parity"]
    assert payload["baseline_vs_ours_k4"]["success_rate_delta"] == 0.5
    assert payload["baseline_vs_ours_k4"]["fail_to_success"] == 1
