from __future__ import annotations

import csv
import json

from tools.openpi.aggregate_pi05_v0_streaming_eval import SUITES, aggregate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_requires_and_combines_exact_paired_rows(tmp_path):
    for suite_index, suite in enumerate(SUITES):
        _write_json(
            tmp_path / suite / "protocol" / "evaluation_preflight.json",
            {
                "training": {"checkpoint_step": 7000},
                "evaluation_harness": {"combined_sha256": "harness"},
            },
        )
        for row in ("original", "v0"):
            row_root = tmp_path / suite / row
            successes = 500 - suite_index - (1 if row == "v0" else 0)
            _write_json(
                row_root / "summary.json",
                {
                    "complete": True,
                    "rollouts": 500,
                    "method_label": row,
                    "suite": suite,
                    "successes": successes,
                    "success_rate": successes / 500,
                    "elapsed_seconds": 10.0,
                    "final_evaluation_manifest_sha256": "manifest",
                    "source_lock_id": "source",
                    "efficiency": {
                        "policy_queries": 100,
                        "actually_executed_actions": 500,
                        "operation_call_matrix": {
                            "full_prefix_refreshes": 100 if row == "original" else 25,
                            "action_expert_calls": 100,
                        },
                        "latency_ms": {
                            "infer_ms": {
                                "count": 100,
                                "mean": 4.0 if row == "original" else 2.0,
                            }
                        },
                        "peak_vram_bytes": 1000,
                    },
                },
            )
            row_root.mkdir(parents=True, exist_ok=True)
            with (row_root / "episode_outcomes.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("suite", "task_id", "trial", "success"),
                )
                writer.writeheader()
                for task_id in range(10):
                    for trial in range(50):
                        flat_index = task_id * 50 + trial
                        writer.writerow(
                            {
                                "suite": suite,
                                "task_id": task_id,
                                "trial": trial,
                                "success": flat_index < successes,
                            }
                        )

    result = aggregate(tmp_path)

    assert result["V0_STREAMING_PAIRED_EVALUATION_COMPLETE"] is True
    assert result["rows"]["original"]["successes"] == 1994
    assert result["rows"]["v0"]["successes"] == 1990
    assert result["paired"]["contingency"]["original_success_v0_failure"] == 4
    assert result["rows"]["original"]["efficiency"]["actual_mean_k_q"] == 1.0
    assert result["rows"]["v0"]["efficiency"]["actual_mean_k_q"] == 4.0
    assert (tmp_path / "paired_episode_outcomes.csv").is_file()
    assert (tmp_path / "per_task_summary.csv").is_file()
    assert (tmp_path / "combined_summary.json").is_file()
