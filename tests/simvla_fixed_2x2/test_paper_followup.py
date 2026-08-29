from __future__ import annotations

import csv
import json
from pathlib import Path

from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    expected_call_counts,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.paper_followup import (
    CLASSIFICATION,
    ROWS,
    SEEDS,
    aggregate,
    build_plan,
    validate_cell,
)


MANIFESTS = {
    "seed01": "1" * 64,
    "seed02": "2" * 64,
    "seed03": "3" * 64,
}


def _write_cell(root: Path, seed: str, row: str, *, failed_trial: int = -1) -> None:
    root.mkdir(parents=True)
    rows = []
    successes = 0
    for task_id in range(10):
        for trial_id in range(50):
            success = not (task_id == 9 and trial_id == failed_trial)
            successes += int(success)
            queries = 2
            counts = expected_call_counts(row, queries)
            rows.append(
                {
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "success": success,
                    "episode_length": 10,
                    "num_policy_queries": queries,
                    "num_full_vlm_calls": counts["full_vlm_calls"],
                    "num_condition_updater_calls": counts[
                        "condition_updater_calls"
                    ],
                    "num_full_action_transformer_evaluations": counts[
                        "full_action_transformer_calls"
                    ],
                    "num_generation_loop_updates": counts[
                        "generation_loop_updates"
                    ],
                    "num_integration_updates": counts["integration_updates"],
                }
            )
    with (root / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "row_summary.json").write_text(
        json.dumps(
            {
                "verdict": "FIXED_2X2_ROW_PASS",
                "row": row,
                "episodes": 500,
                "successes": successes,
                "success_rate": successes / 500.0,
                "manifest_sha256": MANIFESTS[seed],
                "inference_seed": seed,
                "classification": CLASSIFICATION,
                "paper_runtime_match": True,
                "latency_per_executed_action_ms": 10.0,
                "latency_per_policy_query_ms": 50.0,
                "model_vlm_encoder_per_call_ms": 30.0,
                "model_condition_updater_per_call_ms": 3.0,
                "model_action_generation_per_query_ms": 20.0,
            }
        ),
        encoding="utf-8",
    )


def test_followup_contract_is_eight_rows_by_three_seeds() -> None:
    assert len(ROWS) == 8
    assert len(SEEDS) == 3
    assert len(set(ROWS)) == len(ROWS)


def test_validate_cell_accepts_boolean_success_and_rejects_counter_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cell"
    _write_cell(root, "seed01", "condition_kc2_ng3")
    report = validate_cell(
        "seed01",
        "condition_kc2_ng3",
        root,
        manifest_sha256=MANIFESTS["seed01"],
    )
    assert report["verdict"] == "PAPER_FOLLOWUP_CELL_PASS"
    rows = list(csv.DictReader((root / "episode_metrics.csv").open()))
    rows[-1]["num_integration_updates"] = "1"
    with (root / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert (
        validate_cell(
            "seed01",
            "condition_kc2_ng3",
            root,
            manifest_sha256=MANIFESTS["seed01"],
        )["verdict"]
        == "PAPER_FOLLOWUP_CELL_FAIL"
    )


def test_plan_and_aggregate_exact_24_cells(tmp_path: Path) -> None:
    roots = {}
    for seed_index, seed in enumerate(SEEDS):
        for row_index, row in enumerate(ROWS):
            root = tmp_path / seed / row
            failed_trial = 0 if row_index == seed_index else -1
            _write_cell(root, seed, row, failed_trial=failed_trial)
            roots[(seed, row)] = root
    plan = build_plan(roots, MANIFESTS)
    assert plan["verdict"] == "PAPER_FOLLOWUP_COMPLETE"
    assert plan["complete_cell_count"] == 24
    assert plan["missing_cell_count"] == 0

    external = tmp_path / "external.json"
    external.write_text(json.dumps({"verdict": "EXTERNAL_PASS"}), encoding="utf-8")
    result = aggregate(
        roots,
        MANIFESTS,
        tmp_path / "aggregate",
        external_artifacts={"external": external},
    )
    assert result["verdict"] == "PAPER_FOLLOWUP_THREE_INFERENCE_SEED_COMPLETE"
    assert result["cell_count"] == 24
    assert len(result["row_summaries"]) == 8
    assert result["external_evidence"]["external"]["verdict"] == "EXTERNAL_PASS"
    assert (
        tmp_path / "aggregate" / "paper_followup_three_seed_summary.json"
    ).is_file()
