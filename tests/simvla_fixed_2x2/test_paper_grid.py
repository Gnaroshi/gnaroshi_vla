from __future__ import annotations

import csv
import json

import pytest

from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    COUPLED_CONFIGS,
    coupled_row_name,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    PAPER_ANCHOR_ROWS,
    PAPER_COUPLED_ROWS,
    PAPER_GRID_ROWS,
    PAPER_LEARNED_ROWS,
    PAPER_NAIVE_ROWS,
    expected_call_counts,
    generation_row_name,
    naive_generation_row_name,
    row_spec,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.paper_grid import (
    CLASSIFICATION,
    INFERENCE_SEED,
    MANIFEST_SHA256,
    _table_row,
    validate_row,
)


def test_paper_grid_is_exactly_27_unique_scientific_rows() -> None:
    assert len(PAPER_GRID_ROWS) == 27
    assert len(set(PAPER_GRID_ROWS)) == 27
    assert len(PAPER_ANCHOR_ROWS) == 3
    assert len(PAPER_NAIVE_ROWS) == 9
    assert len(PAPER_LEARNED_ROWS) == 9
    assert len(PAPER_COUPLED_ROWS) == 6
    assert set(COUPLED_CONFIGS) == {
        (2, 2),
        (2, 3),
        (2, 5),
        (3, 2),
        (3, 3),
        (3, 5),
    }


def test_ng1_is_disabled_and_kc1_coupling_is_not_a_grid_row() -> None:
    with pytest.raises(ValueError):
        generation_row_name(1)
    with pytest.raises(ValueError):
        naive_generation_row_name(1)
    with pytest.raises(ValueError):
        coupled_row_name(1, 3)
    assert all("kc1" not in row or not row.endswith("_coupled") for row in PAPER_GRID_ROWS)


def test_every_grid_row_has_exact_compute_counters() -> None:
    for row in PAPER_GRID_ROWS:
        spec = row_spec(row)
        counts = expected_call_counts(row, 7)
        assert counts["full_action_transformer_calls"] == 7 * spec.n_g
        if spec.uses_generation:
            assert counts["generation_loop_updates"] == 7 * (10 - spec.n_g)
            assert counts["integration_updates"] == 70
        else:
            assert counts["generation_loop_updates"] == 0
            assert counts["integration_updates"] == 7 * spec.n_g


def test_validate_row_requires_exact_seed02_10x50_and_counters(tmp_path) -> None:
    row = "full_nfe10"
    rows = []
    for task_id in range(10):
        for trial_id in range(50):
            counts = expected_call_counts(row, 2)
            rows.append(
                {
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "num_policy_queries": 2,
                    "num_full_vlm_calls": counts["full_vlm_calls"],
                    "num_condition_updater_calls": counts["condition_updater_calls"],
                    "num_full_action_transformer_evaluations": counts[
                        "full_action_transformer_calls"
                    ],
                    "num_generation_loop_updates": counts["generation_loop_updates"],
                    "num_integration_updates": counts["integration_updates"],
                }
            )
    with (tmp_path / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "row_summary.json").write_text(
        json.dumps(
            {
                "verdict": "GENERATION_CONTROL_ROW_PASS",
                "row": row,
                "episodes": 500,
                "successes": 479,
                "success_rate": 0.958,
                "manifest_sha256": MANIFEST_SHA256,
                "inference_seed": INFERENCE_SEED,
                "classification": CLASSIFICATION,
                "paper_runtime_match": True,
                "latency_per_executed_action_ms": 1.25,
                "latency_per_policy_query_ms": 6.25,
            }
        ),
        encoding="utf-8",
    )
    report = validate_row(row, tmp_path)
    assert report["verdict"] == "PAPER_GRID_ROW_PASS"
    table_row = _table_row(row, tmp_path)
    assert table_row["full_vlm_calls"] == 1_000
    assert table_row["condition_updater_calls"] == 0
    assert table_row["full_action_transformer_evaluations"] == 10_000
    assert table_row["generation_loop_updates"] == 0
    rows[-1]["num_integration_updates"] = 9
    with (tmp_path / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert validate_row(row, tmp_path)["verdict"] == "PAPER_GRID_ROW_FAIL"
