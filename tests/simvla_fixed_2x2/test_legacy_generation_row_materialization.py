from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_aggregate import (
    materialize_legacy_row,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_GENERATION_SOURCE_SHA256,
    FULL_ROW,
    GENERATION_ROW,
)


@pytest.mark.parametrize(
    ("canonical_row", "legacy_row", "full_calls", "decoder_steps"),
    [
        (FULL_ROW, "baseline_k1", 10, 0),
        (GENERATION_ROW, GENERATION_ROW, 3, 7),
    ],
)
def test_materialize_legacy_generation_row(
    tmp_path: Path,
    canonical_row: str,
    legacy_row: str,
    full_calls: int,
    decoder_steps: int,
) -> None:
    manifest_sha = "a" * 64
    source = tmp_path / "legacy"
    shard = source / "shard_rank0_tasks_0_9"
    shard.mkdir(parents=True)
    (source / "row_summary.json").write_text(
        json.dumps(
            {
                "row": legacy_row,
                "episodes": 500,
                "successes": 500,
                "success_rate": 1.0,
                "manifest_sha256": manifest_sha,
                "source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for task_id in range(10):
        for trial_id in range(50):
            rows.append(
                {
                    "row": legacy_row,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "success": True,
                    "episode_length": 5,
                    "num_policy_queries": 1,
                    "num_full_vlm_calls": 1,
                    "num_action_transformer_flow_iterations": full_calls,
                    "num_generation_decoder_only_steps": decoder_steps,
                    "latency_per_executed_action_ms": 2.0,
                    "normalized_second_difference": 0.1,
                    "switch_disagreement": 0.0,
                }
            )
    with (shard / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "canonical"
    result = materialize_legacy_row(
        argparse.Namespace(
            source=str(source),
            output=str(output),
            row=canonical_row,
            inference_seed="seed02",
            classification="RB2_CONFIRMATORY_EGL",
            expected_manifest_sha256=manifest_sha,
            paper_runtime_match=True,
        )
    )

    assert result["verdict"] == "GENERATION_CONTROL_ROW_PASS"
    assert result["row"] == canonical_row
    assert result["episodes"] == 500
    assert result["policy_queries"] == 500
    assert result["full_action_transformer_evaluations"] == 500 * full_calls
    assert result["generation_loop_updates"] == 500 * decoder_steps
    assert result["compatibility_provenance"]["source_artifacts_modified"] is False
    assert (output / "episode_metrics.csv").is_file()
    assert (output / "compatibility_provenance.json").is_file()
