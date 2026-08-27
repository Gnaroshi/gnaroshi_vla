from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

from tools.simvla import paper_suite_matrix as matrix


def _base_manifest() -> dict:
    payload = {
        "schema_version": "simvla_generation_libero_long_v1",
        "source_combined_sha256": matrix.EXPECTED_GENERATION_SOURCE,
        "checkpoint_revision": matrix.EXPECTED_REVISION,
        "suite": "libero_10",
        "trials_per_task": 50,
        "episodes_per_row": 500,
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "num_wait_steps": 10,
        "max_policy_actions": 900,
        "client_resize_size": 224,
        "model_image_size": 384,
        "environment_resolution": 256,
        "determinism_seed": 20260815,
        "action_noise_seed_base": 6828326409295398833,
        "environment_seed": 7,
        "inference_seed_replica": "seed01",
        "renderer": {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONHASHSEED": "20260815",
            "SIMVLA_RENDER_AXIS": "rb2_egl_long500_seed01_v1",
        },
        "episodes": [
            {
                "suite": "libero_10",
                "task_id": task,
                "trial_id": trial,
                "init_state_index": trial,
                "environment_seed": 7,
                "physical_gpu_id": 0,
            }
            for task in range(10)
            for trial in range(50)
        ],
    }
    payload["manifest_sha256"] = matrix.canonical_sha256(payload)
    return payload


def test_prepare_nonlong_manifest_uses_official_step_limit(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    output = tmp_path / "spatial.json"
    matrix.atomic_json(base, _base_manifest())
    result = matrix.prepare_manifest(
        Namespace(
            base_manifest=str(base),
            output=str(output),
            suite="libero_spatial",
            seed="seed03",
        )
    )
    payload = matrix.load_json(output)
    assert result["verdict"] == "PAPER_SUITE_MANIFEST_PASS"
    assert payload["max_policy_actions"] == 800
    assert payload["determinism_seed"] == 20260817
    assert payload["action_noise_seed_base"] == 6828326409295398835
    assert {episode["suite"] for episode in payload["episodes"]} == {
        "libero_spatial"
    }


def test_validate_row_accepts_legacy_baseline_alias(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    matrix.atomic_json(manifest_path, _base_manifest())
    row_root = tmp_path / "baseline"
    shard = row_root / "shard_rank0_tasks_0_9"
    shard.mkdir(parents=True)
    fields = (
        "row",
        "task_id",
        "trial_id",
        "success",
        "episode_length",
        "latency_per_executed_action_ms",
    )
    metrics = shard / "episode_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in range(10):
            for trial in range(50):
                writer.writerow(
                    {
                        "row": "baseline_k1",
                        "task_id": task,
                        "trial_id": trial,
                        "success": task != 0,
                        "episode_length": 100,
                        "latency_per_executed_action_ms": 20.0,
                    }
                )
    matrix.atomic_json(
        shard / "shard_summary.json",
        {
            "row": "baseline_k1",
            "episodes": 500,
            "successes": 450,
            "manifest_sha256": _base_manifest()["manifest_sha256"],
        },
    )
    report, rows = matrix.validate_row_data(
        row_root, manifest_path, "full_nfe10"
    )
    assert len(rows) == 500
    assert report["verdict"] == "PAPER_ROW_PASS"
    assert report["successes"] == 450
    assert report["latency_per_executed_action_ms"] == 20.0


def test_exact_mcnemar_is_symmetric() -> None:
    assert matrix._mcnemar(3, 9) == matrix._mcnemar(9, 3)
    assert matrix._mcnemar(0, 0) == 1.0
