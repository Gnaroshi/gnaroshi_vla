"""CPU-only tests for K1 replay validation and disjoint task merging."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from architectures.simvla.adapters.latentloop.k1_parity_revalidator import (
    ADAPTER_ROW,
    BASELINE_ROW,
    analyze_k1_parity,
)
from architectures.simvla.adapters.latentloop.task_shard_merger import (
    ACTION_DIAGNOSTIC_FIELDS,
    PROTOCOL_A_ROWS,
    merge_task_shards,
)


def _trace(condition: str, noise: str, action: str, seed: int) -> dict[str, object]:
    return {
        "condition_hash": condition,
        "action_noise_hash": noise,
        "action_noise_seed": seed,
        "action_chunk_hash": action,
    }


def _raw_k1_summary() -> dict[str, object]:
    return {
        "matrix": "k1_parity",
        "rows": {
            BASELINE_ROW: {"counters": {}},
            ADAPTER_ROW: {"counters": {}},
        },
        "k1_parity": {
            "exact_action_chunk_equality": False,
            "max_abs_action_chunk_diff": 0.2,
        },
    }


def test_k1_revalidation_ignores_actions_after_input_divergence() -> None:
    outcomes = {
        BASELINE_ROW: {(0, 0): True},
        ADAPTER_ROW: {(0, 0): True},
    }
    traces = {
        BASELINE_ROW: {
            (0, 0, 0): _trace("condition-0", "noise-0", "action-0", 10),
            (0, 0, 1): _trace("condition-left", "noise-1", "action-left", 11),
        },
        ADAPTER_ROW: {
            (0, 0, 0): _trace("condition-0", "noise-0", "action-0", 10),
            (0, 0, 1): _trace("condition-right", "noise-1", "action-right", 11),
        },
    }
    result = analyze_k1_parity(
        raw_summary=_raw_k1_summary(),
        outcomes=outcomes,
        traces=traces,
    )
    assert result["K1_PARITY_PASS"] is True
    assert result["exact_action_chunk_equality"] is True
    assert result["matched_input_query_keys"] == 1
    assert result["condition_diverged_query_keys"] == 1
    assert result["raw_action_hash_mismatch_query_keys"] == 1


def test_k1_revalidation_rejects_action_difference_on_identical_input() -> None:
    outcomes = {
        BASELINE_ROW: {(0, 0): True},
        ADAPTER_ROW: {(0, 0): True},
    }
    traces = {
        BASELINE_ROW: {(0, 0, 0): _trace("condition", "noise", "left", 10)},
        ADAPTER_ROW: {(0, 0, 0): _trace("condition", "noise", "right", 10)},
    }
    result = analyze_k1_parity(
        raw_summary=_raw_k1_summary(),
        outcomes=outcomes,
        traces=traces,
    )
    assert result["K1_PARITY_PASS"] is False
    assert result["action_hash_mismatch_on_matched_input_keys"] == 1


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _make_shard(path: Path, task_id: int) -> None:
    path.mkdir(parents=True)
    rows = {}
    for row_name in PROTOCOL_A_ROWS:
        rows[row_name] = {
            "prediction_horizon": 10,
            "execution_horizon": 1,
            "full_condition_interval": 1 if row_name == "full_k1" else 2,
            "full_condition_environment_action_gap": 1 if row_name == "full_k1" else 2,
            "counters": {
                "num_policy_queries": 10,
                "num_full_vlm_calls": 5,
                "num_env_steps": 10,
            },
            "latency_ms": {
                "policy_total_ms": {
                    "count": 10,
                    "mean": float(task_id + 1),
                    "max": float(task_id + 2),
                }
            },
            "condition_action_tracking_by_query_age": {},
            "teacher_tracking_enabled": True,
        }
    summary = {
        "matrix": "protocol_a_screening",
        "task_ids": [task_id],
        "episodes_per_row": 1,
        "rows": rows,
    }
    config = {
        "matrix": "protocol_a_screening",
        "checkpoint": "checkpoint",
        "smolvlm_model_path": "backbone",
        "norm_stats": "/norm.json",
        "suite": "libero_10",
        "execution_horizon": 1,
        "num_trials": 1,
        "max_env_actions": 900,
        "num_wait_steps": 10,
        "flow_steps": 10,
        "client_resize_size": 224,
        "image_size": 384,
        "resolution": 256,
        "experiment_seed": None,
        "seed": 7,
        "action_noise_seed_base": 17,
        "bootstrap_seed": 19,
        "task_order": "official_reverse",
        "teacher_tracking": True,
        "effective_seed_plan": {
            "experiment_seed": None,
            "process_seed": 7,
            "environment_seed_base": 7,
            "action_noise_seed_base": 17,
            "bootstrap_seed": 19,
            "derivation": "legacy_explicit_seed_tuple",
        },
        "rows": [{"name": name} for name in PROTOCOL_A_ROWS],
        "environment_action_gap_by_row": {name: 1 for name in PROTOCOL_A_ROWS},
        "resolved_task_ids": [task_id],
    }
    source = {
        "root_commit": "root",
        "simvla_upstream_commit": "upstream",
        "checkpoint": {"identifier": "checkpoint", "revision": "revision"},
        "norm_stats_sha256": "norm",
        "conda_env": "simvla_libero",
        "python": "3.10",
        "python_executable": "/python",
        "torch": "2.7",
        "torch_cuda": "12.6",
        "packages": {"mujoco": "2.3.7"},
    }
    _write_json(path / "online_summary.json", summary)
    _write_json(path / "eval_config.json", config)
    _write_json(path / "source_lock.json", source)
    determinism = {
        "protocol": "simvla_online_determinism_v1",
        "scope": {"exact": ["trajectory"], "excluded": ["latency"]},
        "runtime_contract": {"test": True},
        "runtime_sha256": "runtime",
        "run_contract": {
            "runtime_sha256": "runtime",
            "seed_plan": config["effective_seed_plan"],
            "semantic_config": {
                "matrix": "protocol_a_screening",
                "task_ids": [task_id],
                "suite": "libero_10",
            },
            "task_assets": {str(task_id): {"hash": f"task-{task_id}"}},
        },
        "run_contract_sha256": f"run-{task_id}",
    }
    _write_json(path / "determinism_manifest.json", determinism)
    deterministic_rows = [
        {
            "row": row_name,
            "task_id": task_id,
            "episode": 0,
            "success": True,
            "episode_trace_hash": f"{row_name}-{task_id}",
        }
        for row_name in PROTOCOL_A_ROWS
    ]
    _write_json(
        path / "deterministic_results.json",
        {
            "protocol": determinism["protocol"],
            "runtime_sha256": "runtime",
            "run_contract_sha256": f"run-{task_id}",
            "episodes": deterministic_rows,
            "trajectory_sha256": f"trajectory-{task_id}",
        },
    )
    (path / "query_trace.jsonl").write_text("{}\n", encoding="utf-8")
    fields = [
        "row",
        "task_id",
        "episode",
        "success",
        *ACTION_DIAGNOSTIC_FIELDS,
    ]
    with (path / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_name in PROTOCOL_A_ROWS:
            writer.writerow(
                {
                    "row": row_name,
                    "task_id": task_id,
                    "episode": 0,
                    "success": True,
                    **{name: 0.1 for name in ACTION_DIAGNOSTIC_FIELDS},
                }
            )


def test_task_shard_merger_validates_and_reconstructs_matrix(tmp_path: Path) -> None:
    left = tmp_path / "tasks_1"
    right = tmp_path / "tasks_0"
    _make_shard(left, 1)
    _make_shard(right, 0)
    output = tmp_path / "merged"
    result = merge_task_shards(
        shard_dirs=[left, right],
        output_dir=output,
        expected_task_ids=(1, 0),
    )
    assert result["task_ids"] == [1, 0]
    assert result["episodes_per_row"] == 2
    assert len(result["rows"]) == 13
    assert result["rows"]["full_k1"]["successes"] == 2
    assert result["paired_vs_full"]["chunk_aware_latentloop_k4"]["pairs"] == 2
    assert (output / "query_trace_shards.json").is_file()
    assert (output / "determinism_manifest.json").is_file()
    assert (output / "deterministic_results.json").is_file()
