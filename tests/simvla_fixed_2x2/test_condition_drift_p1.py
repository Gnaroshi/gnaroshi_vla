from __future__ import annotations

import copy
import json

import pytest
import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.condition_drift_p1 import (
    ACTION_FIELDS,
    AGES,
    CONDITION_FIELDS,
    PATHS,
    _aggregate_output,
    _write_csv,
    action_fidelity_metrics,
    condition_fidelity_metrics,
    teacher_forced_updates,
    validate_scientific_source_compatibility,
)
from methods.latentloop.modules.native_simvla_v0 import NativeV0UpdateOutput


def test_condition_metrics_include_cosine_and_normalized_mse() -> None:
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])
    prediction = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [-8.0, 9.0]]])
    valid = torch.tensor([[True, True, False]])

    metrics = condition_fidelity_metrics(prediction, target, valid)

    assert metrics["tokens"] == 2
    assert metrics["condition_token_cosine_mean"] == pytest.approx(0.5)
    assert metrics["condition_flat_cosine"] == pytest.approx(0.5)
    assert metrics["condition_raw_mse"] == pytest.approx(0.5)
    assert metrics["condition_normalized_mse"] > 0.0
    assert metrics["condition_max_abs"] == pytest.approx(1.0)


def test_action_metrics_capture_gripper_sign_and_switch_tail() -> None:
    target = torch.zeros(1, 10, 7)
    prediction = target.clone()
    target[0, :5, 6] = torch.tensor([-1.0, -1.0, 1.0, 1.0, -1.0])
    prediction[0, :5, 6] = torch.tensor([-1.0, 1.0, 1.0, -1.0, -1.0])

    metrics = action_fidelity_metrics(prediction, target)

    assert metrics["action_gripper_sign_mismatch_count"] == 2
    assert metrics["action_gripper_sign_mismatch_rate"] == pytest.approx(0.4)
    assert metrics["action_gripper_switch_mismatch_count"] == 4
    assert metrics["action_gripper_switch_mismatch_rate"] == pytest.approx(1.0)
    assert metrics["action_gripper_switch_false_positive_count"] == 2
    assert metrics["action_gripper_switch_false_negative_count"] == 2
    assert metrics["action_gripper_l1"] == pytest.approx(0.8)


def test_teacher_forced_path_uses_exact_previous_condition() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.previous_means: list[float] = []

        def update_once(self, previous_condition, pair, **kwargs):  # type: ignore[no-untyped-def]
            del pair
            self.previous_means.append(float(previous_condition.mean().item()))
            age = int(kwargs["age"])
            condition = previous_condition + float(age)
            return NativeV0UpdateOutput(
                condition=condition,
                residual=torch.zeros_like(condition),
                gate=torch.zeros_like(condition[..., :1]),
            )

    model = FakeModel()
    batch = {
        "anchor_condition": torch.zeros(1, 2, 3),
        "teacher_conditions": torch.stack(
            (
                torch.full((1, 2, 3), 10.0),
                torch.full((1, 2, 3), 20.0),
                torch.full((1, 2, 3), 30.0),
            ),
            dim=1,
        ),
        "image_sequence": torch.zeros(1, 4, 2, 3, 4, 4),
        "proprio_sequence": torch.zeros(1, 4, 8),
    }

    updates = teacher_forced_updates(
        model,  # type: ignore[arg-type]
        batch,
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        group_ids=torch.ones(1, 2, dtype=torch.long),
    )

    assert model.previous_means == [0.0, 10.0, 20.0]
    assert [float(update.condition.mean()) for update in updates] == [1.0, 12.0, 23.0]


def _source_payload() -> dict[str, object]:
    return {
        "checkpoint": "official",
        "combined_sha256": "checkpoint-derived",
        "complete_source_lock": {"path": "old"},
        "selected_physical_gpu_ids": [4, 5],
        "norm_stats_path": "/old/norm.json",
        "norm_stats_sha256": "norm-hash",
        "cache_manifest_path": "/cache/manifest.json",
        "cache_manifest_sha256": "cache-hash",
        "libero_root": "/old/LIBERO",
        "libero_commit": "libero-commit",
        "critical_file_sha256": {"model.py": "model-hash"},
        "environment": {"torch": "2.6.0"},
    }


def test_source_lock_allows_only_verified_path_and_gpu_rebinding() -> None:
    checkpoint = _source_payload()
    current = copy.deepcopy(checkpoint)
    current.update(
        {
            "combined_sha256": "current-derived",
            "complete_source_lock": {"path": "new"},
            "selected_physical_gpu_ids": [4, 5, 6, 7],
            "norm_stats_path": "/new/norm.json",
            "libero_root": "/new/LIBERO",
        }
    )

    result = validate_scientific_source_compatibility(current, checkpoint)

    assert result["scientific_differences"] == {}
    assert set(result["verified_path_relocations"]) == {
        "norm_stats_path",
        "libero_root",
    }

    changed_source = copy.deepcopy(current)
    changed_source["critical_file_sha256"] = {"model.py": "different"}
    with pytest.raises(RuntimeError, match="scientific source mismatch"):
        validate_scientific_source_compatibility(changed_source, checkpoint)

    changed_norm = copy.deepcopy(current)
    changed_norm["norm_stats_sha256"] = "different"
    with pytest.raises(RuntimeError, match="relocation identity mismatch"):
        validate_scientific_source_compatibility(changed_norm, checkpoint)


def _sequence_row(dataset_index: int, age: int) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_index": dataset_index,
        "task_id": dataset_index,
        "episode_id": f"episode-{dataset_index}",
        "anchor_query_index": 4 * dataset_index,
        "age": age,
    }
    for path in PATHS:
        scale = float(age) if path == "recursive" else 0.5 * float(age)
        for field in CONDITION_FIELDS:
            row[f"{path}/{field}"] = (
                1.0 - 0.01 * scale if "cosine" in field else 0.01 * scale
            )
        for field in ACTION_FIELDS:
            row[f"{path}/{field}"] = 0.02 * scale
        row[f"{path}/condition_token_cosine_p05"] = 1.0 - 0.02 * scale
        row[f"{path}/condition_max_abs"] = 0.03 * scale
        row[f"{path}/tokens"] = 2
        row[f"{path}/action_gripper_sign_mismatch_count"] = 0
        row[f"{path}/action_gripper_switch_mismatch_count"] = 0
        row[f"{path}/action_gripper_switch_false_positive_count"] = 0
        row[f"{path}/action_gripper_switch_false_negative_count"] = 0
        row[f"{path}/teacher_gripper_switch_count"] = 0
        row[f"{path}/predicted_gripper_switch_count"] = 0
        row[f"{path}/gate_mean"] = 0.1
        row[f"{path}/residual_rms"] = 0.2
    direct_scale = 0.0 if age == 1 else float(age)
    row.update(
        {
            "recursive_vs_teacher/condition_token_cosine_mean": 1.0 - 0.01 * direct_scale,
            "recursive_vs_teacher/condition_flat_cosine": 1.0 - 0.01 * direct_scale,
            "recursive_vs_teacher/condition_normalized_mse": 0.01 * direct_scale,
            "recursive_vs_teacher/condition_raw_mse": 0.01 * direct_scale,
            "recursive_vs_teacher/condition_max_abs": 0.02 * direct_scale,
            "recursive_vs_teacher/action_max_abs": 0.03 * direct_scale,
            "recursive_vs_teacher/action_first5_l1": 0.01 * direct_scale,
        }
    )
    return row


def test_completed_shards_aggregate_with_age1_parity(tmp_path) -> None:
    output = tmp_path / "diagnostic"
    shard = output / "shards" / "rank_0"
    shard.mkdir(parents=True)
    sequence_rows = [
        _sequence_row(dataset_index, age)
        for dataset_index in range(2)
        for age in AGES
    ]
    group_rows = []
    for row in sequence_rows:
        for path in PATHS:
            group_rows.append(
                {
                    "dataset_index": row["dataset_index"],
                    "task_id": row["task_id"],
                    "episode_id": row["episode_id"],
                    "anchor_query_index": row["anchor_query_index"],
                    "age": row["age"],
                    "path": path,
                    "group_id": 1,
                    "group_name": "image_view_0",
                    **{field: row[f"{path}/{field}"] for field in CONDITION_FIELDS},
                    "tokens": 2,
                }
            )
    _write_csv(shard / "sequence_age_metrics.csv", sequence_rows)
    _write_csv(shard / "token_group_metrics.csv", group_rows)
    (shard / "shard_summary.json").write_text(
        json.dumps({"verdict": "CONDITION_DRIFT_P1_SHARD_COMPLETE"}) + "\n",
        encoding="utf-8",
    )
    (output / "run_contract.json").write_text(
        json.dumps(
            {
                "world_size": 1,
                "evaluated_sequences": 2,
                "dataset_split": {"split": "heldout"},
                "source_compatibility": {"verdict": "PASS"},
                "diagnostic_source_lock": {"git_commit": "test"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _aggregate_output(output)

    assert summary["verdict"] == "P1_CONDITION_DRIFT_DIAGNOSTIC_COMPLETE"
    assert summary["rows"] == 6
    assert summary["checks"]["age1_paths_exact"] is True
    assert summary["recursive_teacher_direct_summary"]["2"][
        "condition_token_cosine_mean"
    ]["mean"] == pytest.approx(0.98)
    assert (output / "sequence_age_metrics.csv").is_file()
    assert (output / "token_group_metrics.csv").is_file()
    assert (output / "top100_recursive_gripper_tail.csv").is_file()
