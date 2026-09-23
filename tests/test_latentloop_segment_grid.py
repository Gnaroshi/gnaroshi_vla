from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from architectures.seer.adapters.latentloop_segment_grid import (
    LatentLoopSegmentExecutor,
    canonical_predicted_horizon_token_indices,
    simulate_segment_execution,
)
from methods.latentloop_segment_grid.decision import (
    apply_commitment_feedback_decision,
    apply_pi05_port_readiness,
)
from methods.latentloop_segment_grid.feedback_schedule import (
    actual_feedback_density,
    build_feedback_plan,
    feedback_mask,
)
from methods.latentloop_segment_grid.metrics import (
    distribution_profile,
    paired_flip_counts,
    paired_hierarchical_bootstrap_interval,
    wilson_interval,
)
from methods.latentloop_segment_grid.serialization import atomic_write_json, read_json
from methods.latentloop_segment_grid.stochasticity import summarize_repeated_outputs
from tools.seer.check_latentloop_k1_parity import compare_k1_rows
from tools.seer.analyze_latentloop_segment_grid import (
    _load_parity,
    _paired_rows,
    _row_summary,
)
from tools.seer.register_latentloop_segment_row import (
    validate_segment_step_contract,
)
from tools.seer.verify_latentloop_k1_parity_artifact import (
    verify_parity_artifact,
)


def test_feedback_masks_and_actual_density() -> None:
    assert feedback_mask(4, "dense") == (1, 1, 1)
    assert feedback_mask(8, "alternate") == (1, 0, 1, 0, 1, 0, 1)
    assert feedback_mask(4, "none") == (0, 0, 0)
    assert actual_feedback_density(()) is None
    assert actual_feedback_density(feedback_mask(8, "alternate")) == pytest.approx(4 / 7)
    assert actual_feedback_density(feedback_mask(10, "none")) == 0.0


def test_l1_identity_and_full_seer_schedule() -> None:
    events = simulate_segment_execution(1, "dense", 6)
    assert all(event["full_forward_called"] == 1 for event in events)
    assert all(event["updater_called"] == 0 for event in events)
    assert all(event["feedback_mask"] is None for event in events)

    executor = LatentLoopSegmentExecutor(4, "dense")
    decisions = [
        executor.decision(step, has_latent_cache=step > 0) for step in range(9)
    ]
    assert [decision.full_refresh for decision in decisions] == [
        True, False, False, False, True, False, False, False, True
    ]
    assert [decision.feedback_enabled for decision in decisions] == [
        None, True, True, True, None, True, True, True, None
    ]
    assert [decision.use_zero_feature for decision in decisions] == [
        False, False, False, False, False, False, False, False, False
    ]


def test_cache_advances_and_updater_runs_on_all_intermediate_steps() -> None:
    events = simulate_segment_execution(8, "alternate", 16)
    assert sum(event["full_forward_called"] for event in events) == 2
    assert sum(event["updater_called"] for event in events) == 14
    assert sum(event["observation_cache_advanced"] for event in events) == 16
    assert sum(event["observation_conditioned_updater_called"] for event in events) == 8
    assert sum(event["zero_feature_updater_called"] for event in events) == 6
    assert all(
        event["updater_called"]
        == event["observation_conditioned_updater_called"]
        + event["zero_feature_updater_called"]
        for event in events
    )


def test_production_step_contract_validation() -> None:
    rows = []
    for event in simulate_segment_execution(4, "alternate", 8):
        feedback = event["feedback_mask"]
        rows.append(
            {
                **event,
                "feedback_mask": "" if feedback is None else feedback,
                "lrnode_update_called": event["updater_called"],
                "observation_conditioned_update_called": event[
                    "observation_conditioned_updater_called"
                ],
                "zero_feature_update_called": event[
                    "zero_feature_updater_called"
                ],
                "fast_encoder_called": event[
                    "observation_conditioned_updater_called"
                ],
            }
        )
    assert validate_segment_step_contract(
        rows, segment_length=4, feedback_schedule="alternate"
    ) == []

    rows[2]["observation_cache_advanced"] = 0
    assert validate_segment_step_contract(
        rows, segment_length=4, feedback_schedule="alternate"
    ) == ["observation_cache_not_advanced"]


def test_none_schedule_matches_legacy_no_delta_semantics() -> None:
    plan = build_feedback_plan(4, "none")
    assert plan.mask == (0, 0, 0)
    for offset in range(1, 4):
        decision = LatentLoopSegmentExecutor(4, "none").decision(
            offset, has_latent_cache=True
        )
        assert decision.use_zero_feature is True
    events = simulate_segment_execution(4, "none", 4)
    assert sum(event["zero_feature_updater_called"] for event in events) == 3
    assert sum(event["observation_conditioned_updater_called"] for event in events) == 0


def test_predicted_horizon_replay_uses_existing_token_indices_only() -> None:
    assert canonical_predicted_horizon_token_indices(4, 3) == [0, 1, 2]
    with pytest.raises(ValueError, match="provides action_pred_steps=3"):
        canonical_predicted_horizon_token_indices(8, 3)


def test_metric_aggregation_and_paired_bootstrap() -> None:
    low, high = wilson_interval(90, 100)
    assert low < 0.9 < high
    assert distribution_profile([1, 2, 3, 4])["p50"] == 2.5
    flips = paired_flip_counts(
        {(0, 0): 1, (0, 1): 0},
        {(0, 0): 0, (0, 1): 1},
    )
    assert flips["baseline_success_candidate_failure"] == 1
    assert flips["baseline_failure_candidate_success"] == 1
    bootstrap = paired_hierarchical_bootstrap_interval(
        [
            {"task_id": 0, "baseline_success": 0, "candidate_success": 1},
            {"task_id": 0, "baseline_success": 1, "candidate_success": 1},
            {"task_id": 1, "baseline_success": 0, "candidate_success": 0},
            {"task_id": 1, "baseline_success": 1, "candidate_success": 1},
        ],
        iterations=200,
        seed=7,
    )
    assert bootstrap["paired_count"] == 4
    assert bootstrap["mean_difference"] == pytest.approx(0.25)


def test_row_summary_uses_executed_call_counters() -> None:
    record = {
        "row_id": "ckpt33_L4_dense",
        "stage": "dense_screening",
        "checkpoint_id": 33,
        "adapter_id": 39,
        "segment_length": 4,
        "feedback_schedule": "dense",
        "planned_feedback_density": 1.0,
        "baseline_kind": "dense_latentloop",
        "ablation_mode": "stepwise",
        "summary_path": "/tmp/summary.json",
    }
    episodes = [{"success": "1"}, {"success": "0"}]
    summary = {
        "num_full_forward_calls": 3,
        "num_env_steps": 10,
        "full_query_reduction_ratio": 0.7,
        "avg_policy_step_latency_ms": 12.5,
        "query_reduction": {"num_lrnode_update_calls": 7},
        "lrnode": {
            "segment_grid": {
                "actual_feedback_density": 6.0 / 7.0,
                "observation_conditioned_updater_calls": 6,
                "zero_feature_updater_calls": 1,
            }
        },
        "renderer_backend": {
            "effective_backend": "osmesa",
            "all_ranks_actual_context_verified": True,
        },
    }
    row = _row_summary(record, episodes, summary)
    assert row["full_seer_call_ratio"] == pytest.approx(0.3)
    assert row["full_query_reduction_ratio"] == pytest.approx(0.7)
    assert row["actual_feedback_density"] == pytest.approx(6.0 / 7.0)


def test_paired_rows_reject_missing_episode_keys() -> None:
    records = [
        {
            "row_id": "baseline",
            "checkpoint_id": 33,
            "segment_length": 1,
            "feedback_schedule": "not_applicable",
            "baseline_kind": "full_replanning",
        },
        {
            "row_id": "candidate",
            "checkpoint_id": 33,
            "segment_length": 4,
            "feedback_schedule": "dense",
            "baseline_kind": "dense_latentloop",
        },
    ]
    episode_rows = {
        "baseline": [
            {"task_id": "0", "episode_id": "0", "seed": "1", "success": "1"},
            {"task_id": "0", "episode_id": "1", "seed": "1", "success": "0"},
        ],
        "candidate": [
            {"task_id": "0", "episode_id": "0", "seed": "1", "success": "1"}
        ],
    }
    with pytest.raises(RuntimeError, match="Paired episode keys differ"):
        _paired_rows(records, episode_rows, bootstrap_iterations=10)


def _decision_summary(*, none_matches: bool = False, pareto_only: bool = False):
    rows = []
    for checkpoint, baseline in ((36, 0.80), (38, 0.82)):
        rows.append(
            {
                "checkpoint_id": checkpoint,
                "segment_length": 1,
                "feedback_schedule": "not_applicable",
                "baseline_kind": "full_replanning",
                "success_rate": baseline,
                "full_query_reduction_ratio": 0.0,
            }
        )
        dense_l4 = baseline + 0.005 if pareto_only else baseline + 0.03
        none_l4 = dense_l4 if none_matches else dense_l4 - (0.01 if pareto_only else 0.04)
        for length, dense in ((4, dense_l4), (8, baseline + 0.02), (10, baseline)):
            rows.append(
                {
                    "checkpoint_id": checkpoint,
                    "segment_length": length,
                    "feedback_schedule": "dense",
                    "baseline_kind": "dense_latentloop",
                    "success_rate": dense,
                    "full_query_reduction_ratio": 1.0 - 1.0 / length,
                }
            )
        rows.append(
            {
                "checkpoint_id": checkpoint,
                "segment_length": 4,
                "feedback_schedule": "none",
                "baseline_kind": "no_observation_latent_dynamics",
                "success_rate": none_l4,
                "full_query_reduction_ratio": 0.75,
            }
        )
    return {
        "rows": rows,
        "k1_parity": {"36": {"pass": True}, "38": {"pass": True}},
    }


def test_scientific_and_pi05_decision_rules() -> None:
    supported = apply_commitment_feedback_decision(_decision_summary())
    assert supported["verdict"] == "COMMITMENT_FEEDBACK_SUPPORTED"
    assert apply_pi05_port_readiness(supported)["verdict"] == "PI05_PORT_JUSTIFIED"

    pareto = apply_commitment_feedback_decision(
        _decision_summary(pareto_only=True)
    )
    assert pareto["verdict"] == "PARETO_IMPROVEMENT_ONLY"
    assert apply_pi05_port_readiness(pareto)["verdict"] == "PI05_PORT_DIAGNOSTIC_ONLY"

    rejected = apply_commitment_feedback_decision(
        _decision_summary(none_matches=True)
    )
    assert rejected["verdict"] == "COMMITMENT_FEEDBACK_NOT_SUPPORTED"
    assert apply_pi05_port_readiness(rejected)["verdict"] == "PI05_PORT_NOT_JUSTIFIED"

    inconclusive = {"verdict": "COMMITMENT_FEEDBACK_INCONCLUSIVE"}
    assert (
        apply_pi05_port_readiness(inconclusive)["verdict"]
        == "PI05_PORT_NOT_JUSTIFIED"
    )


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_k1_parity_helper(tmp_path: Path) -> None:
    episode_rows = [
        {"task_id": 0, "episode_id": 0, "seed": 42, "success": 1},
        {"task_id": 0, "episode_id": 1, "seed": 42, "success": 0},
    ]
    action_rows = [
        {
            "task_id": 0,
            "episode_id": episode,
            "timestep": 0,
            **{f"action_{index}": 0.1 * index for index in range(7)},
        }
        for episode in (0, 1)
    ]
    records = []
    for name in ("base", "adapter"):
        episode_path = tmp_path / name / "episodes.csv"
        step_dir = tmp_path / name / "steps"
        _write_csv(episode_path, episode_rows)
        _write_csv(step_dir / "trace.csv", action_rows)
        records.append(
            {
                "episode_metrics_path": str(episode_path),
                "step_log_dir": str(step_dir),
            }
        )
    result = compare_k1_rows(records[0], records[1], action_tolerance=1e-6)
    assert result["pass"] is True
    assert result["max_executed_action_difference"] == 0.0


def test_serialization_and_stochasticity_summary(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    atomic_write_json(path, {"b": 2, "a": 1}, refuse_overwrite=True)
    assert read_json(path) == {"a": 1, "b": 2}
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"a": 3}, refuse_overwrite=True)

    repeated = summarize_repeated_outputs(
        [np.zeros((2,)), np.asarray([0.0, 0.25])],
        [np.zeros((1,)), np.asarray([0.5])],
        [np.zeros((1,)), np.zeros((1,))],
    )
    assert repeated["max_latent_difference"] == pytest.approx(0.25)
    assert repeated["max_raw_action_difference"] == pytest.approx(0.5)
    assert repeated["max_executed_action_difference"] == 0.0


def test_existing_parity_artifact_gate(tmp_path: Path) -> None:
    artifact = tmp_path / "k1_parity.json"
    atomic_write_json(
        artifact,
        {
            "checkpoint_id": 33,
            "checkpoint_profile": "local_best91",
            "baseline_checkpoint_sha256": "baseline-sha",
            "adapter_checkpoint_sha256": "adapter-sha",
            "pass": True,
            "outcome_parity": True,
            "action_keys_identical": True,
        },
    )
    verify_parity_artifact(
        artifact,
        33,
        checkpoint_profile="local_best91",
        baseline_sha256="baseline-sha",
        adapter_sha256="adapter-sha",
    )
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        verify_parity_artifact(artifact, 36)
    with pytest.raises(RuntimeError, match="checkpoint_profile mismatch"):
        verify_parity_artifact(
            artifact,
            33,
            checkpoint_profile="official",
        )


def test_analysis_accepts_external_parity_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "parity" / "k1_parity.json"
    atomic_write_json(
        artifact,
        {
            "checkpoint_id": 33,
            "pass": True,
            "outcome_parity": True,
            "action_keys_identical": True,
        },
    )
    parity = _load_parity([], [artifact])
    assert parity["33"]["pass"] is True
    assert parity["33"]["path"] == str(artifact.resolve())
