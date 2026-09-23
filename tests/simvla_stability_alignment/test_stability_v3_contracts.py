from __future__ import annotations

import json
import math

import pytest
import torch

from architectures.simvla.adapters.latentloop.stability_alignment import (
    v3_continuation,
    v3_pipeline,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_diagnostic_bundle import (
    build as build_v3_diagnostic_bundle,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_contracts import (
    V3_GRADIENT_TARGETS,
    V3_HARD_POOL_SCHEMA,
    V3_LOSS_NAMES,
    calibrate_v3_gradient_weights,
    evaluate_v3_moving_window,
    evaluate_v3_scientific_gate,
    evaluate_v3_stage_gate,
    select_v2_checkpoint,
    true_count_reduction,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_data import (
    V3PoolSampler,
    build_v3_hard_pool_contract,
    gripper_event_contract,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_objectives import (
    gripper_transition_loss_with_boundaries,
    masked_nrms_per_sequence,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_trainer import (
    build_parser as build_v3_trainer_parser,
)


def _metrics() -> dict[str, object]:
    return {
        "age2_recurrence_improvement": 0.25,
        "age3_recurrence_improvement": 0.35,
        "age3_first_r_p95_ratio": 0.95,
        "age3_first_r_p99_ratio": 1.10,
        "age1_first_r_ratio_to_parent": 1.02,
        "exact_ng3_ratio_to_parent": 1.00,
        "parent_age3_mismatch_sequences": 20,
        "candidate_age3_mismatch_sequences": 15,
        "parent_age3_switch_mismatch_sequences": 8,
        "candidate_age3_switch_mismatch_sequences": 8,
        "original_simvla_frozen": True,
    }


def test_true_count_gate_does_not_use_unit_denominator() -> None:
    assert true_count_reduction(20, 15) == pytest.approx(0.25)
    assert true_count_reduction(0, 0) is None
    result = evaluate_v3_scientific_gate(_metrics())
    assert result.passed
    assert result.measurements[
        "age3_mismatch_sequence_true_relative_reduction"
    ] == pytest.approx(0.25)


def test_progressive_gates_are_distinct_from_strict_10k_gate() -> None:
    five_hundred = _metrics()
    five_hundred["age2_recurrence_improvement"] = 0.01
    five_hundred["age3_recurrence_improvement"] = 0.01
    five_hundred["age3_first_r_p95_ratio"] = 1.0
    gate_500 = evaluate_v3_stage_gate(five_hundred, optimizer_step=500)
    assert gate_500.passed
    assert gate_500.verdict == "STABILITY_V3_500_CONTINUE"
    two_k = _metrics()
    two_k["age2_recurrence_improvement"] = 0.01
    two_k["age3_recurrence_improvement"] = 0.01
    two_k["age3_first_r_p95_ratio"] = 1.0
    assert evaluate_v3_stage_gate(two_k, optimizer_step=2_000).passed
    five_k = _metrics()
    assert evaluate_v3_stage_gate(
        five_k, optimizer_step=5_000, previous_metrics=two_k
    ).passed
    assert evaluate_v3_stage_gate(five_k, optimizer_step=10_000).passed


def test_v2_selection_is_lexicographic_and_validation_only() -> None:
    common = {
        "split": "checkpoint_validation",
        "candidate_age3_mismatch_sequences": 10,
        "candidate_age3_first_r_p95": 0.2,
        "candidate_age3_recurrence_excess_mean": 0.1,
        "age1_first_r_ratio_to_parent": 1.0,
    }
    rows = [
        {**common, "optimizer_step": 2_000, "checkpoint": "2k.pt"},
        {
            **common,
            "optimizer_step": 4_000,
            "checkpoint": "4k.pt",
            "candidate_age3_mismatch_sequences": 9,
            "candidate_age3_first_r_p95": 9.0,
        },
        {
            **common,
            "split": "final_offline",
            "optimizer_step": 10_000,
            "checkpoint": "10k.pt",
            "candidate_age3_mismatch_sequences": 0,
        },
    ]
    selected = select_v2_checkpoint(rows)
    assert selected["selected_step"] == 4_000
    assert selected["selected_checkpoint"] == "4k.pt"


def test_gradient_calibration_uses_total_update_and_warns_on_pairwise_conflict() -> None:
    norms = {name: float(index + 1) for index, name in enumerate(V3_LOSS_NAMES)}
    positive_total = {
        "recurrence_gain": 0.75,
        "frozen_ng3_execution": 0.38,
        "exact_teacher_group": 0.75,
        "gripper_transition": 0.15,
    }
    approved = calibrate_v3_gradient_weights(
        norms,
        recurrence_cosine_with_ng3=-0.10,
        recurrence_cosine_with_reference_group=0.10,
        weighted_total_cosines_with_protected_losses=positive_total,
    )
    assert approved["approved"]
    assert approved["weighted_gradient_shares"] == pytest.approx(
        V3_GRADIENT_TARGETS
    )
    warned = calibrate_v3_gradient_weights(
        norms,
        recurrence_cosine_with_ng3=-0.31,
        recurrence_cosine_with_reference_group=0.10,
        weighted_total_cosines_with_protected_losses=positive_total,
    )
    assert warned["approved_for_bounded_pilot"]
    assert (
        warned["verdict"]
        == "STABILITY_V3_BOUNDED_PILOT_APPROVED_WITH_ALIGNMENT_WARNING"
    )
    assert "recurrence_vs_frozen_ng3" in warned["pairwise_conflict_warnings"]

    warned_total = calibrate_v3_gradient_weights(
        norms,
        recurrence_cosines_with_dominant_losses={
            "recurrence_vs_frozen_ng3": -0.10,
            "recurrence_vs_exact_teacher_group": 0.10,
            "recurrence_vs_gripper_transition": -0.31,
        },
        weighted_total_cosines_with_protected_losses={
            **positive_total,
            "frozen_ng3_execution": -0.01,
        },
    )
    assert (
        warned_total["verdict"]
        == "STABILITY_V3_BOUNDED_PILOT_APPROVED_WITH_ALIGNMENT_WARNING"
    )
    assert warned_total["approved_for_bounded_pilot"]
    assert warned_total["weighted_total_alignment_warnings"] == {
        "frozen_ng3_execution": -0.01
    }

    incomplete = calibrate_v3_gradient_weights(
        norms,
        recurrence_cosine_with_ng3=-0.10,
        recurrence_cosine_with_reference_group=0.10,
    )
    assert incomplete["verdict"] == "STABILITY_V3_WEIGHTS_BLOCKED"
    assert not incomplete["approved_for_bounded_pilot"]


def test_moving_window_contract_checks_gradient_and_clipping_shares() -> None:
    rows = [dict(V3_GRADIENT_TARGETS) for _ in range(16)]
    result = evaluate_v3_moving_window(rows, clipping_flags=[False] * 200)
    assert result["passed"]
    assert result["target_balance_passed"]
    assert result["hard_safety_passed"]
    imbalanced = [
        {
            **V3_GRADIENT_TARGETS,
            "recurrence_gain": 0.17,
            "frozen_ng3_execution": 0.35,
        }
        for _ in range(16)
    ]
    warning = evaluate_v3_moving_window(
        imbalanced, clipping_flags=[False] * 200
    )
    assert not warning["passed"]
    assert not warning["target_balance_passed"]
    assert warning["hard_safety_passed"]
    assert warning["continuation_requires_offline_multimetric_gate"]
    clipped = evaluate_v3_moving_window(rows, clipping_flags=[True] * 50 + [False] * 150)
    assert not clipped["passed"]
    assert not clipped["hard_safety_passed"]


def test_combined_stage_decision_uses_offline_gate_and_numerical_safety(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(v3_pipeline, "RESULT_ROOT", tmp_path)
    training_path = tmp_path / "training.json"
    offline_path = tmp_path / "offline.json"
    training_path.write_text(
        json.dumps(
            {
                "verdict": (
                    "STABILITY_V3_TRAINING_SEGMENT_COMPLETE_WITH_AUDIT_WARNING"
                ),
                "optimizer_step": 500,
                "moving_window_audit": {
                    "passed": False,
                    "target_balance_passed": False,
                    "hard_safety_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    offline_path.write_text(
        json.dumps({"verdict": "STABILITY_V3_500_CONTINUE", "passed": True}),
        encoding="utf-8",
    )
    train_job = v3_pipeline.Job(
        name="train",
        command=(),
        success_file=training_path,
        validator=lambda _: True,
        output_root=tmp_path / "train",
    )
    offline_job = v3_pipeline.Job(
        name="offline",
        command=(),
        success_file=offline_path,
        validator=lambda _: True,
        output_root=tmp_path / "offline",
    )
    assert v3_pipeline._stage_survivors(
        step=500,
        train_jobs={"R50": train_job},
        offline_jobs={"R50": offline_job},
    ) == ["R50"]
    decision = json.loads(
        (tmp_path / "stage_decisions/step_000500.json").read_text()
    )
    assert decision["branches"]["R50"]["gradient_share_target_warning"]

    unsafe = json.loads(training_path.read_text())
    unsafe["moving_window_audit"]["hard_safety_passed"] = False
    training_path.write_text(json.dumps(unsafe), encoding="utf-8")
    assert not v3_pipeline._stage_survivors(
        step=2_000,
        train_jobs={"R50": train_job},
        offline_jobs={"R50": offline_job},
    )


def test_continuation_ignores_soft_gate_but_not_hard_safety(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(v3_pipeline, "RESULT_ROOT", tmp_path)
    training = tmp_path / "training/r50/run_summary_step_000500.json"
    gate = tmp_path / "offline/r50/500_validation/offline_gate.json"
    training.parent.mkdir(parents=True)
    gate.parent.mkdir(parents=True)
    training.write_text(
        json.dumps(
            {
                "verdict": "STABILITY_V3_TRAINING_SEGMENT_COMPLETE_WITH_AUDIT_WARNING",
                "optimizer_step": 500,
                "checkpoint_sha256": "checkpoint",
                "source_combined_sha256": "source",
                "moving_window_audit": {
                    "hard_safety_passed": True,
                    "target_balance_passed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    gate.write_text(
        json.dumps(
            {
                "verdict": "STABILITY_V3_500_STOP",
                "passed": False,
                "candidate_sha256": "checkpoint",
                "source_combined_sha256": "source",
                "gate": {"measurements": {"original_simvla_frozen": True}},
            }
        ),
        encoding="utf-8",
    )
    decision = v3_continuation._hard_safety(500, gate)
    assert decision["passed"]
    assert not decision["soft_stage_gate_passed"]

    payload = json.loads(training.read_text())
    payload["moving_window_audit"]["hard_safety_passed"] = False
    training.write_text(json.dumps(payload), encoding="utf-8")
    assert not v3_continuation._hard_safety(500, gate)["passed"]


def test_v3_trajectory_bundle_preserves_diagnostic_classification(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate = tmp_path / "offline_gate.json"
    gate.write_text(
        json.dumps(
            {
                "branch": "R50",
                "optimizer_step": 500,
                "candidate_sha256": sha256_file(checkpoint),
                "verdict": "STABILITY_V3_500_STOP",
                "passed": False,
                "gate": {"checks": {"soft_metric": False}},
            }
        ),
        encoding="utf-8",
    )
    norm = tmp_path / "norm.json"
    norm.write_text("{}", encoding="utf-8")
    training = tmp_path / "training"
    training.mkdir()
    (training / "source_lock.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "bundle"
    ready = build_v3_diagnostic_bundle(
        checkpoint=checkpoint,
        offline_gate=gate,
        norm_stats=norm,
        training_root=training,
        output=output,
        branch="R50",
        optimizer_step=500,
    )
    assert ready["classification"] == "DIAGNOSTIC_ONLY"
    assert ready["offline_gate_passed"] is False
    assert ready["measured_stage_gate_passed"] is False
    assert ready["online_must_not_select_checkpoint"] is True


def test_full_r50_job_graph_is_cli_parseable_and_resume_connected(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(v3_pipeline, "RESULT_ROOT", tmp_path)
    parser = build_v3_trainer_parser()
    train_500 = v3_pipeline._train_job("R50", 500, None, 29760)
    checkpoint_500 = (
        tmp_path / "training/r50/checkpoints/stability_v3_step_000500.pt"
    )
    train_2k = v3_pipeline._train_job("R50", 2_000, checkpoint_500, 29780)
    checkpoint_2k = (
        tmp_path / "training/r50/checkpoints/stability_v3_step_002000.pt"
    )
    train_5k = v3_pipeline._train_job("R50", 5_000, checkpoint_2k, 29800)
    checkpoint_5k = (
        tmp_path / "training/r50/checkpoints/stability_v3_step_005000.pt"
    )
    train_10k = v3_pipeline._train_job("R50", 10_000, checkpoint_5k, 29820)
    offline_jobs = (
        v3_pipeline._offline_job(
            "R50", 500, "checkpoint_validation", None, 29770
        ),
        v3_pipeline._offline_job(
            "R50",
            2_000,
            "checkpoint_validation",
            tmp_path / "offline/r50/500_validation/offline_gate.json",
            29790,
        ),
        v3_pipeline._offline_job(
            "R50",
            5_000,
            "checkpoint_validation",
            tmp_path / "offline/r50/2k_validation/offline_gate.json",
            29810,
        ),
        v3_pipeline._offline_job(
            "R50", 10_000, "checkpoint_validation", None, 29830
        ),
        v3_pipeline._offline_job("R50", 10_000, "final_offline", None, 29840),
    )
    jobs = (train_500, train_2k, train_5k, train_10k, *offline_jobs)
    for job in jobs:
        module_index = job.command.index(v3_pipeline.TRAINER_MODULE)
        parser.parse_args(job.command[module_index + 1 :])
    assert train_500.validator(
        {
            "verdict": (
                "STABILITY_V3_TRAINING_SEGMENT_COMPLETE_WITH_AUDIT_WARNING"
            ),
            "optimizer_step": 500,
        }
    )
    assert train_2k.preserve_existing
    assert str(checkpoint_500) in train_2k.command
    assert str(checkpoint_2k) in train_5k.command
    assert str(checkpoint_5k) in train_10k.command


def _pool_contract() -> dict[str, object]:
    return {
        "schema_version": V3_HARD_POOL_SCHEMA,
        "pool_indices": {
            "base": list(range(100)),
            "gripper_transition": list(range(100, 120)),
            "recurrence_action_tail": list(range(120, 140)),
        },
    }


def test_pool_sampler_is_deterministic_and_exact_70_15_15_per_20() -> None:
    sampler = V3PoolSampler(_pool_contract(), seed=7, start_step=0, stop_step=40)
    names = [sampler.stream_name(step) for step in range(20)]
    assert names.count("base") == 14
    assert names.count("gripper_transition") == 3
    assert names.count("recurrence_action_tail") == 3
    assert list(sampler) == list(
        V3PoolSampler(_pool_contract(), seed=7, start_step=0, stop_step=40)
    )


def test_hard_pool_reports_cross_query_and_parent_mismatch_counts() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "dataset_index": index,
                "tail_score": float(index),
                "has_any_gripper_event": index % 2 == 0,
                "has_cross_query_gripper_event": index % 4 == 0,
                "within_query_event_positions": index % 3,
                "cross_query_event_positions": index % 2,
                "parent_age3_sign_mismatch_sequence": index in {1, 3, 5},
                "parent_age3_sign_mismatch_positions": int(index in {1, 3, 5}),
            }
        )
    contract = build_v3_hard_pool_contract(
        rows, dataset_contract={"split": "train"}, source_lock={"hash": "x"}
    )
    assert contract["pool_counts"]["recurrence_action_tail"] == 2
    assert contract["event_coverage"]["cross_query_event_sequences"] == 5
    assert contract["event_coverage"]["parent_age3_mismatch_sequences"] == 3
    pools = contract["pool_indices"]
    assert not (set(pools["base"]) & set(pools["gripper_transition"]))
    assert not (set(pools["base"]) & set(pools["recurrence_action_tail"]))
    assert not (
        set(pools["gripper_transition"]) & set(pools["recurrence_action_tail"])
    )
    assert contract["explicit_pools_pairwise_disjoint"]


def test_cross_query_gripper_boundary_is_indexed_and_supervised() -> None:
    anchor = torch.ones(10, 7)
    targets = torch.ones(3, 10, 7)
    targets[0, 0, 6] = -1.0
    event = gripper_event_contract(anchor, targets)
    assert event["has_cross_query_event"]
    batched_targets = tuple(targets[index].unsqueeze(0) for index in range(3))
    exact_loss, exact_diagnostics = gripper_transition_loss_with_boundaries(
        batched_targets,
        batched_targets,
        anchor.unsqueeze(0),
    )
    assert math.isfinite(float(exact_loss.item()))
    assert int(exact_diagnostics["mismatch_sequences"].sum().item()) == 0


def test_masked_nrms_is_per_sequence() -> None:
    target = torch.ones(2, 3, 4)
    prediction = target.clone()
    prediction[1] += 1.0
    mask = torch.ones(2, 3, dtype=torch.bool)
    values = masked_nrms_per_sequence(prediction, target, mask)
    assert values.shape == (2,)
    assert values[0].item() < 1e-5
    assert values[1].item() == pytest.approx(1.0)
