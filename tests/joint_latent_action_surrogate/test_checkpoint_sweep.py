from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from methods.joint_latent_action_surrogate.checkpoint_sweep import (
    EXACT_ENSEMBLE_METRIC_AVAILABLE,
    EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
    build_fidelity_eligibility,
    rank_checkpoint_rows,
    select_eligible_checkpoint,
    utility_verdict,
)


def row(
    checkpoint_id: str,
    *,
    eligible: bool,
    exact: float,
    arm_p95: float,
    horizon: float,
    microbatches: int,
    ensemble: float | None = None,
) -> dict[str, object]:
    return {
        "checkpoint_id": checkpoint_id,
        "fidelity_eligible": eligible,
        "exact_ensemble_status": (
            EXACT_ENSEMBLE_METRIC_AVAILABLE
            if ensemble is not None
            else EXACT_ENSEMBLE_METRIC_UNAVAILABLE
        ),
        "exact_ensemble_executed_action_l1_mean": ensemble,
        "exact_first_token_l1_mean_age_average": exact,
        "exact_arm_first_token_l1_p95_age_average": arm_p95,
        "exact_full_horizon_l1_mean_age_average": horizon,
        "global_microbatches": microbatches,
    }


def test_eligible_checkpoint_ranks_before_lower_error_failure() -> None:
    ranked, selected = select_eligible_checkpoint(
        [
            row(
                "failed", eligible=False, exact=0.01, arm_p95=0.01,
                horizon=0.01, microbatches=10,
            ),
            row(
                "passed", eligible=True, exact=0.02, arm_p95=0.02,
                horizon=0.02, microbatches=20,
            ),
        ]
    )
    assert [item["checkpoint_id"] for item in ranked] == ["passed", "failed"]
    assert selected is not None
    assert selected["checkpoint_id"] == "passed"


def test_canonical_metrics_then_budget_define_tie_breaking() -> None:
    ranked = rank_checkpoint_rows(
        [
            row(
                "later", eligible=True, exact=0.01, arm_p95=0.02,
                horizon=0.03, microbatches=20,
            ),
            row(
                "exact", eligible=True, exact=0.005, arm_p95=0.03,
                horizon=0.04, microbatches=30,
            ),
            row(
                "earlier", eligible=True, exact=0.01, arm_p95=0.02,
                horizon=0.03, microbatches=10,
            ),
        ]
    )
    assert [item["checkpoint_id"] for item in ranked] == [
        "exact",
        "earlier",
        "later",
    ]


def test_exact_ensemble_is_primary_when_available() -> None:
    ranked = rank_checkpoint_rows(
        [
            row(
                "raw_best", eligible=True, exact=0.01, arm_p95=0.01,
                horizon=0.01, microbatches=10, ensemble=0.03,
            ),
            row(
                "ensemble_best", eligible=True, exact=0.02, arm_p95=0.02,
                horizon=0.02, microbatches=20, ensemble=0.01,
            ),
        ]
    )
    assert ranked[0]["checkpoint_id"] == "ensemble_best"


def test_no_gate_pass_returns_no_selection() -> None:
    _, selected = select_eligible_checkpoint(
        [
            row(
                "failed", eligible=False, exact=0.01, arm_p95=0.01,
                horizon=0.01, microbatches=10,
            )
        ]
    )
    assert selected is None


def test_duplicate_checkpoint_is_rejected() -> None:
    duplicate = row(
        "same", eligible=False, exact=0.01, arm_p95=0.01,
        horizon=0.01, microbatches=10,
    )
    with pytest.raises(ValueError, match="Duplicate checkpoint_id"):
        rank_checkpoint_rows([duplicate, duplicate])


def test_age_two_failure_rejects_fidelity() -> None:
    gate = build_fidelity_eligibility(
        exact_first_token_mean_by_age={1: 0.01, 2: 0.03},
        hold_first_token_mean_by_age={1: 0.02, 2: 0.02},
        exact_arm_first_token_p95_by_age={1: 0.01, 2: 0.01},
        hold_arm_first_token_p95_by_age={1: 0.01, 2: 0.01},
        gripper_finite_by_age={1: True, 2: True},
        gripper_noncollapsed_by_age={1: True, 2: True},
        identity_checks={"source_hash": True, "tensor_shape": True},
        ensemble_status=EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
    )
    assert gate["eligible"] is False
    assert gate["checks"]["age2_exact_first_token_better_than_hold"] is False


def test_ensemble_unavailable_does_not_fabricate_gate() -> None:
    gate = build_fidelity_eligibility(
        exact_first_token_mean_by_age={1: 0.01, 2: 0.01},
        hold_first_token_mean_by_age={1: 0.02, 2: 0.02},
        exact_arm_first_token_p95_by_age={1: 0.01, 2: 0.01},
        hold_arm_first_token_p95_by_age={1: 0.01, 2: 0.01},
        gripper_finite_by_age={1: True, 2: True},
        gripper_noncollapsed_by_age={1: True, 2: True},
        identity_checks={"source_hash": True, "tensor_shape": True},
        ensemble_status=EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
    )
    assert gate["eligible"] is True
    assert gate["ensemble_check_applicable"] is False
    assert "exact_ensemble_better_than_hold" not in gate["checks"]


def test_utility_verdicts_are_frozen() -> None:
    selected = {"checkpoint_id": "candidate"}
    assert utility_verdict(
        None,
        surrogate_incremental_latency_ms=None,
        exact_action_head_latency_ms=None,
        additional_observation_encoder_calls=0,
        projected_hybrid_latency_delta_ms=None,
    ) == "STOP_SEER_SURROGATE"
    assert utility_verdict(
        selected,
        surrogate_incremental_latency_ms=0.4,
        exact_action_head_latency_ms=0.5,
        additional_observation_encoder_calls=0,
        projected_hybrid_latency_delta_ms=-0.1,
    ) == "SEER_DEPLOYMENT_CANDIDATE"
    assert utility_verdict(
        selected,
        surrogate_incremental_latency_ms=0.6,
        exact_action_head_latency_ms=0.5,
        additional_observation_encoder_calls=0,
        projected_hybrid_latency_delta_ms=0.1,
    ) == "TRANSFER_ONLY_CANDIDATE"


def test_frozen_v2_contract_parses_and_matches_implementation() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    contract_path = (
        repo_root
        / "codex_outputs/joint_latent_action_surrogate_20260816"
        / "stage_a_one_shot_decision_v2/stage_a_checkpoint_sweep_contract_v2.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["protocol"] == "joint_stage_a_one_shot_checkpoint_sweep_v2"
    assert len(contract["candidates"]) == 5
    paths = {
        "evaluator_sha256": repo_root
        / "tools/seer/evaluate_joint_stage_a_checkpoint_sweep.py",
        "ranking_module_sha256": repo_root
        / "methods/joint_latent_action_surrogate/checkpoint_sweep.py",
    }
    for key, path in paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == contract["implementation"][key]
