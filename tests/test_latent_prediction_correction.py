"""Lightweight contract tests for every-step latent prediction-correction."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "architectures" / "seer" / "upstream"))

from architectures.seer.adapters.latent_prediction_correction import (  # noqa: E402
    SeerLatentFilterAdapter,
)
from methods.latent_prediction_correction.decision import (  # noqa: E402
    apply_decision_rule,
)
from methods.latent_prediction_correction.fusion import (  # noqa: E402
    ema_latent,
    fixed_latent_fusion,
    select_latent,
)
from methods.latent_prediction_correction.metrics import (  # noqa: E402
    checkpoint_hierarchical_paired_bootstrap_ci,
    continuous_action_metrics,
    gripper_continuity_metrics,
    hierarchical_paired_bootstrap_ci,
    paired_outcome_counts,
    wilson_interval,
)
from utils.lrnode_mechanism_utils import (  # noqa: E402
    load_trace_shard,
    save_trace_episode,
    temporal_ensemble_probability,
)


def test_fixed_filter_alpha_zero_is_exact_prior() -> None:
    prior = torch.randn(2, 3, 4)
    full = torch.randn_like(prior)
    assert fixed_latent_fusion(prior, full, 0.0) is prior


def test_fixed_filter_alpha_one_is_exact_full() -> None:
    prior = torch.randn(2, 3, 4)
    full = torch.randn_like(prior)
    assert fixed_latent_fusion(prior, full, 1.0) is full


def test_fixed_filter_interpolation_preserves_shape_and_dtype() -> None:
    prior = torch.zeros(2, 3, 4, dtype=torch.float64)
    full = torch.ones_like(prior)
    fused = fixed_latent_fusion(prior, full, 0.25)
    assert fused.shape == prior.shape
    assert fused.dtype == prior.dtype
    assert torch.equal(fused, torch.full_like(prior, 0.25))


def test_ema_beta_endpoints_are_exact() -> None:
    previous = torch.randn(1, 3, 4)
    full = torch.randn_like(previous)
    assert ema_latent(previous, full, 0.0) is previous
    assert ema_latent(previous, full, 1.0) is full


def test_first_step_initializes_all_modes_from_full() -> None:
    full = torch.randn(1, 3, 4)
    for mode in (
        "raw_full",
        "recurrent_prior",
        "fixed_filter",
        "full_latent_ema",
    ):
        result = select_latent(mode, z_full=full)
        assert result.latent is full
        assert result.initialized_from_full
        assert result.reused_full_action


def test_raw_full_and_recurrent_prior_identities() -> None:
    previous = torch.randn(1, 3, 4)
    full = torch.randn_like(previous)
    prior = torch.randn_like(previous)
    assert select_latent(
        "raw_full",
        z_full=full,
        z_previous=previous,
    ).latent is full
    assert select_latent(
        "recurrent_prior",
        z_full=full,
        z_previous=previous,
        z_prior=prior,
    ).latent is prior


def test_adapter_skips_updater_for_exact_full_endpoints() -> None:
    raw = SeerLatentFilterAdapter("raw_full")
    alpha_one = SeerLatentFilterAdapter("fixed_filter", alpha=1.0)
    assert not raw.requires_recurrent_prior(has_previous=True)
    assert not alpha_one.requires_recurrent_prior(has_previous=True)


def test_fixed_filter_alpha_one_selects_full_without_prior() -> None:
    adapter = SeerLatentFilterAdapter("fixed_filter", alpha=1.0)
    previous = torch.randn(1, 3, 4)
    full = torch.randn_like(previous)
    selected = adapter.select(full, previous, z_prior=None)
    assert selected.latent is full
    assert not selected.used_recurrent_prior
    assert selected.reused_full_action
    assert not selected.initialized_from_full


def test_diagnostics_do_not_change_selected_output_or_rng() -> None:
    adapter = SeerLatentFilterAdapter("fixed_filter", alpha=0.5)
    previous = torch.randn(1, 3, 4)
    prior = torch.randn_like(previous)
    full = torch.randn_like(previous)
    selected = adapter.select(full, previous, prior)
    expected = selected.latent.clone()
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    expected_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    metrics = adapter.diagnostics(selected, full, prior)
    actual_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    assert torch.equal(selected.latent, expected)
    assert "latent_prior_vs_full_l2" in metrics
    assert actual_rng[0] == expected_rng[0]
    assert np.array_equal(actual_rng[1][1], expected_rng[1][1])
    assert torch.equal(actual_rng[2], expected_rng[2])


def test_separate_temporal_ensemble_buffers() -> None:
    left = torch.zeros(8, 11, 7)
    right = torch.zeros_like(left)
    left_action = torch.full((1, 3, 7), 0.2)
    right_action = torch.full((1, 3, 7), 0.8)
    left_result, _ = temporal_ensemble_probability(
        left_action,
        0,
        left,
        0.01,
    )
    right_result, _ = temporal_ensemble_probability(
        right_action,
        0,
        right,
        0.01,
    )
    assert torch.equal(left_result, left_action[:, 0].to(dtype=torch.float64))
    assert torch.equal(right_result, right_action[:, 0].to(dtype=torch.float64))
    assert not torch.equal(left, right)


def test_continuous_metrics_exclude_gripper() -> None:
    actions = np.zeros((6, 7), dtype=np.float64)
    actions[:, 6] = [-1, 1, -1, 1, -1, 1]
    continuous = continuous_action_metrics(actions)
    gripper = gripper_continuity_metrics(actions)
    assert continuous["translation_second_difference_mean"] == 0.0
    assert continuous["rotation_second_difference_p95"] == 0.0
    assert gripper["gripper_switches_per_100_steps"] > 0.0
    assert gripper["gripper_reverse_within_1_per_100_steps"] > 0.0


def test_pairing_wilson_and_hierarchical_bootstrap() -> None:
    baseline = {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1}
    candidate = {(0, 0): 1, (0, 1): 1, (1, 0): 0, (1, 1): 0}
    counts = paired_outcome_counts(baseline, candidate)
    assert counts["fail_to_success"] == 1
    assert counts["success_to_fail"] == 1
    low, high = wilson_interval(3, 4)
    assert 0.0 <= low < 0.75 < high <= 1.0
    rows = [
        {
            "task_id": key[0],
            "baseline_success": baseline[key],
            "candidate_success": candidate[key],
        }
        for key in baseline
    ]
    ci_low, ci_high = hierarchical_paired_bootstrap_ci(
        rows,
        samples=100,
        seed=3,
    )
    assert ci_low <= 0.0 <= ci_high

    checkpoint_rows = [
        {
            **row,
            "checkpoint_id": checkpoint,
        }
        for checkpoint in (36, 38)
        for row in rows
    ]
    ci_low, ci_high = checkpoint_hierarchical_paired_bootstrap_ci(
        checkpoint_rows,
        samples=100,
        seed=3,
    )
    assert ci_low <= 0.0 <= ci_high


def _decision_rows(
    fixed_rates: tuple[float, float],
    ema_rates: tuple[float, float],
) -> list[dict]:
    rows = []
    for index, checkpoint in enumerate((36, 38)):
        rows.extend(
            [
                {
                    "checkpoint_id": checkpoint,
                    "mode": "raw_full",
                    "success_rate": 0.87,
                },
                {
                    "checkpoint_id": checkpoint,
                    "mode": "fixed_filter",
                    "success_rate": fixed_rates[index],
                    "paired_net_flip_vs_raw": 2,
                    "translation_second_difference_p95": 0.7,
                    "rotation_second_difference_p95": 0.7,
                    "gripper_reverse_within_5_per_100_steps": 0.5,
                },
                {
                    "checkpoint_id": checkpoint,
                    "mode": "full_latent_ema",
                    "success_rate": ema_rates[index],
                    "translation_second_difference_p95": 1.0,
                    "rotation_second_difference_p95": 1.0,
                    "gripper_reverse_within_5_per_100_steps": 0.5,
                },
            ]
        )
    return rows


def _decision_rule() -> dict:
    return {
        "heldout_checkpoints": [36, 38],
        "fixed_vs_raw": {
            "minimum_mean_sr_gain_pp": 2.0,
            "maximum_checkpoint_drop_pp": 1.0,
        },
        "ema_exclusion": {
            "minimum_sr_advantage_pp": 1.0,
            "sr_equivalence_margin_pp": 1.0,
            "continuity_reduction_fraction": 0.20,
        },
    }


def test_decision_rule_confirmed_thresholds() -> None:
    decision = apply_decision_rule(
        _decision_rows((0.90, 0.89), (0.87, 0.87)),
        {"alpha_1_exact": True},
        _decision_rule(),
    )
    assert decision["verdict"] == "LATENT_FILTER_CONFIRMED"


def test_decision_rule_endpoint_failure_is_not_confirmed() -> None:
    decision = apply_decision_rule(
        _decision_rows((0.90, 0.89), (0.87, 0.87)),
        {"alpha_1_exact": False},
        _decision_rule(),
    )
    assert decision["verdict"] == "LATENT_FILTER_NOT_CONFIRMED"


def test_decision_rule_enforces_all_declared_endpoint_checks() -> None:
    rule = _decision_rule()
    rule["endpoint_requirements"] = {
        "alpha_1_exact": True,
        "diagnostics_invariant": True,
    }
    decision = apply_decision_rule(
        _decision_rows((0.90, 0.89), (0.87, 0.87)),
        {"alpha_1_exact": True, "diagnostics_invariant": False},
        rule,
    )
    assert decision["verdict"] == "LATENT_FILTER_NOT_CONFIRMED"


def test_trace_save_reload_round_trip(tmp_path: Path) -> None:
    paths = save_trace_episode(
        tmp_path,
        "filter_episode",
        [{"timestep": 0}, {"timestep": 1}],
        [
            {"z_filter": torch.ones(1, 3, 4)},
            {"z_filter": torch.zeros(1, 3, 4)},
        ],
        {"success": 1, "every_step_filter_mode": "fixed_filter"},
    )
    metadata, rows, tensors = load_trace_shard(Path(paths["json"]))
    assert metadata["num_steps"] == 2
    assert rows[1]["timestep"] == "1"
    assert tensors["z_filter"].shape == (2, 1, 3, 4)
