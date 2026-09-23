from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from architectures.seer.adapters.latentloop_horizon_regeneration import (
    save_hierarchical_trace,
)
from architectures.seer.upstream.utils.lrnode_mechanism_utils import (
    temporal_ensemble_probability,
)
from methods.latentloop_horizon_regeneration import (
    ExecutionLevel,
    HierarchicalSchedule,
    HorizonProvenance,
    LevelCallCounts,
    assert_level_call_contract,
    evaluate_decision,
)


def test_exact_hybrid_schedule() -> None:
    schedule = HierarchicalSchedule("hybrid", 8, 3)
    assert schedule.cycle() == [2, 0, 0, 1, 0, 0, 1, 0, 2]


def test_explicit_endpoint_schedules_match_legacy_routing() -> None:
    latent = HierarchicalSchedule("pure_latentloop", 8, 3)
    action = HierarchicalSchedule("pure_action_correction", 8, 3)
    assert [int(latent.level(t, has_latent_cache=t > 0)) for t in range(9)] == [
        2, 1, 1, 1, 1, 1, 1, 1, 2
    ]
    assert [int(action.level(t, has_latent_cache=t > 0)) for t in range(9)] == [
        2, 0, 0, 0, 0, 0, 0, 0, 2
    ]


def test_level_call_contracts_and_forbidden_calls() -> None:
    assert_level_call_contract(2, LevelCallCounts(1, 1, 0, 0), mode="hybrid")
    assert_level_call_contract(1, LevelCallCounts(0, 1, 1, 0), mode="hybrid")
    assert_level_call_contract(0, LevelCallCounts(0, 0, 1, 1), mode="hybrid")
    assert_level_call_contract(
        0, LevelCallCounts(0, 0, 0, 1), mode="pure_action_correction"
    )
    with pytest.raises(RuntimeError):
        assert_level_call_contract(0, LevelCallCounts(0, 1, 1, 1), mode="hybrid")
    with pytest.raises(RuntimeError):
        assert_level_call_contract(1, LevelCallCounts(1, 1, 1, 0), mode="hybrid")
    with pytest.raises(RuntimeError):
        assert_level_call_contract(2, LevelCallCounts(1, 2, 0, 0), mode="hybrid")


def test_provenance_cache_replacement_and_synthetic_prevention() -> None:
    provenance = HorizonProvenance(3)
    assert np.all(provenance.reset_full()[:, 0] == 1.0)
    first = provenance.correct()
    assert np.array_equal(first[:, 1:], np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0]]))
    second = provenance.correct()
    assert np.array_equal(second[:, 2], np.asarray([0.0, 1.0, 1.0]))
    previous_generation = provenance.generation
    regenerated = provenance.reset_regenerated()
    assert provenance.generation == previous_generation + 1
    assert provenance.fully_synthetic_horizons_prevented == 1
    assert np.all(regenerated[:, 3] == 1.0)
    assert np.all(regenerated[:, :3] == 0.0)


def test_canonical_temporal_ensemble_contract() -> None:
    buffer = torch.zeros(5, 8, 7, dtype=torch.float32)
    first = torch.full((1, 3, 7), 0.2)
    second = torch.full((1, 3, 7), 0.8)
    temporal_ensemble_probability(first, 0, buffer, 0.01)
    actual, count = temporal_ensemble_probability(second, 1, buffer, 0.01)
    candidates = torch.stack((first[0, 1], second[0, 0]))
    weights = np.exp(-0.01 * np.arange(2))
    weights /= weights.sum()
    expected = (candidates * torch.from_numpy(weights).unsqueeze(1)).sum(
        dim=0, keepdim=True
    )
    assert count == 2
    assert torch.equal(actual, expected)


def test_predeclared_decision_rules() -> None:
    supported = {
        "k1_and_endpoint_parity_pass": True,
        "hybrid_vs_action_delta_pp": 50.0,
        "hybrid_vs_action_ci_lower_pp": 20.0,
        "hybrid_vs_latentloop_ci_lower_pp": -2.9,
        "action_head_reduction_fraction": 0.50,
        "gripper_reversal_increase_fraction": 0.20,
        "tasks_regressing_over_20pp": 1,
    }
    assert evaluate_decision(supported) == "HORIZON_REGENERATION_INTERVENTION_SUPPORTED"
    recovered = dict(supported, hybrid_vs_latentloop_ci_lower_pp=-3.1)
    assert evaluate_decision(recovered) == "REGENERATION_RECOVERS_ACTION_CORRECTION_ONLY"
    failed = dict(supported, hybrid_vs_action_delta_pp=0.0, hybrid_vs_action_ci_lower_pp=-1.0)
    assert evaluate_decision(failed) == "HORIZON_REGENERATION_NOT_SUPPORTED"


def test_trace_serialization(tmp_path: Path) -> None:
    output = save_hierarchical_trace(
        tmp_path,
        "episode",
        [{"step": 0, "level": int(ExecutionLevel.FULL_SEER)}],
        {"mode": "hybrid"},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["metadata"]["mode"] == "hybrid"
    assert payload["steps"] == [{"step": 0, "level": 2}]


def test_trace_serialization_does_not_advance_rng(tmp_path: Path) -> None:
    torch.manual_seed(91)
    np.random.seed(91)
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    save_hierarchical_trace(
        tmp_path,
        "rng",
        [{"step": 1, "level": int(ExecutionLevel.ACTION_CORRECTION)}],
        {"mode": "hybrid"},
    )
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
