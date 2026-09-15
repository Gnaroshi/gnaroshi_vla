from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import (
    VARIANTS, action_metrics, completed_unit, intervention, observation_codes, tensor_hash,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism_environment import InterventionPolicy, assert_same_prefix
from methods.latentloop.modules.native_simvla_v0 import TokenSharedConditionUpdater


@pytest.fixture
def state():
    torch.manual_seed(42)
    updater = TokenSharedConditionUpdater(condition_dim=8, delta_dim=3, max_tokens=5)
    with torch.no_grad():
        updater.up.weight.normal_(0, .1)
        updater.up.bias.fill_(.2)
        updater.gate_head.weight.normal_(0, .1)
    previous = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, True, True, False, False]])
    groups = torch.zeros(1, 5, dtype=torch.long)
    code = torch.randn(1, 3)
    codes = {k: code for k in ("full", "stale_images", "stale_encoder_proprio", "repeated_observation")}
    codes["zero"] = torch.zeros_like(code)
    return SimpleNamespace(condition_updater=updater), previous, codes, mask, groups


def test_full_and_zero_match_actual_updater(state):
    adapter, previous, codes, mask, groups = state
    original = previous.clone()
    for variant, key in (("full_update", "full"), ("zero_feature", "zero")):
        expected = adapter.condition_updater(previous, codes[key], valid_mask=mask, group_ids=groups, age=1)
        actual, gate, residual = intervention(adapter, previous, codes, mask, groups, 1, variant)
        assert torch.equal(actual, expected.condition)
        assert torch.equal(gate, expected.gate)
        assert torch.equal(residual, expected.residual)
    assert torch.equal(previous, original)
    zero, _, _ = intervention(adapter, previous, codes, mask, groups, 1, "zero_feature")
    assert not torch.equal(zero, previous)


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_interventions_preserve_padding(state, variant):
    adapter, previous, codes, mask, groups = state
    means = torch.ones(3, 5, 8)
    actual, gate, residual = intervention(adapter, previous, codes, mask, groups, 2, variant, means)
    assert torch.equal(actual[~mask], previous[~mask])
    assert torch.count_nonzero(gate[~mask]) == 0
    assert torch.count_nonzero(residual[~mask]) == 0


def test_crossed_gate_and_residual_use_same_previous(state):
    adapter, previous, codes, mask, groups = state
    full = adapter.condition_updater(previous, codes["full"], valid_mask=mask, group_ids=groups, age=1)
    zero = adapter.condition_updater(previous, codes["zero"], valid_mask=mask, group_ids=groups, age=1)
    a, _, _ = intervention(adapter, previous, codes, mask, groups, 1, "full_gate_zero_residual")
    b, _, _ = intervention(adapter, previous, codes, mask, groups, 1, "zero_gate_full_residual")
    assert torch.equal(a, previous + full.gate * zero.residual)
    assert torch.equal(b, previous + zero.gate * full.residual)


def test_information_paths_are_separate():
    seen = []
    def encode(pair):
        seen.append(pair)
        return torch.ones(1, 3)
    sequence = {"image_sequence": torch.arange(4).view(1, 4, 1, 1, 1, 1),
                "proprio_sequence": torch.arange(4).view(1, 4, 1)}
    codes = observation_codes(SimpleNamespace(delta_encoder=encode), sequence, 2)
    assert seen[0].current_images.item() == 2
    assert seen[1].current_images.item() == 1 and seen[1].current_proprio.item() == 2
    assert seen[2].current_images.item() == 2 and seen[2].current_proprio.item() == 1
    assert seen[3].current_images.item() == 1 and seen[3].current_proprio.item() == 1
    assert torch.count_nonzero(codes["zero"]) == 0
    assert not torch.equal(codes["zero"], codes["repeated_observation"])


def test_action_metrics_separate_executed_prefix_and_gripper():
    a = torch.zeros(1, 10, 7)
    b = a.clone()
    b[:, 5:] = 2
    m = action_metrics(a, b)
    assert m["first5_action_l1"] == 0 and m["full_chunk_action_l1"] == 1
    b[:, :5, 6] = 1
    assert action_metrics(a, b)["gripper_sign_disagreement"] == 1


def test_resume_rejects_foreign_or_incomplete_unit(tmp_path):
    p = tmp_path / "unit.json"
    assert completed_unit(p, "a") is None
    p.write_text(json.dumps({"identity": "a", "complete": True, "rows": []}))
    assert completed_unit(p, "a")["complete"]
    with pytest.raises(RuntimeError):
        completed_unit(p, "b")
    p.write_text(json.dumps({"identity": "a", "complete": False}))
    with pytest.raises(RuntimeError):
        completed_unit(p, "a")


def test_hash_includes_shape_and_dtype():
    assert tensor_hash(torch.ones(2)) != tensor_hash(torch.ones(1, 2))
    assert tensor_hash(np.ones(2, np.float32)) != tensor_hash(np.ones(2, np.float64))


def test_pairing_rejects_changed_controller_state():
    reference = {"state": np.zeros(3), "ctrl": np.ones(2)}
    assert_same_prefix(reference, reference)
    with pytest.raises(RuntimeError):
        assert_same_prefix(reference, {"state": np.zeros(3), "ctrl": np.zeros(2)})


def test_launcher_preserves_shell_and_exposes_strict_status():
    root = Path(__file__).resolve().parents[2]
    wrapper = root / "architectures/simvla/wrappers/run_condition_mechanism_rb2.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    env = dict(os.environ, SIMVLA_PYTHON="/bin/false", SIMVLA_STRICT_EXIT="0")
    result = subprocess.run(["bash", str(wrapper)], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and "rc=1" in result.stdout
    env["SIMVLA_STRICT_EXIT"] = "1"
    assert subprocess.run(["bash", str(wrapper)], env=env, capture_output=True).returncode == 1


@pytest.mark.parametrize("variant,refresh_count", [("baseline", 4), ("hold", 2), ("zero_feature", 2), ("full_update", 2)])
def test_native_queue_and_query_schedule(state, variant, refresh_count):
    import collections
    adapter, previous, codes, mask, groups = state
    calls = []
    policy = object.__new__(InterventionPolicy)
    policy.variant = variant
    policy.refresh_every = 1 if variant == "baseline" else 2
    policy.query_index = 0
    policy.query_trace = []
    policy.action_queue = collections.deque()
    policy.metrics = SimpleNamespace(counters=collections.defaultdict(int))
    policy.cached_condition = previous
    policy.cached_raw_rgb = torch.zeros(1, 2, 1, 1, 3)
    policy.cached_proprio = torch.zeros(1, 8)
    policy.condition_layout = SimpleNamespace(valid_mask=mask, group_ids=groups)
    policy.native_v0 = SimpleNamespace(condition_updater=adapter.condition_updater, delta_dim=3, delta_encoder=lambda pair: codes["full"])
    chunk = torch.randn(1, 10, 7)
    def full(batch, policy_query_index):
        calls.append(policy_query_index)
        return previous, chunk, 123
    policy._full_refresh = full
    policy._decode = lambda c, q, policy_query_index: (chunk, 123)
    batch = {"raw_rgb": policy.cached_raw_rgb, "proprio": policy.cached_proprio}
    for _ in range(4):
        policy._refill_action_queue(batch)
        assert len(policy.action_queue) == 5
        assert all(torch.equal(action, chunk[0, i]) for i, (action, _) in enumerate(policy.action_queue))
    assert len(calls) == refresh_count
    assert policy.query_index == 4
    assert np.array_equal(policy.first_intervention_chunk, chunk[0].numpy())
