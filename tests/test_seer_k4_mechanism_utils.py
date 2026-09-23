import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "architectures" / "seer" / "upstream"))

from utils.lrnode_mechanism_utils import (  # noqa: E402
    action_second_differences,
    capture_rng_state,
    classify_transition,
    counterfactual_requires_skip_shadow,
    fuse_latents,
    load_trace_shard,
    matched_random_latent,
    mix_action_tokens,
    preserve_rng_state,
    rng_states_equal,
    save_trace_episode,
    temporal_ensemble_probability,
)


def test_diagnostics_off_temporal_ensemble_matches_legacy_formula():
    new_buffer = torch.zeros(8, 11, 7)
    legacy_buffer = torch.zeros_like(new_buffer)
    for timestep in range(4):
        sequence = torch.rand(1, 3, 7) + 0.1
        actual, count = temporal_ensemble_probability(
            sequence, timestep, new_buffer, 0.01
        )
        legacy_buffer[timestep:timestep + 1, timestep:timestep + 3] = sequence
        candidates = legacy_buffer[:, timestep]
        candidates = candidates[torch.all(candidates != 0, dim=1)]
        weights = np.exp(-0.01 * np.arange(len(candidates)))
        weights /= weights.sum()
        expected = (
            candidates
            * torch.from_numpy(weights).to(candidates.device).unsqueeze(1)
        ).sum(dim=0, keepdim=True)
        assert count == len(candidates)
        assert torch.equal(actual, expected)
        assert actual.dtype == torch.float64


def test_temporal_ensemble_preserves_legacy_all_axis_sentinel():
    buffer = torch.zeros(8, 11, 7)
    sequence = torch.full((1, 3, 7), 0.25)
    sequence[..., 2] = 0.0

    action, count = temporal_ensemble_probability(sequence, 0, buffer, 0.01)

    assert count == 0
    assert action.dtype == torch.float64
    assert torch.equal(action, torch.zeros(1, 7, dtype=torch.float64))


def test_diagnostic_action_decoder_matches_existing_decoder():
    from models.seer_model import SeerAgent

    model = SeerAgent.__new__(SeerAgent)
    torch.nn.Module.__init__(model)
    model.hidden_dim = 4
    model.action_pred_steps = 3
    model.action_decoder = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.ReLU(),
    )
    model.arm_action_decoder = torch.nn.Sequential(
        torch.nn.Linear(5, 6),
        torch.nn.Tanh(),
    )
    model.gripper_action_decoder = torch.nn.Sequential(
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid(),
    )
    latent = torch.randn(2, 3, 4)
    old_arm, old_gripper = model.decode_action_from_latent(latent)
    diagnostic = model.decode_action_diagnostics_from_latent(latent)
    assert torch.equal(diagnostic["arm"], old_arm)
    assert torch.equal(diagnostic["gripper_probability"], old_gripper)
    assert torch.equal(
        torch.sigmoid(diagnostic["gripper_logit"]),
        diagnostic["gripper_probability"],
    )


def test_shadow_rng_context_restores_all_cpu_rngs():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = capture_rng_state(include_cuda=False)
    with preserve_rng_state(include_cuda=False):
        random.random()
        np.random.randn(4)
        torch.randn(4)
    after = capture_rng_state(include_cuda=False)
    assert rng_states_equal(before, after)


def test_separate_temporal_ensemble_buffers_do_not_cross_contaminate():
    left = torch.zeros(8, 11, 7)
    right = torch.zeros_like(left)
    lr = torch.full((1, 3, 7), 0.25)
    full = torch.full((1, 3, 7), 0.75)
    lr_out, _ = temporal_ensemble_probability(lr, 0, left, 0.01)
    full_out, _ = temporal_ensemble_probability(full, 0, right, 0.01)
    assert torch.equal(lr_out, torch.full((1, 7), 0.25, dtype=torch.float64))
    assert torch.equal(full_out, torch.full((1, 7), 0.75, dtype=torch.float64))
    assert not torch.equal(left, right)


def test_shadow_branch_does_not_change_executed_action():
    executed_buffer = torch.zeros(8, 11, 7)
    shadow_buffer = torch.zeros_like(executed_buffer)
    lr = torch.full((1, 3, 7), 0.25)
    full = torch.full((1, 3, 7), 0.75)
    expected, _ = temporal_ensemble_probability(
        lr, 0, executed_buffer.clone(), 0.01
    )
    temporal_ensemble_probability(full, 0, shadow_buffer, 0.01)
    actual, _ = temporal_ensemble_probability(lr, 0, executed_buffer, 0.01)
    assert torch.equal(actual, expected)


def test_arm_gripper_mixing_uses_exact_dimensions():
    lr = torch.zeros(1, 3, 7)
    full = torch.ones_like(lr)
    mixed, arm_source, gripper_source = mix_action_tokens(
        lr, full, "lr_arm_full_gripper"
    )
    assert torch.equal(mixed[..., :6], lr[..., :6])
    assert torch.equal(mixed[..., 6:7], full[..., 6:7])
    assert (arm_source, gripper_source) == ("lr", "full")


def test_arm_jerk_excludes_discrete_gripper():
    actions = np.zeros((4, 7), dtype=np.float64)
    actions[:, 6] = [-1, 1, -1, 1]
    metrics = action_second_differences(actions)
    assert np.all(metrics["arm"] == 0)
    assert np.any(metrics["gripper_discrete_second_difference"] != 0)


def test_transition_labels():
    assert classify_transition(None, True) == "episode_start_full"
    assert classify_transition(True, False) == "full_to_skip"
    assert classify_transition(False, False) == "skip_to_skip"
    assert classify_transition(False, True) == "skip_to_full"


def test_skip_shadow_requirement_matches_executed_counterfactual():
    assert not counterfactual_requires_skip_shadow("standard", "every_step")
    assert not counterfactual_requires_skip_shadow(
        "lr_arm_lr_gripper", "every_step"
    )
    assert not counterfactual_requires_skip_shadow(
        "latent_fusion", "soft_reset_only"
    )
    assert counterfactual_requires_skip_shadow("latent_fusion", "every_step")
    assert counterfactual_requires_skip_shadow("matched_random", "every_step")
    assert counterfactual_requires_skip_shadow(
        "lr_arm_full_gripper", "every_step"
    )


def test_latent_fusion_endpoints():
    lr = torch.zeros(2, 3, 4)
    full = torch.ones_like(lr)
    assert torch.equal(fuse_latents(lr, full, 0), lr)
    assert torch.equal(fuse_latents(lr, full, 1), full)


def test_matched_random_preserves_per_token_norm():
    full = torch.randn(2, 3, 8)
    lr = full + torch.randn_like(full)
    _, learned, random_delta = matched_random_latent(lr, full, seed=17)
    assert torch.allclose(
        learned.norm(dim=-1), random_delta.norm(dim=-1), atol=1e-5, rtol=1e-5
    )


def test_trace_round_trip(tmp_path):
    paths = save_trace_episode(
        tmp_path,
        "episode",
        [{"timestep": 0, "mode": "full"}, {"timestep": 1, "mode": "stepwise"}],
        [{"z_full": torch.ones(1, 3, 4)}, {"z_full": None}],
        {"success": 1},
    )
    metadata, rows, tensors = load_trace_shard(Path(paths["json"]))
    assert metadata["num_steps"] == 2
    assert rows[1]["mode"] == "stepwise"
    assert tensors["z_full"].shape == (2, 1, 3, 4)
    assert tensors["z_full__present"].tolist() == [1, 0]
