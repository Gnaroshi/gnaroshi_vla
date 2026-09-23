"""CPU-only contract tests for the Seer LatentLoop Q1/Q2 comparison."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from architectures.seer.adapters.latentloop_comparison.action_token_alignment import (
    action_token_time_mapping,
    verified_overlap_pairs,
)
from architectures.seer.adapters.latentloop_comparison.seer_action_correction import (
    SeerActionCorrectionAdapter,
)
from architectures.seer.adapters.latentloop_comparison.seer_nonrecurrent_latent import (
    SeerNonRecurrentLatentAdapter,
)
from architectures.seer.adapters.latentloop_comparison.seer_teacher_pairs import (
    build_shifted_context_pair,
)
from methods.latentloop_comparison.action_space_correction import (
    ActionCorrectionCache,
    MatchedActionSpaceCorrection,
    action_correction_parameter_count,
    shift_action_horizon_with_mask,
)
from methods.latentloop_comparison.decisions import apply_comparison_decisions
from methods.latentloop_comparison.fairness import (
    assert_optimizer_exactly_matches_trainable,
    assert_parameter_match,
    assert_zero_or_missing_gradients,
    validate_fairness_manifest,
    write_refuse_overwrite,
)
from methods.latentloop_comparison.nonrecurrent_latent import (
    NonRecurrentAnchorState,
    NonRecurrentAnchorToCurrentBridge,
    cyclic_k4_offset,
    nonrecurrent_parameter_count,
)
from methods.latentloop_comparison.training_losses import (
    ActionCorrectionLossWeights,
    action_correction_loss,
    continuous_gripper_probability,
)
from tools.seer.evaluate_latentloop_comparison_offline import (
    _parse_legacy_seer_runtime_args,
    _run_with_seer_upstream_cwd,
)
from tools.seer.lock_latentloop_comparison_source import _runtime_dataset_path


class TinyMatchedEncoder(nn.Module):
    """Small synthetic stand-in for the shared FastVisualDeltaEncoder contract."""

    def __init__(self, motion_dim: int = 8) -> None:
        super().__init__()
        self.projection = nn.Linear(6, motion_dim)

    def forward(self, previous, current, *, q_key, q_cur):
        camera = torch.stack(
            [
                (current[index] - previous[index]).flatten(1).mean(dim=1)
                for index in range(2)
            ],
            dim=-1,
        )
        proprio = (q_cur - q_key)[:, :4]
        return self.projection(torch.cat((camera, proprio), dim=-1))


def _images(batch: int = 2):
    previous = [torch.zeros(batch, 3, 4, 4), torch.ones(batch, 3, 4, 4)]
    current = [torch.ones(batch, 3, 4, 4), torch.full((batch, 3, 4, 4), 3.0)]
    q_key = torch.zeros(batch, 8)
    q_cur = torch.ones(batch, 8)
    return previous, current, q_key, q_cur


def test_exact_action_token_shift_and_mask() -> None:
    horizon = torch.arange(21, dtype=torch.float32).reshape(1, 3, 7)
    shifted = shift_action_horizon_with_mask(horizon)
    assert torch.equal(shifted.values[:, 0], horizon[:, 1])
    assert torch.equal(shifted.values[:, 1], horizon[:, 2])
    assert torch.equal(shifted.values[:, 2], horizon[:, 2])
    assert shifted.valid_mask.shape == (1, 3, 1)
    assert shifted.valid_mask.squeeze(-1).tolist() == [[True, True, False]]
    assert verified_overlap_pairs(3) == [(1, 0), (2, 1)]
    assert [row["intended_execution_time"] for row in action_token_time_mapping(5, 3)] == [5, 6, 7]


def test_action_horizon_shapes_and_separate_heads() -> None:
    module = MatchedActionSpaceCorrection(
        action_pred_steps=3, motion_dim=8, hidden_dim=32
    )
    output = module(
        torch.zeros(2, 3, 6),
        torch.zeros(2, 3, 1),
        torch.ones(2, 8),
        age=1.0,
    )
    assert output.arm.shape == (2, 3, 6)
    assert output.gripper_logit.shape == (2, 3, 1)
    assert output.gripper_probability.shape == (2, 3, 1)
    assert module.arm_head is not module.gripper_head
    assert module.arm_head.out_features == 6
    assert module.gripper_head.out_features == 1


def test_continuous_gripper_target_and_loss() -> None:
    logits = torch.tensor([[[-1.0], [0.2], [2.0]]])
    probability = continuous_gripper_probability(teacher_logit=logits)
    assert torch.allclose(probability, torch.sigmoid(logits))
    with pytest.raises(ValueError, match="binary"):
        continuous_gripper_probability(
            teacher_probability=torch.tensor([[[0.0], [1.0], [0.0]]])
        )
    bundle = action_correction_loss(
        predicted_arm=torch.zeros(1, 3, 6, requires_grad=True),
        predicted_gripper_logit=torch.zeros(1, 3, 1, requires_grad=True),
        teacher_arm=torch.ones(1, 3, 6),
        teacher_gripper_logit=logits,
        arm_residual=torch.ones(1, 3, 6),
        gripper_logit_residual=torch.ones(1, 3, 1),
        weights=ActionCorrectionLossWeights(1.0, 1.0, 0.5, 0.01),
    )
    assert set(bundle.raw) == {
        "arm",
        "gripper",
        "executed_token",
        "residual_regularization",
    }
    assert torch.isfinite(bundle.total)


def test_full_refresh_initializes_recursive_action_cache() -> None:
    cache = ActionCorrectionCache()
    with pytest.raises(RuntimeError, match="not been initialized"):
        cache.tensors()
    arm = torch.randn(1, 3, 6, requires_grad=True)
    grip = torch.randn(1, 3, 1, requires_grad=True)
    cache.initialize_from_full(arm, grip)
    cached_arm, cached_grip = cache.tensors()
    assert cached_arm.grad_fn is None and cached_grip.grad_fn is None
    module = MatchedActionSpaceCorrection(3, 8, 32)
    output = module(cached_arm, cached_grip, torch.ones(1, 8), age=1.0)
    cache.update_from_prediction(output)
    updated_arm, updated_grip = cache.tensors()
    assert torch.equal(updated_arm, output.arm.detach())
    assert torch.equal(updated_grip, output.gripper_logit.detach())


def test_nonrecurrent_signature_fixed_anchor_and_offsets() -> None:
    parameters = inspect.signature(
        NonRecurrentAnchorToCurrentBridge.forward
    ).parameters
    assert list(parameters) == [
        "self",
        "anchor_latent",
        "anchor_to_current_feature",
        "age",
    ]
    assert [cyclic_k4_offset(index) for index in range(9)] == [1, 2, 3] * 3
    state = NonRecurrentAnchorState()
    observation = {"primary": torch.zeros(1, 3), "state": torch.zeros(1, 8)}
    latent = torch.zeros(1, 3, 16)
    state.replace_at_full_refresh(latent, observation, full_step=4)
    fixed_latent, fixed_observation, fixed_step = state.require()
    module = NonRecurrentAnchorToCurrentBridge(16, 8, 3, 32)
    _ = module(fixed_latent, torch.ones(1, 8), age=1.0)
    _ = module(fixed_latent, torch.ones(1, 8), age=3.0)
    after_latent, after_observation, after_step = state.require()
    assert torch.equal(after_latent, fixed_latent)
    assert torch.equal(after_observation["primary"], fixed_observation["primary"])
    assert after_step == fixed_step == 4


def test_shifted_teacher_pairs_cover_actual_k4_offsets() -> None:
    tensors = [torch.arange(2 * 10).reshape(2, 10, 1).float() for _ in range(4)]
    for offset in (1, 2, 3):
        pair = build_shifted_context_pair(
            *tensors,
            sequence_length=7,
            selected_step=6,
            offset=offset,
        )
        assert pair.anchor_primary.shape[1] == 7
        assert pair.current_primary.shape[1] == 7
        assert pair.anchor_observation_index == 6
        assert pair.current_observation_index == 6 + offset


def test_parameter_matching_and_exact_count_formulas() -> None:
    action = MatchedActionSpaceCorrection(3, 128, 256)
    assert sum(p.numel() for p in action.parameters()) == action_correction_parameter_count(256)
    nonrecurrent = NonRecurrentAnchorToCurrentBridge(384, 128, 3, 256)
    assert sum(p.numel() for p in nonrecurrent.parameters()) == nonrecurrent_parameter_count(256)
    action_adapter = SeerActionCorrectionAdapter(
        TinyMatchedEncoder(128),
        target_predictor_parameters=338690,
        action_pred_steps=3,
        motion_dim=128,
    )
    nonrecurrent_adapter = SeerNonRecurrentLatentAdapter(
        TinyMatchedEncoder(128),
        target_predictor_parameters=338690,
        latent_dim=384,
        action_pred_steps=3,
        motion_dim=128,
    )
    assert_parameter_match(
        338690, action_adapter.parameter_match.actual_parameters, tolerance=0.05
    )
    assert_parameter_match(
        338690,
        nonrecurrent_adapter.parameter_match.actual_parameters,
        tolerance=0.05,
    )


def test_optimizer_isolation_and_frozen_action_head_gradients() -> None:
    adapter = SeerNonRecurrentLatentAdapter(
        TinyMatchedEncoder(8),
        target_predictor_parameters=8000,
        latent_dim=16,
        action_pred_steps=3,
        motion_dim=8,
        maximum_relative_error=0.10,
    )
    frozen_seer = nn.Linear(16, 16).requires_grad_(False)
    frozen_action_head = nn.Linear(16, 7).requires_grad_(False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    assert_optimizer_exactly_matches_trainable(adapter, optimizer)
    previous, current, q_key, q_cur = _images()
    feature = adapter.encode_anchor_to_current(
        previous[0], previous[1], current[0], current[1], q_key, q_cur
    )
    output = adapter.forward_from_feature(torch.randn(2, 3, 16), feature, age=2.0)
    loss = frozen_action_head(output.latent).square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    assert_zero_or_missing_gradients([frozen_seer, frozen_action_head])


def _fairness_row() -> dict:
    return {
        "trainable_parameters": 1,
        "parameter_groups": ["adapter"],
        "training_dataset_manifest_sha256": "abc",
        "training_examples": 1,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "batch_size": 512,
        "precision": "fp32",
        "optimizer_steps": 10400,
        "checkpoint_frequency": 260,
        "validation_split": "deterministic_5_percent",
        "validation_metric": "validation_total_loss",
        "checkpoint_selection_rule": "minimum_validation_total_loss",
        "wall_clock_training_seconds": None,
        "gpu_count": 4,
        "environment": "seer_libero",
        "source_sha256": "def",
    }


def test_training_manifest_and_trace_roundtrip(tmp_path: Path) -> None:
    manifest = {
        "methods": {
            "canonical_latentloop": _fairness_row(),
            "matched_action_space_correction": _fairness_row(),
            "nonrecurrent_anchor_to_current_latent": _fairness_row(),
        }
    }
    validate_fairness_manifest(manifest)
    path = tmp_path / "trace.json"
    write_refuse_overwrite(path, manifest)
    assert json.loads(path.read_text()) == manifest
    with pytest.raises(FileExistsError):
        write_refuse_overwrite(path, manifest)


def test_k1_evaluator_bypasses_intermediate_update() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "architectures/seer/upstream/utils/eval_utils_libero.py").read_text()
    assert "return timestep % self.lrnode_query_interval != 0" in source
    assert all(not (step % 1 != 0) for step in range(100))


def _decision_inputs() -> dict:
    return {
        "latentloop_sr": 0.91,
        "action_correction_sr": 0.84,
        "nonrecurrent_sr": 0.83,
        "latent_minus_action_ci_low": 0.01,
        "action_minus_latent_ci_low": -0.12,
        "latent_minus_nonrecurrent_ci_low": 0.02,
        "nonrecurrent_minus_latent_ci_low": -0.13,
        "latentloop_parameters": 470146,
        "action_correction_parameters": 470000,
        "nonrecurrent_parameters": 470000,
        "latentloop_skip_p50_ms": 4.0,
        "action_correction_skip_p50_ms": 4.0,
        "nonrecurrent_skip_p50_ms": 4.0,
        "latentloop_better_k8_stability": True,
        "latentloop_better_k8_than_nonrecurrent": True,
    }


def test_predeclared_decision_rule_logic() -> None:
    supported = apply_comparison_decisions(_decision_inputs())
    assert supported["latent_location"]["verdict"] == "LATENT_LOCATION_SUPPORTED"
    assert supported["recurrence"]["verdict"] == "RECURRENCE_SUPPORTED"
    assert supported["combined_verdict"] == "LATENT_DYNAMICS_JUSTIFIED"
    inputs = _decision_inputs()
    inputs.update(
        {
            "action_correction_sr": 0.90,
            "action_minus_latent_ci_low": -0.02,
            "nonrecurrent_sr": 0.90,
            "nonrecurrent_minus_latent_ci_low": -0.02,
        }
    )
    sufficient = apply_comparison_decisions(inputs)
    assert sufficient["latent_location"]["verdict"] == "ACTION_CORRECTION_SUFFICIENT"
    assert sufficient["recurrence"]["verdict"] == "RECURRENCE_NOT_NEEDED"
    assert sufficient["combined_verdict"] == "NO_LATENT_ADVANTAGE"


def test_offline_evaluator_isolates_legacy_seer_parser_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The Seer parser must not consume the offline evaluator's arguments."""

    def legacy_get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--save_checkpoint_path", required=True)
        parser.add_argument("--phase", required=True)
        parser.add_argument("--marker", default="preserved")
        parser.parse_args()
        return parser

    outer_argv = ["offline", "--repo-root", "/tmp/repo", "--mode", "anchor_bridge"]
    monkeypatch.setattr(sys, "argv", outer_argv)
    parsed = _parse_legacy_seer_runtime_args(legacy_get_parser, tmp_path)

    assert sys.argv is outer_argv
    assert parsed.save_checkpoint_path == str(tmp_path)
    assert parsed.phase == "finetune"
    assert parsed.marker == "preserved"


def test_offline_evaluator_uses_upstream_cwd_and_restores_caller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Seer's relative data-info path must resolve for the full callback."""

    repo = tmp_path / "repo"
    upstream = repo / "architectures/seer/upstream"
    data_info = upstream / "data_info/libero_10_converted.json"
    data_info.parent.mkdir(parents=True)
    data_info.write_text("[]\n", encoding="utf-8")
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    runtime_dataset = launch_dir / "dataset/libero_10_converted"
    (runtime_dataset / "episodes").mkdir(parents=True)
    (runtime_dataset / "meta_info.h5").write_bytes(b"meta")
    monkeypatch.chdir(launch_dir)
    args = argparse.Namespace(
        repo_root=repo,
        source_lock=Path("inputs/source_lock.json"),
        teacher=Path("inputs/teacher.pth"),
        canonical_adapter=None,
        adapter_checkpoint=None,
        dataset_root=Path("dataset"),
        vit_checkpoint=Path("inputs/vit.pth"),
        libero_path=Path("libero"),
        output=Path("outputs/result.json"),
    )

    def verify_runtime(runtime_args: argparse.Namespace) -> dict[str, Any]:
        assert Path.cwd() == upstream
        assert Path("./data_info/libero_10_converted.json").is_file()
        assert runtime_args.output == launch_dir / "outputs/result.json"
        assert runtime_args.dataset_root == launch_dir / "dataset"
        return {"status": "PASS"}

    result = _run_with_seer_upstream_cwd(args, verify_runtime)

    assert result == {"status": "PASS"}
    assert Path.cwd() == launch_dir


def test_source_lock_dataset_root_matches_seer_runtime_lookup(tmp_path: Path) -> None:
    root_dir = tmp_path / "LIBERO_DATASETS/libero_10_converted"
    assert _runtime_dataset_path(root_dir) == (
        root_dir / "libero_10_converted"
    )
