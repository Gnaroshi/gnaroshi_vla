from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from architectures.seer.adapters.joint_latent_action_surrogate.factory import (
    attach_joint_latent_action_surrogate,
)
from architectures.seer.adapters.joint_latent_action_surrogate.seer_joint import (
    SeerJointLatentActionAdapter,
)
from architectures.seer.upstream.models.lrnode_modules import (
    ControlledLatentNODE,
    FastVisualDeltaEncoder,
)
from methods.joint_latent_action_surrogate import (
    ImmutableActionAnchorCache,
    JointExecutionLevel,
    JointHierarchySchedule,
    JointLatentAnchoredActionSurrogate,
    JointLossWeights,
    JointVerdict,
    WideLatentCapacityControl,
    action_surrogate_error,
    align_immutable_anchor,
    apply_joint_decision_rule,
    joint_surrogate_loss,
    joint_surrogate_parameter_count,
    latent_state_error,
    wide_control_parameter_count,
)


def _inputs(batch: int = 2):
    return {
        "anchor_arm": torch.randn(batch, 3, 6),
        "anchor_gripper_logit": torch.randn(batch, 3, 1),
        "anchor_latent": torch.randn(batch, 3, 384),
        "current_latent": torch.randn(batch, 3, 384),
        "shared_feature": torch.randn(batch, 128),
    }


def test_tail_alignment_is_zero_and_masked() -> None:
    anchor = torch.arange(21, dtype=torch.float32).reshape(1, 3, 7)
    age1 = align_immutable_anchor(anchor, 1)
    assert age1.valid_mask.reshape(-1).tolist() == [True, True, False]
    assert torch.equal(age1.values[:, :2], anchor[:, 1:])
    assert torch.count_nonzero(age1.values[:, 2]) == 0
    age2 = align_immutable_anchor(anchor, 2)
    assert age2.valid_mask.reshape(-1).tolist() == [True, False, False]
    assert torch.equal(age2.values[:, 0], anchor[:, 2])
    assert torch.count_nonzero(age2.values[:, 1:]) == 0
    exhausted = align_immutable_anchor(anchor, 3)
    assert not bool(exhausted.valid_mask.any())
    assert torch.count_nonzero(exhausted.values) == 0


def test_surrogate_is_anchor_relative_and_nonrecursive() -> None:
    torch.manual_seed(0)
    module = JointLatentAnchoredActionSurrogate()
    inputs = _inputs()
    first = module(**inputs, elapsed=1)
    unrelated_previous_surrogate = torch.randn_like(first.horizon)
    del unrelated_previous_surrogate
    second = module(**inputs, elapsed=1)
    assert torch.equal(first.horizon, second.horizon)
    signature = inspect.signature(module.forward)
    assert "previous_surrogate" not in signature.parameters


def test_exact_anchor_is_immutable_until_exact_write_and_serializes() -> None:
    cache = ImmutableActionAnchorCache()
    inputs = _inputs(batch=1)
    cache.write_exact(
        inputs["anchor_arm"],
        inputs["anchor_gripper_logit"],
        inputs["anchor_latent"],
        0,
    )
    before = cache.snapshot()
    module = JointLatentAnchoredActionSurrogate()
    module(**inputs, elapsed=1)
    after = cache.snapshot()
    for left, right in zip(before[:3], after[:3]):
        assert torch.equal(left, right)
    assert before[3:] == after[3:]
    restored = ImmutableActionAnchorCache()
    restored.load_state_dict(cache.state_dict())
    restored_snapshot = restored.snapshot()
    for left, right in zip(after[:3], restored_snapshot[:3]):
        assert torch.equal(left, right)
    assert after[3:] == restored_snapshot[3:]


def test_exact_decoder_target_and_error_reset_losses() -> None:
    inputs = _inputs(batch=1)
    module = JointLatentAnchoredActionSurrogate()
    output = module(**inputs, elapsed=2)
    exact_arm = output.arm.detach() + 0.1
    exact_grip = output.gripper_logit.detach() - 0.2
    bundle = joint_surrogate_loss(
        predicted_arm=output.arm,
        predicted_gripper_logit=output.gripper_logit,
        exact_arm=exact_arm,
        exact_gripper_logit=exact_grip,
        valid_mask=output.valid_mask,
        residual=output.residual,
        weights=JointLossWeights(1, 1, 1, 1, 1, 1, 1),
        latent_loss=torch.tensor(0.3),
        latent_action_loss=torch.tensor(0.4),
    )
    assert bundle.raw["surrogate"] > 0
    exact_horizon = torch.cat((exact_arm, torch.sigmoid(exact_grip)), dim=-1)
    assert torch.count_nonzero(action_surrogate_error(exact_horizon, exact_horizon)) == 0
    latent = torch.randn(1, 3, 384)
    assert torch.count_nonzero(latent_state_error(latent, latent)) == 0


def test_training_exact_target_uses_existing_decoder_and_is_detached() -> None:
    model_source = Path(
        "architectures/seer/upstream/models/seer_model.py"
    ).read_text(encoding="utf-8")
    start = model_source.index("        if joint_surrogate_compute:")
    stop = model_source.index("        if latentloop_plan_compute:", start)
    joint_forward = model_source[start:stop]
    assert "exact = self.decode_action_diagnostics_from_latent(z_current)" in joint_forward

    loss_source = Path(
        "methods/joint_latent_action_surrogate/losses.py"
    ).read_text(encoding="utf-8")
    assert "exact_arm = exact_arm.detach()" in loss_source
    assert "exact_gripper_logit = exact_gripper_logit.detach()" in loss_source


def test_offline_hold_anchor_gate_uses_the_executed_token_metric() -> None:
    source = Path(
        "tests/fixtures/seer/retired_protocols/apply_joint_surrogate_offline_gates.py.txt"
    ).read_text(encoding="utf-8")
    assert '"train/joint/age1/hold_anchor_executed_token_l1"' in source
    assert 'hold_exec = values(rows, "train/joint/age1/hold_anchor_l1")' not in source


def test_shared_encoder_is_not_duplicated_inside_seer_adapter() -> None:
    adapter = SeerJointLatentActionAdapter(
        mode="joint",
        latent_dim=384,
        motion_dim=128,
        action_pred_steps=3,
        surrogate_hidden_dim=192,
        wide_hidden_dim=96,
    )
    names = tuple(name for name, _ in adapter.named_modules())
    assert not any("encoder" in name for name in names)
    assert not any("dynamics" in name for name in names)


def test_runtime_joint_fast_step_reuses_one_canonical_encoder_call() -> None:
    source = Path(
        "architectures/seer/upstream/utils/eval_utils_libero.py"
    ).read_text(encoding="utf-8")
    start = source.index("    def _joint_fast_surrogate(")
    stop = source.index("    def _wide_exact_update(", start)
    method = source[start:stop]
    assert method.count("self._update_from_lrnode_cache(") == 1
    assert "self.lrnode_encode_delta(" not in method
    update_start = source.index("    def _update_from_lrnode_cache(")
    update_stop = source.index("\n    def ", update_start + 8)
    update_method = source[update_start:update_stop]
    assert update_method.count("base_model.lrnode_encode_delta(") == 1


def test_parity_wrapper_exercises_joint_fast_diagnostics() -> None:
    wrapper = Path(
        "tests/fixtures/seer/retired_protocols/eval_joint_latent_action_surrogate.sh.txt"
    ).read_text(encoding="utf-8")
    assert "joint_k8_diagnostics_off" in wrapper
    assert "joint_k8_diagnostics_on" in wrapper
    assert '"${JOINT_CKPT}" joint off off 0 "" 0 0' in wrapper
    assert '"${JOINT_CKPT}" joint off off 0 "" 0 1' in wrapper


def test_online_aggregator_checks_manifest_task_names() -> None:
    source = Path(
        "tests/fixtures/seer/retired_protocols/aggregate_joint_latent_action_surrogate.py.txt"
    ).read_text(encoding="utf-8")
    assert "expected_manifest[episode_key(row)]" in source
    assert "task-name manifest mismatch" in source


def test_source_lock_covers_reused_runtime_dependencies() -> None:
    source = Path(
        "tests/fixtures/seer/retired_protocols/audit_joint_latent_action_surrogate.py.txt"
    ).read_text(encoding="utf-8")
    required = (
        "architectures/seer/upstream/models/lrnode_modules.py",
        "architectures/seer/upstream/data_info/libero_10_converted.json",
        "methods/latentloop_horizon_regeneration/schedule.py",
        "methods/latentloop_plan_continuation/action_correction.py",
        "architectures/seer/adapters/latentloop_plan_continuation/seer_action_correction.py",
    )
    for relative in required:
        assert f'"{relative}"' in source


def test_sequential_wrapper_uses_the_nested_converted_dataset_contract() -> None:
    wrapper = Path(
        "tests/fixtures/seer/retired_protocols/run_joint_latent_action_surrogate_sequential.sh.txt"
    ).read_text(encoding="utf-8")
    expected_root = (
        "${REPO_ROOT}/architectures/seer/upstream/"
        "LIBERO_DATASETS/libero_10_converted"
    )
    assert expected_root in wrapper
    assert 'export ROOT_DIR="${JOINT_DATASET_ROOT}"' in wrapper
    assert 'require_file "${ROOT_DIR}/${DATASET}/meta_info.h5"' in wrapper
    assert '"FREEZE ${stage}"' in wrapper


def test_sd1_seer_asset_resolver_uses_canonical_paths() -> None:
    source = Path("scripts/configure_seer_server_paths.sh").read_text(encoding="utf-8")
    assert "seer_node2" not in source
    assert (
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/"
        "LIBERO_DATASETS"
    ) in source
    assert (
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/"
        "vit_mae/mae_pretrain_vit_base.pth"
    ) in source


def test_full_refresh_and_exact_regeneration_resets_are_explicit() -> None:
    source = Path(
        "architectures/seer/upstream/utils/eval_utils_libero.py"
    ).read_text(encoding="utf-8")
    cache_start = source.index("    def _cache_full_forward_state(")
    cache_stop = source.index("    def _cache_executed_env_action(", cache_start)
    cache_method = source[cache_start:cache_stop]
    assert "self.lrnode_cached_latent = action_latent[:, selected_step].detach()" in cache_method
    assert "self.lrnode_cached_age = 0" in cache_method

    exact_start = source.index("    def _joint_exact_regeneration(")
    exact_stop = source.index("    def _joint_fast_surrogate(", exact_start)
    exact_method = source[exact_start:exact_stop]
    assert '"joint_action_surrogate_error": 0.0' in exact_method
    assert '"joint_executed_token_error": 0.0' in exact_method
    assert '"joint_tail_token_error": 0.0' in exact_method


class _AttachmentFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_lrnode_latent_update = True
        self.hidden_dim = 384
        self.lrnode_motion_dim = 128
        self.action_pred_steps = 3
        self.lrnode_delta_encoder = FastVisualDeltaEncoder(
            motion_dim=128, proprio_dim=8
        )
        self.lrnode_dynamics = ControlledLatentNODE(
            latent_dim=384,
            motion_dim=128,
            hidden_dim=256,
            gate_init_bias=-4.0,
            action_pred_steps=3,
            use_post_layernorm=0,
        )
        self.existing_action_head = nn.Linear(384, 7)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def test_joint_attachment_preserves_rng_and_existing_k1_action_head() -> None:
    torch.manual_seed(17)
    model = _AttachmentFixture()
    latent = torch.randn(2, 384)
    before_action = model.existing_action_head(latent).detach().clone()
    before_state = {
        name: value.detach().clone()
        for name, value in model.existing_action_head.state_dict().items()
    }
    rng_before = torch.random.get_rng_state().clone()
    report = attach_joint_latent_action_surrogate(
        model,
        SimpleNamespace(
            joint_latent_action_surrogate_mode="joint",
            joint_latent_action_surrogate_hidden_dim=192,
            joint_latent_action_surrogate_parameter_match_tolerance=0.02,
            seed=42,
        ),
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert report["attached"] is True
    assert torch.equal(model.existing_action_head(latent), before_action)
    for name, value in model.existing_action_head.state_dict().items():
        assert torch.equal(value, before_state[name])


def test_parameter_budget_and_wide_match() -> None:
    canonical = 470_146
    joint_added = joint_surrogate_parameter_count()
    wide_added = wide_control_parameter_count()
    assert joint_added == 89_671
    assert wide_added == 89_920
    assert (canonical + joint_added) / canonical <= 1.25
    assert abs(wide_added - joint_added) / joint_added <= 0.02


def test_wide_zero_initialization_preserves_latent() -> None:
    module = WideLatentCapacityControl()
    latent = torch.randn(2, 3, 384)
    feature = torch.randn(2, 128)
    assert torch.equal(module(latent, feature, age=1), latent)


def test_joint_and_wide_module_state_serialization() -> None:
    torch.manual_seed(9)
    joint = JointLatentAnchoredActionSurrogate().eval()
    inputs = _inputs(batch=1)
    expected = joint(**inputs, elapsed=1).horizon.detach()
    restored_joint = JointLatentAnchoredActionSurrogate().eval()
    restored_joint.load_state_dict(joint.state_dict(), strict=True)
    assert torch.equal(restored_joint(**inputs, elapsed=1).horizon, expected)

    wide = WideLatentCapacityControl().eval()
    restored_wide = WideLatentCapacityControl().eval()
    restored_wide.load_state_dict(wide.state_dict(), strict=True)
    latent = torch.randn(1, 3, 384)
    feature = torch.randn(1, 128)
    assert torch.equal(wide(latent, feature, age=2), restored_wide(latent, feature, age=2))


def test_primary_schedule() -> None:
    schedule = JointHierarchySchedule(8, 3)
    assert schedule.cycle() == [
        JointExecutionLevel.FULL_SEER,
        JointExecutionLevel.FAST_SURROGATE,
        JointExecutionLevel.FAST_SURROGATE,
        JointExecutionLevel.EXACT_ACTION_HEAD,
        JointExecutionLevel.FAST_SURROGATE,
        JointExecutionLevel.FAST_SURROGATE,
        JointExecutionLevel.EXACT_ACTION_HEAD,
        JointExecutionLevel.FAST_SURROGATE,
        JointExecutionLevel.FULL_SEER,
    ]


def test_decision_rules() -> None:
    supported = {
        "k1_parity": True,
        "joint_vs_canonical_ci_low_pp": -2.9,
        "joint_vs_recursive_ci_low_pp": 1.0,
        "action_head_reduction_fraction": 0.5,
        "parameter_ratio": 1.2,
        "latency_reduction_fraction": 0.05,
        "supported_advantage_at_latency_overhead": False,
        "latency_overhead_fraction": 0.0,
        "gripper_short_reversal_ratio": 1.2,
        "joint_better_pareto_than_wide": True,
        "tasks_regressed_over_20pp": 1,
        "joint_materially_improves_naive": True,
        "offline_gates_pass": True,
        "wide_matches_or_exceeds_joint": False,
        "recovers_recursive_collapse": True,
    }
    assert apply_joint_decision_rule(supported) == JointVerdict.SUPPORTED
    capacity = dict(supported, joint_better_pareto_than_wide=False, wide_matches_or_exceeds_joint=True)
    assert apply_joint_decision_rule(capacity) == JointVerdict.CAPACITY_ONLY
    assert apply_joint_decision_rule({}) == JointVerdict.INCONCLUSIVE
