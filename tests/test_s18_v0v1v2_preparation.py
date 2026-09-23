from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from four_gpu_launcher_guard import parse_devices
from tools.seer.check_process_visibility import validate as validate_process_visibility
from methods.latentloop_v1v2.losses import compute_v1_losses
from methods.latentloop_v1v2.protocol import Operation, OperationCounters, periodic_schedule
from methods.latentloop_v1v2.scheduler import (
    AdaptiveRefreshScheduler,
    RefreshDecision,
    SchedulerState,
)
from methods.latentloop_v1v2.selection import select_v1_budget
from methods.latentloop_v1v2.splits import assert_disjoint_split_manifests
from methods.latentloop_v1v2.transition import (
    ActionGroundedConditioner,
    ExecutedActionSequenceEncoder,
    TransitionOutput,
    VariableTimeLatentLoopTransition,
    count_trainable_parameters,
)
from architectures.seer.adapters.latentloop_v1v2.evaluation import _wrapper_class


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DummyDelta(nn.Module):
    def forward(self, key, current, q_key=None, q_cur=None):
        camera = sum((cur - old).mean(dim=(1, 2, 3)) for old, cur in zip(key, current))
        state = (q_cur - q_key).mean(dim=1)
        scalar = camera + state
        return torch.stack((scalar, scalar.square(), scalar + 1.0, scalar - 1.0), dim=-1)


class DummyDynamics(nn.Module):
    def forward(self, latent, feature, dt=1.0, age=1.0):
        del age
        dt_tensor = torch.as_tensor(dt, dtype=latent.dtype, device=latent.device)
        if dt_tensor.ndim == 1:
            dt_tensor = dt_tensor[:, None, None]
        update = feature[:, None]
        if update.shape[-1] < latent.shape[-1]:
            update = torch.nn.functional.pad(
                update, (0, latent.shape[-1] - update.shape[-1])
            )
        return latent + update[..., : latent.shape[-1]] * dt_tensor


def transition_inputs(interval: int, extra_future: bool = False):
    torch.manual_seed(9)
    length = interval + 1 + int(extra_future)
    return {
        "anchor_latent": torch.randn(2, 3, 4),
        "primary_sequence": torch.randn(2, length, 3, 2, 2),
        "wrist_sequence": torch.randn(2, length, 3, 2, 2),
        "state_sequence": torch.randn(2, length, 7),
        "executed_actions": torch.randn(2, interval, 7),
        "interval": interval,
    }


def test_source_lock_fails_closed_before_git_or_checkpoint_load(tmp_path: Path):
    contract = {
        "status": "LOCKED_CONTRACT",
        "expected_host": socket.gethostname(),
        "target_source_tree": str(tmp_path.resolve()),
        "forbidden_scientific_source": str((tmp_path.parent / "forbidden").resolve()),
        "source_lock_relative_path": ".canonical/missing.json",
        "source_lock_sha256": "0" * 64,
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    result = subprocess.run(
        [sys.executable, str(ROOT / "source_gate.py"), "--repo-root", str(tmp_path), "--contract", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "source-lock" in result.stderr


def test_four_gpu_guard_accepts_only_two_four_gpu_lanes():
    assert parse_devices("0,1,2,3") == ("0", "1", "2", "3")
    assert parse_devices("4,5,6,7") == ("4", "5", "6", "7")
    for invalid in ("0", "0,1,2,3,4,5,6,7", "1,2,3,4", "0,0,1,2"):
        with pytest.raises(ValueError):
            parse_devices(invalid)


def test_teacher_adapter_identity_is_locked_without_loading_checkpoints():
    contract = json.loads((ROOT / "s18_canonical_source_lock.json").read_text())
    assert contract["artifacts"]["teacher33"]["sha256"] == "a999bf839acfb6f77beb8b86576933254f1981d2bacd1f0d269da093d7205cc5"
    assert contract["artifacts"]["adapter39"]["sha256"] == "badc74e135003fee91ccc69c76fe4f225aece856f487236ca7f626424504f132"


def test_k1_bypass_and_k4_periodic_schedule():
    assert periodic_schedule(8, 1) == (Operation.FULL,) * 8
    assert periodic_schedule(8, 4) == (
        Operation.FULL, Operation.UPDATE, Operation.UPDATE, Operation.UPDATE,
        Operation.FULL, Operation.UPDATE, Operation.UPDATE, Operation.UPDATE,
    )
    assert periodic_schedule(4, 4, updater_enabled=False) == (Operation.FULL,) * 4


def test_process_visibility_contract_rejects_report_and_date_paths():
    validate_process_visibility([
        "/home/mingyujung/private/gnaroshi_vla_latentloop_canonical",
        "/results/seer/latentloop/teacher33_v0v1v2",
        "latentloop_v1_e20_seed42",
    ])
    for value in (
        "/repo/codex_outputs/runtime.py",
        "/results/teacher33_20260821",
        "latentloop_20260821_120000",
    ):
        with pytest.raises(ValueError):
            validate_process_visibility([value])


@pytest.mark.parametrize("interval", [1, 2, 3])
def test_variable_intervals_and_direct_composed_shapes(interval: int):
    model = VariableTimeLatentLoopTransition(DummyDelta(), DummyDynamics(), motion_dim=4)
    output = model(**transition_inputs(interval))
    assert output.direct.shape == (2, 3, 4)
    assert output.composed.shape == (2, 3, 4)
    assert len(output.composed_features) == interval


def test_online_step_consumes_exactly_one_executed_action():
    model = VariableTimeLatentLoopTransition(DummyDelta(), DummyDynamics(), motion_dim=4)
    values = transition_inputs(1)
    output, feature = model.forward_step(
        previous_latent=values["anchor_latent"],
        previous_primary=values["primary_sequence"][:, 0],
        previous_wrist=values["wrist_sequence"][:, 0],
        previous_state=values["state_sequence"][:, 0],
        current_primary=values["primary_sequence"][:, 1],
        current_wrist=values["wrist_sequence"][:, 1],
        current_state=values["state_sequence"][:, 1],
        executed_action=values["executed_actions"][:, 0],
        age=1,
    )
    assert output.shape == values["anchor_latent"].shape
    assert feature.shape == (2, 4)


def test_v1_online_wrapper_advances_observation_and_actual_action_cache():
    class FakeModel:
        def __init__(self):
            self.latentloop_plan_adapter = VariableTimeLatentLoopTransition(
                DummyDelta(), DummyDynamics(), motion_dim=4
            )
            self.latentloop_plan_adapter.mode = "v1_transition"

        def decode_action_diagnostics_from_latent(self, latent):
            return {
                "arm": latent[..., :6],
                "gripper_logit": latent[..., 6:],
                "gripper_probability": torch.sigmoid(latent[..., 6:]),
            }

    class FakeBase:
        def __init__(self):
            self.model = FakeModel()
            self.lrnode_cached_latent = torch.zeros(1, 3, 7)
            self.lrnode_cached_image_primary = torch.zeros(1, 1, 3, 2, 2)
            self.lrnode_cached_image_wrist = torch.zeros(1, 1, 3, 2, 2)
            self.lrnode_cached_state = torch.zeros(1, 1, 7)
            self.lrnode_cached_env_action = torch.arange(7).numpy()
            self.lrnode_cached_age = 0
            self.lrnode_update_calls = 0
            self.num_policy_steps = 0

        def _base_model(self):
            return self.model

        def _cache_executed_env_action(self, action):
            self.lrnode_cached_env_action = torch.as_tensor(action).numpy().copy()

        def step(self, *args, **kwargs):
            del args, kwargs
            return torch.full((7,), 9.0).numpy()

        def get_lrnode_stats(self):
            return {}

    wrapper = _wrapper_class(FakeBase)()
    primary = torch.ones(1, 1, 3, 2, 2)
    wrist = torch.ones(1, 1, 3, 2, 2) * 2
    state = torch.ones(1, 1, 7)
    action_seq, debug = wrapper._update_from_lrnode_cache(
        primary, wrist, state, timestep=1
    )
    assert action_seq.shape == (1, 3, 7)
    assert debug["v1_direct_transition_called"] == 0
    assert debug["observation_cache_advanced"] == 1
    assert wrapper.lrnode_cached_age == 1
    wrapper.step({}, "goal", 1)
    assert wrapper.lrnode_cached_env_action.tolist() == [9.0] * 7


def test_direct_path_ignores_observation_after_requested_interval():
    model = VariableTimeLatentLoopTransition(DummyDelta(), DummyDynamics(), motion_dim=4)
    values = transition_inputs(2, extra_future=True)
    first = model(**values)
    values["primary_sequence"][:, -1].fill_(10000)
    values["wrist_sequence"][:, -1].fill_(-10000)
    values["state_sequence"][:, -1].fill_(5000)
    second = model(**values)
    assert torch.equal(first.direct, second.direct)
    assert torch.equal(first.composed, second.composed)


def test_executed_action_sequence_order_changes_feature():
    torch.manual_seed(4)
    encoder = ExecutedActionSequenceEncoder(action_dim=2, max_interval=3, output_dim=8).eval()
    actions = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]])
    assert not torch.equal(encoder(actions, 3), encoder(actions.flip(1), 3))


def test_composition_loss_sends_gradients_to_both_estimators():
    direct = torch.randn(2, 3, 7, requires_grad=True)
    composed = torch.randn(2, 3, 7, requires_grad=True)
    output = TransitionOutput(direct, composed, torch.empty(0), (), torch.ones(2, dtype=torch.long))

    def action_generator(latent):
        return {"arm": latent[..., :6], "gripper_probability": latent[..., 6:]}

    weights = {name: 0.0 for name in ("direct_latent", "composed_latent", "direct_action", "composed_action", "smooth")}
    weights["composition"] = 1.0
    loss = compute_v1_losses(output, torch.zeros_like(direct), torch.zeros_like(direct), action_generator, weights)
    loss.total.backward()
    assert direct.grad is not None and direct.grad.abs().sum() > 0
    assert composed.grad is not None and composed.grad.abs().sum() > 0


def budget_row(scale: float, collapse: bool = False):
    return {
        "split_role": "checkpoint_validation",
        "uses_online_sr": False,
        "split_manifest_sha256": "1" * 64,
        "source_lock_sha256": "2" * 64,
        "initialization_sha256": "3" * 64,
        "independent_run_id": "run_placeholder",
        "budget_epochs": 20,
        "training_seed": 42,
        "world_size": 4,
        "effective_batch": 512,
        "cosine_horizon_epochs": 20,
        "warmup_fraction": 0.05,
        "direct_latent_mse": 1.0 * scale,
        "composed_latent_mse": 1.1 * scale,
        "direct_action_l1": 0.2 * scale,
        "composed_action_l1": 0.22 * scale,
        "composition_defect": 0.3 * scale,
        "hold_latent_mse": 2.0,
        "hold_action_l1": 0.5,
        "gripper_collapse": collapse,
    }


def test_e20_e40_validation_selection_is_frozen():
    def rows(e20_scale, e40_scale):
        e20, e40 = budget_row(e20_scale), budget_row(e40_scale)
        e20["independent_run_id"] = "e20"
        e40.update(independent_run_id="e40", budget_epochs=40, cosine_horizon_epochs=40)
        return e20, e40

    assert select_v1_budget(*rows(1.04, 1.0))["selected_budget_epochs"] == 20
    assert select_v1_budget(*rows(1.06, 1.0))["selected_budget_epochs"] == 40
    leaked = budget_row(1.0)
    leaked["uses_online_sr"] = True
    with pytest.raises(ValueError):
        select_v1_budget(leaked, rows(1.0, 1.0)[1])


def test_all_scientific_splits_are_episode_disjoint():
    assert_disjoint_split_manifests({
        "train": {"episode_keys": ["a", "b"]},
        "checkpoint_validation": {"episode_keys": ["c"]},
        "defect_fit": {"episode_keys": ["d"]},
        "defect_validation": {"episode_keys": ["e"]},
        "scheduler_calibration": {"episode_keys": ["f"]},
        "final": {"episode_keys": ["g"]},
    })
    with pytest.raises(ValueError):
        assert_disjoint_split_manifests({"fit": {"episode_keys": ["x"]}, "validation": {"episode_keys": ["x"]}})


def test_v1_parameter_increment_stays_under_locked_caps():
    increment = count_trainable_parameters(ActionGroundedConditioner(motion_dim=128))
    total = 470146 + increment
    assert total <= int(1.25 * 470146)
    assert total <= 600000


def test_no_composition_control_is_a_matched_trained_ablation():
    source = (
        ROOT
        / "architectures/seer/adapters/latentloop_v1v2/training.py"
    ).read_text(encoding="utf-8")
    assert 'no_composition_weights["composition"] = 0.0' in source
    assert "no_composition_optimizer.step()" in source
    assert '"kind": "matched_training_ablation"' in source
    assert "selected_no_composition_control.pth" in source
    assert "model.get_action_head_modules()" in source
    assert "model.lrnode_action_head_modules()" not in source


def test_conditional_v1_templates_are_explicit_and_load_v0_initialization():
    wrapper_root = ROOT / "architectures/seer/wrappers/latentloop_v0v1v2"
    for name, interval in (("v1_k8_template.sh", 8), ("v1_k12_template.sh", 12)):
        source = (wrapper_root / name).read_text(encoding="utf-8")
        assert "--adapter-init \"${ADAPTER}\"" in source
        assert "--allow-conditional-interval" in source
        assert f"--query-interval {interval}" in source
        assert "V1_ONLINE_PASS" in source


def test_importer_externalizes_large_artifacts_to_shared_storage():
    source = (ROOT / "s18_import_canonical_bundle.sh").read_text(encoding="utf-8")
    assert 'mv "${TMP}/repo/.canonical/artifacts" "${ARTIFACT_STAGING}/payload"' in source
    assert 'mv "${ARTIFACT_STAGING}/payload" "${ARTIFACT_TARGET}"' in source
    assert 'ln -s "${ARTIFACT_TARGET}" "${TMP}/repo/.canonical/artifacts"' in source
    assert 'ln -s "${ARTIFACT_TARGET}" "${STABLE_ARTIFACT}"' in source
    assert "SOURCE_SWAPPED=1" in source
    assert "IMPORT_COMMITTED=1" in source
    assert 'mv "${TARGET}" "${TARGET}.failed_import"' in source


def test_exporter_preserves_git_status_leading_space():
    source = (ROOT / "sd1_export_canonical_bundle.sh").read_text(encoding="utf-8")
    assert '"status", "--short", "--untracked-files=all"' in source
    assert 'text=True,\n).splitlines()' in source
    assert 'status != [" M libero/libero/__init__.py"]' in source
    assert 'run("status", "--short", "--untracked-files=all")' not in source
    assert 'CANONICAL_EXPORT_PREFLIGHT_PASS' in source


def test_canonical_bundle_checksum_is_location_independent():
    exporter = (ROOT / "sd1_export_canonical_bundle.sh").read_text(encoding="utf-8")
    importer = (ROOT / "s18_import_canonical_bundle.sh").read_text(encoding="utf-8")
    assert 'sha256sum "$(basename "${ARCHIVE}")"' in exporter
    assert 'EXPECTED_BUNDLE_SHA="$(awk' in importer
    assert 'ACTUAL_BUNDLE_SHA="$(sha256sum "${BUNDLE}"' in importer
    assert 'sha256sum -c "$(basename "${BUNDLE}").sha256"' not in importer
    assert 'DATASET_META="${DATASET_ROOT}/libero_10_converted/meta_info.h5"' in importer
    assert 'CANONICAL_IMPORT_PREFLIGHT_PASS' in importer


def test_runtime_import_dependency_closure_is_locked_and_exported():
    contract = json.loads((ROOT / "s18_canonical_source_lock.json").read_text())
    expected = {
        "methods/latentloop_plan_continuation/cqpc_loss.py",
        "methods/latentloop_plan_continuation/feedback_source.py",
        "architectures/seer/adapters/latentloop_plan_continuation/__init__.py",
        "architectures/seer/adapters/latentloop_plan_continuation/trace_adapter.py",
    }
    assert contract["source_scope"]["core_locked_files"] == 38
    assert contract["source_scope"]["runtime_dependency_files"] == 14
    assert contract["source_scope"]["total_locked_files"] == 52
    assert len(contract["source_sha256"]) == 52
    runtime_paths = {
        relative
        for relative in contract["source_sha256"]
        if "latentloop_plan_continuation" in relative
    }
    assert len(runtime_paths) == 14
    assert expected <= runtime_paths
    for relative in runtime_paths:
        digest = contract["source_sha256"][relative]
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256_file(path) == digest, relative
    exporter = (ROOT / "sd1_export_canonical_bundle.sh").read_text()
    assert 'json.loads(contract_path.read_text())["source_sha256"]' in exporter


def test_level1_and_level2_scheduler_resets_and_counters():
    scheduler = AdaptiveRefreshScheduler(1.0, 1.0, 0.1, 3, 5)
    assert scheduler.decide(predicted_seq_error=0.8, predicted_direct_error=0.2, sequential_age=1, full_age=1) is RefreshDecision.DIRECT_REANCHOR
    state = SchedulerState()
    full = torch.ones(1, 3, 4)
    state.apply(RefreshDecision.FULL_SEER, full, full, full)
    state.apply(RefreshDecision.KEEP_SEQUENTIAL, full + 1, full + 2)
    state.apply(RefreshDecision.DIRECT_REANCHOR, full + 3, full + 4)
    assert state.sequential_age == 0 and state.full_age == 2
    assert torch.equal(state.current, full + 4)
    state.apply(RefreshDecision.FULL_SEER, full, full, full + 9)
    assert state.sequential_age == state.full_age == 0
    assert torch.equal(state.full_anchor, full + 9)
    assert state.to_dict()["direct_calls"] == 2


def test_target_k_budget_counters_and_json_serialization():
    counters = OperationCounters()
    for operation in periodic_schedule(8, 4):
        counters.record(operation, direct_was_evaluated=operation is Operation.UPDATE)
    payload = counters.to_dict()
    assert payload["effective_k"] == 4.0
    assert payload["full_query_reduction"] == 0.75
    assert payload["action_generator_calls"] == 8
    assert payload["direct_transition_calls"] == 6
    assert json.loads(json.dumps(payload)) == payload
