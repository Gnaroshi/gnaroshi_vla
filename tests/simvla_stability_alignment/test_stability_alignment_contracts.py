from __future__ import annotations

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    condition_update_with_code,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    GenerationLoopConfig,
)
from architectures.simvla.adapters.latentloop.stability_alignment import rb2_pipeline
from architectures.simvla.adapters.latentloop.stability_alignment.sd1_pipeline import (
    _gate_branch_decision,
    _select_short_span_summary,
)
from architectures.simvla.adapters.latentloop.stability_alignment.age_encoding import (
    ParityPreservingAgeEncoding,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    BUNDLE_SCHEMA,
    CONTRIBUTION_TARGETS,
    GENERATION_NG3_FULL_INDICES,
    atomic_write_json,
    AtomicStageState,
    condition_only_2k_continuation,
    evaluate_2k_gate,
    evaluate_10k_gate,
    free_gpu_pairs,
    gpu_is_free,
    kc_schedule,
    load_json,
    naive_nfe3_contract,
    rotating_condition_age,
    select_condition_only_parent,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment.data import (
    ReplicatedEventAwareSampler,
    compose_q0_q7_windows,
    teacher_gripper_event,
)
from architectures.simvla.adapters.latentloop.stability_alignment.diagnostic_bundle import (
    READY_NAME,
    build as build_diagnostic_bundle,
)
from architectures.simvla.adapters.latentloop.stability_alignment.long_span import (
    UNROLL_PATTERN,
    variable_unroll_length,
)
from architectures.simvla.adapters.latentloop.stability_alignment.model import (
    StabilityAlignedModules,
    configure_condition_only_stage,
    optimizer_parameter_groups,
    zero_code_parity,
)
from architectures.simvla.adapters.latentloop.stability_alignment.objectives import (
    ConditionPaths,
    LOSS_NAMES,
    bounded_top_cvar,
    calibrate_loss_weights,
    condition_paths,
    gripper_transition_loss,
    stability_pair_loss,
    stability_raw_losses,
)
from architectures.simvla.adapters.latentloop.stability_alignment.online_eval import (
    StabilityAlignedPolicy,
    _renderer_contract,
)
from architectures.simvla.adapters.latentloop.stability_alignment.trainer import (
    benchmark_numerical_stability,
    deterministic_parameter_probe,
)
from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0ObservationPair,
)


def test_active_jobs_are_never_classified_free_or_killed() -> None:
    assert not gpu_is_free(memory_used_mib=0, utilization_percent=0, compute_pids=[123])
    assert free_gpu_pairs((2, 3, 4, 5), (3,), max_simultaneous_pairs=2) == ((2, 4),)
    source = inspect.getsource(rb2_pipeline)
    assert "pkill" not in source
    assert "SIGKILL" not in source


def test_failed_gate_diagnostic_bundle_preserves_classification(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    checkpoint = train_root / "checkpoints/stability_step_010000.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {"optimizer_step": 10_000, "parent_identity": {"condition": "S50"}},
        checkpoint,
    )
    for name in (
        "source_lock.json",
        "training_contract.json",
        "determinism.json",
        "parameter_audit.json",
    ):
        atomic_write_json(train_root / name, {"name": name})
    offline_gate = tmp_path / "offline_gate.json"
    atomic_write_json(
        offline_gate,
        {
            "verdict": "STABILITY_10K_GATE_FAIL",
            "optimizer_step": 10_000,
            "gate": {"passed": False, "checks": {"recurrence": False}},
        },
    )
    norm_stats = tmp_path / "libero_norm.json"
    loss_weights = tmp_path / "loss_weights.json"
    atomic_write_json(norm_stats, {"norm": "locked"})
    atomic_write_json(loss_weights, {"weights": "locked"})
    output = tmp_path / "bundle"

    ready = build_diagnostic_bundle(
        SimpleNamespace(
            checkpoint=str(checkpoint),
            offline_gate=str(offline_gate),
            norm_stats=str(norm_stats),
            loss_weights=str(loss_weights),
            output=str(output),
        )
    )

    assert ready["diagnostic_only"] is True
    assert ready["offline_gate_passed"] is False
    assert ready["kc3_offline_ready"] is False
    assert ready["kc4_offline_ready"] is False
    assert load_json(output / READY_NAME) == ready
    assert load_json(output / "diagnostic_contract.json")[
        "online_result_must_not_be_reclassified_as_gate_passing"
    ] is True


def test_gpu_pool_never_double_assigns_running_pairs() -> None:
    pairs = free_gpu_pairs(
        (2, 3, 4, 5, 6, 7), (), running_pairs=((2, 3),), max_simultaneous_pairs=2
    )
    assert pairs == ((4, 5),)


def test_exact_cache_identity_and_atomic_serialization(tmp_path: Path) -> None:
    cache_manifest = tmp_path / "manifest.json"
    atomic_write_json(cache_manifest, {"windows": 6525, "queries": 26100})
    first = sha256_file(cache_manifest)
    assert first == sha256_file(cache_manifest)
    atomic_write_json(cache_manifest, {"windows": 6525, "queries": 26101})
    assert first != sha256_file(cache_manifest)


def test_stage_ledger_can_resume_stale_running_stage(tmp_path: Path) -> None:
    state = AtomicStageState(tmp_path / "state.json", ("S0", "S1"))
    state.set("S0", "RUNNING", pid=1)
    state.set("S0", "RUNNING", pid=2, resumed=True)
    assert state.payload["stages"]["S0"] == {
        "state": "RUNNING",
        "pid": 2,
        "resumed": True,
    }


class _Delta(nn.Module):
    def forward(self, pair: NativeV0ObservationPair) -> torch.Tensor:
        return pair.current_proprio - pair.previous_proprio


class _Updater(nn.Module):
    def forward(self, previous, delta, *, valid_mask, group_ids, age):
        residual = delta[:, :1].unsqueeze(1).expand_as(previous) + float(age) / 10.0
        return SimpleNamespace(condition=previous + residual, residual=residual)


class _Adapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta_encoder = _Delta()
        self.condition_updater = _Updater()


def test_teacher_forced_and_recursive_paths_are_distinct() -> None:
    batch = {
        "anchor_condition": torch.zeros(1, 2, 1),
        "teacher_conditions": torch.tensor(
            [[[[10.0], [10.0]], [[20.0], [20.0]], [[30.0], [30.0]]]]
        ),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "group_ids": torch.zeros(1, 2, dtype=torch.long),
        "image_sequence": torch.zeros(1, 4, 1),
        "proprio_sequence": torch.arange(4.0).reshape(1, 4, 1),
    }
    paths = condition_paths(_Adapter(), batch)
    assert torch.equal(paths.recursive[0], paths.teacher_forced[0])
    assert not torch.equal(paths.recursive[1], paths.teacher_forced[1])
    assert torch.allclose(paths.teacher_forced[1], torch.full((1, 2, 1), 11.2))


def test_stability_target_is_stop_gradient() -> None:
    recursive = torch.ones(1, 2, 3, requires_grad=True)
    teacher = torch.zeros(1, 2, 3, requires_grad=True)
    loss = stability_pair_loss(recursive, teacher, torch.ones(1, 2, dtype=torch.bool))
    loss.backward()
    assert recursive.grad is not None
    assert teacher.grad is None


def test_event_sampler_is_replicated_and_75_25() -> None:
    index = {"natural_indices": [0, 1, 2, 3], "event_indices": [7, 8]}
    a = ReplicatedEventAwareSampler(index, seed=11, start_step=0, stop_step=40)
    b = ReplicatedEventAwareSampler(index, seed=11, start_step=0, stop_step=40)
    assert list(a) == list(b)
    assert sum(a.index(step) in {7, 8} for step in range(40)) == 10
    actions = torch.ones(3, 10, 7)
    actions[1, 2:, 6] = -1
    assert teacher_gripper_event(actions)


def test_tail_loss_uses_worst_ten_percent() -> None:
    values = torch.arange(1.0, 11.0)
    assert bounded_top_cvar(values, 0.10).item() == 10.0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for autocast policy"
)
def test_gripper_transition_loss_is_cuda_autocast_safe() -> None:
    prediction = torch.randn(2, 10, 7, device="cuda", requires_grad=True)
    target = torch.randn(2, 10, 7, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, details = gripper_transition_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(details["switch_event"])
    assert prediction.grad is not None


def test_deterministic_parameter_probe_restores_exact_state() -> None:
    module = nn.Linear(8, 4)
    originals = tuple(value.detach().clone() for value in module.parameters())
    with deterministic_parameter_probe(
        tuple(module.parameters()), relative_norm=1e-3, seed=17
    ) as audit:
        assert audit["relative_norm_observed"] == pytest.approx(1e-3, rel=1e-4)
        assert any(
            not torch.equal(value.detach(), original)
            for value, original in zip(module.parameters(), originals)
        )
    assert audit["restored_exactly"]
    assert all(
        torch.equal(value.detach(), original)
        for value, original in zip(module.parameters(), originals)
    )


def test_exposing_real_cj_preserves_condition_output() -> None:
    torch.manual_seed(7)
    adapter = NativeSimVLAV0()
    adapter.eval()
    previous = torch.randn(1, 4, 960)
    pair = NativeV0ObservationPair(
        previous_images=torch.randn(1, 2, 3, 16, 16),
        current_images=torch.randn(1, 2, 3, 16, 16),
        previous_proprio=torch.randn(1, 8),
        current_proprio=torch.randn(1, 8),
    )
    mask = torch.ones(1, 4, dtype=torch.bool)
    groups = torch.zeros(1, 4, dtype=torch.long)
    with torch.no_grad():
        direct = adapter.update_once(previous, pair, valid_mask=mask, group_ids=groups, age=1)
        exposed = condition_update_with_code(
            adapter, previous, pair, valid_mask=mask, group_ids=groups, age=1
        )
    assert torch.equal(direct.condition, exposed.update.condition)
    assert bool((exposed.condition_change_code.norm(dim=-1) > 0).all())


def test_ng3_indices_and_zero_code_parent_preservation() -> None:
    assert GENERATION_NG3_FULL_INDICES == (0, 4, 8)
    parent = GenerationLoopConfig().build()
    candidate = copy.deepcopy(parent)
    result = zero_code_parity(parent, candidate, device=torch.device("cpu"))
    assert result["verdict"] == "ZERO_CODE_PARENT_PARITY_PASS"


def test_parent_preservation_loss_is_zero_for_identical_actions() -> None:
    condition = tuple(torch.ones(1, 2, 3) for _ in range(3))
    action = tuple(torch.zeros(1, 10, 7) for _ in range(3))
    paths = ConditionPaths(
        condition,
        condition,
        tuple(torch.ones(1, 128) for _ in range(3)),
        condition,
    )
    raw, _ = stability_raw_losses(
        paths=paths,
        parent_paths=paths,
        exact_conditions=condition,
        rotating_recursive_full_action=action[0],
        rotating_teacher_forced_action=action[0],
        rotating_age_index=0,
        joint_actions=action,
        parent_joint_actions=action[:2],
        exact_actions=action,
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert raw["parent_preservation"].item() <= 1.1e-6


def test_s50_s150_selection_and_gate_contracts() -> None:
    common = {
        "gate_passed": True,
        "age2_recursive_first_r_mean": 1.0,
        "age3_recursive_first_r_mean": 1.0,
        "age3_recursive_first_r_p95": 1.0,
        "age3_gripper_tail": 1.0,
        "exact_ng3_error": 1.0,
        "parameter_count": 10,
    }
    assert select_condition_only_parent(common, common)["selected"] == "S50"
    assert _select_short_span_summary([{**common, "branch": "S150"}]) == {
        "verdict": "S150_SELECTED_ONLY_2K_SURVIVOR",
        "selected": "S150",
    }
    assert evaluate_2k_gate(
        {
            "frozen_base_gradients_zero": True,
            "age1_first_r_ratio_to_parent": 1.0,
            "exact_ng3_ratio_to_parent": 1.0,
            "no_gripper_collapse": True,
            "p99_ratio_to_parent": 1.0,
            "stability_slope": -0.1,
        }
    ).passed
    assert evaluate_10k_gate(
        {
            "age2_recurrence_improvement": 0.20,
            "age3_recurrence_improvement": 0.30,
            "age3_gripper_sign_improvement": 0.25,
            "age3_first_r_p95_ratio": 0.99,
            "teacher_forced_first_r_ratio": 1.0,
            "age1_final_system_ratio": 1.0,
            "exact_ng3_ratio_to_parent": 1.0,
            "no_p99_or_gripper_collapse": True,
            "original_simvla_frozen": True,
        }
    ).passed


def test_gate_branch_decision_continues_safe_trend_warning_branch(tmp_path: Path) -> None:
    s50 = tmp_path / "s50.json"
    s150 = tmp_path / "s150.json"
    atomic_write_json(
        s50,
        {
            "verdict": "STABILITY_2K_GATE_FAIL",
            "gate": {
                "passed": False,
                "checks": {
                    "finite_losses": True,
                    "frozen_base_gradients_zero": True,
                    "age1_first_r_within_5pct": True,
                    "exact_ng3_within_5pct": True,
                    "no_gripper_collapse": True,
                    "no_p99_explosion": True,
                    "stability_loss_decreasing": False,
                },
            },
            "optimizer_step": 2000,
        },
    )
    atomic_write_json(
        s150,
        {
            "verdict": "STABILITY_2K_GATE_PASS",
            "gate": {
                "passed": True,
                "checks": {
                    "finite_losses": True,
                    "frozen_base_gradients_zero": True,
                    "age1_first_r_within_5pct": True,
                    "exact_ng3_within_5pct": True,
                    "no_gripper_collapse": True,
                    "no_p99_explosion": True,
                    "stability_loss_decreasing": True,
                },
            },
            "optimizer_step": 2000,
        },
    )
    decision = _gate_branch_decision(
        {"S50": s50, "S150": s150},
        pass_verdict="STABILITY_2K_GATE_PASS",
        fail_verdict="STABILITY_2K_GATE_FAIL",
    )
    assert decision["passing_branches"] == ["S150"]
    assert decision["continuing_branches"] == ["S50", "S150"]
    assert decision["trend_warning_branches"] == ["S50"]
    assert decision["stopped_branches"] == []
    fallback = condition_only_2k_continuation(load_json(s50))
    assert fallback["passed"]
    assert fallback["condition_only_fallback"]


def test_kc_schedules_and_conditional_kc8_age_parity() -> None:
    assert kc_schedule(1) == (0,)
    assert kc_schedule(3) == (0, 1, 2)
    assert kc_schedule(4) == (0, 1, 2, 3)
    assert kc_schedule(8) == tuple(range(8))
    embedding = nn.Embedding(4, 8)
    continuous = ParityPreservingAgeEncoding.from_embedding(embedding)
    ages = torch.tensor([1, 2, 3])
    assert torch.equal(embedding(ages), continuous(ages))
    assert continuous(torch.tensor([4, 7])).shape == (2, 8)


def test_rotating_full_nfe_age_is_deterministic() -> None:
    assert tuple(rotating_condition_age(step) for step in range(7)) == (
        1,
        2,
        3,
        1,
        2,
        3,
        1,
    )


def test_condition_only_stage_freezes_generation_and_projection() -> None:
    modules = StabilityAlignedModules(
        NativeSimVLAV0(), GenerationLoopConfig().build()
    )
    audit = configure_condition_only_stage(modules)
    assert audit["training_mode"] == "condition_only"
    assert all(not parameter.requires_grad for parameter in modules.generation.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in modules.generation.condition_code_projection.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in modules.condition.condition_updater.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in modules.condition.delta_encoder.parameters()
    )
    groups = optimizer_parameter_groups(modules, base_lr=1e-4, weight_decay=0.0)
    assert [group["name"] for group in groups] == [
        "condition_updater",
        "observation_change_encoder",
    ]
    assert [group["peak_lr"] for group in groups] == [1e-4, 4e-5]


def test_q0_q7_catalog_reuses_contiguous_query_ids_and_short_age_fraction() -> None:
    class Store:
        manifest = {
            "windows": [
                [f"q{index}" for index in range(4)],
                [f"q{index}" for index in range(4, 8)],
            ]
        }

        @staticmethod
        def query(query_id: str) -> dict[str, object]:
            return {
                "metadata": {
                    "task_id": 2,
                    "episode_id": "episode-7",
                    "query_index": int(query_id[1:]),
                }
            }

    windows = compose_q0_q7_windows(Store())
    assert windows == (tuple(f"q{index}" for index in range(8)),)
    assert tuple(variable_unroll_length(step) for step in range(8)) == UNROLL_PATTERN
    assert sum(age <= 3 for age in UNROLL_PATTERN) / len(UNROLL_PATTERN) == 0.5
    assert set(UNROLL_PATTERN) == set(range(1, 8))


def test_naive_nfe3_is_exact_source_three_step_control() -> None:
    assert (
        naive_nfe3_contract((1.0, 2 / 3, 1 / 3), -1 / 3)["verdict"]
        == "NAIVE_NFE3_CONTRACT_PASS"
    )
    assert (
        naive_nfe3_contract((1.0, 0.5, 0.0), -0.5)["verdict"]
        == "NAIVE_NFE3_CONTRACT_FAIL"
    )


def test_severe_gradient_conflict_is_balanced_before_step_zero() -> None:
    raw = {name: 1.0 for name in LOSS_NAMES}
    gradients = {name: 1.0 for name in LOSS_NAMES}
    gradients["parent_preservation"] = 1e-9
    targets = {
        "recursive_reference": 0.30,
        "teacher_forced_preservation": 0.15,
        "recursive_stability": 0.20,
        "end_to_end_execution": 0.20,
        "gripper_transition": 0.08,
        "tail_cvar": 0.04,
        "rotating_full_nfe_execution": 0.02,
        "parent_preservation": 0.01,
    }
    calibrated = calibrate_loss_weights(raw, gradients, targets)
    assert calibrated["severe_gradient_scale_conflict_observed"]
    assert calibrated["gradient_norm_balancing_applied"]
    assert calibrated["gradient_norm_spread"] <= 30.0 + 1e-6
    assert calibrated["weighted_gradient_l1_after_global_scale"] <= 1.0 + 1e-9


def test_calibration_preserves_semantic_ratios_under_global_gradient_budget() -> None:
    raw = {name: 1.0 for name in LOSS_NAMES}
    gradients = {name: 10.0 for name in LOSS_NAMES}
    calibrated = calibrate_loss_weights(raw, gradients, CONTRIBUTION_TARGETS)
    assert calibrated["global_weight_scale"] == pytest.approx(0.1)
    assert calibrated["weighted_gradient_l1_after_global_scale"] == pytest.approx(1.0)
    assert calibrated["weighted_raw_fractions"] == pytest.approx(CONTRIBUTION_TARGETS)


def test_calibration_rejects_unmeasured_zero_gradient() -> None:
    raw = {name: 1.0 for name in LOSS_NAMES}
    gradients = {name: 1.0 for name in LOSS_NAMES}
    gradients["parent_preservation"] = 0.0
    targets = {
        "recursive_reference": 0.30,
        "teacher_forced_preservation": 0.15,
        "recursive_stability": 0.20,
        "end_to_end_execution": 0.20,
        "gripper_transition": 0.08,
        "tail_cvar": 0.04,
        "rotating_full_nfe_execution": 0.02,
        "parent_preservation": 0.01,
    }
    with pytest.raises(ValueError, match="measured nonzero gradient"):
        calibrate_loss_weights(raw, gradients, targets)


def test_benchmark_numerical_gate_rejects_parent_dominance() -> None:
    balanced = {
        name: [float(CONTRIBUTION_TARGETS[name])] * 4 for name in LOSS_NAMES
    }
    passed = benchmark_numerical_stability(
        total_losses=[1.0] * 4,
        gradient_norms=[0.5] * 4,
        weighted_fractions=balanced,
    )
    assert passed["passed"]

    dominated = {name: [0.01 / 7.0] * 4 for name in LOSS_NAMES}
    dominated["parent_preservation"] = [0.99] * 4
    failed = benchmark_numerical_stability(
        total_losses=[1000.0] * 4,
        gradient_norms=[1e5] * 4,
        weighted_fractions=dominated,
    )
    assert not failed["passed"]
    assert not failed["checks"]["parent_mean_at_most_5_percent"]
    assert not failed["checks"]["gradient_clipping_below_90_percent"]


def test_rb2_bundle_contract_and_remote_parser(tmp_path: Path) -> None:
    assert rb2_pipeline._remote_parts("sd1:/tmp/bundle") == ("sd1", "/tmp/bundle")
    bundle = tmp_path / "short_span"
    bundle.mkdir()
    checkpoint = bundle / "selected.pt"
    checkpoint.write_bytes(b"checkpoint")
    atomic_write_json(bundle / "payload.json", {"value": 1})
    atomic_write_json(
        bundle / "SHA256_MANIFEST.json",
        {
            "selected.pt": sha256_file(checkpoint),
            "payload.json": sha256_file(bundle / "payload.json"),
        },
    )
    atomic_write_json(
        bundle / "READY_SHORT_FOR_RB2.json",
        {
            "schema_version": BUNDLE_SCHEMA,
            "verdict": "READY_SHORT_FOR_RB2",
            "checkpoint": "selected.pt",
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    assert (
        rb2_pipeline._validate_bundle(bundle, "READY_SHORT_FOR_RB2.json")["verdict"]
        == "READY_SHORT_FOR_RB2"
    )
    incoming = tmp_path / "incoming"
    (incoming / "short_span").mkdir(parents=True)
    atomic_write_json(incoming / "short_span/READY_SHORT_FOR_RB2.json", {"ready": True})
    assert rb2_pipeline.remote_ready(str(incoming))
    (incoming / "long_span").mkdir()
    long_ready = {"ready": True, "checkpoint_sha256": "abc"}
    atomic_write_json(incoming / "long_span/READY_KC8_FOR_RB2.json", long_ready)
    assert rb2_pipeline.remote_ready(str(incoming), "READY_KC8_FOR_RB2.json")
    assert rb2_pipeline.remote_payload(
        str(incoming), span="long_span", filename="READY_KC8_FOR_RB2.json"
    ) == long_ready


def test_completed_frontier_rows_are_not_in_rb2_execution_queue() -> None:
    lock = rb2_pipeline.load_json(rb2_pipeline.COMPLETED_FRONTIER_LOCK)
    assert lock["rerun_allowed"] is False
    assert set(lock["rows"]) == {
        "condition_kc2_naive_nfe2",
        "condition_kc2_naive_nfe3",
        "condition_kc2_ng2",
        "condition_kc3_naive_nfe3",
    }
    run_source = inspect.getsource(rb2_pipeline.run)
    assert "_run_fixed_control" not in run_source
    assert '_run_selected_row(bundle, 2, "learned_ng3")' in run_source


def test_condition_only_online_generation_uses_validated_zero_code_lane() -> None:
    source = inspect.getsource(StabilityAlignedPolicy._decode)
    assert "condition.new_zeros((condition.shape[0], 128))" in source
    assert "code = self._active_code" not in source


def test_egl_contract_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    with pytest.raises(RuntimeError, match="EGL contract failed"):
        _renderer_contract({"renderer": {}})
