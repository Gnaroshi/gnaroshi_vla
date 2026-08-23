from __future__ import annotations

import ast
import hashlib
import inspect
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch import nn

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    CACHE_MARKER_SCHEMA,
    CACHE_SCHEMA,
    CACHE_SHARD_SCHEMA,
    GENERATION_SCHEDULES,
    STAGE_GRAPH,
    balanced_mode_d_age,
    canonical_sha256,
    effective_batch_contract,
    libero_long_500_episode_keys,
    mode_ab_pass,
    mode_d_age_counts,
    native_nfe_time_grid,
    project_exact_teacher_cache,
    reference_noninterference,
    require_gate_payload,
    stage_graph_payload,
    stage_readiness,
    validate_generation_schedule,
    validate_query_windows,
    wallclock_projection,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import (
    install_exact_uint8_delta_path,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.decisions import (
    command_mode_d_not_required,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherSequenceDataset,
    ExactTeacherStore,
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.efficient_multirate import (
    exact_teacher_cache,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    load_frozen_simvla,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    hidden_hook_parity_report,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.lineage_bridge import (
    lineage_require_gate,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.gpu_contract import (
    parse_gpu_ids,
)
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.simvla_generation_loop import (
    SimVLAGenerationHiddenUpdater,
    SimVLAGenerationLoop,
)
from methods.latentloop.training.native_simvla_v0 import lr_multiplier


def test_source_gate_fails_closed(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps({"verdict": "PASS", "source_combined_sha256": "source-a"}),
        encoding="utf-8",
    )
    assert require_gate_payload(
        gate, verdicts=("PASS",), source_combined_sha256="source-a"
    )["verdict"] == "PASS"
    with pytest.raises(RuntimeError, match="different source lock"):
        require_gate_payload(gate, verdicts=("PASS",), source_combined_sha256="source-b")
    with pytest.raises(RuntimeError, match="expected"):
        require_gate_payload(gate, verdicts=("OTHER",), source_combined_sha256="source-a")

    noninterference = reference_noninterference(
        active_result_root=tmp_path / "streaming",
        optimized_result_root=tmp_path / "efficient",
        active_source_files=(tmp_path / "native_train.py",),
        modified_files=(tmp_path / "efficient_train.py",),
    )
    assert noninterference["verdict"] == "STREAMING_REFERENCE_UNTOUCHED"
    assert reference_noninterference(
        active_result_root=tmp_path / "streaming",
        optimized_result_root=tmp_path / "streaming" / "child",
        active_source_files=(tmp_path / "native_train.py",),
        modified_files=(tmp_path / "native_train.py",),
    )["passed"] is False

    child = {
        "combined_sha256": "child",
        "parent_source_combined_sha256": "parent",
    }
    parent_gate = tmp_path / "parent_gate.json"
    parent_gate.write_text(
        json.dumps(
            {
                "verdict": "K1_HOOK_PARITY_PASS",
                "source_combined_sha256": "parent",
            }
        ),
        encoding="utf-8",
    )
    assert lineage_require_gate(
        parent_gate,
        verdicts=("K1_HOOK_PARITY_PASS",),
        source_combined_sha256="child",
        child_source=child,
    )["verdict"] == "K1_HOOK_PARITY_PASS"
    parent_gate.write_text(
        json.dumps(
            {
                "verdict": "OFFLINE_K4_GATE_PASS",
                "source_combined_sha256": "parent",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="outside the child/parent"):
        lineage_require_gate(
            parent_gate,
            verdicts=("OFFLINE_K4_GATE_PASS",),
            source_combined_sha256="child",
            child_source=child,
        )


def test_two_gpu_parser_is_exact_and_does_not_choose_for_user() -> None:
    assert parse_gpu_ids("6,7") == (6, 7)
    for invalid in (None, "6", "6,7,8", "6,6", "-1,7"):
        with pytest.raises(ValueError):
            parse_gpu_ids(invalid)


def test_exact_cache_projection_and_query_deduplication() -> None:
    projection = project_exact_teacher_cache(
        query_count=26_100,
        window_count=6_525,
        free_bytes_before=2_360_234_930_176,
    )
    assert projection.condition_tokens == 122
    assert projection.condition_dim == 960
    assert projection.condition_dtype == "torch.float32"
    assert projection.projected_permanent_bytes < 80 * 2**30
    assert projection.projected_peak_temporary_bytes < 150 * 2**30
    assert projection.exact_fp32_storage_gate_pass is True
    windows = [[f"q{4 * row + age}" for age in range(4)] for row in range(3)]
    assert validate_query_windows(windows)["passed"] is True
    assert validate_query_windows([windows[0], windows[0]])["passed"] is False


def test_exact_cache_model_load_calls_match_runtime_signature() -> None:
    tree = ast.parse(inspect.getsource(exact_teacher_cache))
    allowed = set(inspect.signature(load_frozen_simvla).parameters)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_frozen_simvla"
    ]
    assert len(calls) == 2
    for call in calls:
        keywords = {item.arg for item in call.keywords if item.arg is not None}
        assert keywords == allowed


def _fake_exact_cache(root: Path) -> None:
    rank_root = root / "rank_00"
    rank_root.mkdir(parents=True)
    metadata = [
        {
            "query_id": f"q{index}",
            "task_id": 0,
            "episode_id": "episode",
            "query_index": index,
            "language_instruction": "task",
            "raw_rgb_ref": {},
            "noise_key": {
                "checkpoint": "checkpoint",
                "task_id": 0,
                "episode_id": "episode",
                "policy_query_index": index,
                "seed_base": 1,
                "seed": index,
            },
        }
        for index in range(4)
    ]
    shard = {
        "schema_version": CACHE_SHARD_SCHEMA,
        "source_combined_sha256": "source",
        "conditions": torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        "valid_masks": torch.ones(4, 2, dtype=torch.bool),
        "group_ids": torch.ones(4, 2, dtype=torch.uint8),
        "teacher_actions": torch.arange(4 * 10 * 7, dtype=torch.float32).reshape(4, 10, 7),
        "proprio": torch.zeros(4, 8),
        "noise_seeds": torch.arange(4, dtype=torch.int64),
        "metadata": metadata,
    }
    shard_path = rank_root / "queries_00000.pt"
    torch.save(shard, shard_path)
    digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    marker = {
        "schema_version": CACHE_MARKER_SCHEMA,
        "source_combined_sha256": "source",
        "file": "rank_00/queries_00000.pt",
        "sha256": digest,
        "query_ids": [f"q{index}" for index in range(4)],
        "complete": True,
    }
    (rank_root / "queries_00000.complete.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    manifest = {
        "schema_version": CACHE_SCHEMA,
        "source_combined_sha256": "source",
        "complete": True,
        "query_count": 4,
        "window_count": 1,
        "query_index": [
            {"query_id": f"q{index}", "file": marker["file"], "offset": index}
            for index in range(4)
        ],
        "windows": [["q0", "q1", "q2", "q3"]],
        "shards": [marker],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_exact_cache_identity_split_and_serialization(tmp_path: Path) -> None:
    _fake_exact_cache(tmp_path)
    store = ExactTeacherStore(tmp_path)
    assert torch.equal(
        store.query("q2")["teacher_action"],
        torch.arange(280, dtype=torch.float32).reshape(4, 10, 7)[2],
    )
    assert validate_exact_cache(tmp_path, verify_checksums=True)["passed"] is True
    dataset = ExactTeacherSequenceDataset(
        tmp_path, split="all", heldout_fraction=0.2, split_seed=20260822
    )
    payload = json.dumps(
        ((0, "episode", 0),), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    assert dataset.split_sha256 == hashlib.sha256(payload).hexdigest()
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_recursive_age_path_and_exact_uint8_conversion() -> None:
    torch.manual_seed(7)
    parent = NativeSimVLAV0(condition_dim=16, delta_dim=8, rank_dim=64, max_tokens=8)
    efficient = NativeSimVLAV0(condition_dim=16, delta_dim=8, rank_dim=64, max_tokens=8)
    efficient.load_state_dict(parent.state_dict())
    install_exact_uint8_delta_path(efficient)
    images = torch.randint(0, 256, (1, 4, 2, 12, 12, 3), dtype=torch.uint8)
    proprio = torch.randn(1, 4, 8)
    condition = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.bool)
    groups = torch.ones(1, 5, dtype=torch.long)
    expected = parent(condition, images, proprio, valid_mask=mask, group_ids=groups)
    observed = efficient(condition, images, proprio, valid_mask=mask, group_ids=groups)
    for left, right in zip(expected.conditions, observed.conditions):
        assert torch.equal(left, right)
    assert observed.conditions[1].grad_fn is not None
    assert observed.conditions[2].grad_fn is not None


def test_mode_ab_mode_d_batch_lr_and_wallclock_contracts() -> None:
    mode_ab = mode_ab_pass(
        {
            "max_total_loss_relative_difference": 0.001,
            "max_first5_loss_relative_difference": 0.001,
            "min_gradient_cosine": 0.9999,
            "max_gradient_relative_error": 0.001,
            "all_ages_represented": True,
            "all_finite": True,
            "median_speedup": 2.0,
            "mode_b_peak_vram_fits": True,
        }
    )
    assert mode_ab["verdict"] == "MODE_B_APPROVED"
    assert [balanced_mode_d_age(index) for index in range(6)] == [1, 2, 3, 1, 2, 3]
    assert mode_d_age_counts(0, 1000) == {1: 334, 2: 333, 3: 333}
    selected = effective_batch_contract(
        local_unique_batch=1,
        gradient_accumulation_steps=1,
        world_size=2,
        replicated_logical_sample=True,
    )
    assert selected["effective_unique_global_batch"] == 1
    assert selected["preserves_reference"] is True
    assert effective_batch_contract(
        local_unique_batch=2,
        gradient_accumulation_steps=1,
        world_size=2,
        replicated_logical_sample=True,
    )["preserves_reference"] is False
    assert lr_multiplier(0) == 0.0
    assert lr_multiplier(7_500) == 1.0
    assert lr_multiplier(150_000) == pytest.approx(0.1)
    assert wallclock_projection(
        mean_step_seconds=0.25,
        measured_steps=1000,
        scientific_parity_gates_pass=True,
        objective_mode_approved=True,
    )["verdict"] == "TRAIN_150K_APPROVED"
    assert wallclock_projection(
        mean_step_seconds=0.68,
        measured_steps=1000,
        scientific_parity_gates_pass=True,
        objective_mode_approved=True,
    )["verdict"] == "TRAINING_OPTIMIZATION_INSUFFICIENT"


def test_mode_d_not_required_is_measured_and_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"combined_sha256": "source"}), encoding="utf-8")
    summary = tmp_path / "mode_b.json"
    summary.write_text(
        json.dumps(
            {
                "source_combined_sha256": "source",
                "measured_steps": 1000,
                "mean_measured_step_seconds": 0.2,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "not_required.json"
    result = command_mode_d_not_required(
        Namespace(
            output=str(output),
            source_lock=str(source),
            mode_b_summary=str(summary),
            amortized_overhead_seconds=0.0,
        )
    )
    assert result["verdict"] == "MODE_D_NOT_REQUIRED"
    summary.write_text(
        json.dumps(
            {
                "source_combined_sha256": "source",
                "measured_steps": 1000,
                "mean_measured_step_seconds": 0.4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Mode D benchmark is required"):
        command_mode_d_not_required(
            Namespace(
                output=str(tmp_path / "blocked.json"),
                source_lock=str(source),
                mode_b_summary=str(summary),
                amortized_overhead_seconds=0.0,
            )
        )


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_adaln = False
        self.action_decoder = nn.Linear(8, 7)

    def forward(
        self,
        *,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        del vlm_features, proprio
        hidden = torch.cat((action_with_noise, t[:, None, None].expand(-1, 10, 1)), dim=-1)
        return self.action_decoder(hidden)


def test_generator_hidden_hook_nfe_schedules_and_generation_ages() -> None:
    transformer = _FakeTransformer()
    report = hidden_hook_parity_report(
        transformer,
        condition=torch.zeros(2, 3, 8),
        noisy_action=torch.zeros(2, 10, 7),
        proprio=torch.zeros(2, 8),
        tau=torch.ones(2),
        dt=-0.1,
    )
    assert report["verdict"] == "GENERATOR_HIDDEN_HOOK_PASS"
    assert native_nfe_time_grid(5) == (1.0, 0.8, 0.6, 0.4, 0.19999999999999996)
    for n_g, schedule in GENERATION_SCHEDULES.items():
        assert validate_generation_schedule(n_g, schedule)["matches_contract"] is True

    updater = SimVLAGenerationHiddenUpdater(
        hidden_dim=8, condition_dim=8, condition_code_dim=4, rank_dim=8
    )
    loop = SimVLAGenerationLoop(updater, transformer.action_decoder)

    def full_step(action: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.cat((action, tau[:, None, None].expand(-1, 10, 1)), dim=-1)
        return hidden, transformer.action_decoder(hidden)

    trace = loop(
        torch.zeros(2, 10, 7),
        full_step=full_step,
        full_step_indices=(0, 4, 8),
        proprio=torch.zeros(2, 8),
        condition=torch.zeros(2, 3, 8),
        condition_valid_mask=torch.ones(2, 3, dtype=torch.bool),
        condition_change_code=torch.zeros(2, 4),
    )
    assert trace.skipped_ages == (1, 2, 3, 1, 2, 3, 1)
    assert updater.parameter_audit()["under_hard_cap"] is True
    assert all(not parameter.requires_grad for parameter in loop.decoder.parameters())


def test_stage_graph_and_fixed_500_episode_manifest() -> None:
    graph = stage_graph_payload()
    assert graph["automatic_stage_launch"] is False
    assert graph["dynamic_n_g_enabled"] is False
    assert len(graph["stages"]) == len(STAGE_GRAPH)
    assert {item.stage for item in STAGE_GRAPH} == {
        *(str(index) for index in range(11)),
        *(f"G{index}" for index in range(6)),
        *(f"C{index}" for index in range(5)),
    }
    episodes = libero_long_500_episode_keys()
    assert len(episodes) == 500
    assert len({(item["task_id"], item["trial_id"]) for item in episodes}) == 500
    assert all(item["suite"] == "libero_10" for item in episodes)
    assert stage_readiness("1", {})["verdict"] == "STAGE_BLOCKED"
    assert stage_readiness("1", {"0": "STAGE0_AUDIT_PASS"})["verdict"] == "STAGE_READY"
    assert stage_readiness("1", {"0": "WRONG"})["verdict"] == "STAGE_BLOCKED"
