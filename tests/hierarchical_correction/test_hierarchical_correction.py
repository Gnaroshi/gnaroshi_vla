"""CPU-only invariants for Hierarchical Latent-Action Correction."""

from __future__ import annotations

import json
import csv
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from architectures.simvla.adapters.hierarchical_correction.cache_state import (
    SimVLAHybridCache,
)
from architectures.simvla.adapters.hierarchical_correction.simvla_hybrid_policy import (
    RealSimVLAHierarchicalCorrectionPolicy,
)
from architectures.simvla.adapters.hierarchical_correction.source_locked_loading import (
    load_source_locked_processor,
    load_source_locked_simvla,
)
from architectures.simvla.adapters.hierarchical_correction.offline_replay import (
    _anchor_endpoint_policy,
    _endpoint_policies,
    _run_endpoint_lightweight_query,
)
from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.query_cache_state import (
    SimVLAQueryObservation,
    tensor_hash,
)
from architectures.simvla.adapters.latentloop.condition_adapter import (
    LatentLoopAdapterConfig,
    SimVLAChunkAwareAdapter,
)
from methods.hierarchical_correction.decisions import (
    HybridGateInputs,
    NativeHybridDecisionInputs,
    RegenerationCandidateGateInputs,
    evaluate_hybrid_gate,
    evaluate_native_hybrid_decision,
    evaluate_r5_regeneration_gate,
)
from methods.hierarchical_correction.horizon_provenance import (
    derive_provenance_schedule,
    original_tokens_remaining,
    r1_hybrid_readiness_verdict,
    simulate_token_provenance,
)
from methods.hierarchical_correction.policy_state import (
    HierarchicalPolicyState,
    validate_trace_record,
)
from methods.hierarchical_correction.provenance import hierarchical_source_manifest
from methods.hierarchical_correction.schedules import (
    ExecutionLevel,
    HierarchicalSchedule,
)
from methods.latentloop.modules.action_chunk_correction import shift_action_chunk
from tools.simvla.analyze_hierarchical_correction import _merge_scientific
from tools.simvla.analyze_native_horizon_results import run as _run_native_analysis


def _observation(value: float) -> SimVLAQueryObservation:
    return SimVLAQueryObservation(
        raw_rgb=torch.full((1, 2, 4, 4, 3), value),
        proprio=torch.full((1, 8), value),
    )


def _state_record(
    state: HierarchicalPolicyState,
    query: int,
    condition: torch.Tensor,
    action: torch.Tensor,
) -> dict[str, object]:
    level = state.schedule.level(query)
    previous = state.action_chunk_cache_hash
    return state.apply_query(
        policy_query_index=query,
        level=level,
        condition_cache_hash=tensor_hash(condition),
        action_chunk_cache_hash=tensor_hash(action),
        flow_noise_hash=f"noise-{query}" if level != ExecutionLevel.ACTION_CORRECTION else None,
        action_correction_input_chunk_hash=(
            previous if level == ExecutionLevel.ACTION_CORRECTION else None
        ),
        action_correction_residual={"all_l1": 0.1}
        if level == ExecutionLevel.ACTION_CORRECTION
        else None,
    )


def test_exact_primary_schedule_and_level_call_counts() -> None:
    schedule = HierarchicalSchedule(4, 2, 1)
    assert [int(level) for level in schedule.levels(9)] == [2, 0, 1, 0, 2, 0, 1, 0, 2]
    assert schedule.expected_calls(8) == {
        "num_policy_queries": 8,
        "num_level2_queries": 2,
        "num_level1_queries": 2,
        "num_level0_queries": 4,
        "num_full_vlm_calls": 2,
        "num_condition_updater_calls": 6,
        "num_action_transformer_decodes": 4,
        "num_action_correction_calls": 4,
    }


def test_action_shift_and_original_lineage_accounting_are_exact() -> None:
    chunk = torch.arange(70, dtype=torch.float32).reshape(1, 10, 7)
    shifted = shift_action_chunk(chunk, 5)
    assert torch.equal(shifted.actions[:, :5], chunk[:, 5:])
    assert torch.equal(shifted.actions[:, 5:], torch.zeros_like(chunk[:, 5:]))
    assert shifted.validity_mask.tolist() == [[True] * 5 + [False] * 5]
    assert [original_tokens_remaining(10, 5, age) for age in range(4)] == [10, 5, 0, 0]
    assert [original_tokens_remaining(10, 1, age) for age in range(5)] == [10, 9, 8, 7, 6]


def test_r1_provenance_rejects_unneeded_hybrid() -> None:
    pure_action = simulate_token_provenance(
        action_horizon=10,
        execution_horizon=1,
        levels=(2, 0, 0, 0, 2),
    )
    hybrid = simulate_token_provenance(
        action_horizon=10,
        execution_horizon=1,
        levels=(2, 0, 1, 0, 2),
    )
    assert [row["executed_generator_backed"] for row in pure_action] == [1, 1, 1, 1, 1]
    assert hybrid[2]["generator_backed_before_query"] == 9
    assert hybrid[2]["level1_provenance_reset"] is True
    assert r1_hybrid_readiness_verdict(
        implementation_valid=True,
        lineage_exhausted_before_regeneration=False,
        measured_regeneration_need=False,
        pure_action_dominates=True,
        regeneration_adds_action_transformer_compute=True,
    ) == "R1_HYBRID_NOT_JUSTIFIED"


def test_native_schedule_is_derived_from_h10_r5_and_level1_resets_lineage() -> None:
    schedule = derive_provenance_schedule(
        action_horizon=10,
        execution_horizon=5,
        full_refresh_interval=4,
    )
    assert schedule.first_exhaustion_query == 2
    assert schedule.regeneration_interval == 2
    assert schedule.levels_through_next_full_refresh == (2, 0, 1, 0, 2)

    pure_action = simulate_token_provenance(
        action_horizon=10,
        execution_horizon=5,
        levels=(2, 0, 0),
    )
    assert [row["executed_generator_backed"] for row in pure_action] == [5, 5, 0]

    hybrid = simulate_token_provenance(
        action_horizon=10,
        execution_horizon=5,
        levels=schedule.levels_through_next_full_refresh,
    )
    assert hybrid[2]["generator_backed_before_query"] == 5
    assert hybrid[2]["generator_backed_after_query"] == 10
    assert hybrid[2]["executed_generator_backed"] == 5
    assert hybrid[2]["level1_provenance_reset"] is True


def test_schedule_endpoints_match_existing_action_production_routes() -> None:
    condition_endpoint = HierarchicalSchedule(4, 1, 1)
    action_endpoint = HierarchicalSchedule(4, 4, 1)
    assert condition_endpoint.levels(4) == (
        ExecutionLevel.FULL_REFRESH,
        ExecutionLevel.CONDITION_REGENERATION,
        ExecutionLevel.CONDITION_REGENERATION,
        ExecutionLevel.CONDITION_REGENERATION,
    )
    assert action_endpoint.levels(4) == (
        ExecutionLevel.FULL_REFRESH,
        ExecutionLevel.ACTION_CORRECTION,
        ExecutionLevel.ACTION_CORRECTION,
        ExecutionLevel.ACTION_CORRECTION,
    )


def test_policy_state_enforces_level_specific_calls_and_serializes(tmp_path: Path) -> None:
    state = HierarchicalPolicyState(HierarchicalSchedule(4, 2, 1))
    records = []
    for query in range(4):
        condition = torch.full((1, 2, 3), float(query))
        action = torch.full((1, 10, 7), float(query))
        record = _state_record(state, query, condition, action)
        assert validate_trace_record(record) == []
        records.append(record)
    assert records[1]["condition_updater_called"]
    assert not records[1]["action_transformer_called"]
    assert not records[2]["full_condition_called"]
    assert records[2]["action_transformer_called"]
    assert records[3]["previous_action_chunk_cache_hash"] == records[2]["action_chunk_cache_hash"]
    assert state.counters == state.schedule.expected_calls(4)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state.snapshot(), sort_keys=True), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["last_action_regeneration_query"] == 2


def test_stale_action_correction_input_is_rejected() -> None:
    state = HierarchicalPolicyState(HierarchicalSchedule(4, 2, 1))
    _state_record(state, 0, torch.zeros(1, 2, 3), torch.zeros(1, 10, 7))
    with pytest.raises(AssertionError, match="stale"):
        state.apply_query(
            policy_query_index=1,
            level=ExecutionLevel.ACTION_CORRECTION,
            condition_cache_hash="condition",
            action_chunk_cache_hash="action",
            flow_noise_hash=None,
            action_correction_input_chunk_hash="wrong",
        )


def test_tensor_cache_advances_condition_and_resets_action_after_level1() -> None:
    cache = SimVLAHybridCache()
    condition0 = torch.zeros(1, 2, 960)
    action0 = torch.zeros(1, 10, 7)
    cache.full_refresh(condition0, action0, _observation(0), policy_query_index=0)
    cache.record_executed_subchunk(torch.zeros(1, 1, 7))
    inputs1 = cache.lightweight_inputs(_observation(1))
    action1 = torch.ones(1, 10, 7)
    cache.commit_lightweight(
        level=ExecutionLevel.ACTION_CORRECTION,
        condition=condition0 + 1,
        action_chunk=action1,
        observation=_observation(1),
        policy_query_index=1,
    )
    cache.record_executed_subchunk(torch.ones(1, 1, 7))
    inputs2 = cache.lightweight_inputs(_observation(2))
    assert torch.equal(inputs2["previous_query_observation"].raw_rgb, _observation(1).raw_rgb)
    regenerated = torch.full((1, 10, 7), 2.0)
    cache.commit_lightweight(
        level=ExecutionLevel.CONDITION_REGENERATION,
        condition=condition0 + 2,
        action_chunk=regenerated,
        observation=_observation(2),
        policy_query_index=2,
    )
    cache.record_executed_subchunk(torch.full((1, 1, 7), 2.0))
    inputs3 = cache.lightweight_inputs(_observation(3))
    assert inputs1["previous_action_chunk_source"] == "full_action_transformer"
    assert inputs3["previous_action_chunk_source"] == "updated_condition_action_transformer"
    assert inputs3["previous_action_chunk_hash"] == tensor_hash(regenerated)
    assert cache.last_action_regeneration_query == 2


def test_real_policy_constructor_initializes_hybrid_episode_state() -> None:
    class FakeActionSpace:
        dim_action = 7

    class FakeModel:
        num_actions = 10
        action_space = FakeActionSpace()

    shared = {
        "condition_dim": 8,
        "condition_tokens": 2,
        "observation_dim": 4,
        "action_feature_dim": 4,
        "context_dim": 4,
        "fusion_hidden_dim": 8,
        "dynamics_hidden_dim": 8,
        "rank_dim": 2,
        "action_encoder_hidden_dim": 8,
    }
    condition = SimVLAChunkAwareAdapter(
        LatentLoopAdapterConfig(variant="chunk_aware_latentloop", **shared)
    )
    correction = SimVLAChunkAwareAdapter(
        LatentLoopAdapterConfig(
            variant="action_chunk_correction",
            action_correction_hidden_dim=8,
            **shared,
        )
    )
    policy = RealSimVLAHierarchicalCorrectionPolicy(
        model=FakeModel(),
        processor=object(),
        condition_adapter=condition,
        action_correction_adapter=correction,
        full_refresh_interval=4,
        action_regeneration_interval=2,
        execution_horizon=1,
        checkpoint_id="checkpoint",
        flow_steps=10,
        image_size=32,
        client_resize_size=32,
        device=torch.device("cpu"),
        suite="libero_10",
        row_name="hierarchical_hybrid_r1_kf4_kg2",
        task_id=9,
        episode_id="episode",
        action_noise_seed_base=20260804,
    )
    assert policy.hybrid_state.next_query_index == 0
    assert policy.hybrid_cache.action_chunk is None
    assert [int(level) for level in policy.hierarchical_schedule.levels(4)] == [2, 0, 1, 0]


def test_diagnostic_noise_does_not_change_rng_or_actions() -> None:
    key = ActionNoiseKey("checkpoint", 9, "episode", 2, 20260804)
    before = torch.get_rng_state().clone()
    action_without_diagnostic = explicit_action_noise(
        key,
        batch_size=1,
        action_horizon=10,
        action_dim=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    _ = explicit_action_noise(
        ActionNoiseKey("checkpoint", 9, "teacher", 2, 20260804),
        batch_size=1,
        action_horizon=10,
        action_dim=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    action_with_diagnostic = explicit_action_noise(
        key,
        batch_size=1,
        action_horizon=10,
        action_dim=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(action_without_diagnostic, action_with_diagnostic)


def _gate_inputs(**overrides: object) -> HybridGateInputs:
    payload: dict[str, object] = {
        "k1_parity_pass": True,
        "offline_gate_pass": True,
        "hybrid_minus_action_ci95_pp": (-2.0, 1.0),
        "hybrid_minus_condition_pp": 1.0,
        "hybrid_action_transformer_calls": 50,
        "condition_action_transformer_calls": 100,
        "hybrid_amortized_policy_ms": 80.0,
        "condition_amortized_policy_ms": 100.0,
        "action_amortized_policy_ms": 60.0,
        "hybrid_gripper_reversals": 9.0,
        "action_gripper_reversals": 10.0,
        "hybrid_translation_second_difference": 0.09,
        "action_translation_second_difference": 0.10,
        "hybrid_rotation_second_difference": 0.04,
        "action_rotation_second_difference": 0.05,
        "catastrophic_task_regressions_gt20pp": 1,
        "regeneration_recovers_condition_path": True,
    }
    payload.update(overrides)
    return HybridGateInputs(**payload)  # type: ignore[arg-type]


def test_predeclared_decision_rules_and_result_serialization() -> None:
    supported = evaluate_hybrid_gate(_gate_inputs())
    assert supported["verdict"] == "HYBRID_K4_SUPPORTED"
    assert supported["k8_diagnostic_allowed"]
    json.dumps(supported, sort_keys=True)

    pure_action = evaluate_hybrid_gate(
        _gate_inputs(
            hybrid_amortized_policy_ms=95.0,
            hybrid_gripper_reversals=10.0,
            hybrid_translation_second_difference=0.10,
            hybrid_rotation_second_difference=0.05,
        )
    )
    assert pure_action["verdict"] == "PURE_ACTION_PREFERRED"

    pure_condition = evaluate_hybrid_gate(
        _gate_inputs(
            hybrid_minus_action_ci95_pp=(-8.0, -4.0),
            hybrid_minus_condition_pp=-4.0,
            action_amortized_policy_ms=120.0,
            regeneration_recovers_condition_path=False,
        )
    )
    assert pure_condition["verdict"] == "PURE_CONDITION_PREFERRED"


def _regeneration_inputs(**overrides: object) -> RegenerationCandidateGateInputs:
    payload: dict[str, object] = {
        "name": "recurrent_age2",
        "finite": True,
        "mean_prefix_l1": 0.02,
        "candidate_minus_hold_prefix_l1_ci95": (-0.02, -0.01),
        "candidate_minus_old_observation_prefix_l1_ci95": (-0.01, 0.0),
        "gripper_noncollapsed": True,
        "prefix_l1_p99": 0.08,
        "old_observation_prefix_l1_p99": 0.08,
        "level1_provenance_reset": True,
        "latency_ms_mean": 30.0,
    }
    payload.update(overrides)
    return RegenerationCandidateGateInputs(**payload)  # type: ignore[arg-type]


def test_r5_offline_gate_requires_every_frozen_check_and_selects_best_passer() -> None:
    recurrent = _regeneration_inputs()
    nonrecurrent = _regeneration_inputs(
        name="nonrecurrent_anchor_age2",
        mean_prefix_l1=0.015,
        latency_ms_mean=25.0,
    )
    passed = evaluate_r5_regeneration_gate(
        (recurrent, nonrecurrent),
        k1_parity_pass=True,
        exact_age2_pairs_present=True,
        cache_continuity_pass=True,
        same_noise_teacher_reload_pass=True,
    )
    assert passed["ONLINE_R5_GATE_PASS"] is True
    assert passed["selected_candidate"] == "nonrecurrent_anchor_age2"

    failed = evaluate_r5_regeneration_gate(
        (_regeneration_inputs(gripper_noncollapsed=False),),
        k1_parity_pass=True,
        exact_age2_pairs_present=True,
        cache_continuity_pass=True,
        same_noise_teacher_reload_pass=True,
    )
    assert failed["ONLINE_R5_GATE_PASS"] is False
    assert failed["selected_candidate"] is None


def _native_decision_inputs(**overrides: object) -> NativeHybridDecisionInputs:
    payload: dict[str, object] = {
        "k1_parity_pass": True,
        "offline_r5_gate_pass": True,
        "hybrid_minus_action_ci95_pp": (-2.0, 3.0),
        "hybrid_materially_better_after_exhaustion": True,
        "hybrid_full_vlm_calls": 50,
        "k1_full_vlm_calls": 200,
        "hybrid_action_transformer_decodes": 100,
        "condition_action_transformer_decodes": 200,
        "hybrid_gripper_reversals": 11.0,
        "better_endpoint_gripper_reversals": 10.0,
        "pure_action_as_successful_or_better": False,
        "pure_action_faster": True,
        "pure_action_long_gap_failure": True,
        "regeneration_recovers_long_gap": True,
        "scientific_matrix_complete": True,
        "hybrid_improves_success_compute_tradeoff": True,
    }
    payload.update(overrides)
    return NativeHybridDecisionInputs(**payload)  # type: ignore[arg-type]


def test_native_scientific_verdict_logic() -> None:
    assert (
        evaluate_native_hybrid_decision(_native_decision_inputs())["verdict"]
        == "NATIVE_HYBRID_SUPPORTED"
    )
    assert (
        evaluate_native_hybrid_decision(
            _native_decision_inputs(
                hybrid_materially_better_after_exhaustion=False,
                pure_action_as_successful_or_better=True,
                pure_action_long_gap_failure=False,
            )
        )["verdict"]
        == "PURE_ACTION_PREFERRED"
    )
    assert (
        evaluate_native_hybrid_decision(
            _native_decision_inputs(
                offline_r5_gate_pass=False,
                regeneration_recovers_long_gap=False,
            )
        )["verdict"]
        == "REGENERATION_MODEL_INADEQUATE"
    )
    assert (
        evaluate_native_hybrid_decision(
            _native_decision_inputs(
                hybrid_materially_better_after_exhaustion=False,
                pure_action_long_gap_failure=False,
                hybrid_improves_success_compute_tradeoff=False,
            )
        )["verdict"]
        == "NATIVE_HYBRID_NOT_SUPPORTED"
    )


def test_invalid_non_nested_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="divide"):
        HierarchicalSchedule(4, 3, 1)


def test_hierarchical_source_manifest_is_complete_and_stable() -> None:
    root = Path(__file__).resolve().parents[2]
    first = hierarchical_source_manifest(root)
    second = hierarchical_source_manifest(root)
    assert first["missing"] == []
    assert first["combined_sha256"] == second["combined_sha256"]
    assert "methods/hierarchical_correction/provenance.py" in first["files"]


def test_source_locked_loader_overrides_nested_smolvlm_path(tmp_path: Path) -> None:
    model_snapshot = tmp_path / "simvla"
    processor_snapshot = tmp_path / "smolvlm"
    model_snapshot.mkdir()
    processor_snapshot.mkdir()
    source = {
        "checkpoint": {"snapshot_path": str(model_snapshot)},
        "processor_checkpoint": {"snapshot_path": str(processor_snapshot)},
    }

    class FakeConfig:
        smolvlm_model_path = "unpinned"

        @classmethod
        def from_pretrained(cls, path: str, *, local_files_only: bool) -> "FakeConfig":
            assert path == str(model_snapshot)
            assert local_files_only
            return cls()

    class FakeModel:
        config_class = FakeConfig

        @classmethod
        def from_pretrained(
            cls,
            path: str,
            *,
            config: FakeConfig,
            local_files_only: bool,
        ) -> "FakeModel":
            assert path == str(model_snapshot)
            assert config.smolvlm_model_path == str(processor_snapshot)
            assert local_files_only
            return cls()

        def to(self, device: torch.device) -> "FakeModel":
            assert device.type == "cpu"
            return self

    class FakeProcessor:
        def __init__(self, *, smolvlm_model_path: str) -> None:
            assert smolvlm_model_path == str(processor_snapshot)
            self.path = smolvlm_model_path

    assert isinstance(
        load_source_locked_simvla(FakeModel, source, device=torch.device("cpu")),
        FakeModel,
    )
    assert load_source_locked_processor(FakeProcessor, source).path == str(processor_snapshot)


def test_endpoint_policy_objects_match_existing_lightweight_routes() -> None:
    class FakeActionSpace:
        dim_action = 7

        @staticmethod
        def normalize_state(value: torch.Tensor) -> torch.Tensor:
            return value

        @staticmethod
        def postprocess(value: torch.Tensor) -> torch.Tensor:
            return value

    class FakeModel:
        num_actions = 10
        action_space = FakeActionSpace()

        def eval(self) -> "FakeModel":
            return self

        @staticmethod
        def transformer(**kwargs: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(kwargs["action_with_noise"])

    shared = {
        "condition_dim": 8,
        "condition_tokens": 2,
        "observation_dim": 4,
        "action_feature_dim": 4,
        "context_dim": 4,
        "fusion_hidden_dim": 8,
        "dynamics_hidden_dim": 8,
        "rank_dim": 2,
        "action_encoder_hidden_dim": 8,
    }
    condition = SimVLAChunkAwareAdapter(
        LatentLoopAdapterConfig(variant="chunk_aware_latentloop", **shared)
    ).eval()
    correction = SimVLAChunkAwareAdapter(
        LatentLoopAdapterConfig(
            variant="action_chunk_correction",
            action_correction_hidden_dim=8,
            **shared,
        )
    ).eval()
    policies = _endpoint_policies(
        model=FakeModel(),
        processor=object(),
        condition_adapter=condition,
        correction_adapter=correction,
        args=Namespace(
            checkpoint="checkpoint",
            flow_steps=2,
            image_size=32,
            client_resize_size=32,
            device="cpu",
            suite="libero_10",
            action_noise_seed_base=7,
        ),
        task_id=9,
        episode_id="episode",
    )
    anchor_condition = torch.randn(1, 2, 8)
    anchor_action = torch.randn(1, 10, 7)
    anchor_rgb = torch.zeros(1, 2, 32, 32, 3, dtype=torch.uint8)
    anchor_proprio = torch.zeros(1, 8)
    lightweight_names = (
        "condition_reference",
        "condition_hybrid",
        "action_reference",
        "action_hybrid",
    )
    for name in lightweight_names:
        policy = policies[name]
        _anchor_endpoint_policy(
            policy,
            condition=anchor_condition,
            action=anchor_action,
            observation_rgb=anchor_rgb,
            proprio=anchor_proprio,
            query_index=0,
            flow_noise_hash="noise",
        )
    record = {
        "executed_subchunk": torch.zeros(1, 7),
        "next_raw_rgb": torch.ones(2, 32, 32, 3, dtype=torch.uint8),
        "next_proprio": torch.ones(8),
    }
    with torch.no_grad():
        for name in lightweight_names:
            policy = policies[name]
            _run_endpoint_lightweight_query(
                policy,
                record=record,
                device=torch.device("cpu"),
            )
    assert torch.equal(
        policies["condition_reference"].cached_condition,
        policies["condition_hybrid"].cached_condition,
    )
    assert torch.equal(
        policies["condition_reference"].cached_action_chunk,
        policies["condition_hybrid"].cached_action_chunk,
    )
    assert torch.equal(
        policies["action_reference"].cached_action_chunk,
        policies["action_hybrid"].cached_action_chunk,
    )


def test_scientific_task_shards_merge_to_exact_200_episode_rows(tmp_path: Path) -> None:
    row_names = (
        "full_k1",
        "chunk_aware_latentloop_k4",
        "action_chunk_correction_k4",
        "hierarchical_hybrid_r1_kf4_kg2",
    )
    shards = []
    for shard_index, tasks in enumerate((range(0, 5), range(5, 10))):
        shard = tmp_path / f"shard{shard_index}"
        shard.mkdir()
        shards.append(shard)
        source = {
            "root_commit": "root",
            "simvla_upstream_commit": "upstream",
            "norm_stats_sha256": "norm",
            "checkpoint": {"revision": "revision", "hf_blob_key_sha256": "blob"},
            "hierarchical_checkpoints": {"condition": "c", "action": "a"},
            "packages": {"mujoco": "2.3.7"},
            "torch": "2",
            "torch_cuda": "12",
        }
        (shard / "source_lock.json").write_text(json.dumps(source), encoding="utf-8")
        summary_rows = {
            name: {
                "counters": {
                    "num_env_steps": len(tasks) * 20,
                    "num_action_transformer_decodes": len(tasks) * 20,
                }
            }
            for name in row_names
        }
        summary = {
            "matrix": "scientific_r1_k4",
            "suite": "libero_10",
            "task_ids": list(tasks),
            "rows": summary_rows,
            "parameter_audit": {"combined_adapter_parameters": 686011},
            "query_trace_jsonl": str(shard / "query_trace.jsonl"),
        }
        (shard / "online_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with (shard / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("row", "task_id", "episode", "success"))
            writer.writeheader()
            for name in row_names:
                for task in tasks:
                    for episode in range(20):
                        writer.writerow(
                            {
                                "row": name,
                                "task_id": task,
                                "episode": episode,
                                "success": episode != 0,
                            }
                        )
        samples = {
            "schema_version": "simvla_hierarchical_metric_samples_v2",
            "latencies": {
                name: {"policy_total_ms": [1.0] * (len(tasks) * 20)} for name in row_names
            },
            "action_diagnostics": {
                name: {
                    "gripper_reversals": [1.0] * (len(tasks) * 20),
                    "translation_second_difference": [0.1] * (len(tasks) * 20),
                    "rotation_second_difference": [0.05] * (len(tasks) * 20),
                }
                for name in row_names
            },
            "condition_action_tracking_by_query_age": {name: {} for name in row_names},
            "correction_residual_records": {name: [] for name in row_names},
            "condition_drift_records": {
                name: [
                    {"query_age": 1, "condition_cache_drift": {"l1": 0.01}}
                    for _ in range(len(tasks) * 20)
                ]
                for name in row_names
            },
        }
        torch.save(samples, shard / "metric_samples.pt")
        (shard / "query_trace.jsonl").write_text("", encoding="utf-8")

    output = tmp_path / "merged"
    result = _merge_scientific(
        Namespace(output=str(output), shard=[str(path) for path in shards], bootstrap_seed=7)
    )
    assert result["episodes_per_row"] == 200
    assert all(row["episodes"] == 200 for row in result["rows"].values())
    assert (output / "online_summary.json").is_file()
    assert json.loads((output / "online_summary.json").read_text())["task_ids"] == list(range(10))


def test_native_result_aggregation_applies_frozen_late_episode_rule(tmp_path: Path) -> None:
    row_names = (
        "native_full_simvla_k1",
        "native_action_correction_kf4",
        "native_condition_regeneration_kf4",
        "native_horizon_hybrid_kf4_kg2",
        "native_stale_action_chunk_kf4",
    )
    episode_path = tmp_path / "episodes.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("row", "task_id", "episode", "success", "policy_queries"),
        )
        writer.writeheader()
        for row in row_names:
            for task in range(10):
                for episode in range(20):
                    success = row != "native_action_correction_kf4"
                    writer.writerow(
                        {
                            "row": row,
                            "task_id": task,
                            "episode": episode,
                            "success": success,
                            "policy_queries": 3,
                        }
                    )
    rows = {
        row: {
            "episodes": 200,
            "success_rate": 0.0 if row == "native_action_correction_kf4" else 1.0,
            "amortized_policy_ms_per_environment_action": {
                "native_full_simvla_k1": 100.0,
                "native_action_correction_kf4": 30.0,
                "native_condition_regeneration_kf4": 80.0,
                "native_horizon_hybrid_kf4_kg2": 50.0,
                "native_stale_action_chunk_kf4": 10.0,
            }[row],
            "counters": {
                "num_full_vlm_calls": 200 if row == "native_full_simvla_k1" else 50,
                "num_action_transformer_decodes": (
                    200 if row == "native_condition_regeneration_kf4" else 100
                ),
            },
            "action_diagnostics": {"gripper_reversals": {"mean": 10.0}},
        }
        for row in row_names
    }
    signature = {"source": "same"}
    summary = {
        "matrix": "native_r5",
        "suite": "libero_10",
        "task_ids": list(range(10)),
        "episodes_per_row": 200,
        "rows": rows,
        "paired": {
            "hybrid_minus_action_correction": {
                "task_hierarchical_paired_ci95_pp": [-2.0, 5.0]
            }
        },
        "source_signature": signature,
        "episode_metrics_csv": str(episode_path),
    }
    gate = {
        "ONLINE_R5_GATE_PASS": True,
        "R5_REGENERATION_GATE_PASS": True,
        "first_provenance_exhaustion_query": 2,
        "prerequisites": {"k1_parity": True},
        "source_signature": signature,
    }
    summary_path = tmp_path / "online.json"
    gate_path = tmp_path / "gate.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    result = _run_native_analysis(
        Namespace(
            output=str(tmp_path / "decision"),
            online_summary=str(summary_path),
            r5_gate=str(gate_path),
            bootstrap_seed=7,
            minimum_late_pairs=50,
            minimum_late_improvement_pp=3.0,
        )
    )
    assert result["verdict"] == "NATIVE_HYBRID_SUPPORTED"
    assert result["late_paired_outcomes"]["pairs"] == 200
