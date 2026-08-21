from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "openpi"))

from architectures.openpi.adapters.latentloop.cache_contract_v2 import (  # noqa: E402
    SPLIT_ROLES,
    SUITES,
    array_hash,
    canonical_payload_hash,
    load_final_evaluation_manifest,
    load_split_contract,
    resolve_task_identity,
    tensor_contract_from_record,
    tree_hash,
    validate_record_v2,
)
from architectures.openpi.adapters.latentloop.cache_io import EpisodeCacheWriter  # noqa: E402
from architectures.openpi.adapters.latentloop.full_cache_contract_v2 import (  # noqa: E402
    build_full_cache_inventory,
    expected_query_spec,
    load_full_cache_inventory,
)
from architectures.openpi.adapters.latentloop.losses import composition_loss  # noqa: E402
from architectures.openpi.adapters.latentloop.prefix_kv_hook import (  # noqa: E402
    PrefixEmbeddingState,
    PrefixKVState,
)
from architectures.openpi.adapters.latentloop.streaming_teacher import (  # noqa: E402
    StreamingTeacherConfig,
    build_streaming_episode_plan,
    build_streaming_provenance,
    deterministic_episode_order,
    iter_rolling_v0_examples,
)
from architectures.openpi.adapters.latentloop.transition_core import OpenPITransitionOutput  # noqa: E402
from architectures.openpi.adapters.latentloop.v0_recursive_unroll import (  # noqa: E402
    V0AgeInput,
    recursive_v0_unroll,
)
from architectures.openpi.adapters.latentloop.v2_schedule_state import V2ScheduleState  # noqa: E402
from architectures.openpi.adapters.latentloop.variable_time_policy import (  # noqa: E402
    VariableTimeStateManager,
)
from methods.variable_time_latentloop.decisions import RefreshDecision  # noqa: E402
from methods.variable_time_latentloop.operation_counters_v2 import (  # noqa: E402
    OperationCountersV2,
    full_hook_query,
    latent_query,
)
from simulate_dynamic_budget_v2 import aggregate_simulations, simulate_episode  # noqa: E402
import defect_split_common  # noqa: E402
import aggregate_pi05_scientific_row_v2  # noqa: E402
import pi05_stage_gate_v2  # noqa: E402
import source_lock_v2  # noqa: E402
import verify_pi05_dynamic_threshold_lock_v2  # noqa: E402


def _state(value: float = 0.0) -> PrefixKVState:
    embeddings = torch.full((1, 3, 8), value)
    return PrefixKVState(
        embeddings=embeddings,
        pad_mask=torch.ones(1, 3, dtype=torch.bool),
        attention_pattern=torch.zeros(1, 3, dtype=torch.bool),
        position_ids=torch.arange(3).view(1, -1),
        pre_rope_keys=(torch.full((1, 1, 3, 4), value),) * 2,
        values=(torch.full((1, 1, 3, 4), value),) * 2,
    )


def _prefix(value: float) -> PrefixEmbeddingState:
    state = _state(value)
    return PrefixEmbeddingState(
        state.embeddings, state.pad_mask, state.attention_pattern, state.position_ids
    )


def _final_manifest() -> dict:
    tasks = []
    episodes = []
    for suite_offset, suite in enumerate(SUITES):
        for task in range(10):
            tasks.append(
                {
                    "suite": suite,
                    "benchmark_task_index": task,
                    "dataset_task_index": suite_offset * 10 + task,
                    "canonical_task_name": f"{suite}/task{task}",
                    "canonical_instruction": f"instruction {suite_offset} {task}",
                }
            )
            for trial in range(50):
                episodes.append(
                    {
                        "suite": suite,
                        "benchmark_task_index": task,
                        "episode_namespace": "final_scientific_evaluation",
                        "trial": trial,
                        "environment_seed": 7,
                        "initial_state_identifier": f"final-{suite}-{task}-{trial}",
                        "query_noise_key_prefix": f"7:{suite}:{task}:{trial}:",
                        "max_episode_steps": 520 if suite == "libero_10" else 300,
                    }
                )
    payload = {
        "schema_version": 2,
        "frozen": True,
        "source_lock_id": "lock-v2",
        "protocol": {},
        "tasks": tasks,
        "episodes": episodes,
    }
    payload["manifest_id"] = canonical_payload_hash(payload, "manifest_id")
    return payload


def _split_contract(final: dict) -> dict:
    assignments = []
    for index, role in enumerate(SPLIT_ROLES):
        assignments.append(
            {
                "suite": "libero_10",
                "benchmark_task_index": index,
                "episode_namespace": "teacher_demonstration",
                "episode_id": str(index),
                "environment_seed": "not-recorded",
                "initial_state_identifier": f"teacher-{index}",
                "dataset_frame_start": index * 100,
                "dataset_frame_stop": index * 100 + 26,
                "role": role,
            }
        )
    payload = {
        "schema_version": 2,
        "frozen": True,
        "source_lock_id": final["source_lock_id"],
        "final_manifest_id": final["manifest_id"],
        "required_cache_roles": list(SPLIT_ROLES),
        "assignments": assignments,
    }
    payload["split_contract_id"] = canonical_payload_hash(payload, "split_contract_id")
    return payload


def _record() -> tuple[dict, dict]:
    raw = {"base": torch.zeros(8, 8, 3, dtype=torch.uint8)}
    preprocessed = {"base": torch.zeros(3, 8, 8, dtype=torch.float16)}
    noise = torch.arange(320, dtype=torch.float32).view(10, 32) / 100
    executed = torch.zeros(5, 7)
    record = {
        "suite": "libero_10",
        "benchmark_task_index": 0,
        "dataset_task_index": 30,
        "canonical_task_name": "libero_10/task0",
        "canonical_instruction": "instruction 3 0",
        "episode_namespace": "teacher_demonstration",
        "episode_id": "demo-0",
        "environment_seed": "not-recorded",
        "initial_state_identifier": "teacher-state-0",
        "policy_query_index": 0,
        "absolute_environment_step": 0,
        "raw_images": raw,
        "raw_image_identity": tree_hash(raw),
        "preprocessed_images": preprocessed,
        "preprocessed_image_hash": tree_hash(preprocessed),
        "robot_state_raw": torch.zeros(8),
        "robot_state_normalized": torch.zeros(32),
        "prefix_embeddings": torch.zeros(5, 11),
        "pre_rope_keys": torch.zeros(3, 2, 5, 7),
        "values": torch.zeros(3, 2, 5, 7),
        "prefix_pad_mask": torch.ones(5, dtype=torch.bool),
        "prefix_attention_pattern": torch.zeros(5, dtype=torch.bool),
        "prefix_position_ids": torch.arange(5),
        "action_noise": noise,
        "action_noise_seed": 17,
        "action_noise_hash": array_hash(noise),
        "teacher_action_chunk_normalized": torch.zeros(10, 32),
        "teacher_action_chunk_postprocessed": torch.zeros(10, 7),
        "executed_actions_postprocessed": executed,
        "executed_action_length": 5,
        "gripper_conversion": "LiberoOutputs continuous source-correct 7D action; no binary target rewrite",
        "next_query_observation": {"state": torch.zeros(8)},
        "source_lock_id": "lock-v2",
        "source_hashes": {
            name: name
            for name in ("source", "checkpoint", "norm_stats", "config", "preprocessing", "postprocessing")
        },
        "timing_ms": {},
    }
    catalog = {
        ("libero_10", 0): {
            "dataset_task_index": 30,
            "canonical_task_name": "libero_10/task0",
            "canonical_instruction": "instruction 3 0",
        }
    }
    return record, catalog


def test_source_lock_v2_fails_closed_on_mismatch(tmp_path: Path, monkeypatch):
    sections = {
        "schema_version": 2,
        "repository": {"head": "a"},
        "upstream": {"head": "b"},
        "nested_libero": {"head": "c"},
        "ours_and_upstream_source": {"combined_sha256": "c"},
        "checkpoint": {"directory": "/x", "config_path": "/y"},
        "normalization": {"path": "/z"},
        "environment": {"python": "3.11"},
        "native_intervals": {"action_horizon_h": 10, "execution_horizon_r": 5},
        "preprocessing": {"combined_sha256": "p"},
        "postprocessing": {"combined_sha256": "q"},
    }
    locked = dict(sections)
    locked["source_lock_id"] = source_lock_v2._canonical_hash(locked)
    locked["source_lock_v2_pass"] = True
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(locked))
    observed = dict(locked)
    monkeypatch.setattr(source_lock_v2, "collect_current", lambda **_kwargs: observed)
    assert source_lock_v2.verify_lock(path)["source_lock_v2_pass"]
    observed = {**observed, "environment": {"python": "3.12"}}
    monkeypatch.setattr(source_lock_v2, "collect_current", lambda **_kwargs: observed)
    with pytest.raises(source_lock_v2.SourceLockError, match="environment_mismatch"):
        source_lock_v2.verify_lock(path)
    with pytest.raises(source_lock_v2.SourceLockError, match="missing evidence"):
        source_lock_v2.verify_lock(tmp_path / "missing.json")


def test_source_lock_requires_trusted_vendored_libero_loader_opt_in(monkeypatch):
    common = (ROOT / "architectures/openpi/wrappers/latentloop_common.sh").read_text()
    assert "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in common
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    with pytest.raises(source_lock_v2.SourceLockError, match="environment mismatch"):
        source_lock_v2.environment_identity()
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    assert source_lock_v2.environment_identity()["torch_force_no_weights_only_load"] == "1"


def test_dynamic_cache_contract_and_finiteness_are_not_hard_coded():
    record, catalog = _record()
    contract = tensor_contract_from_record(record)
    assert contract["layer_count"] == 3
    assert contract["prefix_sequence_length"] == 5
    assert contract["head_dim"] == 7
    validate_record_v2(
        record,
        expected_contract=contract,
        expected_source_lock_id="lock-v2",
        expected_source_hashes=record["source_hashes"],
        task_catalog=catalog,
        execution_horizon=5,
    )
    record["values"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_record_v2(
            record,
            expected_contract=contract,
            expected_source_lock_id="lock-v2",
            expected_source_hashes=record["source_hashes"],
            task_catalog=catalog,
            execution_horizon=5,
        )


def test_cache_record_position_ids_follow_mask_cumsum():
    record, catalog = _record()
    contract = tensor_contract_from_record(record)
    record["prefix_position_ids"][-1] += 1
    with pytest.raises(ValueError, match="cumsum"):
        validate_record_v2(
            record,
            expected_contract=contract,
            expected_source_lock_id="lock-v2",
            expected_source_hashes=record["source_hashes"],
            task_catalog=catalog,
            execution_horizon=5,
        )


def test_schema_v2_cache_manifest_has_a_self_hash(tmp_path: Path):
    record = {
        key: None
        for key in __import__(
            "architectures.openpi.adapters.latentloop.cache_io",
            fromlist=["REQUIRED_RECORD_KEYS"],
        ).REQUIRED_RECORD_KEYS
    }
    record.update(suite="libero_10", task_id=0, episode_id=0)
    writer = EpisodeCacheWriter(tmp_path / "cache", {"schema_version": 2})
    writer.write_episode(
        [record], suite="libero_10", task_id=0, episode_id=0, split="train"
    )
    manifest = json.loads(writer.finalize().read_text())
    expected = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "cache_manifest_id"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert manifest["cache_manifest_id"] == expected


def test_suite_task_identity_and_final_manifest_exclusion(tmp_path: Path):
    final = _final_manifest()
    final_path = tmp_path / "final.json"
    final_path.write_text(json.dumps(final))
    loaded = load_final_evaluation_manifest(final_path)
    task = resolve_task_identity(30, " Instruction   3  0 ", loaded)
    assert task["suite"] == "libero_10" and task["benchmark_task_index"] == 0
    split = _split_contract(final)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    mapping, _ = load_split_contract(split_path, loaded)
    assert len(mapping) == 5
    split["assignments"][0]["initial_state_identifier"] = final["episodes"][1500][
        "initial_state_identifier"
    ]
    split["assignments"][0]["environment_seed"] = 7
    split["split_contract_id"] = canonical_payload_hash(split, "split_contract_id")
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="physically overlaps"):
        load_split_contract(split_path, loaded)


def test_split_contract_must_name_exact_final_manifest(tmp_path: Path):
    final = _final_manifest()
    final_path = tmp_path / "final.json"
    final_path.write_text(json.dumps(final))
    split = _split_contract(final)
    split["final_manifest_id"] = "another-final-manifest"
    split["split_contract_id"] = canonical_payload_hash(split, "split_contract_id")
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="does not name"):
        load_split_contract(split_path, load_final_evaluation_manifest(final_path))


def test_full_cache_inventory_reconstructs_every_query_and_detects_tampering(tmp_path: Path):
    final = _final_manifest()
    split = _split_contract(final)
    final_path = tmp_path / "final.json"
    split_path = tmp_path / "split.json"
    final_path.write_text(json.dumps(final))
    split_path.write_text(json.dumps(split))
    payload = build_full_cache_inventory(
        source_lock_id="lock-v2",
        split_contract=split,
        final_manifest=final,
        split_contract_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        final_manifest_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
        num_shards=4,
    )
    assert payload["statistics"]["episodes"] == 5
    assert payload["statistics"]["queries"] == 25
    first = payload["episodes"][0]
    query = expected_query_spec(first, 4, execution_horizon=5, noise_seed_base=20260820)
    assert query["absolute_environment_step"] == 20
    assert query["dataset_frame_index"] == first["dataset_frame_start"] + 20
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(payload))
    loaded = load_full_cache_inventory(
        inventory_path,
        source_lock_id="lock-v2",
        split_contract=split,
        final_manifest=final,
        split_contract_path=split_path,
        final_manifest_path=final_path,
    )
    assert loaded["inventory_id"] == payload["inventory_id"]
    payload["episodes"][0]["query_count"] += 1
    payload["inventory_id"] = canonical_payload_hash(payload, "inventory_id")
    inventory_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="deterministic split-derived"):
        load_full_cache_inventory(
            inventory_path,
            source_lock_id="lock-v2",
            split_contract=split,
            final_manifest=final,
            split_contract_path=split_path,
            final_manifest_path=final_path,
        )


def test_streaming_v0_plan_is_deterministic_and_uses_no_persistent_teacher_tensors(
    tmp_path: Path,
):
    final = _final_manifest()
    split = _split_contract(final)
    final_path = tmp_path / "final.json"
    split_path = tmp_path / "split.json"
    final_path.write_text(json.dumps(final))
    split_path.write_text(json.dumps(split))
    source_lock = {
        "source_lock_id": "lock-v2",
        "checkpoint": {
            "model_sha256": "model-sha",
            "config_sha256": "config-sha",
        },
        "normalization": {"sha256": "norm-sha"},
    }
    config = StreamingTeacherConfig()
    plan = build_streaming_episode_plan(split)
    assert len(plan) == 5
    assert sum(row.query_count for row in plan) == 25
    assert sum(row.window_count for row in plan) == 10
    assert deterministic_episode_order(plan, seed=42, epoch=0) == deterministic_episode_order(
        plan, seed=42, epoch=0
    )
    provenance = build_streaming_provenance(
        source_lock=source_lock,
        checkpoint=tmp_path / "checkpoint",
        final_manifest=final,
        final_manifest_path=final_path,
        split_contract=split,
        split_contract_path=split_path,
        config=config,
    )
    assert provenance["training_source_mode"] == "online_frozen_teacher_rolling_v0"
    assert provenance["persistent_teacher_tensor_bytes"] == 0
    assert provenance["rolling_window_teacher_records"] == 4
    assert provenance["maximum_transient_teacher_records_in_trainer"] == 10
    assert provenance["statistics"]["v0_windows"] == 10


def test_streaming_v0_rolling_window_contains_exact_four_consecutive_queries():
    records = [{"query_index": index} for index in range(6)]
    examples = list(iter_rolling_v0_examples(records))
    assert [[row["query_index"] for row in example["records"]] for example in examples] == [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ]
    assert all(example["delta_q"] == 3 for example in examples)


def test_streaming_v0_stages_and_wrappers_do_not_require_a_tensor_cache():
    raw_requirements = set(pi05_stage_gate_v2.STAGE_REQUIREMENTS["stage3_v0_streaming_raw_loss"])
    assert raw_requirements == {
        "REAL_KV_ROUNDTRIP_PASS",
        "K1_ACTION_PARITY_PASS",
        "K1_EPISODE_PARITY_PASS",
        "BASE_FREEZE_PASS",
    }
    train_requirements = set(pi05_stage_gate_v2.STAGE_REQUIREMENTS["stage3_v0_streaming"])
    assert "V0_STREAMING_LOSS_WEIGHTS_APPROVED" in train_requirements
    acceptance = (
        ROOT / "architectures/openpi/wrappers/accept_pi05_v0_streaming.sh"
    ).read_text()
    training = (
        ROOT / "architectures/openpi/wrappers/train_pi05_v0_streaming.sh"
    ).read_text()
    cli = (ROOT / "tools/openpi/train_pi05_v0_streaming.py").read_text()
    assert "generate_pi05_latentloop_cache" not in acceptance
    assert "FULL_CACHE_SCHEMA_V2_PASS" not in acceptance
    assert "--raw-loss-only" in acceptance
    assert "--cache" not in training
    assert "--action-execution-mode" in training
    assert "validate_pi05_cache_v2" not in cli


class _RecursiveAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.increment = nn.Parameter(torch.tensor(0.25))
        self.previous_states = []

    def forward(self, previous_state, current_prefix, *_args, **_kwargs):
        self.previous_states.append(previous_state)
        keys = tuple(value + self.increment for value in previous_state.pre_rope_keys)
        values = tuple(value + self.increment for value in previous_state.values)
        state = PrefixKVState(
            embeddings=current_prefix.embeddings,
            pad_mask=current_prefix.pad_mask,
            attention_pattern=current_prefix.attention_pattern,
            position_ids=current_prefix.position_ids,
            pre_rope_keys=keys,
            values=values,
        )
        encoded = torch.stack(keys, dim=1)
        return OpenPITransitionOutput(state, encoded, encoded, self.increment.view(1), self.increment.view(1))


def test_v0_ages_two_and_three_consume_predicted_states_with_full_gradient():
    adapter = _RecursiveAdapter()
    anchor = _state(0.0)
    steps = [
        V0AgeInput(
            current_prefix=_prefix(float(age)),
            executed_actions=torch.full((1, 5, 7), float(age)),
            robot_state=torch.zeros(1, 4),
            target_state=_state(float(age)),
            action_noise=torch.zeros(1, 10, 32),
            teacher_action_chunk=torch.zeros(1, 10, 32),
        )
        for age in (1, 2, 3)
    ]
    outputs = recursive_v0_unroll(adapter, anchor, steps)
    assert adapter.previous_states[1] is outputs[0].transition.state
    assert adapter.previous_states[2] is outputs[1].transition.state
    assert [item.consumed_predicted_previous for item in outputs] == [False, True, True]
    outputs[-1].transition.state.pre_rope_keys[0].sum().backward()
    assert adapter.increment.grad is not None and adapter.increment.grad.item() != 0


class _CausalAdapter:
    def __init__(self):
        self.calls = []

    def __call__(self, previous_state, current_prefix, _previous_embeddings, actions, _robot, **kwargs):
        self.calls.append({"previous_state": previous_state, "actions": actions.clone(), **kwargs})
        value = float(kwargs["delta_q"])
        state = _state(value)
        state = PrefixKVState(
            current_prefix.embeddings,
            current_prefix.pad_mask,
            current_prefix.attention_pattern,
            current_prefix.position_ids,
            state.pre_rope_keys,
            state.values,
        )
        encoded = torch.full((1, 2, 3, 4), value)
        return OpenPITransitionOutput(state, encoded, encoded, torch.ones(1), actions.mean(dim=1))


def test_v1_uses_actual_ordered_actions_and_level1_resets_only_recurrent_state():
    adapter = _CausalAdapter()
    manager = VariableTimeStateManager(adapter, execution_horizon=5)
    anchor = _state(0.0)
    manager.reset(anchor, 0)
    actions1 = torch.arange(35, dtype=torch.float32).view(1, 5, 7)
    first = manager.step(_prefix(1.0), torch.zeros(1, 4), actions1, 1)
    assert torch.equal(adapter.calls[0]["actions"], actions1)
    assert torch.equal(adapter.calls[1]["actions"], actions1)
    manager.direct_reanchor(first, query_index=1)
    assert manager.anchor.state is anchor
    assert manager.sequential_state is first.direct.state
    assert manager.sequential_action_history == []
    actions2 = torch.full((1, 5, 7), 9.0)
    second = manager.step(_prefix(2.0), torch.zeros(1, 4), actions2, 2)
    assert adapter.calls[-2]["actions"].shape[1] == 5
    assert adapter.calls[-1]["actions"].shape[1] == 10
    assert second.input_provenance["forbidden_inputs_present"] == {
        "current_full_kv": False,
        "future_observation": False,
        "future_action": False,
        "final_success_label": False,
    }


def test_v1_composition_loss_remains_connected_to_trainable_values():
    direct = nn.Parameter(torch.tensor([[[[1.0, 2.0]]]]))
    composed = direct * torch.tensor(1.5)
    loss = composition_loss(direct, composed)
    loss.backward()
    assert direct.grad is not None
    assert torch.isfinite(direct.grad).all() and direct.grad.abs().sum() > 0


def test_defect_fit_validity_scheduler_and_final_roles_are_disjoint(tmp_path: Path):
    cache_split = tmp_path / "cache_split.json"
    final_manifest = tmp_path / "final_manifest.json"
    cache_split.write_text("cache-split")
    final_manifest.write_text("final-manifest")
    roles = {}
    for index, role in enumerate(defect_split_common.ROLES):
        roles[role] = [
            {
                "suite": "libero_10",
                "benchmark_task_index": index,
                "episode_namespace": role,
                "episode_id": "0",
            }
        ]
    payload = {
        "schema_version": 2,
        "frozen": True,
        "source_lock_id": "x",
        "cache_split_contract": str(cache_split),
        "cache_split_contract_id": "cache-split-id",
        "cache_split_contract_sha256": hashlib.sha256(cache_split.read_bytes()).hexdigest(),
        "final_evaluation_manifest": str(final_manifest),
        "final_manifest_id": "final-manifest-id",
        "final_evaluation_manifest_sha256": hashlib.sha256(
            final_manifest.read_bytes()
        ).hexdigest(),
        "roles": roles,
    }
    payload["defect_split_contract_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "defect.json"
    path.write_text(json.dumps(payload))
    role_sets, _ = defect_split_common.load_contract(path)
    assert all(role_sets[left].isdisjoint(role_sets[right]) for left in roles for right in roles if left != right)
    payload["roles"]["defect_validity"] = payload["roles"]["defect_fit"]
    payload["defect_split_contract_id"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "defect_split_contract_id"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="overlap"):
        defect_split_common.load_contract(path)


def test_offline_summary_is_bound_to_role_metrics_and_producer_inputs(tmp_path: Path):
    cache_split = tmp_path / "cache_split.json"
    final_manifest = tmp_path / "final_manifest.json"
    adapter = tmp_path / "adapter.pt"
    cache_manifest = tmp_path / "cache_manifest.json"
    metrics = tmp_path / "metrics.csv"
    for path, content in (
        (cache_split, "cache-split"),
        (final_manifest, "final-manifest"),
        (adapter, "adapter"),
        (cache_manifest, "cache"),
        (metrics, "suite,benchmark_task_index,episode_namespace,episode_id\nlibero_10,0,teacher_demonstration,0\n"),
    ):
        path.write_text(content)
    roles = {
        role: [
            {
                "suite": "libero_10",
                "benchmark_task_index": index,
                "episode_namespace": (
                    "final_scientific_evaluation"
                    if role == "final_scientific_evaluation"
                    else "teacher_demonstration"
                ),
                "episode_id": "0",
            }
        ]
        for index, role in enumerate(defect_split_common.ROLES)
    }
    contract = {
        "schema_version": 2,
        "frozen": True,
        "source_lock_id": "lock-v2",
        "cache_split_contract": str(cache_split),
        "cache_split_contract_id": "cache-id",
        "cache_split_contract_sha256": hashlib.sha256(cache_split.read_bytes()).hexdigest(),
        "final_evaluation_manifest": str(final_manifest),
        "final_manifest_id": "final-id",
        "final_evaluation_manifest_sha256": hashlib.sha256(
            final_manifest.read_bytes()
        ).hexdigest(),
        "roles": roles,
    }
    contract["defect_split_contract_id"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract_path = tmp_path / "defect_contract.json"
    contract_path.write_text(json.dumps(contract))
    summary = {
        "complete": True,
        "split": "defect_fit",
        "source_lock_id": "lock-v2",
        "offline_metrics": str(metrics),
        "offline_metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
        "split_contract": str(cache_split),
        "split_contract_id": "cache-id",
        "split_contract_sha256": hashlib.sha256(cache_split.read_bytes()).hexdigest(),
        "adapter_checkpoint": str(adapter),
        "adapter_checkpoint_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "base_checkpoint": "/frozen/base",
        "base_checkpoint_sha256": "base-sha",
        "cache_manifest": str(cache_manifest),
        "cache_manifest_id": "cache-manifest-id",
        "cache_manifest_sha256": hashlib.sha256(cache_manifest.read_bytes()).hexdigest(),
        "final_evaluation_manifest": str(final_manifest),
        "final_evaluation_manifest_sha256": hashlib.sha256(
            final_manifest.read_bytes()
        ).hexdigest(),
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    observed = defect_split_common.verify_offline_summary(
        summary_path,
        metrics_path=metrics,
        contract_path=contract_path,
        role="defect_fit",
        source_lock_id="lock-v2",
    )
    assert observed["adapter_checkpoint"] == str(adapter)
    summary["split"] = "defect_validity"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="not role defect_fit"):
        defect_split_common.verify_offline_summary(
            summary_path,
            metrics_path=metrics,
            contract_path=contract_path,
            role="defect_fit",
            source_lock_id="lock-v2",
        )


def test_dynamic_threshold_consumer_revalidates_budget_and_inputs(tmp_path: Path, monkeypatch):
    cache_split = tmp_path / "cache_split.json"
    final_manifest = tmp_path / "final_manifest.json"
    scheduler_metrics = tmp_path / "scheduler.csv"
    scheduler_summary = tmp_path / "scheduler_summary.json"
    defect_fit = tmp_path / "fit.json"
    defect_validity = tmp_path / "validity.json"
    adapter = tmp_path / "adapter.pt"
    for path in (
        cache_split,
        final_manifest,
        scheduler_metrics,
        scheduler_summary,
        defect_fit,
        defect_validity,
        adapter,
    ):
        path.write_text(path.name)
    roles = {
        role: [
            {
                "suite": "libero_10",
                "benchmark_task_index": index,
                "episode_namespace": role,
                "episode_id": "0",
            }
        ]
        for index, role in enumerate(defect_split_common.ROLES)
    }
    split = {
        "schema_version": 2,
        "frozen": True,
        "source_lock_id": "lock-v2",
        "cache_split_contract": str(cache_split),
        "cache_split_contract_id": "cache-id",
        "cache_split_contract_sha256": hashlib.sha256(cache_split.read_bytes()).hexdigest(),
        "final_evaluation_manifest": str(final_manifest),
        "final_manifest_id": "final-id",
        "final_evaluation_manifest_sha256": hashlib.sha256(
            final_manifest.read_bytes()
        ).hexdigest(),
        "roles": roles,
    }
    split["defect_split_contract_id"] = hashlib.sha256(
        json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    adapter_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
    payload = {
        "schema_version": 2,
        "frozen": True,
        "DYNAMIC_BUDGET_LOCK_PASS": True,
        "source_lock_id": "lock-v2",
        "model_checkpoint": str(adapter),
        "model_checkpoint_sha256": adapter_hash,
        "scheduler_calibration_manifest_sha256": hashlib.sha256(
            split_path.read_bytes()
        ).hexdigest(),
        "scheduler_metrics": str(scheduler_metrics),
        "scheduler_metrics_sha256": hashlib.sha256(scheduler_metrics.read_bytes()).hexdigest(),
        "scheduler_summary": str(scheduler_summary),
        "scheduler_summary_sha256": hashlib.sha256(scheduler_summary.read_bytes()).hexdigest(),
        "split_contract": str(split_path),
        "defect_fit": str(defect_fit),
        "defect_fit_sha256": hashlib.sha256(defect_fit.read_bytes()).hexdigest(),
        "defect_validity": str(defect_validity),
        "defect_validity_sha256": hashlib.sha256(defect_validity.read_bytes()).hexdigest(),
        "split_contract_id": split["defect_split_contract_id"],
        "target": {"K_q": 4.0, "minimum": 3.8, "maximum": 4.2},
        "selected": {
            "M_seq": 2,
            "M_full": 4,
            "K_q_hat": 4.0,
            "N_q": 40,
            "N_F": 10,
            "direct_reanchor_threshold": 0.1,
            "full_refresh_threshold": 0.2,
        },
        "calibration": {"low_threshold": 0.1, "high_threshold": 0.2},
        "producer": {
            "adapter_checkpoint": str(adapter),
            "adapter_checkpoint_sha256": adapter_hash,
        },
    }
    payload["dynamic_threshold_lock_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "dynamic.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        verify_pi05_dynamic_threshold_lock_v2,
        "verify_lock",
        lambda _path: {"source_lock_id": "lock-v2"},
    )
    result = verify_pi05_dynamic_threshold_lock_v2.verify_dynamic_threshold_lock(
        path,
        source_lock_path=tmp_path / "source.json",
        adapter_checkpoint=adapter,
    )
    assert result["DYNAMIC_THRESHOLD_LOCK_V2_VERIFIED"]
    payload["selected"]["K_q_hat"] = 5.0
    payload["dynamic_threshold_lock_id"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "dynamic_threshold_lock_id"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="outside"):
        verify_pi05_dynamic_threshold_lock_v2.verify_dynamic_threshold_lock(
            path,
            source_lock_path=tmp_path / "source.json",
            adapter_checkpoint=adapter,
        )


def test_scientific_aggregator_reopens_underlying_suite_artifacts(tmp_path: Path):
    manifest = _final_manifest()
    suite = "libero_10"
    episodes = [row for row in manifest["episodes"] if row["suite"] == suite]
    outcomes_path = tmp_path / "episode_outcomes.csv"
    query_path = tmp_path / "query_metrics.jsonl"
    episode_manifest_path = tmp_path / "episode_manifest.json"
    environment_path = tmp_path / "environment_metadata.json"
    summary_path = tmp_path / "summary.json"

    outcomes = [
        {
            "suite": suite,
            "task_id": row["benchmark_task_index"],
            "trial": row["trial"],
            "policy_path": "v0",
            "success": True,
        }
        for row in episodes
    ]
    with outcomes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    query_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": row["benchmark_task_index"],
                    "episode_id": row["trial"],
                    "query_index": 0,
                    "policy_path": "v0",
                },
                sort_keys=True,
            )
            + "\n"
            for row in episodes
        ),
        encoding="utf-8",
    )
    episode_manifest_path.write_text(json.dumps(episodes), encoding="utf-8")
    manifest_sha = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    environment_path.write_text(
        json.dumps(
            {
                "render_backend": "egl",
                "policy_server_metadata": {
                    "source_lock_id": "lock-v2",
                    "final_evaluation_manifest_sha256": manifest_sha,
                    "final_evaluation_manifest_id": manifest["manifest_id"],
                    "suite": suite,
                },
            }
        ),
        encoding="utf-8",
    )

    def artifact_pair(path: Path) -> tuple[str, str]:
        return str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest()

    outcomes_ref, outcomes_sha = artifact_pair(outcomes_path)
    query_ref, query_sha = artifact_pair(query_path)
    episode_ref, episode_sha = artifact_pair(episode_manifest_path)
    environment_ref, environment_sha = artifact_pair(environment_path)
    summary = {
        "complete": True,
        "suite": suite,
        "rollouts": 500,
        "tasks": 10,
        "trials_per_task": 50,
        "method_label": "v0",
        "source_lock_id": "lock-v2",
        "final_evaluation_manifest_sha256": manifest_sha,
        "final_evaluation_manifest_id": manifest["manifest_id"],
        "seed": 7,
        "replan_steps": 5,
        "resize_size": 224,
        "wait_steps": 10,
        "successes": 500,
        "protocol_artifacts": {
            "episode_outcomes": outcomes_ref,
            "episode_outcomes_sha256": outcomes_sha,
            "query_metrics": query_ref,
            "query_metrics_sha256": query_sha,
            "episode_manifest": episode_ref,
            "episode_manifest_sha256": episode_sha,
            "environment_metadata": environment_ref,
            "environment_metadata_sha256": environment_sha,
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    validated = aggregate_pi05_scientific_row_v2.validate_suite_summary(
        summary_path,
        method="v0",
        source_lock_id="lock-v2",
        manifest=manifest,
        manifest_sha256=manifest_sha,
    )
    assert validated["successes"] == 500 and validated["query_rows"] == 500

    outcomes[0]["success"] = False
    with outcomes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    summary["protocol_artifacts"]["episode_outcomes_sha256"] = hashlib.sha256(
        outcomes_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="success total"):
        aggregate_pi05_scientific_row_v2.validate_suite_summary(
            summary_path,
            method="v0",
            source_lock_id="lock-v2",
            manifest=manifest,
            manifest_sha256=manifest_sha,
        )


def test_v2_level_resets_and_maximum_ages():
    state = V2ScheduleState(m_seq=2, m_full=4)
    assert state.forced_decision(0) is RefreshDecision.FULL_PREFIX
    state.commit(RefreshDecision.FULL_PREFIX, 0, 5)
    assert state.forced_decision(1) is None
    assert state.forced_decision(2) is RefreshDecision.DIRECT_REANCHOR
    state.commit(RefreshDecision.DIRECT_REANCHOR, 2, 5)
    assert state.last_full_query == 0
    assert state.last_sequential_anchor_query == 2
    assert state.forced_decision(4) is RefreshDecision.FULL_PREFIX
    state.commit(RefreshDecision.FULL_PREFIX, 4, 2)
    assert state.last_full_query == state.last_sequential_anchor_query == 4


def _budget_rows(count: int, terminal_actions: int = 2):
    return [
        {
            "suite": "libero_10",
            "benchmark_task_index": 0,
            "episode_namespace": "teacher_demonstration",
            "episode_id": "budget-0",
            "query_index": query,
            "latent_defect": 0.1,
            "sequential_executed_mse": 0.1,
            "direct_executed_mse": 0.05,
            "full_executed_mse": 0.0,
            "executed_actions_actual": terminal_actions if query == count - 1 else 5,
            "terminal": query == count - 1,
        }
        for query in range(count)
    ]


def test_episode_ordered_budget_counts_initial_full_and_terminal_partial_chunk():
    result = simulate_episode(
        _budget_rows(4),
        predicted_error=lambda score: score,
        direct_threshold=1.0,
        full_threshold=2.0,
        m_seq=3,
        m_full=4,
    )
    aggregate = aggregate_simulations([result])
    assert result.decisions == ("full_prefix", "sequential", "sequential", "direct_reanchor")
    assert aggregate["N_q"] == 4 and aggregate["N_F"] == 1
    assert aggregate["K_q_hat"] == 4.0 and aggregate["K_a_hat"] == 17.0
    assert aggregate["operation_counters"]["latentloop_sequential_calls"] == 3
    assert aggregate["operation_counters"]["latentloop_direct_calls"] == 3


def test_operation_counter_names_and_level2_costs_are_explicit():
    total = OperationCountersV2()
    total.add(full_hook_query(10))
    full_after_candidates = latent_query(10, direct=True)
    full_after_candidates.prefix_transformer_calls += 1
    full_after_candidates.full_prefix_refreshes += 1
    total.add(full_after_candidates)
    assert set(total.to_dict()) == {
        "vision_encoder_calls",
        "prefix_embedding_calls",
        "prefix_transformer_calls",
        "latentloop_sequential_calls",
        "latentloop_direct_calls",
        "direct_reanchor_events",
        "full_prefix_refreshes",
        "action_expert_calls",
        "flow_iterations",
        "cache_rebuild_calls",
    }
    assert total.latentloop_sequential_calls == total.latentloop_direct_calls == 1


def test_stage_gate_missing_evidence_leaves_no_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        pi05_stage_gate_v2,
        "verify_lock",
        lambda _path: {"source_lock_id": "lock-v2"},
    )
    candidate = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="missing evidence"):
        pi05_stage_gate_v2.verify_stage(
            "stage2_cache_smoke",
            tmp_path / "lock.json",
            [],
            output_candidate=candidate,
        )
    assert not candidate.exists()


def test_evaluation_wrapper_is_one_suite_and_gates_before_residue():
    wrapper = (
        ROOT / "architectures" / "openpi" / "wrappers" / "eval_pi05_latentloop.sh"
    ).read_text()
    assert ': "${SUITE:?--suite is required}"' in wrapper
    assert "suites=(" not in wrapper
    assert "for index in 0 1 2 3" not in wrapper
    assert wrapper.index("pi05_stage_gate_v2.py") < wrapper.index('mkdir -p "${OUTPUT}')
    assert "ONLINE_SUITE_SHARD_COMPLETE" in wrapper


def test_full_cache_wrapper_stops_before_v0_and_merge_forbids_symlink_fallback():
    wrapper = (
        ROOT / "architectures" / "openpi" / "wrappers" / "generate_pi05_full_cache_4gpu.sh"
    ).read_text()
    merge = (ROOT / "tools" / "openpi" / "merge_pi05_latentloop_cache.py").read_text()
    assert "stage2_full_cache" in wrapper
    assert "V0 remains blocked" in wrapper
    assert "train_pi05_latentloop.py" not in wrapper
    assert "destination.symlink_to" not in merge
    assert "same-filesystem hard links" in merge


def test_k1_server_records_effective_kq_one():
    source = (ROOT / "tools" / "openpi" / "serve_pi05_latentloop.py").read_text()
    assert 'effective_k_q = 1 if args.mode == "k1" else args.k_q' in source
    assert '"effective_k_q": effective_k_q' in source


def test_latent_bridge_label_is_explicitly_style_only():
    serialization = (
        ROOT / "architectures" / "openpi" / "adapters" / "latentloop" / "serialization.py"
    ).read_text()
    wrapper = (
        ROOT / "architectures" / "openpi" / "wrappers" / "train_pi05_latent_bridge.sh"
    ).read_text()
    assert "Latent Bridge-style KV baseline" in serialization
    assert "DISABLED" in wrapper


def test_real_k1_audit_requires_explicit_user_run_gate():
    source = (ROOT / "tools" / "openpi" / "audit_pi05_k1_equivalence.py").read_text()
    assert 'require_run(args.run, "OPENPI_LATENTLOOP_K1_RUN")' in source
    assert "adapter_resident_during_k1" in source
