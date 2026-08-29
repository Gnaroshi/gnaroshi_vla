from __future__ import annotations

import ast
import csv
import inspect
import textwrap
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate import (
    BASELINE_ROW,
    COMBINED_ROW,
    CONDITION_ROW,
    GENERATION_ROW,
    _summarize,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import (
    SynchronizedConditionNaiveNFEPolicy,
    SynchronizedConditionK_C2Policy,
    evaluate_shard,
    _validate_sd1_fixed_shard,
    _validate_fixed_2x2_counters,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    COUPLED_ROW,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_aggregate import (
    _load_npz_records,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    condition_row_name,
    expected_call_counts,
    naive_condition_row_name,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.row_postprocess_recovery import (
    recover_row,
)


def test_fixed_2x2_counter_contracts() -> None:
    for queries in (1, 2, 5, 18):
        full_vlm = (queries + 1) // 2
        condition = queries // 2
        full_gate = _validate_fixed_2x2_counters(
            BASELINE_ROW,
            policy_queries=queries,
            full_vlm_calls=queries,
            condition_updater_calls=0,
            full_action_transformer_calls=10 * queries,
            generation_loop_updates=0,
        )
        condition_gate = _validate_fixed_2x2_counters(
            CONDITION_ROW,
            policy_queries=queries,
            full_vlm_calls=full_vlm,
            condition_updater_calls=condition,
            full_action_transformer_calls=10 * queries,
            generation_loop_updates=0,
        )
        generation_gate = _validate_fixed_2x2_counters(
            GENERATION_ROW,
            policy_queries=queries,
            full_vlm_calls=queries,
            condition_updater_calls=0,
            full_action_transformer_calls=3 * queries,
            generation_loop_updates=7 * queries,
        )
        combined_gate = _validate_fixed_2x2_counters(
            COMBINED_ROW,
            policy_queries=queries,
            full_vlm_calls=full_vlm,
            condition_updater_calls=condition,
            full_action_transformer_calls=3 * queries,
            generation_loop_updates=7 * queries,
        )
        coupled_gate = _validate_fixed_2x2_counters(
            COUPLED_ROW,
            policy_queries=queries,
            full_vlm_calls=full_vlm,
            condition_updater_calls=condition,
            full_action_transformer_calls=3 * queries,
            generation_loop_updates=7 * queries,
        )
        for gate in (full_gate, condition_gate, generation_gate, combined_gate, coupled_gate):
            assert gate["verdict"] == "FIXED_2X2_COUNTER_PASS"


def test_kc3_kc4_counter_contracts() -> None:
    for k_c in (3, 4):
        for n_g in (10, 3):
            row = condition_row_name(k_c, n_g)
            for queries in (1, 2, 3, 4, 5, 8):
                counts = expected_call_counts(row, queries)
                gate = _validate_fixed_2x2_counters(
                    row,
                    policy_queries=queries,
                    full_vlm_calls=counts["full_vlm_calls"],
                    condition_updater_calls=counts["condition_updater_calls"],
                    full_action_transformer_calls=counts[
                        "full_action_transformer_calls"
                    ],
                    generation_loop_updates=counts["generation_loop_updates"],
                )
                assert gate["verdict"] == "KC_FRONTIER_COUNTER_PASS"
                assert counts["integration_updates"] == 10 * queries


def test_joint_learned_and_naive_nfe_counter_contracts() -> None:
    learned = condition_row_name(2, 2)
    naive_rows = (
        naive_condition_row_name(2, 3),
        naive_condition_row_name(2, 2),
        naive_condition_row_name(3, 3),
    )
    for queries in (1, 2, 5, 18):
        learned_counts = expected_call_counts(learned, queries)
        assert learned_counts["full_action_transformer_calls"] == 2 * queries
        assert learned_counts["generation_loop_updates"] == 8 * queries
        assert learned_counts["integration_updates"] == 10 * queries
        for row in naive_rows:
            counts = expected_call_counts(row, queries)
            nfe = 2 if row.endswith("nfe2") else 3
            assert counts["full_action_transformer_calls"] == nfe * queries
            assert counts["generation_loop_updates"] == 0
            assert counts["integration_updates"] == nfe * queries
            gate = _validate_fixed_2x2_counters(
                row,
                policy_queries=queries,
                full_vlm_calls=counts["full_vlm_calls"],
                condition_updater_calls=counts["condition_updater_calls"],
                full_action_transformer_calls=counts[
                    "full_action_transformer_calls"
                ],
                generation_loop_updates=0,
            )
            assert gate["verdict"] == "KC_FRONTIER_COUNTER_PASS"


def test_evaluate_shard_does_not_shadow_row_contract_with_episode_spec() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(evaluate_shard)))
    shadowing_targets = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.comprehension)):
            shadowing_targets.extend(
                target.id
                for target in ast.walk(node.target)
                if isinstance(target, ast.Name) and target.id == "row_contract"
            )
    assert shadowing_targets == []


def test_completed_row_postprocessing_is_recoverable(tmp_path) -> None:
    row_name = naive_condition_row_name(2, 3)
    shard = tmp_path / "shard"
    merged = tmp_path / "merged"
    shard.mkdir()
    rows = []
    for task_id in range(10):
        for trial_id in range(50):
            queries = 2 + trial_id % 3
            counts = expected_call_counts(row_name, queries)
            rows.append(
                {
                    "row": row_name,
                    "classification": "HOST_LOCAL_EGL_DIAGNOSTIC",
                    "inference_seed": "seed02",
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "success": 1,
                    "episode_length": queries * 5,
                    "num_policy_queries": queries,
                    "num_full_vlm_calls": counts["full_vlm_calls"],
                    "num_condition_updater_calls": counts[
                        "condition_updater_calls"
                    ],
                    "num_full_action_transformer_evaluations": counts[
                        "full_action_transformer_calls"
                    ],
                    "num_generation_loop_updates": 0,
                    "num_integration_updates": counts["integration_updates"],
                    "latency_per_policy_query_ms": 30.0,
                    "latency_per_executed_action_ms": 6.0,
                    "model_vlm_encoder_per_query_ms": 20.0,
                    "model_condition_updater_per_update_ms": 2.0,
                    "model_action_generation_per_query_ms": 8.0,
                    "policy_wall_time_seconds": queries * 0.03,
                    "episode_wall_time_seconds": queries * 0.05,
                    "counter_gate": "KC_FRONTIER_COUNTER_PASS",
                }
            )
    with (shard / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_sha = "a" * 64
    (shard / "manifest_validation.json").write_text(
        '{"verdict":"EPISODE_MANIFEST_PASS","observed_manifest_sha256":"'
        + manifest_sha
        + '"}\n',
        encoding="utf-8",
    )
    (shard / "host_shard_contract.json").write_text(
        '{"verdict":"SD1_FIXED_SHARD_PASS","physical_gpu_id":4,'
        '"task_ids":[0,1,2,3,4,5,6,7,8,9]}\n',
        encoding="utf-8",
    )
    (shard / "frozen_provenance.json").write_text(
        '{"verdict":"FROZEN_PROVENANCE_PASS","paper_runtime_match":false}\n',
        encoding="utf-8",
    )
    np.savez_compressed(
        shard / "action_chunks.npz",
        task_id=np.arange(500, dtype=np.int16),
        trial_id=np.zeros(500, dtype=np.int16),
        policy_query_index=np.zeros(500, dtype=np.int32),
        action_noise_seed=np.zeros(500, dtype=np.uint64),
        action_chunk=np.zeros((500, 10, 7), dtype=np.float32),
    )
    recovered = recover_row(
        row_name=row_name,
        shard=shard,
        merged=merged,
        expected_manifest_sha256=manifest_sha,
    )
    assert recovered["verdict"] == "ROW_POSTPROCESS_RECOVERED"
    assert (shard / "shard_summary.json").is_file()
    assert (merged / "row_summary.json").is_file()


def test_condition_naive_policy_uses_native_reduced_nfe_and_paired_noise() -> None:
    calls = {}

    class FakeActionAdapter:
        def decode_action_from_condition(self, condition, proprio, **kwargs):  # type: ignore[no-untyped-def]
            calls.update(kwargs)
            return SimpleNamespace(
                action=torch.zeros(condition.shape[0], 10, 7),
                debug={"iterations": kwargs["steps"]},
            )

    policy = object.__new__(SynchronizedConditionNaiveNFEPolicy)
    policy.device = torch.device("cpu")
    policy.nfe = 3
    policy.action_adapter = FakeActionAdapter()
    policy.metrics = SimpleNamespace(
        latencies=defaultdict(list), counters=defaultdict(int)
    )
    paired_noise = torch.ones(1, 10, 7)
    policy._paired_initial_noise = lambda *args: (paired_noise, 12345)
    action, seed = policy._decode(
        torch.zeros(1, 4, 960),
        torch.zeros(1, 8),
        policy_query_index=7,
    )
    assert action.shape == (1, 10, 7)
    assert seed == 12345
    assert calls["steps"] == 3
    assert calls["initial_noise"] is paired_noise
    assert policy.metrics.counters["num_action_transformer_calls"] == 3
    assert policy.metrics.counters["num_generation_decoder_only_steps"] == 0


def test_condition_policy_uses_recursive_age_schedule() -> None:
    for k_c in (3, 4):
        policy = object.__new__(SynchronizedConditionK_C2Policy)
        policy.k_c = k_c
        policy.query_index = 0
        policy.action_queue = deque()
        policy.query_trace = []
        policy.action_chunk_records = []
        policy.log_action_chunks = False
        policy.suite = "libero_10"
        policy.task_id = 0
        policy.trial_id = 0
        policy.row_name = condition_row_name(k_c, 10)
        policy.mode = policy.row_name
        policy.metrics = SimpleNamespace(counters=defaultdict(int))
        observed: list[tuple[str, int]] = []

        def full_refresh(batch, *, policy_query_index):  # type: ignore[no-untyped-def]
            del batch
            observed.append(("full", policy_query_index % k_c))
            policy.metrics.counters["num_full_vlm_calls"] += 1
            return None, torch.zeros(1, 10, 7), policy_query_index

        def update(batch, *, age, policy_query_index):  # type: ignore[no-untyped-def]
            del batch
            observed.append(("update", age))
            policy.metrics.counters["num_condition_updater_calls"] += 1
            return None, torch.zeros(1, 10, 7), policy_query_index

        policy._full_refresh = full_refresh
        policy._v0_update = update
        for _ in range(2 * k_c):
            policy._refill_action_queue({})
        expected = [
            ("full", 0) if query % k_c == 0 else ("update", query % k_c)
            for query in range(2 * k_c)
        ]
        assert observed == expected


def test_kc_frontier_aggregate_exact_10x50() -> None:
    for k_c in (3, 4):
        for n_g in (10, 3):
            row_name = condition_row_name(k_c, n_g)
            rows = []
            for task_id in range(10):
                for trial_id in range(50):
                    queries = 3 + trial_id % 5
                    counts = expected_call_counts(row_name, queries)
                    rows.append(
                        {
                            "row": row_name,
                            "task_id": task_id,
                            "trial_id": trial_id,
                            "success": int((task_id + trial_id) % 7 != 0),
                            "episode_length": queries * 5,
                            "num_policy_queries": queries,
                            "num_full_vlm_calls": counts["full_vlm_calls"],
                            "num_condition_updater_calls": counts[
                                "condition_updater_calls"
                            ],
                            "num_full_action_transformer_evaluations": counts[
                                "full_action_transformer_calls"
                            ],
                            "num_generation_loop_updates": counts[
                                "generation_loop_updates"
                            ],
                            "num_integration_updates": counts["integration_updates"],
                            "latency_per_policy_query_ms": 100.0,
                            "model_vlm_encoder_per_query_ms": 50.0,
                            "model_condition_updater_per_update_ms": 5.0,
                            "model_action_generation_per_query_ms": 40.0,
                            "policy_wall_time_seconds": queries * 0.1,
                        }
                    )
            summary = _summarize(row_name, rows)
            assert summary["episodes"] == 500
            assert summary["integration_updates"] == 10 * summary["policy_queries"]
            assert 1.0 < summary["effective_k_c"] <= float(k_c)


def _rows(row_name: str) -> list[dict[str, int | float | str]]:
    output = []
    for task_id in range(10):
        for trial_id in range(50):
            queries = 3 + trial_id % 3
            full_vlm = queries if row_name in {BASELINE_ROW, GENERATION_ROW} else (queries + 1) // 2
            condition = 0 if row_name in {BASELINE_ROW, GENERATION_ROW} else queries // 2
            transformer = queries * (10 if row_name in {BASELINE_ROW, CONDITION_ROW} else 3)
            generation = queries * (0 if row_name in {BASELINE_ROW, CONDITION_ROW} else 7)
            output.append(
                {
                    "row": row_name,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "success": int((task_id + trial_id) % 7 != 0),
                    "episode_length": queries * 5,
                    "num_policy_queries": queries,
                    "num_full_vlm_calls": full_vlm,
                    "num_condition_updater_calls": condition,
                    "num_full_action_transformer_evaluations": transformer,
                    "num_generation_loop_updates": generation,
                    "num_integration_updates": transformer + generation,
                    "latency_per_policy_query_ms": 100.0,
                    "model_vlm_encoder_per_query_ms": 50.0,
                    "model_condition_updater_per_update_ms": 5.0,
                    "model_action_generation_per_query_ms": 40.0,
                    "policy_wall_time_seconds": queries * 0.1,
                }
            )
    return output


def test_all_four_rows_aggregate_exact_10x50() -> None:
    for row_name in (BASELINE_ROW, CONDITION_ROW, GENERATION_ROW, COMBINED_ROW, COUPLED_ROW):
        summary = _summarize(row_name, _rows(row_name))
        assert summary["episodes"] == 500
        assert summary["integration_updates"] == 10 * summary["policy_queries"]
        if row_name in {CONDITION_ROW, COMBINED_ROW, COUPLED_ROW}:
            assert 1.0 < summary["effective_k_c"] <= 2.0
        else:
            assert summary["effective_k_c"] == 1.0


def test_sd1_fixed_rows_use_gpus_2_through_7_and_all_tasks() -> None:
    for gpu in range(2, 8):
        gate = _validate_sd1_fixed_shard(gpu, tuple(range(10)))
        assert gate["verdict"] == "SD1_FIXED_SHARD_PASS"
    assert _validate_sd1_fixed_shard(1, tuple(range(10)))["verdict"] == "SD1_FIXED_SHARD_FAIL"
    assert _validate_sd1_fixed_shard(2, tuple(range(5)))["verdict"] == "SD1_FIXED_SHARD_FAIL"


def test_rb2_three_seed_launcher_uses_confirmatory_gpu_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root
        / "architectures/simvla/wrappers/run_action_equivalent_refresh_three_seed_rb2.sh"
    ).read_text(encoding="utf-8")
    periodic_call = launcher.split("run_periodic_once()", 1)[1].split(
        "action_complete()", 1
    )[0]

    assert "--classification RB2_CONFIRMATORY_EGL" in periodic_call
    assert "--classification HOST_LOCAL_EGL_DIAGNOSTIC" not in periodic_call


def test_action_chunk_npz_is_loaded_once_and_indexed_correctly(tmp_path) -> None:
    path = tmp_path / "action_chunks.npz"
    action_chunks = np.arange(4 * 10 * 7, dtype=np.float32).reshape(4, 10, 7)
    np.savez_compressed(
        path,
        task_id=np.asarray([0, 0, 1, 1]),
        trial_id=np.asarray([2, 2, 3, 3]),
        policy_query_index=np.asarray([0, 1, 0, 1]),
        action_noise_seed=np.asarray([10, 11, 12, 13]),
        action_chunk=action_chunks,
    )

    records = _load_npz_records(path)

    assert set(records) == {(0, 2, 0), (0, 2, 1), (1, 3, 0), (1, 3, 1)}
    assert records[(1, 3, 1)][0] == 13
    np.testing.assert_array_equal(records[(1, 3, 1)][1], action_chunks[3])
