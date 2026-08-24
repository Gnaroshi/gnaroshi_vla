from __future__ import annotations

import numpy as np

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate import (
    BASELINE_ROW,
    COMBINED_ROW,
    CONDITION_ROW,
    GENERATION_ROW,
    _summarize,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import (
    _validate_sd1_fixed_shard,
    _validate_fixed_2x2_counters,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    COUPLED_ROW,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_aggregate import (
    _load_npz_records,
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
