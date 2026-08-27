from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from types import SimpleNamespace

import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import (
    _validate_fixed_2x2_counters,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    MECHANICAL_CONTROL_MODES,
    expected_call_counts,
    mechanical_control_row_name,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.mechanical_control_aggregate import (
    PRIMARY_ROW,
    aggregate,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.mechanical_control_policy import (
    SynchronizedMechanicalControlPolicy,
)


def test_mechanical_control_counter_contracts() -> None:
    queries = 5
    expected_by_mode = {
        "hold_condition": (3, 0, 15, 35),
        "native_chunk_replay": (3, 0, 9, 21),
        "hold_action": (3, 0, 9, 21),
        "no_observation": (3, 2, 15, 35),
    }
    for mode, expected_values in expected_by_mode.items():
        row = mechanical_control_row_name(mode)
        counts = expected_call_counts(row, queries)
        assert (
            counts["full_vlm_calls"],
            counts["condition_updater_calls"],
            counts["full_action_transformer_calls"],
            counts["generation_loop_updates"],
        ) == expected_values
        gate = _validate_fixed_2x2_counters(
            row,
            policy_queries=queries,
            full_vlm_calls=counts["full_vlm_calls"],
            condition_updater_calls=counts["condition_updater_calls"],
            full_action_transformer_calls=counts["full_action_transformer_calls"],
            generation_loop_updates=counts["generation_loop_updates"],
            action_transformer_decodes=(
                counts["full_action_transformer_calls"] // 3
            ),
            observation_encoder_calls=0,
        )
        assert gate["verdict"] == "MECHANICAL_CONTROL_COUNTER_PASS"


def _bare_policy(mode: str) -> SynchronizedMechanicalControlPolicy:
    policy = object.__new__(SynchronizedMechanicalControlPolicy)
    policy.control_mode = mode
    policy.mode = mechanical_control_row_name(mode)
    policy.row_name = policy.mode
    policy.k_c = 2
    policy.n_g = 3
    policy.query_index = 0
    policy.action_queue = deque()
    policy.query_trace = []
    policy.action_chunk_records = []
    policy.log_action_chunks = False
    policy.cached_condition = torch.ones(1, 2, 3)
    policy.cached_action_chunk = None
    policy.cached_executed_action = None
    policy.suite = "libero_10"
    policy.task_id = 0
    policy.trial_id = 0
    policy.metrics = SimpleNamespace(
        counters=defaultdict(int), latencies=defaultdict(list)
    )

    anchor = torch.arange(70, dtype=torch.float32).reshape(1, 10, 7)

    def full_refresh(batch, *, policy_query_index):  # type: ignore[no-untyped-def]
        del batch, policy_query_index
        policy.metrics.counters["num_full_vlm_calls"] += 1
        policy.metrics.counters["num_action_transformer_calls"] += 3
        policy.metrics.counters["num_generation_decoder_only_steps"] += 7
        policy.cached_action_chunk = anchor.clone()
        return policy.cached_condition, anchor.clone(), 100

    def decode(condition, proprio, *, policy_query_index):  # type: ignore[no-untyped-def]
        del condition, proprio, policy_query_index
        policy.metrics.counters["num_action_transformer_calls"] += 3
        policy.metrics.counters["num_generation_decoder_only_steps"] += 7
        return torch.full((1, 10, 7), 77.0), 101

    def no_observation(batch, *, policy_query_index):  # type: ignore[no-untyped-def]
        del batch, policy_query_index
        policy.metrics.counters["num_condition_updater_calls"] += 1
        policy.metrics.counters["num_no_observation_updates"] += 1
        decoded, seed = decode(None, None, policy_query_index=1)
        return policy.cached_condition, decoded, seed

    policy._full_refresh = full_refresh
    policy._decode = decode
    policy._zero_observation_update = no_observation
    return policy


def test_mechanical_control_skipped_query_semantics() -> None:
    for mode in MECHANICAL_CONTROL_MODES:
        policy = _bare_policy(mode)
        policy._refill_action_queue({"proprio": torch.zeros(1, 8)})
        policy.action_queue.clear()
        policy.cached_executed_action = torch.full((7,), 55.0)
        policy._refill_action_queue({"proprio": torch.zeros(1, 8)})
        queued = torch.stack([action for action, _ in policy.action_queue])
        sources = {source for _, source in policy.action_queue}
        assert len(queued) == 5
        if mode == "native_chunk_replay":
            expected = torch.arange(70, dtype=torch.float32).reshape(10, 7)[5:10]
            assert torch.equal(queued, expected)
            assert sources == {"native_action_chunk"}
        elif mode == "hold_action":
            assert torch.equal(queued, torch.full((5, 7), 55.0))
            assert sources == {"hold_action"}
        elif mode == "hold_condition":
            assert torch.equal(queued, torch.full((5, 7), 77.0))
            assert sources == {"hold_condition"}
        else:
            assert torch.equal(queued, torch.full((5, 7), 77.0))
            assert sources == {"no_observation"}


def test_no_observation_control_injects_exact_zero_feature() -> None:
    captured = {}

    class FakeUpdater:
        def __call__(self, condition, delta, **kwargs):  # type: ignore[no-untyped-def]
            captured["delta"] = delta.detach().clone()
            return SimpleNamespace(condition=condition + 1)

    policy = object.__new__(SynchronizedMechanicalControlPolicy)
    policy.cached_condition = torch.ones(1, 4, 960)
    policy.cached_raw_rgb = torch.zeros(1, 2, 3, 8, 8)
    policy.cached_proprio = torch.zeros(1, 8)
    policy.cached_action_chunk = None
    policy.condition_layout = SimpleNamespace(
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        group_ids=torch.zeros(1, 4, dtype=torch.long),
    )
    policy.native_v0 = SimpleNamespace(
        delta_dim=128, condition_updater=FakeUpdater()
    )
    policy.metrics = SimpleNamespace(
        counters=defaultdict(int), latencies=defaultdict(list)
    )
    policy._sync = lambda: None
    policy._decode = lambda *args, **kwargs: (torch.zeros(1, 10, 7), 123)
    batch = {
        "raw_rgb": torch.randn(1, 2, 3, 8, 8),
        "proprio": torch.randn(1, 8),
    }
    _, action, seed = policy._zero_observation_update(
        batch, policy_query_index=1
    )
    assert torch.count_nonzero(captured["delta"]) == 0
    assert policy.metrics.counters["num_condition_updater_calls"] == 1
    assert policy.metrics.counters["num_observation_encoder_calls"] == 0
    assert action.shape == (1, 10, 7)
    assert seed == 123


def test_mechanical_control_aggregate_uses_exact_paired_episode_set(tmp_path) -> None:
    manifest_sha = "a" * 64
    names = (
        "full_nfe10",
        "generation_ng3",
        "naive_nfe3",
        "condition_kc2_ng10",
        PRIMARY_ROW,
        *(mechanical_control_row_name(mode) for mode in MECHANICAL_CONTROL_MODES),
    )
    row_args = []
    for index, name in enumerate(names):
        root = tmp_path / name
        root.mkdir()
        successes = 480 - index
        (root / "row_summary.json").write_text(
            json.dumps(
                {
                    "verdict": "MECHANICAL_CONTROL_ROW_PASS",
                    "row": name,
                    "manifest_sha256": manifest_sha,
                    "episodes": 500,
                    "successes": successes,
                    "effective_k_c": 2.0,
                    "full_vlm_calls": 250,
                    "condition_updater_calls": 250,
                    "full_action_transformer_evaluations": 1500,
                    "generation_loop_updates": 3500,
                    "latency_per_policy_query_ms": 25.0,
                    "latency_per_executed_action_ms": 5.0,
                }
            ),
            encoding="utf-8",
        )
        rows = [
            {
                "task_id": task,
                "trial_id": trial,
                "success": int(task * 50 + trial < successes),
            }
            for task in range(10)
            for trial in range(50)
        ]
        with (root / "episode_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        row_args.append(f"{name}={root}")

    result = aggregate(
        SimpleNamespace(
            output=str(tmp_path / "comparison"),
            expected_manifest_sha256=manifest_sha,
            row=row_args,
        )
    )
    assert result["verdict"] == "MECHANICAL_CONTROL_COMPARISON_COMPLETE"
    assert set(result["paired_primary_vs_control"]) == {
        mechanical_control_row_name(mode) for mode in MECHANICAL_CONTROL_MODES
    }
