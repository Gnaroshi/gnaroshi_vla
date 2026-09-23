from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from architectures.simvla.adapters.exact_q2.trainer import _restore_rng_state
from architectures.simvla.adapters.exact_q2.simvla_exact_q2_adapter import (
    build_exact_q2_adapter,
    freeze_module,
    load_exact_q2_checkpoint,
    parameter_budget_audit,
    save_exact_q2_checkpoint,
    trainable_parameter_names,
)
from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from methods.simvla_exact_q2.dataset import (
    episode_is_heldout,
    validate_exact_q2_pair,
)
from methods.simvla_exact_q2.decisions import (
    evaluate_exact_q2_offline_gate,
    evaluate_short_budget_gate,
    online_native_r5_enabled,
)
from methods.simvla_exact_q2.losses import per_example_prefix_l1


def _record(task: int = 0, episode: str = "task00_trial000", query: int = 0) -> dict:
    next_rgb = torch.randint(0, 255, (2, 8, 8, 3), dtype=torch.uint8)
    next_q = torch.randn(8)
    next_c = torch.randn(122, 960)
    next_a = torch.randn(10, 7)
    next_noise = torch.randn(10, 7)
    return {
        "task_id": task,
        "episode_id": episode,
        "query_index": query,
        "next_query_index": query + 1,
        "raw_rgb": torch.randint(0, 255, (2, 8, 8, 3), dtype=torch.uint8),
        "proprio": torch.randn(8),
        "full_condition": torch.randn(122, 960),
        "teacher_action_chunk": torch.randn(10, 7),
        "initial_noise": torch.randn(10, 7),
        "action_noise_hash": "a" * 64,
        "next_raw_rgb": next_rgb,
        "next_proprio": next_q,
        "next_full_condition": next_c,
        "next_teacher_action_chunk": next_a,
        "next_initial_noise": next_noise,
        "next_action_noise_hash": "b" * 64,
        "executed_subchunk": torch.randn(5, 7),
        "execution_horizon": 5,
        "elapsed_time": 0.25,
    }


def _consecutive_pair() -> tuple[dict, dict]:
    q0 = _record(query=0)
    q1 = _record(query=1)
    q1["raw_rgb"] = q0["next_raw_rgb"].clone()
    q1["proprio"] = q0["next_proprio"].clone()
    q1["full_condition"] = q0["next_full_condition"].clone()
    q1["teacher_action_chunk"] = q0["next_teacher_action_chunk"].clone()
    q1["initial_noise"] = q0["next_initial_noise"].clone()
    q1["action_noise_hash"] = q0["next_action_noise_hash"]
    return q0, q1


def _batch() -> dict[str, torch.Tensor]:
    return {
        "c0_full": torch.randn(1, 122, 960),
        "q0_raw_rgb": torch.randint(0, 255, (1, 2, 8, 8, 3), dtype=torch.uint8),
        "q1_raw_rgb": torch.randint(0, 255, (1, 2, 8, 8, 3), dtype=torch.uint8),
        "q2_raw_rgb": torch.randint(0, 255, (1, 2, 8, 8, 3), dtype=torch.uint8),
        "q0_proprio": torch.randn(1, 8),
        "q1_proprio": torch.randn(1, 8),
        "q2_proprio": torch.randn(1, 8),
        "x0_executed": torch.randn(1, 5, 7),
        "x1_executed": torch.randn(1, 5, 7),
        "elapsed_q0_to_q1": torch.tensor([0.25]),
        "elapsed_q1_to_q2": torch.tensor([0.25]),
    }


def test_tuple_continuity_and_episode_boundary_rejection() -> None:
    q0, q1 = _consecutive_pair()
    assert validate_exact_q2_pair(q0, q1) == []
    q1["episode_id"] = "task00_trial001"
    assert "episode_id differs" in validate_exact_q2_pair(q0, q1)


def test_episode_split_is_stable_and_episode_disjoint() -> None:
    first = episode_is_heldout(3, "task03_trial007", heldout_fraction=0.2, split_seed=20260804)
    second = episode_is_heldout(3, "task03_trial007", heldout_fraction=0.2, split_seed=20260804)
    assert first == second


def test_candidates_receive_same_fields_and_parameter_budget() -> None:
    batch = _batch()
    expected = set(batch)
    for candidate in ("recurrent_exact_q2", "direct_exact_q2"):
        adapter = build_exact_q2_adapter(candidate)
        assert set(batch) == expected
        output = adapter(batch)
        assert output.c1.shape == output.c2.shape == batch["c0_full"].shape
    audit = parameter_budget_audit()
    assert audit["parameter_match_pass"]
    assert audit["relative_parameter_difference"] == 0.0


def test_recurrent_q2_consumes_predicted_q1_and_direct_never_does() -> None:
    batch = _batch()
    recurrent = build_exact_q2_adapter("recurrent_exact_q2")(batch)
    direct = build_exact_q2_adapter("direct_exact_q2")(batch)
    assert recurrent.q2_input_condition.data_ptr() == recurrent.c1.data_ptr()
    assert direct.q2_input_condition.data_ptr() == batch["c0_full"].data_ptr()
    assert direct.q2_input_condition.data_ptr() != direct.c1.data_ptr()


def test_direct_transition_order_is_not_mean_pooling() -> None:
    torch.manual_seed(7)
    adapter = build_exact_q2_adapter("direct_exact_q2").eval()
    batch = _batch()
    normal = adapter(batch).c2
    swapped = dict(batch)
    swapped["x0_executed"], swapped["x1_executed"] = batch["x1_executed"], batch["x0_executed"]
    assert not torch.equal(normal, adapter(swapped).c2)


def test_explicit_noise_and_prefix_are_exact() -> None:
    key = ActionNoiseKey("checkpoint", 2, "episode", 9, 20260804)
    kwargs = {
        "batch_size": 1,
        "action_horizon": 10,
        "action_dim": 7,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    assert torch.equal(explicit_action_noise(key, **kwargs), explicit_action_noise(key, **kwargs))
    target = torch.zeros(2, 10, 7)
    prediction = target.clone()
    prediction[:, :5] = 2.0
    prediction[:, 5:] = 100.0
    assert torch.equal(per_example_prefix_l1(prediction, target, 5), torch.tensor([2.0, 2.0]))


def test_freeze_optimizer_filter_and_checkpoint_serialization(tmp_path: Path) -> None:
    adapter = build_exact_q2_adapter("recurrent_exact_q2")
    names = trainable_parameter_names(adapter)
    assert names and all(name.startswith("model.") for name in names)
    checkpoint = tmp_path / "adapter.pt"
    save_exact_q2_checkpoint(checkpoint, adapter=adapter, step=2, metadata={"test": True})
    loaded, payload = load_exact_q2_checkpoint(checkpoint)
    assert payload["step"] == 2
    assert all(torch.equal(a, b) for a, b in zip(adapter.parameters(), loaded.parameters()))
    freeze_module(loaded)
    assert trainable_parameter_names(loaded) == []


def test_restore_rng_state_moves_cpu_generator_state_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = torch.get_rng_state()

    class DeviceMappedState:
        cpu_calls = 0

        def cpu(self) -> torch.Tensor:
            self.cpu_calls += 1
            return expected

    mapped_state = DeviceMappedState()
    restored: list[torch.Tensor] = []

    def capture(value: torch.Tensor) -> None:
        assert torch.is_tensor(value)
        restored.append(value)

    monkeypatch.setattr(torch, "set_rng_state", capture)
    _restore_rng_state(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": mapped_state,
        },
        torch.device("cpu"),
    )

    assert mapped_state.cpu_calls == 1
    assert restored == [expected]


def _validation_row(step: int, prefix: float, hold: float = 0.2) -> dict:
    metric = {"count": 10, "mean": prefix, "p50": prefix, "p95": prefix, "p99": prefix, "max": prefix}
    hold_metric = {**metric, "mean": hold, "p99": hold}
    return {
        "partition": "validation",
        "step": step,
        "metrics": {
            "q2_prefix_l1": metric,
            "q1_prefix_l1": {**metric, "mean": max(prefix, 0.1)},
            "q2_gripper_command_l1": metric,
            "q2_chunk_l1": metric,
            "q2_condition_normalized_mse": metric,
            "gripper_noncollapsed": True,
            "finite": True,
        },
        "references": {
            "hold_stale_condition": {"q2_prefix_l1": hold_metric},
            "old_observation_only": {"q2_prefix_l1": hold_metric},
        },
    }


def test_short_and_offline_gates_and_online_default_off() -> None:
    rows = [_validation_row(12_500, 0.18), _validation_row(25_000, 0.15), _validation_row(37_500, 0.10)]
    short = evaluate_short_budget_gate(
        {"recurrent_exact_q2": rows, "direct_exact_q2": rows}
    )
    assert short["verdict"] == "SHORT_BUDGET_PASS"
    final = _validation_row(150_000, 0.10)
    final.update(
        {
            "paired_ci95": {
                "candidate_minus_hold": [-0.2, -0.01],
                "candidate_minus_old_observation": [-0.1, 0.001],
            },
            "prerequisites": {
                "cache_integrity": True,
                "same_noise": True,
                "selected_by_validation_only": True,
            },
        }
    )
    gate = evaluate_exact_q2_offline_gate(
        {"recurrent_exact_q2": final, "direct_exact_q2": final}
    )
    assert gate["verdict"] == "BOTH_PASS"
    assert online_native_r5_enabled(gate)
    assert not online_native_r5_enabled(None)
    assert not online_native_r5_enabled({"EXACT_Q2_OFFLINE_PASS": False})


def test_short_gate_rejects_incomplete_validation_schedule() -> None:
    incomplete = [_validation_row(37_500, 0.01)]
    gate = evaluate_short_budget_gate(
        {"recurrent_exact_q2": incomplete, "direct_exact_q2": incomplete}
    )
    assert gate["verdict"] == "EARLY_STOP_BOTH"
    assert not gate["candidates"]["recurrent_exact_q2"]["checks"][
        "required_short_checkpoints_present"
    ]


def test_train_wrapper_defaults_off() -> None:
    wrapper = Path("architectures/simvla/wrappers/train_exact_q2_regeneration.sh").read_text()
    assert "SIMVLA_EXACT_Q2_TRAIN_RUN:-0" in wrapper
    offline = Path("architectures/simvla/wrappers/eval_exact_q2_offline.sh").read_text()
    assert "SIMVLA_EXACT_Q2_OFFLINE_RUN:-0" in offline
