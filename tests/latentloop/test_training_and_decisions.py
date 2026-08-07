"""CPU-only tests for freezing, losses, serialization, and frozen gates."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pytest import MonkeyPatch
import torch
from torch import nn

from architectures.simvla.adapters.latentloop.checkpoint import (
    freeze_module,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
    trainable_parameter_names,
)
from architectures.simvla.adapters.latentloop.condition_adapter import build_latentloop_adapter
from architectures.simvla.adapters.latentloop.offline_evaluator import _efficiency_snapshot
from architectures.simvla.adapters.latentloop.result_aggregator import aggregate
from architectures.simvla.adapters.latentloop.trainer import (
    _capture_rng_state,
    _restore_rng_state,
    _seed_training,
)
from methods.latentloop.eval.decisions import apply_predeclared_decisions
from methods.latentloop.training.losses import (
    LatentLoopLossWeights,
    LossScaleAccumulator,
    compute_t1_losses,
)
from methods.latentloop.training.sampling import DeterministicStepBatchSampler


def test_freeze_and_optimizer_filtering() -> None:
    teacher = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    adapter = nn.Linear(3, 2)
    freeze_module(teacher)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert trainable_parameter_names(adapter) == ["bias", "weight"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for parameter in adapter.parameters()}
    assert optimizer_ids.isdisjoint({id(parameter) for parameter in teacher.parameters()})


def test_loss_scale_logging_and_prefix() -> None:
    previous = torch.zeros(2, 3, 4)
    predicted = torch.ones(2, 3, 4)
    teacher = torch.zeros(2, 3, 4)
    predicted_action = torch.ones(2, 10, 7)
    teacher_action = torch.zeros(2, 10, 7)
    losses = compute_t1_losses(
        previous_condition=previous,
        predicted_condition=predicted,
        teacher_condition=teacher,
        predicted_action_chunk=predicted_action,
        teacher_action_chunk=teacher_action,
        execution_lengths=torch.tensor([1, 5]),
        weights=LatentLoopLossWeights(1.0, 1.0, 1.0, 1.0),
    )
    accumulator = LossScaleAccumulator()
    accumulator.update(losses)
    accumulator.update(losses)
    summary = accumulator.summary()
    assert "total" not in summary
    assert all(row["count"] == 2 for row in summary.values())
    assert losses["executed_prefix_l1"].item() == 1.0
    restored = LossScaleAccumulator()
    restored.load_state_dict(accumulator.state_dict())
    assert restored.summary() == summary


def test_step_sampler_resume_matches_uninterrupted_order() -> None:
    full = list(
        DeterministicStepBatchSampler(
            dataset_size=11,
            batch_size=3,
            seed=17,
            start_step=0,
            max_steps=10,
        )
    )
    resumed = list(
        DeterministicStepBatchSampler(
            dataset_size=11,
            batch_size=3,
            seed=17,
            start_step=6,
            max_steps=10,
        )
    )
    assert resumed == full[6:]
    assert sorted(index for batch in full[:4] for index in batch) == list(range(11))


def test_training_rng_state_round_trip(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = torch.device("cpu")
    _seed_training(29)
    state = _capture_rng_state(device)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    _seed_training(71)
    _restore_rng_state(state, device)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


def test_offline_efficiency_uses_run_timer(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "architectures.simvla.adapters.latentloop.offline_evaluator.time.perf_counter",
        lambda: 12.5,
    )
    metrics = _efficiency_snapshot(
        run_started_at=10.0,
        processed_records=5,
        device=torch.device("cpu"),
    )
    assert metrics == {
        "elapsed_seconds": 2.5,
        "records_per_second": 2.0,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
    }


def test_predeclared_gate_rules() -> None:
    summary = {
        "k1_exact_action_chunk_equality": True,
        "k1_identical_paired_outcomes": True,
        "k1_updater_calls": 0,
        "k1_observation_encoder_calls": 0,
        "k1_action_encoder_calls": 0,
        "offline_chunk_aware_beats_hold_prefix_error": True,
        "offline_chunk_aware_beats_old_observation_only_prefix_error": True,
        "r1_confirmation_complete": True,
        "r1_k4_paired_ci_lower_pp": -2.9,
        "r1_k4_vs_no_observation_ci_lower_pp": 0.1,
        "r1_k4_full_condition_reduction": 0.75,
        "r1_k4_worse_than_both_matched_baselines": False,
        "r5_k2_paired_ci_lower_pp": -2.9,
        "r5_k2_full_condition_reduction": 0.50,
        "r5_k2_beats_hold": True,
        "r5_k2_beats_no_observation": True,
        "r5_k2_improves_old_observation_only": True,
    }
    decisions = apply_predeclared_decisions(summary)
    assert decisions["K1_PARITY_PASS"]
    assert decisions["OFFLINE_PREFIX_GATE_PASS"]
    assert decisions["R1_K4_PASS"]
    assert decisions["R5_K2_PASS"]
    summary["r1_confirmation_complete"] = False
    assert not apply_predeclared_decisions(summary)["R1_K4_PASS"]


def test_predeclared_scientific_verdicts() -> None:
    unsupported = apply_predeclared_decisions(
        {
            "simvla_r1_credible": False,
            "simvla_r5_credible": False,
            "simvla_screening_complete": True,
        }
    )
    assert unsupported["scientific_verdict"] == "SIMVLA_NOT_SUPPORTED"
    envstep_only = apply_predeclared_decisions(
        {
            "simvla_r1_credible": True,
            "simvla_r5_credible": False,
        }
    )
    assert envstep_only["scientific_verdict"] == "ENVSTEP_ONLY"
    chunk_aware = apply_predeclared_decisions(
        {
            "simvla_r1_credible": False,
            "simvla_r5_credible": True,
            "simvla_r5_chunk_aware_supported": True,
        }
    )
    assert chunk_aware["scientific_verdict"] == "CHUNK_AWARE_SUPPORTED"
    nonrecurrent = apply_predeclared_decisions(
        {
            "simvla_r1_credible": True,
            "simvla_r5_credible": True,
            "nonrecurrent_matches_or_beats_recurrent": True,
        }
    )
    assert nonrecurrent["scientific_verdict"] == "RECURRENCE_NOT_NEEDED"
    action = apply_predeclared_decisions(
        {
            "simvla_r1_credible": True,
            "simvla_r5_credible": True,
            "action_correction_matches_or_beats_recurrent": True,
        }
    )
    assert action["scientific_verdict"] == "ACTION_CORRECTION_SUFFICIENT"


def test_adapter_checkpoint_round_trip(tmp_path: Path) -> None:
    adapter = build_latentloop_adapter("chunk_aware_latentloop")
    path = tmp_path / "adapter.pt"
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    training_state = {"elapsed_seconds": 1.5, "loss_accumulator": {"x": [1.0]}}
    save_adapter_checkpoint(
        path,
        adapter=adapter,
        step=7,
        metadata={"test": True},
        optimizer_state_dict=optimizer.state_dict(),
        training_state=training_state,
    )
    loaded, payload = load_adapter_checkpoint(path)
    assert payload["step"] == 7
    assert loaded.config == adapter.config
    assert payload["optimizer_state_dict"] == optimizer.state_dict()
    assert payload["training_state"] == training_state
    for name, value in adapter.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])


def test_aggregator_uses_gate_pass_not_file_presence_for_credibility(tmp_path: Path) -> None:
    def write(name: str, payload: dict[str, object]) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    parity = {
        "k1_parity": {
            "exact_action_chunk_equality": True,
            "identical_paired_outcomes": True,
            "updater_calls": 0,
            "observation_encoder_calls": 0,
            "action_encoder_calls": 0,
        }
    }
    online_r5 = {
        "episodes_per_row": 100,
        "rows": {
            "chunk_aware_latentloop_k2": {
                "full_condition_reduction_per_policy_query": 0.5,
            }
        },
        "paired_vs_full": {
            "chunk_aware_latentloop_k2": {
                "task_hierarchical_paired_ci95_pp": [-2.0, 1.0],
            }
        },
        "paired_between_rows": {
            "chunk_aware_latentloop_k2": {
                "hold_condition_k2": {"candidate_minus_baseline_pp": 2.0},
                "no_observation_k2": {"candidate_minus_baseline_pp": 1.0},
                "old_observation_only_k2": {"candidate_minus_baseline_pp": 0.0},
                "nonrecurrent_condition_k2": {"candidate_minus_baseline_pp": 1.0},
                "action_chunk_correction_k2": {"candidate_minus_baseline_pp": 1.0},
            }
        },
    }
    result = aggregate(
        SimpleNamespace(
            output=str(tmp_path / "aggregate"),
            k1_r1_summary=write("k1_r1.json", parity),
            k1_r5_summary=write("k1_r5.json", parity),
            offline_r1=write("offline_r1.json", {"gate": {}}),
            offline_r5=write(
                "offline_r5.json",
                {
                    "gate": {
                        "OFFLINE_PREFIX_GATE_PASS": True,
                        "offline_chunk_aware_beats_hold_prefix_error": True,
                        "offline_chunk_aware_beats_old_observation_only_prefix_error": True,
                    }
                },
            ),
            online_r1=write("online_r1.json", {"rows": {"full_k1": {}}}),
            online_r5=write("online_r5.json", online_r5),
            confirmation_r1="",
        )
    )
    assert not result["inputs"]["simvla_r1_credible"]
    assert result["inputs"]["simvla_r5_credible"]
    assert result["decisions"]["R5_K2_PASS"]
    assert result["decisions"]["scientific_verdict"] == "CHUNK_AWARE_SUPPORTED"
