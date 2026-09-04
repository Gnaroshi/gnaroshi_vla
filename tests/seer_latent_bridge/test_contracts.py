from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from architectures.seer.adapters.latent_bridge.action_protocol import SeerTemporalEnsembler
from architectures.seer.adapters.latent_bridge.bridge import (
    SeerFeatureBridge,
    SeerFeatureBridgeConfig,
)
from architectures.seer.adapters.latent_bridge.dataset import (
    BridgeTransition,
    BridgeTransitionDataset,
    BridgeTransitionWriter,
    StreamingBridgeTransitionWriter,
    episode_split,
)
from architectures.seer.adapters.latent_bridge.checkpoint import (
    validate_bridge_runtime_provenance,
)
from architectures.seer.adapters.latent_bridge.provenance import (
    OFFICIAL_COMMIT,
    OFFICIAL_MODEL_SHA256,
    PUBLIC_SEER_33_SHA256,
)
from architectures.seer.adapters.latent_bridge.hooks import SeerBoundaryCapture
from architectures.seer.adapters.latent_bridge.layout import SeerTokenLayout
from methods.latent_bridge import (
    ComputeMatchedTrainingContract,
    TrainingContract,
    bridge_distillation_loss,
    should_full_refresh,
)
from architectures.seer.adapters.latent_bridge.train import ExactDistributedEvalSampler
from tools.seer_latent_bridge.aggregate_evaluations import _validate_runtime_contract
from tools.seer_latent_bridge.validate_eval_row import validate_eval_row
from architectures.seer.adapters.latent_bridge.rendering import configured_renderer_backend


def test_public33_token_layout_is_259_tokens():
    layout = SeerTokenLayout(
        sequence_length=7,
        resampler_queries_per_camera=6,
        observation_prediction_tokens=18,
        action_tokens=3,
    )
    assert layout.conditioning_tokens == 16
    assert layout.tokens_per_timestep == 37
    assert layout.flattened_tokens == 259
    assert layout.per_timestep_slices["action"] == slice(34, 37)
    values = torch.arange(259).reshape(1, 259, 1)
    assert layout.select(values, timestep=-1, group="action").flatten().tolist() == [256, 257, 258]


class _ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.ln_f = nn.LayerNorm(4)

    def forward(self, x):
        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class _ToySeer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_backbone = _ToyBackbone()
        self.action_decoder = nn.Linear(4, 2)

    def forward(self, x):
        hidden = self.transformer_backbone(x)
        return self.action_decoder(hidden[:, -3:])


def test_external_hook_captures_final_and_action_input():
    model = _ToySeer()
    x = torch.randn(2, 9, 4)
    with SeerBoundaryCapture(model) as capture:
        model(x)
        capture.require_complete()
    assert list(capture.layer_outputs) == ["block_00", "block_01"]
    torch.testing.assert_close(capture.final_output[:, -3:], capture.action_head_input)


def test_external_hook_can_capture_only_selected_layers():
    model = _ToySeer()
    with SeerBoundaryCapture(model, layer_indices=(1,)) as capture:
        model(torch.randn(1, 9, 4))
        capture.require_complete()
    assert list(capture.layer_outputs) == ["block_01"]


def test_small_bridge_zero_output_starts_as_exact_copy():
    config = SeerFeatureBridgeConfig.from_preset(
        "small", stable_seq_len=3, stable_layer="block_00", stable_token_group="action"
    )
    model = SeerFeatureBridge(config)
    previous = torch.randn(2, 3, 384)
    stable = torch.randn(2, 3, 384)
    state = torch.randn(2, 8)
    action = torch.randn(2, 7)
    with torch.no_grad():
        delta = model(previous, stable, state, action)
        predicted = model.predict_next(previous, stable, state, action)
    assert torch.count_nonzero(delta).item() == 0
    torch.testing.assert_close(predicted, previous, rtol=0, atol=0)
    assert model.parameter_audit()["zero_initialized_output"] is True


def test_official_training_batch_contract_and_schedule():
    config = TrainingContract(
        stage="R0",
        epochs=200,
        learning_rate=3e-4,
        per_rank_batch=4,
        world_size=4,
        gradient_accumulation_steps=4,
    )
    config.validate()
    assert config.effective_batch == 64
    assert [should_full_refresh(i, 4) for i in range(8)] == [True, False, False, False] * 2


def test_compute_matched_contract_preserves_batch_and_counts_examples():
    contract = ComputeMatchedTrainingContract(
        stage="R0",
        optimizer_steps=50_200,
        learning_rate=3e-4,
        per_rank_batch=16,
        world_size=4,
        gradient_accumulation_steps=1,
    )
    contract.validate()
    assert contract.effective_batch == 64
    assert contract.examples_seen == 3_212_800
    with pytest.raises(ValueError, match="effective batch"):
        ComputeMatchedTrainingContract(
            stage="R0",
            optimizer_steps=1,
            learning_rate=3e-4,
            per_rank_batch=8,
            world_size=4,
            gradient_accumulation_steps=1,
        ).validate()


def test_distillation_loss_is_mse_plus_cosine():
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    predicted = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    total, terms = bridge_distillation_loss(predicted, target, cosine_weight=0.5)
    torch.testing.assert_close(total, terms["mse"] + 0.5 * terms["cosine_loss"])


def test_temporal_ensemble_matches_documented_seer_rule():
    ensemble = SeerTemporalEnsembler(max_steps=8, temperature=0.01)
    row0 = torch.full((1, 3, 7), 0.2)
    row1 = torch.full((1, 3, 7), 0.6)
    first = ensemble.continuous_action(row0, 0)
    second = ensemble.continuous_action(row1, 1)
    torch.testing.assert_close(first, torch.full((1, 7), 0.2).double())
    weights = np.exp(-0.01 * np.arange(2))
    weights /= weights.sum()
    expected = 0.2 * weights[0] + 0.6 * weights[1]
    torch.testing.assert_close(second, torch.full((1, 7), expected).double())


def test_episode_split_is_deterministic_and_dataset_is_hash_locked(tmp_path: Path):
    path = tmp_path / "transitions.h5"
    writer = BridgeTransitionWriter(path)
    for index in range(20):
        writer.append(
            BridgeTransition(
                previous_condition=np.zeros((3, 384), np.float32),
                target_condition=np.ones((3, 384), np.float32),
                stable_context=np.zeros((3, 384), np.float32),
                current_state=np.zeros((8,), np.float32),
                previous_executed_action=np.zeros((7,), np.float32),
                episode_id=f"episode-{index}",
                task_id=index % 10,
                step=1,
                success=1,
                source="sync",
            )
        )
    manifest = writer.close({"test": True})
    train = BridgeTransitionDataset(path, split="train", expected_sha256=manifest["dataset_sha256"])
    validation = BridgeTransitionDataset(
        path, split="validation", expected_sha256=manifest["dataset_sha256"]
    )
    assert len(train) + len(validation) == 20
    assert episode_split("episode-3", validation_fraction=0.1, seed=42) == episode_split(
        "episode-3", validation_fraction=0.1, seed=42
    )
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        BridgeTransitionDataset(path, split="train", expected_sha256=manifest["dataset_sha256"])


def test_preloaded_dataset_is_numerically_identical_to_lazy_dataset(tmp_path: Path):
    path = tmp_path / "transitions.h5"
    writer = BridgeTransitionWriter(path)
    for index in range(20):
        writer.append(
            BridgeTransition(
                previous_condition=np.full((3, 384), index, np.float32),
                target_condition=np.full((3, 384), index + 1, np.float32),
                stable_context=np.full((3, 384), index + 2, np.float32),
                current_state=np.full((8,), index + 3, np.float32),
                previous_executed_action=np.full((7,), index + 4, np.float32),
                episode_id=f"episode-{index}",
                task_id=index % 10,
                step=index,
                success=index % 2,
                source="sync",
            )
        )
    manifest = writer.close({"test": True})
    lazy = BridgeTransitionDataset(
        path, split="train", expected_sha256=manifest["dataset_sha256"]
    )
    preloaded = BridgeTransitionDataset(
        path,
        split="train",
        expected_sha256=manifest["dataset_sha256"],
        preload=True,
    )
    assert len(lazy) == len(preloaded)
    assert preloaded.preloaded_bytes > 0
    for index in range(len(lazy)):
        for key in lazy[index]:
            torch.testing.assert_close(preloaded[index][key], lazy[index][key], rtol=0, atol=0)


def test_streaming_writer_exposes_partial_then_atomically_finalizes(tmp_path: Path):
    path = tmp_path / "stream.h5"
    writer = StreamingBridgeTransitionWriter(path)
    assert not path.exists()
    assert writer.partial_path.exists()
    writer.append(
        BridgeTransition(
            previous_condition=np.zeros((3, 384), np.float32),
            target_condition=np.ones((3, 384), np.float32),
            stable_context=np.zeros((14, 384), np.float32),
            current_state=np.zeros((8,), np.float32),
            previous_executed_action=np.zeros((7,), np.float32),
            episode_id="episode-0",
            task_id=0,
            step=1,
            success=1,
            source="sync",
        )
    )
    manifest = writer.close({"stage": "R0"})
    assert path.exists()
    assert not writer.partial_path.exists()
    assert manifest["num_transitions"] == 1
    dataset = BridgeTransitionDataset(
        path,
        split=episode_split("episode-0", validation_fraction=0.1, seed=42),
        expected_sha256=manifest["dataset_sha256"],
    )
    assert len(dataset) == 1


def test_exact_distributed_eval_sampler_has_no_padding_or_duplicates():
    dataset = list(range(11))
    partitions = [
        list(ExactDistributedEvalSampler(dataset, rank=rank, world_size=4))
        for rank in range(4)
    ]
    flattened = [index for partition in partitions for index in partition]
    assert sorted(flattened) == list(range(len(dataset)))
    assert len(flattened) == len(set(flattened)) == len(dataset)
    assert [len(partition) for partition in partitions] == [3, 3, 3, 2]


def test_runtime_provenance_is_fail_closed():
    valid = {
        "stage": "R1",
        "metadata": {
            "official_source": {
                "commit": OFFICIAL_COMMIT,
                "model_sha256": OFFICIAL_MODEL_SHA256,
            },
            "public_seer_checkpoint_sha256": PUBLIC_SEER_33_SHA256,
            "sync_files": [{"path": "sync.h5", "sha256": "sync"}],
            "dagger_files": [{"path": "dagger.h5", "sha256": "dagger"}],
        },
    }
    audit = validate_bridge_runtime_provenance(valid)
    assert audit["stage"] == "R1"
    invalid = {
        **valid,
        "metadata": {**valid["metadata"], "public_seer_checkpoint_sha256": "wrong"},
    }
    with pytest.raises(RuntimeError, match="base-checkpoint provenance mismatch"):
        validate_bridge_runtime_provenance(invalid)


def test_evaluation_aggregation_fails_closed_on_runtime_contract():
    manifest = {
        "suite": "libero_10",
        "renderer": "osmesa",
        "checkpoint_sha256": PUBLIC_SEER_33_SHA256,
        "policy_contract": {"control_frequency_hz": 20, "max_policy_steps": 600},
    }
    summary = {
        "suite": "libero_10",
        "environment": {"control_hz": 20.0, "eval_max_steps": 600},
        "lrnode": {
            "renderer_backend": "osmesa",
            "method": "seer_latent_bridge",
            "refresh_period": 4,
            "base_checkpoint_sha256": PUBLIC_SEER_33_SHA256,
            "action_protocol": (
                "three-token prediction with temporal ensembling; one executed action"
            ),
            "bridge_calls": 10,
        },
    }
    contract = _validate_runtime_contract("f4", summary, manifest)
    assert contract["refresh_period"] == 4
    tampered = {**summary, "lrnode": {**summary["lrnode"], "renderer_backend": "egl"}}
    with pytest.raises(RuntimeError, match="renderer"):
        _validate_runtime_contract("f4", tampered, manifest)


def test_renderer_contract_rejects_conflicting_environment(monkeypatch):
    for name in ("LIBERO_GL_BACKEND", "MUJOCO_GL", "PYOPENGL_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    assert configured_renderer_backend() == "osmesa"
    monkeypatch.setenv("LIBERO_GL_BACKEND", "egl")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    assert configured_renderer_backend() == "egl"
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    with pytest.raises(RuntimeError, match="conflicting renderer"):
        configured_renderer_backend()


def test_eval_row_validation_checks_exact_episode_identity(tmp_path: Path):
    import csv
    import json

    root = tmp_path / "row"
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    rows = [
        {"task_id": task, "episode_id": episode, "seed": 42, "success": int(episode == 0)}
        for task in range(2)
        for episode in range(2)
    ]
    with (analysis / "eval_episode_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (analysis / "eval_summary.json").write_text(
        json.dumps(
            {
                "success_rate": 0.5,
                "lrnode": {"renderer_backend": "egl"},
                "environment": {
                    "renderer": {
                        "requested_backend": "egl",
                        "effective_backend": "egl",
                        "actual_context_verified": True,
                        "software_renderer": False,
                        "actual_gl_vendor": "NVIDIA Corporation",
                        "actual_gl_renderer": "NVIDIA RTX 3090",
                        "actual_gl_version": "4.6",
                    }
                },
                "task_results": [
                    {"task_id": 0, "num_episodes": 2},
                    {"task_id": 1, "num_episodes": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    (analysis / "eval_latency_profile.json").write_text("{}\n", encoding="utf-8")
    payload = validate_eval_row(
        root, seed=42, episodes_per_task=2, num_tasks=2, renderer="egl"
    )
    assert payload["episodes"] == 4
    assert payload["successes"] == 2
