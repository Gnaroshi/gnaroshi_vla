from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from architectures.simvla.adapters.latent_bridge.checkpoint import (
    load_bridge_checkpoint,
    save_bridge_checkpoint,
)
from architectures.simvla.adapters.latent_bridge.condition_hook import (
    SimVLAConditionWithStableHook,
    resolve_text_layers,
)
from architectures.simvla.adapters.latent_bridge.dataset import (
    DAGGER_SCHEMA,
    SYNC_SCHEMA,
    SimVLALatentBridgeDaggerDataset,
    SimVLALatentBridgeSyncDataset,
    sha256_file,
)
from architectures.simvla.adapters.latent_bridge.model import (
    SimVLALatentBridge,
    SimVLALatentBridgeConfig,
)
from architectures.simvla.adapters.latent_bridge.provenance import (
    OFFICIAL_COMMIT,
    latent_bridge_source_manifest,
    resolve_upstream_root,
    simvla_latent_bridge_integration_manifest,
)
from architectures.simvla.adapters.latentloop.native_v0_dataset import (
    stable_episode_partition,
)


OFFICIAL_ROOT = Path(
    "/home/mingyujung/private/gnaroshi_vla/architectures/latent_bridge/upstream"
)


@pytest.fixture(scope="module")
def official_root() -> Path:
    try:
        root = resolve_upstream_root()
    except FileNotFoundError:
        root = OFFICIAL_ROOT
    if not root.is_dir():
        pytest.skip("official Latent Bridge clone is not installed")
    return root


@pytest.fixture(scope="module")
def bridge(official_root: Path) -> SimVLALatentBridge:
    return SimVLALatentBridge(official_upstream_root=str(official_root))


def test_official_source_is_pinned_clean_and_hash_exact(official_root: Path) -> None:
    manifest = latent_bridge_source_manifest(official_root)
    assert manifest["commit"] == OFFICIAL_COMMIT
    assert all(manifest["checks"].values())
    assert manifest["official_simvla_implementation"] is False


def test_default_bridge_matches_official_capacity(bridge: SimVLALatentBridge) -> None:
    assert bridge.parameter_audit() == {
        "total": 183_584_448,
        "trainable": 183_584_448,
        "total_millions": 183.584448,
    }
    assert bridge.config.feature_dim == 960
    assert bridge.config.sequence_length == 72
    assert bridge.config.token_mode == "image_only"


def test_zero_initialized_bridge_is_identity(official_root: Path) -> None:
    bridge = SimVLALatentBridge(
        SimVLALatentBridgeConfig(
            sequence_length=122,
            hidden_dim=48,
            num_heads=6,
            num_blocks=1,
            token_mode="all",
        ),
        official_upstream_root=str(official_root),
    )
    condition = torch.randn(1, 122, 960)
    stable = torch.randn_like(condition)
    state = torch.randn(1, 8)
    action = torch.randn(1, 7)
    with torch.inference_mode():
        delta = bridge(condition, stable, state, action)
        predicted = bridge.predict_next(condition, stable, state, action)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(predicted, condition)


def test_image_only_mode_uses_reduced_72_token_sequence(official_root: Path) -> None:
    model = SimVLALatentBridge(
        SimVLALatentBridgeConfig(
            sequence_length=72,
            hidden_dim=48,
            num_heads=6,
            num_blocks=1,
            token_mode="image_only",
        ),
        official_upstream_root=str(official_root),
    )
    with torch.no_grad():
        model.final_layer.linear.bias.fill_(1)
    condition = torch.randn(1, 72, 960)
    delta = model(condition, condition, torch.zeros(1, 8), torch.zeros(1, 7))
    assert torch.count_nonzero(delta).item() > 0


class _FakeLayer(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value + self.offset,)


class _FakeSimVLA:
    def __init__(self) -> None:
        layers = nn.ModuleList((_FakeLayer(1), _FakeLayer(2)))
        text_model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        self.vlm = SimpleNamespace(model=SimpleNamespace(text_model=text_model))

    def forward_vlm_efficient(self, _images, _mask, _ids):
        value = torch.zeros(1, 3, 4)
        for layer in self.vlm.model.text_model.model.layers:
            value = layer(value)[0]
        return {"vlm_features": value}


def test_external_hook_captures_middle_layer_without_model_patch() -> None:
    model = _FakeSimVLA()
    layers, path = resolve_text_layers(model)
    assert len(layers) == 2
    assert path.endswith("text_model.model.layers")
    with SimVLAConditionWithStableHook(model, stable_layer_index=0) as hook:
        output = hook.encode(
            input_ids=torch.zeros(1, 1, dtype=torch.long),
            image_input=torch.zeros(1, 1, 3, 2, 2),
            image_mask=torch.ones(1, 1, dtype=torch.bool),
        )
    assert torch.equal(output.stable, torch.ones(1, 3, 4))
    assert torch.equal(output.condition, torch.full((1, 3, 4), 3.0))


def test_checkpoint_roundtrip_preserves_config_and_weights(
    official_root: Path, tmp_path: Path
) -> None:
    bridge = SimVLALatentBridge(
        SimVLALatentBridgeConfig(
            sequence_length=122,
            hidden_dim=48,
            num_heads=6,
            num_blocks=1,
            token_mode="all",
        ),
        official_upstream_root=str(official_root),
    )
    path = save_bridge_checkpoint(
        tmp_path / "bridge.pt",
        bridge,
        provenance={
            "latent_bridge": latent_bridge_source_manifest(official_root),
            "integration": simvla_latent_bridge_integration_manifest(),
        },
        training={"step": 1},
    )
    loaded, payload = load_bridge_checkpoint(
        path,
        device="cpu",
        official_upstream_root=str(official_root),
    )
    assert loaded.config == bridge.config
    assert payload["training"]["step"] == 1
    assert all(
        torch.equal(bridge.state_dict()[name], loaded.state_dict()[name])
        for name in bridge.state_dict()
    )


def test_dagger_dataset_requires_complete_hash_locked_manifest(tmp_path: Path) -> None:
    root = tmp_path / "dagger"
    root.mkdir()
    transition = {
        "condition_input": torch.zeros(122, 960, dtype=torch.bfloat16),
        "condition_target": torch.ones(122, 960, dtype=torch.bfloat16),
        "stable_anchor": torch.zeros(122, 960, dtype=torch.bfloat16),
        "state": torch.zeros(8),
        "previous_action": torch.zeros(7),
        "age": 1,
    }
    shard = root / "task00_trial000.pt"
    torch.save(
        {
            "schema_version": DAGGER_SCHEMA,
            "task_id": 0,
            "trial_id": 0,
            "transitions": [transition],
        },
        shard,
    )
    with pytest.raises(FileNotFoundError, match="complete DAgger manifest"):
        SimVLALatentBridgeDaggerDataset(root)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": DAGGER_SCHEMA,
                "base_checkpoint": "YuankaiLuo/SimVLA-LIBERO",
                "norm_stats_sha256": "test-norm-stats",
                "bridge_checkpoint_sha256": "test-r0-checkpoint",
                "stable_layer_index": 10,
                "token_mode": "image_only",
                "latent_bridge_upstream": {"combined_sha256": "test-official"},
                "simvla_latent_bridge_integration": {
                    "combined_sha256": "test-integration"
                },
                "action_horizon": 10,
                "execution_horizon": 5,
                "flow_steps": 10,
                "episodes": 1,
                "shards": [
                    {
                        "file": shard.name,
                        "sha256": sha256_file(shard),
                        "size_bytes": shard.stat().st_size,
                        "task_id": 0,
                        "trial_id": 0,
                        "transitions": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = SimVLALatentBridgeDaggerDataset(root)
    assert len(dataset) == 1
    assert dataset[0]["condition_t"].shape == (122, 960)


def test_sync_dataset_is_hash_locked_and_episode_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "sync"
    root.mkdir()
    train_trial = next(
        trial
        for trial in range(1000)
        if stable_episode_partition(0, f"sync_trial_{trial}", 42) >= 0.1
    )
    heldout_trial = next(
        trial
        for trial in range(1000)
        if stable_episode_partition(0, f"sync_trial_{trial}", 42) < 0.1
    )
    transition = {
        "condition_input": torch.zeros(122, 960, dtype=torch.bfloat16),
        "condition_target": torch.ones(122, 960, dtype=torch.bfloat16),
        "stable_anchor": torch.zeros(122, 960, dtype=torch.bfloat16),
        "state": torch.zeros(8),
        "previous_action": torch.zeros(7),
        "age": 1,
    }
    entries = []
    for trial in (train_trial, heldout_trial):
        shard = root / f"task00_trial{trial:03d}.pt"
        torch.save(
            {
                "schema_version": SYNC_SCHEMA,
                "task_id": 0,
                "trial_id": trial,
                "transitions": [transition],
            },
            shard,
        )
        entries.append(
            {
                "file": shard.name,
                "sha256": sha256_file(shard),
                "size_bytes": shard.stat().st_size,
                "task_id": 0,
                "trial_id": trial,
                "transitions": 1,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SYNC_SCHEMA,
                "data_role": "on_policy_frozen_full_simvla_rollouts",
                "task_ids": [0],
                "trial_offset": 0,
                "trials_per_task": 2,
                "episodes": 2,
                "shards": entries,
            }
        ),
        encoding="utf-8",
    )
    train = SimVLALatentBridgeSyncDataset(root, split="train")
    heldout = SimVLALatentBridgeSyncDataset(root, split="heldout")
    assert len(train) == len(heldout) == 1
    assert train.contract()["split_sha256"] != heldout.contract()["split_sha256"]
    assert train[0]["condition_t"].shape == (122, 960)
