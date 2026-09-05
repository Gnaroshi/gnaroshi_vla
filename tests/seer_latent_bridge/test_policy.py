from pathlib import Path

import numpy as np
import torch
from torch import nn

from architectures.seer.adapters.latent_bridge.bridge import (
    SeerFeatureBridge,
    SeerFeatureBridgeConfig,
)
from architectures.seer.adapters.latent_bridge.checkpoint import save_bridge_checkpoint
from architectures.seer.adapters.latent_bridge.provenance import (
    OFFICIAL_COMMIT,
    OFFICIAL_MODEL_SHA256,
    PUBLIC_SEER_33_SHA256,
)
from architectures.seer.adapters.latent_bridge.policy import build_latent_bridge_wrapper


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList([nn.Identity()])
        self.ln_f = nn.LayerNorm(384)


class _FakeSeer(nn.Module):
    sequence_length = 7
    NUM_RESAMPLER_QUERY = 6
    NUM_OBS_TOKEN = 18
    obs_pred = True
    action_pred_steps = 3

    def __init__(self):
        super().__init__()
        self.transformer_backbone = _Backbone()
        self.action_decoder = nn.Linear(384, 7)

    def decode_action_from_latent(self, latent):
        value = self.action_decoder(latent)
        return torch.tanh(value[..., :6]), torch.sigmoid(value[..., 6:])


class _FakeBaseWrapper:
    def __init__(self, model, **kwargs):
        self.model = model
        self.device = "cpu"
        self.action_pred_steps = 3
        self.lrnode_cached_latent = None
        self.lrnode_cached_age = 0
        self.lrnode_update_calls = 0
        self.fast_encoder_calls = 0
        self.action_head_calls = 0
        self.lrnode_query_interval = 1
        self.lrnode_eval_shadow_full_forward = False

    def _base_model(self):
        return self.model

    def _sync_cuda(self):
        return None

    def get_lrnode_stats(self):
        return {}

    def reset(self):
        return None


def test_zero_initialized_policy_skip_reuses_condition_and_shared_head(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv(
        "SEER_LATENT_BRIDGE_BASE_CHECKPOINT_SHA256", PUBLIC_SEER_33_SHA256
    )
    config = SeerFeatureBridgeConfig(
        stable_seq_len=3,
        hidden_dim=24,
        num_blocks=1,
        num_heads=6,
        preset="unit",
        stable_layer="block_00",
        stable_token_group="action",
    )
    checkpoint = tmp_path / "bridge.pt"
    save_bridge_checkpoint(
        checkpoint,
        SeerFeatureBridge(config),
        stage="R0",
        epoch=0,
        metadata={
            "official_source": {
                "commit": OFFICIAL_COMMIT,
                "model_sha256": OFFICIAL_MODEL_SHA256,
            },
            "public_seer_checkpoint_sha256": PUBLIC_SEER_33_SHA256,
            "sync_files": [{"path": "unit", "sha256": "unit"}],
        },
    )
    monkeypatch.setenv("SEER_LATENT_BRIDGE_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SEER_LATENT_BRIDGE_BASE_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SEER_LATENT_BRIDGE_REFRESH_PERIOD", "4")
    monkeypatch.setenv("SEER_LATENT_BRIDGE_PRECISION", "fp32")
    monkeypatch.setenv("SEER_LATENT_BRIDGE_COMPILE", "0")
    monkeypatch.setattr(
        "architectures.seer.adapters.latent_bridge.policy.require_file_hash",
        lambda *_args, **_kwargs: PUBLIC_SEER_33_SHA256,
    )
    wrapper_type = build_latent_bridge_wrapper(_FakeBaseWrapper)
    wrapper = wrapper_type(model=_FakeSeer())
    previous = torch.randn(1, 3, 384)
    wrapper.lrnode_cached_latent = previous.clone()
    wrapper._latent_bridge_stable_context = torch.randn(1, 3, 384)
    wrapper._latent_bridge_previous_executed_action = np.zeros(7, dtype=np.float32)
    action_sequence, debug = wrapper._update_from_lrnode_cache(
        image_x=torch.empty(1),
        gripper=torch.empty(1),
        state=torch.randn(1, 1, 8),
    )
    torch.testing.assert_close(wrapper.lrnode_cached_latent, previous, rtol=0, atol=0)
    assert tuple(action_sequence.shape) == (1, 3, 7)
    assert debug["update_norm"] == 0.0
    assert wrapper._latent_bridge_calls == 1
