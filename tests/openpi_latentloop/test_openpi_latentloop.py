from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from architectures.openpi.adapters.latentloop.cache_io import EpisodeCacheWriter, episode_split
from architectures.openpi.adapters.latentloop.policy_io import explicit_policy_noise, policy_noise_seed
from architectures.openpi.adapters.latentloop.prefix_kv_hook import (
    PrefixEmbeddingState,
    PrefixKVHook,
    PrefixKVState,
    apply_rope_to_keys,
)
from architectures.openpi.adapters.latentloop.recurrent_policy import OpenPILatentLoopPolicy
from architectures.openpi.adapters.latentloop.serialization import (
    adapter_checkpoint_config,
    load_adapter_checkpoint,
    prefix_state_from_record,
    prefix_state_to_dict,
)
from architectures.openpi.adapters.latentloop.trainer import freeze_base_model, optimizer_parameter_names
from architectures.openpi.adapters.latentloop.transition_core import (
    OpenPIKVLatentLoop,
    OpenPIKVLatentLoopConfig,
)
from methods.variable_time_latentloop.budget_calibration import BudgetCalibrator
from methods.variable_time_latentloop.composition import compose_one_query_updates
from methods.variable_time_latentloop.defect import evaluate_defect_validity, normalized_latent_defect
from methods.variable_time_latentloop.transition import TransitionConfig


def _state(*, batch: int = 1, layers: int = 2, tokens: int = 3, embed: int = 8, head: int = 4):
    embeddings = torch.randn(batch, tokens, embed)
    return PrefixKVState(
        embeddings=embeddings,
        pad_mask=torch.ones(batch, tokens, dtype=torch.bool),
        attention_pattern=torch.zeros(batch, tokens, dtype=torch.bool),
        position_ids=torch.arange(tokens).expand(batch, -1),
        pre_rope_keys=tuple(torch.randn(batch, 1, tokens, head) for _ in range(layers)),
        values=tuple(torch.randn(batch, 1, tokens, head) for _ in range(layers)),
    )


def _tiny_adapter() -> OpenPIKVLatentLoop:
    transition = TransitionConfig(
        state_width=8,
        observation_width=8,
        action_width=8,
        robot_state_dim=4,
        robot_width=4,
        scalar_width=4,
        layer_width=4,
        hidden_width=16,
        num_blocks=1,
        max_layers=4,
    )
    return OpenPIKVLatentLoop(
        OpenPIKVLatentLoopConfig(
            prefix_embedding_dim=8,
            head_dim=4,
            action_dim=7,
            execution_horizon=5,
            action_horizon=10,
            parameter_cap=100_000,
            transition=transition,
        )
    )


class _FakeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_key_value_heads=1)
        self.head_dim = 4
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeSelfAttention()


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])
        self.config = SimpleNamespace(_attn_implementation="eager")

    def rotary_emb(self, hidden, position_ids):
        angle = position_ids.to(hidden.dtype)[..., None] * 0.1
        return torch.cos(angle).expand(-1, -1, 4), torch.sin(angle).expand(-1, -1, 4)


class _FakePaliGemmaExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = SimpleNamespace(language_model=_FakeLanguageModel())

    def forward(self, *, inputs_embeds, position_ids, **_kwargs):
        from transformers.cache_utils import DynamicCache

        hidden = inputs_embeds[0]
        cos, sin = self.paligemma.language_model.rotary_emb(hidden, position_ids)
        cache = DynamicCache()
        for index, layer in enumerate(self.paligemma.language_model.layers):
            key = layer.self_attn.k_proj(hidden).view(hidden.shape[0], hidden.shape[1], 1, 4).transpose(1, 2)
            value = layer.self_attn.v_proj(hidden).view(hidden.shape[0], hidden.shape[1], 1, 4).transpose(1, 2)
            cache.update(apply_rope_to_keys(key, cos, sin), value, index)
        return (hidden, None), cache


class _FakeHookModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _FakePaliGemmaExpert()

    def _preprocess_observation(self, observation, train=False):
        del train
        return [], [], None, None, observation["robot"]

    def embed_prefix(self, _images, _masks, _tokens, _token_masks):
        embeddings = torch.arange(24, dtype=torch.float32).view(1, 3, 8) / 24
        return embeddings, torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool)

    @staticmethod
    def _prepare_attention_masks_4d(mask):
        return torch.where(mask[:, None], 0.0, -1e9)


def test_exact_synthetic_kv_extraction_and_rebuild():
    hook = PrefixKVHook(_FakeHookModel())
    extracted = hook.extract({"robot": torch.zeros(1, 4)})
    result = hook.cache_allclose(extracted.state, extracted.source_cache)
    assert extracted.state.num_layers == 2
    assert result["passed"]


def test_pre_rope_conversion_matches_definition():
    key = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = torch.zeros(1, 1, 4)
    sin = torch.ones(1, 1, 4)
    assert torch.equal(apply_rope_to_keys(key, cos, sin), torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))


def test_prefix_state_device_move_and_serialization():
    state = _state()
    moved = state.to(torch.device("cpu"))
    assert moved.embeddings.device == moved.pad_mask.device
    record = prefix_state_to_dict(state)
    rebuilt = prefix_state_from_record(record, "cpu")
    assert torch.equal(rebuilt.pre_rope_keys[1], state.pre_rope_keys[1])


def test_horizon_and_execution_semantics_are_pinned():
    model = SimpleNamespace(config=SimpleNamespace(action_horizon=10))
    policy = OpenPILatentLoopPolicy(model, None, k_q=1, execution_horizon=5)
    assert policy.num_flow_steps == 10
    assert policy.execution_horizon == 5
    with pytest.raises(ValueError):
        OpenPILatentLoopPolicy(model, None, k_q=1, execution_horizon=4)


def test_k1_is_a_hard_original_sampler_bypass():
    class Model:
        config = SimpleNamespace(action_horizon=10)

        @staticmethod
        def sample_actions(_device, _observation, noise=None, num_steps=10):
            assert num_steps == 10
            return noise + 1

    policy = OpenPILatentLoopPolicy(Model(), None, k_q=1)
    noise = torch.zeros(1, 10, 32)
    output = policy.query(object(), noise)
    assert torch.equal(output.normalized_actions, noise + 1)
    assert output.metrics["path"] == "k1_exact_bypass"
    assert policy.runtime.updater_calls == 0


def test_executed_actions_change_the_transition_feature():
    torch.manual_seed(1)
    adapter = _tiny_adapter()
    previous = _state()
    current = PrefixEmbeddingState(
        previous.embeddings + 0.1,
        previous.pad_mask,
        previous.attention_pattern,
        previous.position_ids,
    )
    kwargs = dict(
        previous_state=previous,
        current_prefix=current,
        previous_prefix_embeddings=previous.embeddings,
        robot_state=torch.zeros(1, 4),
        delta_q=1,
        delta_a=5,
        full_refresh_age=1,
    )
    zero = adapter(executed_actions=torch.zeros(1, 5, 7), **kwargs)
    one = adapter(executed_actions=torch.ones(1, 5, 7), **kwargs)
    assert not torch.equal(zero.action_feature, one.action_feature)


def test_recurrent_one_query_and_direct_multi_query_shapes():
    adapter = _tiny_adapter()
    anchor = _state()
    current = PrefixEmbeddingState(
        anchor.embeddings + 0.2,
        anchor.pad_mask,
        anchor.attention_pattern,
        anchor.position_ids,
    )
    one = adapter(
        anchor,
        current,
        anchor.embeddings,
        torch.randn(1, 5, 7),
        torch.randn(1, 4),
        delta_q=1,
        delta_a=5,
        full_refresh_age=1,
    )
    direct = adapter(
        anchor,
        current,
        anchor.embeddings,
        torch.randn(1, 15, 7),
        torch.randn(1, 4),
        delta_q=3,
        delta_a=15,
        full_refresh_age=3,
        intermediate_prefix_embeddings=torch.stack(
            (anchor.embeddings + 0.05, anchor.embeddings + 0.1, current.embeddings), dim=1
        ),
        robot_state_history=torch.randn(1, 3, 4),
    )
    assert one.encoded_state.shape == direct.encoded_state.shape == (1, 2, 3, 8)


def test_composition_uses_chronological_inputs():
    seen = []

    def update(state, increment):
        seen.append(int(increment))
        return state + increment

    result = compose_one_query_updates(
        torch.zeros(1),
        ({"increment": torch.tensor([1])}, {"increment": torch.tensor([2])}),
        update,
    )
    assert seen == [1, 2]
    assert result.final_state.item() == 3


def test_same_noise_is_keyed_not_request_order():
    key = policy_noise_seed(7, "libero_10", 2, 3, 4)
    first = explicit_policy_noise((1, 10, 32), seed=key, device="cpu")
    _ = explicit_policy_noise((1, 10, 32), seed=123, device="cpu")
    second = explicit_policy_noise((1, 10, 32), seed=key, device="cpu")
    assert torch.equal(first, second)
    assert key != policy_noise_seed(7, "libero_10", 2, 3, 5)


def test_episode_split_is_deterministic_and_episode_disjoint():
    identities = [("libero_10", task, episode) for task in range(10) for episode in range(20)]
    first = {identity: episode_split(*identity, seed=9) for identity in identities}
    second = {identity: episode_split(*identity, seed=9) for identity in identities}
    assert first == second
    assert set(first) == set(second)
    assert set(first.values()) == {"train", "validation", "calibration"}


def test_defect_and_validity_metrics():
    direct = torch.ones(2, 2, 3)
    sequential = direct.clone()
    sequential[1] += 1
    defect = normalized_latent_defect(sequential, direct)
    assert defect[0] == 0 and defect[1] > 0
    values = np.linspace(0, 1, 100)
    audit = evaluate_defect_validity(
        values,
        values,
        {"age": values[::-1], "action": np.zeros(100), "observation": values * 0.5},
    )
    assert audit.spearman > 0
    assert audit.auroc >= 0.70


def test_budget_calibration_targets_mean_kq_four():
    defect = np.linspace(0, 1, 400)
    calibration = BudgetCalibrator(0.25, bins=32).fit(
        defect,
        sequential_error=defect + 0.1,
        direct_error=0.5 * defect,
    )
    assert calibration.validation_full_prefix_ratio <= 0.25
    assert 3.8 <= 1 / calibration.validation_full_prefix_ratio <= 4.2


def test_base_freezing_optimizer_filter_and_primary_parameter_cap():
    base = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    freeze_base_model(base)
    assert not any(parameter.requires_grad for parameter in base.parameters())
    tiny = _tiny_adapter()
    names = optimizer_parameter_names(tiny)
    assert names and all(name in dict(tiny.named_parameters()) for name in names)
    primary = OpenPIKVLatentLoop()
    assert primary.trainable_parameters == 3_370_081
    assert primary.trainable_parameters <= 19_000_000


def test_adapter_checkpoint_round_trip(tmp_path: Path):
    adapter = _tiny_adapter()
    path = tmp_path / "adapter.pt"
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "config": {"trainer": {"variant": "v0"}, **adapter_checkpoint_config(adapter)},
        },
        path,
    )
    loaded, payload = load_adapter_checkpoint(str(path), "cpu")
    assert payload["config"]["adapter_type"] == "openpi_variable_time_latentloop"
    assert loaded.trainable_parameters == adapter.trainable_parameters


def test_cache_result_serialization_refuses_duplicate_episode(tmp_path: Path):
    record = {
        key: None
        for key in __import__(
            "architectures.openpi.adapters.latentloop.cache_io", fromlist=["REQUIRED_RECORD_KEYS"]
        ).REQUIRED_RECORD_KEYS
    }
    record.update(suite="libero_10", task_id=0, episode_id=0)
    writer = EpisodeCacheWriter(tmp_path / "cache", {"action_horizon_h": 10, "execution_horizon_r": 5})
    writer.write_episode([record], suite="libero_10", task_id=0, episode_id=0, split="train")
    manifest = writer.finalize()
    assert manifest.is_file()
    assert (manifest.parent / "pi05_latentloop_split.json").is_file()
