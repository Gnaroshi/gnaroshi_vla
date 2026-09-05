"""CPU contracts for Seer's training-free VLA-Cache adaptation."""

from __future__ import annotations

import torch
from transformers import GPT2Config

from architectures.seer.upstream.models.gpt2 import GPT2Model
from architectures.seer.upstream.models.vla_cache import (
    VLA_CACHE_COMMIT,
    VLA_CACHE_TRANSFORMERS_COMMIT,
    build_seer_vla_cache_config,
)


def _tiny_model() -> GPT2Model:
    torch.manual_seed(20260905)
    config = GPT2Config(
        hidden_size=32,
        n_layer=4,
        n_head=4,
        vocab_size=1,
        n_positions=64,
        n_ctx=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2Model(config).eval()


def _config(mode: str) -> dict[str, object]:
    return build_seer_vla_cache_config(
        mode=mode,
        pruning_layers="1,2",
        reference_attention_layer=3,
        similarity_threshold=0.996,
        positive_growth_factor=0.55,
        transformer_layers=4,
        sequence_length=2,
        num_resampler_query=1,
        num_obs_token_per_image=0,
        obs_pred=False,
        action_pred_steps=2,
    )


def test_official_source_and_full_seer_layout_are_locked() -> None:
    assert VLA_CACHE_COMMIT == "a4909880573868dee2769343d52e793c0341678b"
    assert VLA_CACHE_TRANSFORMERS_COMMIT == (
        "2302fce58afa3a4f8461625b1394f9e9c8a7f1ea"
    )
    config = build_seer_vla_cache_config(
        mode="reuse",
        pruning_layers="2,6,9,11",
        reference_attention_layer=15,
        similarity_threshold=0.996,
        positive_growth_factor=0.55,
        transformer_layers=24,
        sequence_length=7,
        num_resampler_query=6,
        num_obs_token_per_image=9,
        obs_pred=True,
        action_pred_steps=3,
    )
    assert config["tokens_per_timestep"] == 37
    assert config["total_tokens"] == 259
    assert len(config["visual_positions"]) == 98
    assert len(config["visual_groups"]) == 14
    assert all(len(group) == 7 for group in config["visual_groups"])
    assert config["stable_top_k"] == 4
    assert config["task_relevant_top_k"] == 3
    assert config["action_query_groups"][0] == [34, 35, 36]
    assert config["action_query_groups"][-1] == [256, 257, 258]


def test_off_mode_preserves_default_twelve_layer_seer_configuration() -> None:
    config = build_seer_vla_cache_config(
        mode="off",
        pruning_layers="2,6,9,11",
        reference_attention_layer=15,
        similarity_threshold=0.996,
        positive_growth_factor=0.55,
        transformer_layers=12,
        sequence_length=7,
        num_resampler_query=6,
        num_obs_token_per_image=9,
        obs_pred=True,
        action_pred_steps=3,
    )
    assert config["enabled"] is False


def test_first_query_and_matched_full_are_numerically_identical() -> None:
    model = _tiny_model()
    state_keys = tuple(model.state_dict())
    first = torch.randn(1, 16, 32)
    second = torch.randn(1, 16, 32)
    mask = torch.zeros(16, 16)
    config = _config("matched_full")

    with torch.no_grad():
        baseline_first = model(inputs_embeds=first, attention_mask=mask)
        cache_first = model(
            inputs_embeds=first,
            attention_mask=mask,
            vla_cache_config=config,
            vla_cache_source_embeds=first,
        )
        baseline_second = model(inputs_embeds=second, attention_mask=mask)
        cache_second = model(
            inputs_embeds=second,
            attention_mask=mask,
            vla_cache_config=config,
            vla_cache_source_embeds=second,
        )

    assert torch.equal(cache_first, baseline_first)
    assert torch.equal(cache_second, baseline_second)
    assert model.last_vla_cache_report["actual_kv_reuse"] is False
    assert model.last_vla_cache_report["token_layer_reduction"] == 0.0
    assert tuple(model.state_dict()) == state_keys


def test_reuse_skips_visual_token_layers_and_restores_full_output_shape() -> None:
    model = _tiny_model()
    config = _config("reuse")
    # Force a deterministic mechanics test: every token in each tiny visual
    # group may be stable and none is protected as task-relevant.
    config["stable_top_k"] = 2
    config["task_relevant_top_k"] = 0
    inputs = torch.randn(1, 16, 32)
    mask = torch.zeros(16, 16)

    with torch.no_grad():
        first = model(
            inputs_embeds=inputs,
            attention_mask=mask,
            vla_cache_config=config,
            vla_cache_source_embeds=inputs,
        )
        second = model(
            inputs_embeds=inputs,
            attention_mask=mask,
            vla_cache_config=config,
            vla_cache_source_embeds=inputs,
        )

    assert first.shape == second.shape == inputs.shape
    report = model.last_vla_cache_report
    assert report["reusable_candidates"] == 8
    assert report["removed_final"] > 0
    assert report["actual_kv_reuse"] is True
    assert report["computed_token_layers"] < report["full_token_layers"]
    assert len(report["selection_by_timestep_camera"]) == 4
    stats = model.get_vla_cache_stats()
    assert stats["calls"] == 2
    assert stats["first_queries"] == 1
    assert stats["actual_kv_reuse_calls"] == 1
    assert stats["token_layer_reduction"] > 0.0


def test_changed_visual_condition_tokens_are_not_reused() -> None:
    model = _tiny_model()
    config = _config("reuse")
    config["stable_top_k"] = 2
    config["task_relevant_top_k"] = 0
    first = torch.randn(1, 16, 32)
    second = -first
    mask = torch.zeros(16, 16)
    with torch.no_grad():
        model(
            inputs_embeds=first,
            attention_mask=mask,
            vla_cache_config=config,
            vla_cache_source_embeds=first,
        )
        cached_second = model(
            inputs_embeds=second,
            attention_mask=mask,
            vla_cache_config=config,
            vla_cache_source_embeds=second,
        )
        baseline_second = model(
            inputs_embeds=second,
            attention_mask=mask,
        )
    assert model.last_vla_cache_report["reusable_candidates"] == 0
    assert model.last_vla_cache_report["actual_kv_reuse"] is False
    assert torch.equal(cached_second, baseline_second)


def test_episode_reset_forces_a_new_full_first_query() -> None:
    model = _tiny_model()
    config = _config("reuse")
    inputs = torch.randn(1, 16, 32)
    with torch.no_grad():
        model(
            inputs_embeds=inputs,
            vla_cache_config=config,
            vla_cache_source_embeds=inputs,
        )
        model.reset_vla_cache_state()
        model(
            inputs_embeds=inputs,
            vla_cache_config=config,
            vla_cache_source_embeds=inputs,
        )
    assert model.last_vla_cache_report["first_query"] is True
    assert model.last_vla_cache_report["query_index"] == 0
    assert model.get_vla_cache_stats()["first_queries"] == 2


def test_reuse_rejects_training_mode() -> None:
    model = _tiny_model().train()
    inputs = torch.randn(1, 16, 32)
    try:
        model(
            inputs_embeds=inputs,
            vla_cache_config=_config("reuse"),
            vla_cache_source_embeds=inputs,
        )
    except ValueError as exc:
        assert "inference-only" in str(exc)
    else:
        raise AssertionError("VLA-Cache unexpectedly ran in training mode")
