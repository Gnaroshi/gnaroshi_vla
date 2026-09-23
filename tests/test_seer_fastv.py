"""CPU contract tests for Seer's inference-only FastV path."""

from __future__ import annotations

import torch
from transformers import GPT2Config

from architectures.seer.upstream.models.fastv import build_seer_fastv_config
from architectures.seer.upstream.models.gpt2 import GPT2Model


def _tiny_model() -> GPT2Model:
    torch.manual_seed(20260902)
    config = GPT2Config(
        hidden_size=48,
        n_layer=4,
        n_head=4,
        vocab_size=1,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    return GPT2Model(config).eval()


def _config(
    *,
    enabled: bool,
    ratio: float,
    score_mode: str = "action_mean_first_l",
    retention_diagnostics: bool = False,
) -> dict:
    return build_seer_fastv_config(
        enabled=enabled,
        prune_layer=2,
        prune_ratio=ratio,
        transformer_layers=4,
        sequence_length=3,
        num_resampler_query=2,
        num_obs_token_per_image=0,
        obs_pred=False,
        action_pred_steps=2,
        score_mode=score_mode,
        retention_diagnostics=retention_diagnostics,
    )


def test_full_seer_layout_matches_evaluation_token_topology() -> None:
    config = build_seer_fastv_config(
        enabled=True,
        prune_layer=2,
        prune_ratio=0.5,
        transformer_layers=24,
        sequence_length=7,
        num_resampler_query=6,
        num_obs_token_per_image=9,
        obs_pred=True,
        action_pred_steps=3,
        score_mode="action_mean_first_l",
    )
    assert config["tokens_per_timestep"] == 37
    assert config["tokens_before_pruning"] == 259
    assert config["visual_tokens_before_pruning"] == 98
    assert config["visual_tokens_after_pruning"] == 49
    assert config["tokens_after_pruning"] == 210
    assert config["pruned_transformer_layers"] == 22
    assert config["score_layer_count"] == 2
    assert config["score_query_indices"] == [256, 257, 258]
    assert config["selection_scope"] == "global_visual"
    assert config["retention_diagnostics"] is False
    assert config["visual_camera_names"] == ["primary", "wrist"]
    assert config["original_visual_tokens_by_timestep_camera"] == [[7, 7]] * 7
    assert config["visual_token_timesteps"] == [
        timestep for timestep in range(7) for _ in range(14)
    ]
    assert config["visual_token_cameras"] == [
        camera
        for _ in range(7)
        for camera in ([0] * 6 + [1] * 6 + [0, 1])
    ]
    assert config["visual_token_kinds"] == [
        token_kind
        for _ in range(7)
        for token_kind in (["resampler"] * 12 + ["cls", "cls"])
    ]


def test_full_seer_layout_assigns_exact_queries_for_each_score_contract() -> None:
    common = dict(
        enabled=True,
        prune_layer=2,
        prune_ratio=0.5,
        transformer_layers=24,
        sequence_length=7,
        num_resampler_query=6,
        num_obs_token_per_image=9,
        obs_pred=True,
        action_pred_steps=3,
    )
    text = build_seer_fastv_config(score_mode="text_mean_first_l", **common)
    last = build_seer_fastv_config(score_mode="last_token_at_l", **common)
    alias = build_seer_fastv_config(score_mode="hf_last_action_at_l", **common)
    action = build_seer_fastv_config(score_mode="action_mean_first_l", **common)

    assert text["score_query_indices"] == [0, 37, 74, 111, 148, 185, 222]
    assert last["score_query_indices"] == [258]
    assert alias["score_query_indices"] == [258]
    assert action["score_query_indices"] == [256, 257, 258]


def test_disabled_metadata_reports_the_unpruned_runtime() -> None:
    config = _config(enabled=False, ratio=0.5)
    assert config["tokens_before_pruning"] == config["tokens_after_pruning"]
    assert (
        config["visual_tokens_before_pruning"]
        == config["visual_tokens_after_pruning"]
    )
    assert config["dropped_visual_tokens"] == 0
    assert config["pruned_transformer_layers"] == 0


def test_cpu_zero_ratio_fastv_is_numerically_identical_to_default_path() -> None:
    model = _tiny_model()
    state_keys = tuple(model.state_dict())
    inputs = torch.randn(1, 30, 48)
    attention_mask = torch.zeros(30, 30)
    with torch.no_grad():
        baseline = model(inputs_embeds=inputs, attention_mask=attention_mask)
        for score_mode in (
            "text_mean_first_l",
            "last_token_at_l",
            "action_mean_first_l",
        ):
            fastv_r0 = model(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                fastv_config=_config(
                    enabled=True,
                    ratio=0.0,
                    score_mode=score_mode,
                ),
            )
            assert torch.equal(baseline, fastv_r0)
    assert tuple(model.state_dict()) == state_keys
    assert model.last_fastv_stats["tokens_before_pruning"] == 30
    assert model.last_fastv_stats["tokens_after_pruning"] == 30


def test_positive_ratio_prunes_only_internal_sequence_and_restores_shape() -> None:
    model = _tiny_model()
    inputs = torch.randn(1, 30, 48)
    attention_mask = torch.zeros(30, 30)
    config = _config(
        enabled=True,
        ratio=0.5,
        retention_diagnostics=True,
    )
    with torch.no_grad():
        output = model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            fastv_config=config,
        )
    assert output.shape == inputs.shape
    assert model.last_fastv_stats == {
        "enabled": True,
        "prune_layer": 2,
        "prune_ratio": 0.5,
        "tokens_before_pruning": 30,
        "tokens_after_pruning": 21,
        "visual_tokens_before_pruning": 18,
        "visual_tokens_after_pruning": 9,
        "score_mode": "action_mean_first_l",
        "score_layer_count": 2,
        "selection_scope": "global_visual",
        "retention_diagnostics": True,
    }
    retention = model.get_fastv_retention_stats()
    assert retention["calls"] == 1
    assert len(retention["retained_token_sum_by_timestep_camera"]) == 3
    assert all(
        len(row) == 2
        for row in retention["retained_token_sum_by_timestep_camera"]
    )
    assert sum(
        sum(row)
        for row in retention["retained_token_sum_by_timestep_camera"]
    ) == 9
    assert retention["min_retained_tokens_by_timestep_camera"] == (
        retention["retained_token_sum_by_timestep_camera"]
    )
    assert retention["max_retained_tokens_by_timestep_camera"] == (
        retention["retained_token_sum_by_timestep_camera"]
    )
    action_indices = [8, 9, 18, 19, 28, 29]
    assert torch.count_nonzero(output[:, action_indices]).item() > 0


def test_retention_diagnostics_are_absent_from_latency_mode() -> None:
    model = _tiny_model()
    inputs = torch.randn(1, 30, 48)
    with torch.no_grad():
        model(
            inputs_embeds=inputs,
            fastv_config=_config(enabled=True, ratio=0.5),
        )
    assert model.last_fastv_stats["retention_diagnostics"] is False
    assert model.get_fastv_retention_stats()["calls"] == 0


def test_fastv_rejects_training_mode() -> None:
    model = _tiny_model().train()
    inputs = torch.randn(1, 30, 48)
    try:
        model(inputs_embeds=inputs, fastv_config=_config(enabled=True, ratio=0.5))
    except ValueError as exc:
        assert "inference-only" in str(exc)
    else:
        raise AssertionError("FastV unexpectedly ran while GPT2Model was training")


def test_score_contracts_use_their_declared_query_and_layer_evidence() -> None:
    visual_indices = torch.tensor([0, 1])
    text_queries = torch.tensor([2, 3])
    action_queries = torch.tensor([4, 5])
    layer0 = torch.zeros(1, 1, 6, 6)
    layer1 = torch.zeros(1, 1, 6, 6)
    layer0[:, :, text_queries, 0] = 8.0
    layer1[:, :, text_queries, 1] = 4.0
    layer0[:, :, action_queries, 0] = 4.0
    layer1[:, :, action_queries, 1] = 2.0

    text = GPT2Model._score_fastv_visual_tokens(
        [layer0, layer1],
        visual_indices,
        text_queries,
        "text_mean_first_l",
    )
    assert torch.equal(text, torch.tensor([4.0, 2.0]))

    action = GPT2Model._score_fastv_visual_tokens(
        [layer0, layer1],
        visual_indices,
        action_queries,
        "action_mean_first_l",
    )
    assert torch.equal(action, torch.tensor([2.0, 1.0]))

    layer1[:, :, 5, :2] = torch.tensor([1.0, 3.0])
    last = GPT2Model._score_fastv_visual_tokens(
        [layer0, layer1],
        visual_indices,
        torch.tensor([5]),
        "last_token_at_l",
    )
    alias = GPT2Model._score_fastv_visual_tokens(
        [layer0, layer1],
        visual_indices,
        torch.tensor([5]),
        "hf_last_action_at_l",
    )
    assert torch.equal(last, torch.tensor([1.0, 3.0]))
    assert torch.equal(alias, last)
