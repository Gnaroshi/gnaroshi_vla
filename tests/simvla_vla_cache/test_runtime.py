import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from architectures.simvla.adapters.vla_cache.official_contract import (
    VLA_CACHE_COMMIT,
    VLA_CACHE_TRANSFORMERS_COMMIT,
    VLACacheConfig,
    connector_patch_cosine,
    reusable_visual_positions,
    layer_reuse_schedule,
)
from architectures.simvla.adapters.vla_cache.smolvlm_runtime import (
    IndexedReuseDecoder,
    SimVLAVLACacheBackbone,
)


def _tiny_decoder() -> tuple[LlamaModel, IndexedReuseDecoder]:
    torch.manual_seed(7)
    model_config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    model_config._attn_implementation = "eager"
    model = LlamaModel(model_config).eval()
    cache_config = VLACacheConfig(
        pruning_layers=(2, 6, 9, 11),
        reference_attention_layer=15,
    )
    return model, IndexedReuseDecoder(model, cache_config, diagnostics=True)


def test_official_contract_is_pinned_and_scaled_to_connector_tokens():
    config = VLACacheConfig()
    assert VLA_CACHE_COMMIT == "a4909880573868dee2769343d52e793c0341678b"
    assert VLA_CACHE_TRANSFORMERS_COMMIT == "2302fce58afa3a4f8461625b1394f9e9c8a7f1ea"
    assert config.pruning_layers == (2, 6, 9, 11)
    assert config.stable_top_k == 21
    assert config.task_relevant_top_k == 14
    assert config.similarity_threshold == 0.996


def test_stable_minus_task_relevant_selection_operates_on_real_tokens():
    config = VLACacheConfig()
    images = torch.arange(2 * 3 * 12 * 12, dtype=torch.float32).reshape(2, 3, 12, 12)
    cosine = connector_patch_cosine(images, images.clone(), grid_size=6)
    assert cosine.shape == (2, 36)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6)

    importance = torch.zeros(2, 36)
    importance[:, :14] = 1.0
    positions, report = reusable_visual_positions(
        previous_images=images,
        current_images=images.clone(),
        previous_visual_importance=importance,
        config=config,
    )
    assert positions.ndim == 1
    assert all(position < 72 for position in positions.tolist())
    assert all(item["stable_selected"] == 21 for item in report["per_view"])
    assert all(item["task_relevant_selected"] == 14 for item in report["per_view"])


def test_similarity_uses_rgb_values_not_imagenet_normalized_values():
    rgb = torch.rand(2, 3, 384, 384)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    normalized = (rgb - mean) / std
    restored = SimVLAVLACacheBackbone._rgb_for_similarity(normalized)
    assert torch.allclose(restored, rgb, atol=1e-6)


def test_first_query_is_exact_full_eager_decoder():
    model, runtime = _tiny_decoder()
    inputs = torch.randn(1, 12, 32)
    with torch.no_grad():
        expected = model(inputs_embeds=inputs, use_cache=False).last_hidden_state
        observed = runtime.forward(inputs, reusable_positions=None)
    assert torch.equal(observed.hidden_states, expected)
    assert observed.report["computed_token_layers"] == 16 * 12
    assert observed.report["skipped_token_layers"] == 0
    assert observed.report["actual_kv_reuse"] is False


def test_second_query_skips_compute_and_retains_removed_kv_and_hidden():
    model, runtime = _tiny_decoder()
    first_inputs = torch.randn(1, 12, 32)
    first = runtime.forward(first_inputs, reusable_positions=None)
    old_key = runtime.cache.layers[2].keys[:, :, 0].clone()
    old_hidden = first.hidden_states[:, 0].clone()

    second_inputs = first_inputs + 0.1 * torch.randn_like(first_inputs)
    second = runtime.forward(
        second_inputs,
        reusable_positions=torch.tensor([0, 1]),
    )
    assert second.hidden_states.shape == first.hidden_states.shape
    assert second.report["computed_token_layers"] < second.report["full_token_layers"]
    assert second.report["actual_kv_reuse"] is True
    assert 0 in second.report["selected_positions_per_pruning_layer"]["2"]
    assert torch.equal(runtime.cache.layers[2].keys[:, :, 0], old_key)
    assert torch.equal(second.hidden_states[:, 0], old_hidden)


def test_actual_projection_sizes_and_all_layer_kv_invariants():
    model, runtime = _tiny_decoder()
    with torch.no_grad():
        first = runtime.forward(torch.randn(1, 12, 32), reusable_positions=None)
        old = [(layer.keys.clone(), layer.values.clone()) for layer in runtime.cache.layers]
        observed = [[] for _ in model.layers]
        handles = [layer.self_attn.q_proj.register_forward_pre_hook(
            lambda module, args, i=i: observed[i].append(args[0].shape[1])) for i, layer in enumerate(model.layers)]
        second = runtime.forward(torch.randn(1, 12, 32), reusable_positions=torch.tensor([0, 1, 2]))
        for handle in handles:
            handle.remove()
        assert [x[0] for x in observed] == second.report["active_tokens_per_layer"]
        assert min(x[0] for x in observed) < 12
        for i, cache in enumerate(runtime.cache.layers):
            positions = runtime.previous_query_positions[i]
            removed = torch.tensor([j for j in range(12) if j not in positions], dtype=torch.long)
            assert torch.equal(cache.keys[:, :, removed], old[i][0][:, :, removed])
            assert torch.equal(cache.values[:, :, removed], old[i][1][:, :, removed])
            assert not torch.equal(cache.keys[:, :, 9:], old[i][0][:, :, 9:])
            mask = runtime._causal_mask(positions, sequence_length=12, dtype=torch.float32, batch_size=1)[0, 0]
            assert torch.equal(mask == 0, torch.arange(12)[None] <= positions[:, None])
        removed = torch.tensor([j for j in range(12) if j not in runtime.previous_query_positions[-1]], dtype=torch.long)
        assert torch.equal(second.hidden_states[:, removed], first.hidden_states[:, removed])
        runtime.reset()
        assert runtime.cache is None and runtime.previous_full_hidden is None
        assert runtime.previous_attentions is None and runtime.previous_query_positions is None


def test_matched_full_control_recomputes_every_token():
    _, runtime = _tiny_decoder()
    first_inputs = torch.randn(1, 12, 32)
    runtime.forward(first_inputs, reusable_positions=None)
    second = runtime.forward(
        first_inputs + 0.1 * torch.randn_like(first_inputs),
        reusable_positions=torch.empty(0, dtype=torch.long),
    )
    assert second.report["computed_token_layers"] == second.report["full_token_layers"]
    assert second.report["actual_kv_reuse"] is False


def _official_schedule_reference(maps):
    values = []
    for attention in maps:
        probability = attention.float().mean(1)[0]
        probability = probability / (probability.sum(-1, keepdim=True) + 1e-10)
        values.append(-(probability * (probability + 1e-10).log()).sum(-1).mean())
    entropy = torch.stack(values)
    ratios = (1 - (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-10)).tolist()
    for index in range(1, len(ratios)):
        if ratios[index] > ratios[index - 1]:
            ratios[index] = ratios[index - 1] + (ratios[index] - ratios[index - 1]) * 0.55
    return torch.tensor(ratios)


def test_schedule_uses_every_real_layer_and_rejects_metadata():
    maps = [((1 - w) * torch.eye(4) + w * torch.full((4, 4), 0.25))[None, None]
            for w in (0.0, 0.1, 0.3, 1.0)]
    result = layer_reuse_schedule(maps, growth_factor=0.55)
    assert len(result) == 4
    torch.testing.assert_close(result, _official_schedule_reference(maps), rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="attention maps"):
        layer_reuse_schedule(maps + [torch.arange(4)], growth_factor=0.55)


def test_grouped_schedule_matches_scalar_on_variable_query_lengths():
    maps = [torch.softmax(torch.randn(1, 4, q, 12), dim=-1) for q in (12, 12, 9, 9, 8, 8)]
    grouped = layer_reuse_schedule(maps, growth_factor=0.55)
    reference = layer_reuse_schedule(maps, growth_factor=0.55, grouped=False)
    torch.testing.assert_close(grouped, reference, rtol=0, atol=1e-7)


def test_native_attention_configuration_is_not_mutated():
    model, _ = _tiny_decoder()
    model.config._attn_implementation = "sdpa"
    runtime = IndexedReuseDecoder(model, VLACacheConfig())
    assert model.config._attn_implementation == "sdpa"
    assert all(layer.self_attn.config._attn_implementation == "sdpa" for layer in model.layers)
    assert all(attention.config._attn_implementation == "eager" for attention in runtime.attentions)
    assert runtime.attentions[0].q_proj is model.layers[0].self_attn.q_proj


def test_task_relevance_excludes_padding_queries():
    _, runtime = _tiny_decoder()
    attention = torch.zeros(1, 2, 6, 6)
    attention[:, :, 2, 1] = 1
    attention[:, :, 3:, 0] = 1
    runtime.previous_attentions = [attention] * 16
    runtime.previous_query_positions = [torch.arange(6)] * 16
    runtime.sequence_length = 6
    scores = runtime.previous_visual_importance(
        visual_tokens=2, views=1, text_attention_mask=torch.tensor([[1, 0, 0, 0]]))
    torch.testing.assert_close(scores, torch.tensor([[0., 1.]]))


def test_no_reusable_candidates_do_not_compute_schedule(monkeypatch):
    _, runtime = _tiny_decoder()
    inputs = torch.randn(1, 12, 32)
    with torch.no_grad():
        runtime.forward(inputs, reusable_positions=None)
        def forbidden(*args, **kwargs):
            raise AssertionError("unused schedule executed")
        monkeypatch.setattr("architectures.simvla.adapters.vla_cache.smolvlm_runtime.layer_reuse_schedule", forbidden)
        result = runtime.forward(inputs, reusable_positions=torch.empty(0, dtype=torch.long))
    assert not result.report["actual_kv_reuse"]


def test_optimized_decoder_matches_slow_reference_across_queries():
    model, _ = _tiny_decoder()
    fast = IndexedReuseDecoder(model, VLACacheConfig(), optimized=True, diagnostics=True)
    slow = IndexedReuseDecoder(copy.deepcopy(model), VLACacheConfig(), optimized=False, diagnostics=True)
    with torch.no_grad():
        for step in range(5):
            x = torch.randn(1, 12, 32)
            reusable = None if step == 0 else torch.tensor([0, 1, 3])
            actual = fast.forward(x, reusable_positions=reusable)
            expected = slow.forward(x, reusable_positions=reusable)
            torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=0, atol=1e-6)
            assert actual.report == expected.report
            for left, right in zip(fast.cache.layers, slow.cache.layers):
                torch.testing.assert_close(left.keys, right.keys, rtol=0, atol=1e-6)
                torch.testing.assert_close(left.values, right.values, rtol=0, atol=1e-6)
    fast.reset()
    assert fast.cache is None and fast.reuse_age is None


def test_selection_diagnostics_do_not_change_reusable_tokens():
    images = torch.rand(2, 3, 12, 12)
    kwargs = dict(previous_images=images, current_images=images.clone(),
                  previous_visual_importance=torch.rand(2, 36), config=VLACacheConfig())
    selected, report = reusable_visual_positions(**kwargs, diagnostics=False)
    reference, debug = reusable_visual_positions(**kwargs, diagnostics=True)
    assert torch.equal(selected, reference)
    assert report["reusable_count"] == len(debug["reusable_positions"])
