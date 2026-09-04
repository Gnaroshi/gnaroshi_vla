import torch
from transformers import LlamaConfig, LlamaModel

from architectures.simvla.adapters.vla_cache.official_contract import (
    VLA_CACHE_COMMIT,
    VLA_CACHE_TRANSFORMERS_COMMIT,
    VLACacheConfig,
    connector_patch_cosine,
    reusable_visual_positions,
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
    return model, IndexedReuseDecoder(model, cache_config)


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
