from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from architectures.simvla.adapters.fastv.encoder import (
    FastVConditionEncoder,
    FastVForwardConfig,
    deterministic_visual_keep_indices,
    fastv_attention_scores,
    fastv_text_forward,
    scatter_compact_hidden,
)
from architectures.simvla.adapters.fastv.provenance import (
    OFFICIAL_COMMIT,
    fastv_source_manifest,
    simvla_fastv_integration_manifest,
)
from architectures.simvla.adapters.fastv.recipe import (
    EVALUATION_ROWS,
    evaluation_row,
    scientific_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_primary_recipe_is_evidence_locked() -> None:
    row = evaluation_row("fastv_k2_r50")
    assert row.uses_fastv is True
    assert row.prune_layer == 2
    assert row.prune_ratio == 0.5
    assert row.score_mode == "text_mean_first_k"
    assert row.restore_mode == "zero_scatter"


def test_baseline_recipe_cannot_activate_fastv() -> None:
    row = evaluation_row("baseline_k1")
    assert row.uses_fastv is False
    assert row.prune_layer is None
    assert row.prune_ratio == 0.0


def test_unknown_recipe_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown FastV evaluation row"):
        evaluation_row("fastv_custom_mutable")


def test_scientific_contract_distinguishes_paper_and_hf_code() -> None:
    contract = scientific_contract()
    discrepancy = contract["source_discrepancy"]
    assert "average attention" in discrepancy["paper"]
    assert "last query token" in discrepancy["official_hf_code"]
    assert contract["simvla_interface"]["k2_r50_compact_sequence"] == 86


def test_official_fastv_source_is_pinned_and_clean() -> None:
    manifest = fastv_source_manifest()
    assert manifest["commit"] == OFFICIAL_COMMIT
    assert manifest["checks"] == {
        "commit_matches": True,
        "working_tree_clean": True,
        "file_hashes_match": True,
    }
    assert manifest["official_simvla_implementation"] is False


def test_integration_manifest_binds_frozen_simvla() -> None:
    manifest = simvla_fastv_integration_manifest()
    assert manifest["simvla_upstream_commit"] == (
        "32700d0ad8991996e123e4b685abe370ce6e9aab"
    )
    assert len(manifest["combined_sha256"]) == 64


def test_visual_selection_is_deterministic_with_stable_ties() -> None:
    scores = torch.tensor([0.4, 0.9, 0.9, 0.1, 0.3, 0.2])
    first = deterministic_visual_keep_indices(scores, prune_ratio=0.5)
    second = deterministic_visual_keep_indices(scores, prune_ratio=0.5)
    assert torch.equal(first, second)
    assert first.tolist() == [0, 1, 2]


def test_text_mean_score_uses_all_early_layers_and_text_queries() -> None:
    layer0 = torch.zeros(1, 1, 5, 5)
    layer1 = torch.zeros(1, 1, 5, 5)
    layer0[:, :, 2:, 0] = 1.0
    layer1[:, :, 2:, 1] = 3.0
    scores = fastv_attention_scores(
        [layer0, layer1],
        num_visual_tokens=2,
        score_mode="text_mean_first_k",
    )
    assert scores.shape == (1, 2)
    assert torch.allclose(scores, torch.tensor([[0.5, 1.5]]))


def test_hf_score_uses_only_last_query_at_k() -> None:
    layer0 = torch.zeros(1, 2, 5, 5)
    layer1 = torch.zeros(1, 2, 5, 5)
    layer1[0, 0, -1, :2] = torch.tensor([1.0, 3.0])
    layer1[0, 1, -1, :2] = torch.tensor([3.0, 1.0])
    scores = fastv_attention_scores(
        [layer0, layer1],
        num_visual_tokens=2,
        score_mode="hf_last_token_at_k",
    )
    assert torch.allclose(scores, torch.tensor([[2.0, 2.0]]))


def test_zero_scatter_restores_original_positions() -> None:
    compact = torch.tensor([[[1.0], [2.0], [3.0]]])
    keep = torch.tensor([0, 2, 4])
    restored = scatter_compact_hidden(compact, keep, full_sequence_length=5)
    assert restored.squeeze(-1).tolist() == [[1.0, 0.0, 2.0, 0.0, 3.0]]


def _tiny_llama() -> torch.nn.Module:
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return LlamaModel(config).eval()


def test_ratio_zero_matches_native_text_forward() -> None:
    torch.manual_seed(7)
    model = _tiny_llama()
    inputs = torch.randn(1, 12, 32)
    mask = torch.ones(1, 12, dtype=torch.long)
    with torch.no_grad():
        native = model(
            inputs_embeds=inputs,
            attention_mask=mask,
            use_cache=False,
        ).last_hidden_state
        adapted = fastv_text_forward(
            model,
            inputs_embeds=inputs,
            attention_mask=mask,
            num_visual_tokens=8,
            config=FastVForwardConfig(prune_layer=2, prune_ratio=0.0),
        ).condition
    assert torch.allclose(native, adapted, atol=1e-5, rtol=1e-5)


def test_physical_pruning_reduces_12_to_8_and_zero_restores() -> None:
    torch.manual_seed(11)
    model = _tiny_llama()
    inputs = torch.randn(1, 12, 32)
    mask = torch.ones(1, 12, dtype=torch.long)
    with torch.no_grad():
        result = fastv_text_forward(
            model,
            inputs_embeds=inputs,
            attention_mask=mask,
            num_visual_tokens=8,
            config=FastVForwardConfig(prune_layer=2, prune_ratio=0.5),
        )
    assert result.condition.shape == (1, 12, 32)
    assert result.debug["compact_sequence_length"] == 8
    assert result.debug["visual_tokens_before"] == 8
    assert result.debug["visual_tokens_kept"] == 4
    assert result.debug["visual_tokens_pruned"] == 4
    assert result.debug["nonvisual_tokens_kept"] == 4
    keep = set(result.debug["visual_keep_indices"])
    pruned = [index for index in range(8) if index not in keep]
    assert torch.count_nonzero(result.condition[:, pruned, :]) == 0
    assert torch.count_nonzero(result.condition[:, 8:, :]) > 0
    assert result.debug["swiglu_aware_text_model_reduction"] > 0


def test_condition_encoder_patches_and_restores_upstream_text_forward() -> None:
    class Node:
        pass

    class FakeVLA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vlm = Node()
            self.vlm.model = Node()
            self.vlm.model.text_model = _tiny_llama()
            self.register_buffer("fused", torch.randn(1, 12, 32))

        def forward_vlm_efficient(self, image_input, image_mask, input_ids):
            output = self.vlm.model.text_model(
                inputs_embeds=self.fused,
                attention_mask=torch.ones(1, 12, dtype=torch.long),
                output_hidden_states=True,
                return_dict=True,
            )
            return {"vlm_features": output.last_hidden_state}

    model = FakeVLA().eval()
    original_forward = model.vlm.model.text_model.forward
    encoder = FastVConditionEncoder(
        model,
        FastVForwardConfig(prune_layer=2, prune_ratio=0.5),
    )
    condition = encoder.encode_condition(
        input_ids=torch.ones(1, 4, dtype=torch.long),
        image_input=torch.zeros(1, 2, 3, 4, 4),
        image_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert condition.shape == (1, 12, 32)
    assert encoder.last_debug["compact_sequence_length"] == 8
    assert model.vlm.model.text_model.forward == original_forward


def test_invalid_fastv_contract_is_rejected() -> None:
    config = FastVForwardConfig(prune_layer=0, prune_ratio=0.5)
    with pytest.raises(ValueError, match="prune_layer"):
        config.validate(num_layers=32, num_visual_tokens=72)


def test_wrapper_has_a_heavy_run_guard_and_valid_shell_syntax() -> None:
    wrapper = ROOT / "architectures/simvla/wrappers/simvla_fastv_eval.sh"
    text = wrapper.read_text(encoding="utf-8")
    assert "SIMVLA_FASTV_EVAL_RUN" in text
    subprocess.run(["bash", "-n", str(wrapper)], check=True)


def test_default_rows_do_not_include_hf_diagnostic() -> None:
    assert set(EVALUATION_ROWS) == {
        "baseline_k1",
        "fastv_k2_r50",
        "fastv_k2_r50_hf_last_token",
    }
