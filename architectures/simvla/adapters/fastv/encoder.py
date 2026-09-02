"""Physical visual-token pruning inside SimVLA's frozen SmolVLM text model."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FastVForwardConfig:
    prune_layer: int = 2
    prune_ratio: float = 0.5
    score_mode: str = "text_mean_first_k"
    restore_mode: str = "zero_scatter"

    def validate(self, *, num_layers: int, num_visual_tokens: int) -> None:
        if not 1 <= self.prune_layer < num_layers:
            raise ValueError(
                f"prune_layer must be in [1, {num_layers - 1}], got {self.prune_layer}"
            )
        if not 0.0 <= self.prune_ratio < 1.0:
            raise ValueError(f"prune_ratio must be in [0, 1), got {self.prune_ratio}")
        if self.score_mode not in {"text_mean_first_k", "hf_last_token_at_k"}:
            raise ValueError(f"unsupported FastV score mode: {self.score_mode}")
        if self.restore_mode != "zero_scatter":
            raise ValueError(f"unsupported FastV restore mode: {self.restore_mode}")
        if num_visual_tokens < 1:
            raise ValueError("FastV requires at least one visual token")

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FastVForwardResult:
    condition: torch.Tensor
    debug: dict[str, Any]


def deterministic_visual_keep_indices(
    scores: torch.Tensor,
    *,
    prune_ratio: float,
) -> torch.Tensor:
    """Return ascending visual indices with stable low-index tie breaking."""

    if scores.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got {tuple(scores.shape)}")
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1), got {prune_ratio}")
    keep_count = max(1, round(scores.numel() * (1.0 - prune_ratio)))
    ranking = torch.argsort(scores, descending=True, stable=True)
    return ranking[:keep_count].sort().values


def fastv_attention_scores(
    attentions: list[torch.Tensor],
    *,
    num_visual_tokens: int,
    score_mode: str,
) -> torch.Tensor:
    """Compute FastV image-token scores from early-layer self-attention."""

    if not attentions:
        raise ValueError("at least one attention tensor is required")
    expected_shape = attentions[0].shape
    if len(expected_shape) != 4:
        raise ValueError(f"attention must have shape [B,H,Q,K], got {expected_shape}")
    if any(item.shape != expected_shape for item in attentions):
        raise ValueError("all early-layer attention tensors must share one shape")
    if not 0 < num_visual_tokens < expected_shape[-1]:
        raise ValueError(
            f"invalid visual-token count {num_visual_tokens} for sequence {expected_shape[-1]}"
        )
    if score_mode == "text_mean_first_k":
        stacked = torch.stack(attentions, dim=0)
        text_to_image = stacked[:, :, :, num_visual_tokens:, :num_visual_tokens]
        return text_to_image.mean(dim=(0, 2, 3))
    if score_mode == "hf_last_token_at_k":
        return attentions[-1][:, :, -1, :num_visual_tokens].mean(dim=1)
    raise ValueError(f"unsupported FastV score mode: {score_mode}")


def scatter_compact_hidden(
    compact_hidden: torch.Tensor,
    keep_indices: torch.Tensor,
    *,
    full_sequence_length: int,
) -> torch.Tensor:
    if compact_hidden.ndim != 3 or compact_hidden.shape[0] != 1:
        raise ValueError("zero-scatter restoration currently requires batch size one")
    if keep_indices.ndim != 1 or keep_indices.numel() != compact_hidden.shape[1]:
        raise ValueError("keep_indices do not match compact hidden states")
    restored = compact_hidden.new_zeros(
        compact_hidden.shape[0], full_sequence_length, compact_hidden.shape[-1]
    )
    restored[:, keep_indices, :] = compact_hidden
    return restored


def estimate_fastv_flops(
    *,
    full_tokens: int,
    compact_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    prune_layer: int,
) -> dict[str, float]:
    """Report paper-equation and SwiGLU-aware text-model FLOP estimates."""

    def paper_layer(tokens: int) -> float:
        n, d, m = tokens, hidden_size, intermediate_size
        return float(4 * n * d**2 + 2 * n**2 * d + 2 * n * d * m)

    def swiglu_layer(tokens: int) -> float:
        n, d, m = tokens, hidden_size, intermediate_size
        return float(4 * n * d**2 + 2 * n**2 * d + 3 * n * d * m)

    def reduction(layer_cost: Any) -> float:
        baseline = num_layers * layer_cost(full_tokens)
        fastv = (
            prune_layer * layer_cost(full_tokens)
            + (num_layers - prune_layer) * layer_cost(compact_tokens)
        )
        return 1.0 - fastv / baseline

    return {
        "paper_equation_text_model_reduction": reduction(paper_layer),
        "swiglu_aware_text_model_reduction": reduction(swiglu_layer),
    }


def _additive_causal_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    minimum = torch.finfo(dtype).min
    causal = torch.full(
        (sequence_length, sequence_length), minimum, dtype=dtype, device=device
    )
    causal = torch.triu(causal, diagonal=1)
    causal = causal.view(1, 1, sequence_length, sequence_length).expand(
        batch_size, 1, sequence_length, sequence_length
    )
    if attention_mask is None:
        return causal
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError(
            f"attention mask must be {(batch_size, sequence_length)}, "
            f"got {tuple(attention_mask.shape)}"
        )
    key_padding = attention_mask[:, None, None, :].eq(0)
    return causal.masked_fill(key_padding, minimum)


def _decoder_layer_with_attention(
    layer: Any,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    config = layer.self_attn.config
    previous_backend = getattr(config, "_attn_implementation", "eager")
    config._attn_implementation = "eager"
    try:
        attended, weights = layer.self_attn(
            hidden_states=normalized,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            ),
            position_embeddings=position_embeddings,
        )
    finally:
        config._attn_implementation = previous_backend
    if weights is None:
        raise RuntimeError("FastV early-layer eager attention did not return weights")
    hidden_states = residual + attended
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states, weights


def fastv_text_forward(
    text_model: Any,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_visual_tokens: int,
    config: FastVForwardConfig,
) -> FastVForwardResult:
    """Run early full layers, physically compact, then restore zero holes."""

    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("the validated FastV SimVLA path requires evaluation batch size one")
    layers = text_model.layers
    config.validate(num_layers=len(layers), num_visual_tokens=num_visual_tokens)
    batch_size, full_length, hidden_size = inputs_embeds.shape
    if num_visual_tokens >= full_length:
        raise ValueError("FastV requires non-visual query positions after image tokens")
    position_ids = torch.arange(full_length, device=inputs_embeds.device).unsqueeze(0)
    full_mask = _additive_causal_mask(
        attention_mask,
        batch_size=batch_size,
        sequence_length=full_length,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    hidden_states = inputs_embeds
    full_position_embeddings = text_model.rotary_emb(hidden_states, position_ids)
    attentions: list[torch.Tensor] = []
    for layer_index in range(config.prune_layer):
        hidden_states, weights = _decoder_layer_with_attention(
            layers[layer_index],
            hidden_states,
            attention_mask=full_mask,
            position_ids=position_ids,
            position_embeddings=full_position_embeddings,
        )
        attentions.append(weights)

    scores = fastv_attention_scores(
        attentions,
        num_visual_tokens=num_visual_tokens,
        score_mode=config.score_mode,
    )
    visual_keep = deterministic_visual_keep_indices(
        scores[0], prune_ratio=config.prune_ratio
    )
    nonvisual_keep = torch.arange(
        num_visual_tokens, full_length, device=inputs_embeds.device
    )
    keep_indices = torch.cat((visual_keep, nonvisual_keep)).sort().values
    compact_hidden = hidden_states[:, keep_indices, :]
    compact_length = compact_hidden.shape[1]
    compact_position_ids = keep_indices.unsqueeze(0)
    compact_attention_mask = torch.ones(
        batch_size, compact_length, dtype=torch.long, device=inputs_embeds.device
    )

    from transformers.masking_utils import create_causal_mask

    cache_position = torch.arange(compact_length, device=inputs_embeds.device)
    compact_mask = create_causal_mask(
        config=text_model.config,
        input_embeds=compact_hidden,
        attention_mask=compact_attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=compact_position_ids,
    )
    compact_position_embeddings = text_model.rotary_emb(
        compact_hidden, compact_position_ids
    )
    for layer_index in range(config.prune_layer, len(layers)):
        compact_hidden = layers[layer_index](
            compact_hidden,
            attention_mask=compact_mask,
            position_ids=compact_position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=compact_position_embeddings,
        )
    compact_hidden = text_model.norm(compact_hidden)
    restored = scatter_compact_hidden(
        compact_hidden,
        keep_indices,
        full_sequence_length=full_length,
    )
    flops = estimate_fastv_flops(
        full_tokens=full_length,
        compact_tokens=compact_length,
        hidden_size=hidden_size,
        intermediate_size=int(text_model.config.intermediate_size),
        num_layers=len(layers),
        prune_layer=config.prune_layer,
    )
    pruned_visual = num_visual_tokens - visual_keep.numel()
    score_bytes = scores.detach().cpu().float().contiguous().numpy().tobytes()
    debug = {
        "config": config.serializable(),
        "batch_size": batch_size,
        "full_sequence_length": full_length,
        "compact_sequence_length": compact_length,
        "visual_tokens_before": num_visual_tokens,
        "visual_tokens_kept": visual_keep.numel(),
        "visual_tokens_pruned": pruned_visual,
        "nonvisual_tokens_before": full_length - num_visual_tokens,
        "nonvisual_tokens_kept": full_length - num_visual_tokens,
        "visual_prune_fraction": pruned_visual / num_visual_tokens,
        "total_token_reduction_fraction": 1.0 - compact_length / full_length,
        "visual_keep_indices": visual_keep.detach().cpu().tolist(),
        "visual_scores_sha256": hashlib.sha256(score_bytes).hexdigest(),
        "visual_score_mean": float(scores.float().mean().item()),
        "visual_score_max": float(scores.float().max().item()),
        "visual_score_min": float(scores.float().min().item()),
        **flops,
    }
    return FastVForwardResult(condition=restored, debug=debug)


class FastVConditionEncoder:
    """Reuse upstream vision/sequence assembly while replacing only text forward."""

    def __init__(self, model: Any, config: FastVForwardConfig) -> None:
        self.model = model
        self.config = config
        self.last_debug: dict[str, Any] | None = None

    def encode_condition(
        self,
        *,
        input_ids: torch.Tensor,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        if requires_grad:
            raise ValueError("FastV is a training-free inference adapter")
        if input_ids.shape[0] != 1:
            raise ValueError("the validated SimVLA FastV path requires batch size one")
        text_model = self.model.vlm.model.text_model
        original_forward = text_model.forward
        captured: dict[str, FastVForwardResult] = {}

        def patched_forward(*args: Any, **kwargs: Any) -> Any:
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None:
                raise ValueError("FastV requires the fused inputs_embeds path")
            visual_tokens = inputs_embeds.shape[1] - input_ids.shape[1]
            result = fastv_text_forward(
                text_model,
                inputs_embeds=inputs_embeds,
                attention_mask=kwargs.get("attention_mask"),
                num_visual_tokens=visual_tokens,
                config=self.config,
            )
            captured["result"] = result
            from transformers.modeling_outputs import BaseModelOutputWithPast

            return BaseModelOutputWithPast(last_hidden_state=result.condition)

        self.model.eval()
        text_model.forward = patched_forward
        try:
            with torch.no_grad():
                encoded = self.model.forward_vlm_efficient(
                    image_input, image_mask, input_ids
                )["vlm_features"]
        finally:
            text_model.forward = original_forward
        result = captured.get("result")
        if result is None:
            raise RuntimeError("upstream SimVLA did not invoke the patched text model")
        self.last_debug = result.debug
        return encoded
