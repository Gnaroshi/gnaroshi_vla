"""Actual decoder-token pruning and KV reuse for SimVLA's SmolVLM backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import StaticCache

from .official_contract import (
    VLACacheConfig,
    layer_reuse_schedule,
    reusable_visual_positions,
)


@dataclass
class DecoderResult:
    hidden_states: torch.Tensor
    report: dict[str, Any]


class IndexedReuseDecoder:
    """Run current tokens while retaining prior K/V at removed positions."""

    def __init__(self, text_model: torch.nn.Module, config: VLACacheConfig) -> None:
        self.text_model = text_model
        self.config = config
        if len(text_model.layers) <= max(config.pruning_layers):
            raise ValueError("text decoder does not contain every VLA-Cache pruning layer")
        # The official fork materializes attention probabilities to derive its
        # entropy schedule. Eager attention is therefore part of this baseline.
        text_model.config._attn_implementation = "eager"
        for layer in text_model.layers:
            layer.self_attn.config._attn_implementation = "eager"
        self.reset()

    def reset(self) -> None:
        self.cache: StaticCache | None = None
        self.previous_full_hidden: torch.Tensor | None = None
        self.previous_attentions: list[torch.Tensor] | None = None
        self.previous_query_positions: list[torch.Tensor] | None = None
        self.sequence_length: int | None = None
        self.last_report: dict[str, Any] = {}

    def previous_visual_importance(
        self,
        *,
        visual_tokens: int,
        views: int,
    ) -> torch.Tensor:
        if self.previous_attentions is None or self.previous_query_positions is None:
            raise RuntimeError("visual importance requires a completed decoder query")
        layer = self.config.reference_attention_layer
        attention = self.previous_attentions[layer]
        positions = self.previous_query_positions[layer]
        text_rows = torch.nonzero(positions >= visual_tokens, as_tuple=False).flatten()
        if not text_rows.numel():
            raise RuntimeError("reference layer contains no text query tokens")
        scores = attention[:, :, text_rows, :visual_tokens].float().mean(dim=(0, 1, 2))
        if visual_tokens % views:
            raise ValueError("visual token count must divide evenly across views")
        return scores.reshape(views, visual_tokens // views)

    @staticmethod
    def _causal_mask(
        positions: torch.Tensor,
        *,
        sequence_length: int,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        keys = torch.arange(sequence_length, device=positions.device)
        allowed = keys.unsqueeze(0) <= positions.unsqueeze(1)
        mask = torch.zeros(
            (positions.numel(), sequence_length), dtype=dtype, device=positions.device
        )
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        reusable_positions: torch.Tensor | None,
    ) -> DecoderResult:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise ValueError("VLA-Cache inference supports batch size one")
        sequence_length = int(inputs_embeds.shape[1])
        if self.sequence_length is not None and sequence_length != self.sequence_length:
            raise ValueError("token sequence length changed within a cached rollout")
        self.sequence_length = sequence_length
        full_positions = torch.arange(sequence_length, device=inputs_embeds.device)
        first_query = self.cache is None
        if first_query:
            self.cache = StaticCache(
                config=self.text_model.config,
                max_cache_len=sequence_length,
            )
            reusable = full_positions[:0]
            schedule = None
        else:
            if self.previous_full_hidden is None or self.previous_attentions is None:
                raise RuntimeError("incomplete VLA-Cache state")
            reusable = (
                full_positions[:0]
                if reusable_positions is None
                else torch.unique(reusable_positions.to(full_positions), sorted=True)
            )
            if reusable.numel() and (
                int(reusable.min()) < 0 or int(reusable.max()) >= sequence_length
            ):
                raise ValueError("reusable token position is out of bounds")
            schedule = layer_reuse_schedule(
                self.previous_attentions,
                growth_factor=self.config.positive_growth_factor,
            )

        hidden_states = inputs_embeds
        active_positions = full_positions
        removed = full_positions[:0]
        attentions: list[torch.Tensor] = []
        query_positions: list[torch.Tensor] = []
        active_tokens_per_layer: list[int] = []
        selected_per_pruning_layer: dict[str, list[int]] = {}
        full_position_embeddings = self.text_model.rotary_emb(
            inputs_embeds, full_positions.unsqueeze(0)
        )

        for layer_index, layer in enumerate(self.text_model.layers):
            if (
                not first_query
                and reusable.numel()
                and layer_index in self.config.pruning_layers
            ):
                assert schedule is not None
                proportion = float(schedule[layer_index].item())
                selected_count = max(1, int(proportion * reusable.numel()))
                selected = reusable[:selected_count]
                if removed.numel() <= selected.numel():
                    keep = ~torch.isin(active_positions, selected)
                    hidden_states = hidden_states[:, keep]
                    active_positions = active_positions[keep]
                    removed = selected
                selected_per_pruning_layer[str(layer_index)] = selected.tolist()

            active_tokens_per_layer.append(int(active_positions.numel()))
            query_positions.append(active_positions.detach().clone())
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            position_embeddings = tuple(
                value[:, active_positions] for value in full_position_embeddings
            )
            attention_output, attention_weights = layer.self_attn(
                hidden_states=normalized,
                attention_mask=self._causal_mask(
                    active_positions,
                    sequence_length=sequence_length,
                    dtype=normalized.dtype,
                    batch_size=normalized.shape[0],
                ),
                position_ids=active_positions.unsqueeze(0),
                past_key_values=self.cache,
                use_cache=True,
                cache_position=active_positions,
                position_embeddings=position_embeddings,
                output_attentions=True,
            )
            if attention_weights is None:
                raise RuntimeError("eager attention did not return attention probabilities")
            hidden_states = residual + attention_output
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
            attentions.append(attention_weights.detach())

        hidden_states = self.text_model.norm(hidden_states)
        if first_query:
            full_hidden = hidden_states
        else:
            assert self.previous_full_hidden is not None
            full_hidden = self.previous_full_hidden.clone()
            full_hidden.index_copy_(1, active_positions, hidden_states)
        self.previous_full_hidden = full_hidden.detach()
        self.previous_attentions = attentions
        self.previous_query_positions = query_positions
        full_token_layers = sequence_length * len(self.text_model.layers)
        computed_token_layers = sum(active_tokens_per_layer)
        self.last_report = {
            "first_query": first_query,
            "sequence_length": sequence_length,
            "reusable_candidates": int(reusable.numel()),
            "removed_final": int(removed.numel()),
            "active_tokens_per_layer": active_tokens_per_layer,
            "selected_positions_per_pruning_layer": selected_per_pruning_layer,
            "full_token_layers": full_token_layers,
            "computed_token_layers": computed_token_layers,
            "skipped_token_layers": full_token_layers - computed_token_layers,
            "token_layer_reduction": 1.0 - computed_token_layers / full_token_layers,
            "output_reconstructed_from_previous_hidden": bool(not first_query and removed.numel()),
            "actual_kv_reuse": bool(not first_query and removed.numel()),
        }
        return DecoderResult(full_hidden, dict(self.last_report))


class SimVLAVLACacheBackbone:
    """SmolVLM encoder whose text decoder performs actual VLA-Cache reuse."""

    def __init__(
        self,
        model: Any,
        config: VLACacheConfig | None = None,
        *,
        enable_reuse: bool = True,
    ) -> None:
        self.model = model
        self.config = config or VLACacheConfig()
        self.enable_reuse = bool(enable_reuse)
        self.text_decoder = IndexedReuseDecoder(model.vlm.model.text_model, self.config)
        self.previous_images: torch.Tensor | None = None
        self.last_report: dict[str, Any] = {}

    def reset(self) -> None:
        self.previous_images = None
        self.last_report = {}
        self.text_decoder.reset()

    @staticmethod
    def _rgb_for_similarity(valid_images: torch.Tensor) -> torch.Tensor:
        """Undo SimVLA's ImageNet normalization before RGB patch matching."""

        mean = valid_images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = valid_images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (valid_images * std + mean).clamp_(0.0, 1.0)

    def _inputs_embeds(
        self,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if image_input.ndim == 6:
            image_input = image_input.squeeze(2) if image_input.shape[2] == 1 else image_input[:, :, 0]
        if image_input.ndim != 5 or image_input.shape[0] != 1:
            raise ValueError("real VLA-Cache expects [1,V,C,H,W] images")
        batch, _, _, _, _ = image_input.shape
        valid_mask = image_mask.reshape(-1).bool()
        valid_images = image_input.flatten(0, 1)[valid_mask]
        if valid_images.shape[0] != 2:
            raise ValueError("real VLA-Cache contract requires exterior and wrist views")
        vision_outputs = self.model.vlm.model.vision_model(
            pixel_values=valid_images,
            output_hidden_states=True,
            return_dict=True,
        )
        image_features = self.model.vlm.model.connector(vision_outputs.last_hidden_state)
        expected = self.config.visual_tokens_per_view
        if image_features.shape[1] != expected:
            raise RuntimeError(
                f"SmolVLM connector produced {image_features.shape[1]} tokens, expected {expected}"
            )
        image_features = image_features.reshape(batch, -1, image_features.shape[-1])
        text_features = self.model.vlm.model.text_model.get_input_embeddings()(input_ids)
        return torch.cat([image_features, text_features], dim=1)

    @torch.no_grad()
    def encode_condition(
        self,
        *,
        input_ids: torch.Tensor,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self._inputs_embeds(image_input, image_mask, input_ids)
        valid_images = image_input[0, image_mask[0].bool()].detach()
        similarity_images = self._rgb_for_similarity(valid_images)
        visual_tokens = valid_images.shape[0] * self.config.visual_tokens_per_view
        if self.previous_images is None or not self.enable_reuse:
            reusable = None
            selection: dict[str, Any] = {
                "per_view": [],
                "reusable_positions": [],
            }
        else:
            importance = self.text_decoder.previous_visual_importance(
                visual_tokens=visual_tokens,
                views=int(valid_images.shape[0]),
            )
            reusable, selection = reusable_visual_positions(
                previous_images=self.previous_images,
                current_images=similarity_images,
                previous_visual_importance=importance,
                config=self.config,
            )
        result = self.text_decoder.forward(
            inputs_embeds,
            reusable_positions=reusable,
        )
        self.previous_images = similarity_images.clone()
        self.last_report = {
            "method": (
                "vla_cache_actual_token_pruning_kv_reuse"
                if self.enable_reuse
                else "vla_cache_matched_full_eager_control"
            ),
            "official_contract": self.config.to_dict(),
            "selection": selection,
            "decoder": result.report,
            "condition_shape": list(result.hidden_states.shape),
        }
        return result.hidden_states
