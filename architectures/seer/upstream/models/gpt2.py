import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class GPT2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if attention_mask is not None:
            # Apply the attention 
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    # def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
    #     # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
    #     bsz, num_heads, q_seq_len, dk = query.size()
    #     _, _, k_seq_len, _ = key.size()

    #     # Preallocate attn_weights for `baddbmm`
    #     attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

    #     # Compute Scale Factor
    #     scale_factor = 1.0
    #     if self.scale_attn_weights:
    #         scale_factor /= float(value.size(-1)) ** 0.5

    #     if self.scale_attn_by_inverse_layer_idx:
    #         scale_factor /= float(self.layer_idx + 1)

    #     # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
    #     with autocast(enabled=False):
    #         q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
    #         attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
    #         attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

    #     if not self.is_cross_attention:
    #         # if only "normal" attention layer implements causal mask
    #         query_length, key_length = query.size(-2), key.size(-2)
    #         causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    #         mask_value = torch.finfo(attn_weights.dtype).min
    #         # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    #         # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    #         mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    #         attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    #     if attention_mask is not None:
    #         # Apply the attention mask
    #         attn_weights = attn_weights + attention_mask

    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    #     # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
    #     if attn_weights.dtype != torch.float32:
    #         raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
    #     attn_weights = attn_weights.type(value.dtype)
    #     attn_weights = self.attn_dropout(attn_weights)

    #     # Mask heads if we want to
    #     if head_mask is not None:
    #         attn_weights = attn_weights * head_mask

    #     attn_output = torch.matmul(attn_weights, value)

    #     return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output

    def forward_indexed_reuse(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        active_positions: torch.Tensor,
        previous_key: Optional[torch.Tensor],
        previous_value: Optional[torch.Tensor],
        sequence_length: int,
    ):
        """Compute active queries while retaining prior K/V at inactive positions."""

        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if previous_key is None or previous_value is None:
            expected = torch.arange(sequence_length, device=active_positions.device)
            if not torch.equal(active_positions, expected):
                raise RuntimeError("The first VLA-Cache query must compute every token")
            full_key = key.detach()
            full_value = value.detach()
        else:
            expected_shape = (
                hidden_states.shape[0],
                self.num_heads,
                sequence_length,
                self.head_dim,
            )
            if tuple(previous_key.shape) != expected_shape:
                raise RuntimeError(
                    "VLA-Cache key shape changed: "
                    f"expected={expected_shape}, actual={tuple(previous_key.shape)}"
                )
            if tuple(previous_value.shape) != expected_shape:
                raise RuntimeError(
                    "VLA-Cache value shape changed: "
                    f"expected={expected_shape}, actual={tuple(previous_value.shape)}"
                )
            full_key = previous_key
            full_value = previous_value
            full_key.index_copy_(2, active_positions, key.detach())
            full_value.index_copy_(2, active_positions, value.detach())

        if self.reorder_and_upcast_attn:
            raise RuntimeError(
                "Seer VLA-Cache requires the model's standard eager attention path"
            )
        attn_output, attn_weights = self._attn(
            query,
            full_key,
            full_value,
            attention_mask,
        )
        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights, full_key, full_value


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states,
            attention_mask=attention_mask,
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states  # TODO

        return hidden_states 

    def forward_indexed_reuse(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        active_positions: torch.Tensor,
        previous_key: Optional[torch.Tensor],
        previous_value: Optional[torch.Tensor],
        sequence_length: int,
    ):
        residual = hidden_states
        normalized = self.ln_1(hidden_states)
        attn_output, attention_weights, key, value = (
            self.attn.forward_indexed_reuse(
                normalized,
                attention_mask=attention_mask,
                active_positions=active_positions,
                previous_key=previous_key,
                previous_value=previous_value,
                sequence_length=sequence_length,
            )
        )
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = residual + self.mlp(self.ln_2(hidden_states))
        return hidden_states, attention_weights, key, value


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    load_tf_weights = None
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
"""


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        self.reset_vla_cache_statistics()
        self.reset_vla_cache_state()

    def reset_vla_cache_state(self):
        """Reset rollout-local cache state without discarding aggregate metrics."""

        self._vla_cache_key_cache = None
        self._vla_cache_value_cache = None
        self._vla_cache_previous_full_hidden = None
        self._vla_cache_previous_source = None
        self._vla_cache_previous_importance = None
        self._vla_cache_previous_entropies = None
        self._vla_cache_sequence_length = None
        self._vla_cache_query_index = 0
        self.last_vla_cache_report = {}

    def reset_vla_cache_statistics(self):
        self._vla_cache_statistics = {
            "calls": 0,
            "first_queries": 0,
            "actual_kv_reuse_calls": 0,
            "reusable_candidates": 0,
            "removed_final": 0,
            "full_token_layers": 0,
            "computed_token_layers": 0,
        }

    def get_vla_cache_stats(self):
        stats = dict(self._vla_cache_statistics)
        calls = max(1, int(stats["calls"]))
        full_token_layers = int(stats["full_token_layers"])
        computed_token_layers = int(stats["computed_token_layers"])
        stats.update(
            {
                "avg_reusable_candidates": stats["reusable_candidates"] / calls,
                "avg_removed_final": stats["removed_final"] / calls,
                "token_layer_reduction": (
                    1.0 - computed_token_layers / full_token_layers
                    if full_token_layers
                    else 0.0
                ),
                "last_report": dict(self.last_vla_cache_report),
            }
        )
        return stats

    @staticmethod
    def _vla_cache_attention_rows(attention_mask, active_positions):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask.index_select(0, active_positions)
        if attention_mask.dim() == 3:
            return attention_mask.index_select(1, active_positions)
        if attention_mask.dim() == 4:
            return attention_mask.index_select(2, active_positions)
        raise ValueError(
            "Seer VLA-Cache supports 2D, 3D, or 4D attention masks; "
            f"got shape={tuple(attention_mask.shape)}"
        )

    @staticmethod
    def _vla_cache_attention_entropy(attention):
        probabilities = attention.float().mean(dim=1)
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-10)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        token_entropy = -(
            probabilities * torch.log(probabilities + 1e-10)
        ).sum(dim=-1)
        return token_entropy.mean().detach()

    @staticmethod
    def _vla_cache_layer_schedule(entropies, growth_factor):
        if len(entropies) < 2:
            raise ValueError("VLA-Cache requires at least two attention layers")
        values = torch.stack(list(entropies[:-1])).float()
        normalized = (values - values.min()) / (
            values.max() - values.min() + 1e-10
        )
        reuse = (1.0 - normalized).tolist()
        for index in range(1, len(reuse)):
            delta = reuse[index] - reuse[index - 1]
            if delta > 0:
                reuse[index] = reuse[index - 1] + delta * float(growth_factor)
        return torch.tensor(
            reuse, dtype=torch.float32, device=entropies[0].device
        )

    @staticmethod
    def _vla_cache_select_reusable(
        previous_source,
        current_source,
        previous_importance,
        config,
    ):
        if previous_source.shape != current_source.shape:
            raise ValueError("VLA-Cache source-token shape changed within a rollout")
        if previous_source.ndim != 3 or previous_source.shape[0] != 1:
            raise ValueError("Seer VLA-Cache requires source tokens shaped [1,N,D]")
        sequence_length = current_source.shape[1]
        if previous_importance.shape != (sequence_length,):
            raise ValueError("VLA-Cache visual-importance shape is invalid")

        threshold = float(config["similarity_threshold"])
        stable_top_k = int(config["stable_top_k"])
        relevant_top_k = int(config["task_relevant_top_k"])
        reusable = []
        diagnostics = []
        for group_index, group_values in enumerate(config["visual_groups"]):
            positions = torch.as_tensor(
                group_values, dtype=torch.long, device=current_source.device
            )
            previous = previous_source.index_select(1, positions).float()
            current = current_source.index_select(1, positions).float()
            similarities = nn.functional.cosine_similarity(
                previous, current, dim=-1
            )[0]
            eligible = torch.nonzero(
                similarities >= threshold, as_tuple=False
            ).flatten()
            if eligible.numel():
                order = torch.argsort(
                    similarities.index_select(0, eligible),
                    descending=True,
                    stable=True,
                )
                stable_relative = eligible.index_select(0, order[:stable_top_k])
            else:
                stable_relative = eligible
            importance = previous_importance.index_select(0, positions)
            important_relative = torch.argsort(
                importance, descending=True, stable=True
            )[:relevant_top_k]
            stable_set = set(stable_relative.tolist())
            important_set = set(important_relative.tolist())
            reusable_relative = sorted(stable_set - important_set)
            reusable.extend(
                int(positions[index].item()) for index in reusable_relative
            )
            diagnostics.append(
                {
                    "group_index": group_index,
                    "timestep": int(config["visual_group_timesteps"][group_index]),
                    "camera": str(config["visual_group_cameras"][group_index]),
                    "stable_candidates": int(eligible.numel()),
                    "stable_selected": int(stable_relative.numel()),
                    "task_relevant_selected": int(important_relative.numel()),
                    "reusable_selected": len(reusable_relative),
                    "similarity_mean": float(similarities.mean().item()),
                    "similarity_min": float(similarities.min().item()),
                    "similarity_max": float(similarities.max().item()),
                }
            )
        return (
            torch.tensor(
                sorted(reusable), dtype=torch.long, device=current_source.device
            ),
            diagnostics,
        )

    @staticmethod
    def _vla_cache_action_importance(
        attention,
        active_positions,
        action_query_positions,
        visual_positions,
        sequence_length,
    ):
        action_positions = torch.as_tensor(
            action_query_positions,
            dtype=torch.long,
            device=active_positions.device,
        )
        action_rows = torch.searchsorted(active_positions, action_positions)
        if (
            action_rows.numel() != action_positions.numel()
            or int(action_rows.max()) >= active_positions.numel()
            or not torch.equal(
                active_positions.index_select(0, action_rows), action_positions
            )
        ):
            raise RuntimeError("VLA-Cache removed an action query token")
        visual = torch.as_tensor(
            visual_positions,
            dtype=torch.long,
            device=active_positions.device,
        )
        scores = attention.index_select(-2, action_rows).index_select(-1, visual)
        scores = scores.float().mean(dim=(0, 1, 2))
        full = scores.new_zeros(sequence_length)
        full.index_copy_(0, visual, scores)
        return full.detach()

    def _forward_vla_cache(
        self,
        *,
        inputs_embeds,
        attention_mask,
        source_embeds,
        config,
    ):
        if self.training:
            raise ValueError("VLA-Cache is an inference-only path")
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise ValueError("Seer VLA-Cache requires per-rank batch size one")
        if source_embeds is None or source_embeds.shape != inputs_embeds.shape:
            raise ValueError(
                "Seer VLA-Cache requires pre-position source embeddings with the "
                "same [1,N,D] shape as inputs_embeds"
            )
        sequence_length = int(inputs_embeds.shape[1])
        if sequence_length != int(config["total_tokens"]):
            raise ValueError(
                "VLA-Cache token-layout mismatch: "
                f"runtime={sequence_length}, contract={config['total_tokens']}"
            )
        if (
            self._vla_cache_sequence_length is not None
            and self._vla_cache_sequence_length != sequence_length
        ):
            raise ValueError("VLA-Cache sequence length changed within a rollout")
        self._vla_cache_sequence_length = sequence_length

        hidden_states = self.drop(inputs_embeds)
        full_positions = torch.arange(sequence_length, device=inputs_embeds.device)
        first_query = self._vla_cache_key_cache is None
        reuse_enabled = bool(config.get("reuse_enabled", False))
        if first_query:
            self._vla_cache_key_cache = [None] * len(self.h)
            self._vla_cache_value_cache = [None] * len(self.h)

        selection_diagnostics = []
        if (
            reuse_enabled
            and not first_query
            and self._vla_cache_previous_source is not None
            and self._vla_cache_previous_importance is not None
        ):
            reusable, selection_diagnostics = self._vla_cache_select_reusable(
                self._vla_cache_previous_source,
                source_embeds,
                self._vla_cache_previous_importance,
                config,
            )
            schedule = self._vla_cache_layer_schedule(
                self._vla_cache_previous_entropies,
                config["positive_growth_factor"],
            )
        else:
            reusable = full_positions[:0]
            schedule = None

        active_positions = full_positions
        removed_positions = full_positions[:0]
        active_tokens_per_layer = []
        selected_per_pruning_layer = {}
        current_entropies = []
        current_importance = None
        selected_timestep = min(
            self._vla_cache_query_index, int(config["sequence_length"]) - 1
        )
        pruning_layers = set(int(layer) for layer in config["pruning_layers"])

        for layer_index, block in enumerate(self.h):
            if (
                schedule is not None
                and reusable.numel()
                and layer_index in pruning_layers
            ):
                proportion = float(schedule[layer_index].item())
                selected_count = max(1, int(proportion * reusable.numel()))
                selected = reusable[:selected_count]
                if removed_positions.numel() <= selected.numel():
                    keep = ~torch.isin(active_positions, selected)
                    hidden_states = hidden_states[:, keep]
                    active_positions = active_positions[keep]
                    removed_positions = selected
                selected_per_pruning_layer[str(layer_index)] = selected.tolist()

            active_tokens_per_layer.append(int(active_positions.numel()))
            layer_attention_mask = self._vla_cache_attention_rows(
                attention_mask, active_positions
            )
            hidden_states, attention, key, value = block.forward_indexed_reuse(
                hidden_states,
                attention_mask=layer_attention_mask,
                active_positions=active_positions,
                previous_key=self._vla_cache_key_cache[layer_index],
                previous_value=self._vla_cache_value_cache[layer_index],
                sequence_length=sequence_length,
            )
            self._vla_cache_key_cache[layer_index] = key
            self._vla_cache_value_cache[layer_index] = value
            current_entropies.append(
                self._vla_cache_attention_entropy(attention)
            )
            if layer_index == int(config["reference_attention_layer"]):
                current_importance = self._vla_cache_action_importance(
                    attention,
                    active_positions,
                    config["action_query_groups"][selected_timestep],
                    config["visual_positions"],
                    sequence_length,
                )

        hidden_states = self.ln_f(hidden_states)
        if removed_positions.numel():
            if self._vla_cache_previous_full_hidden is None:
                raise RuntimeError("VLA-Cache cannot reconstruct output without history")
            full_hidden = self._vla_cache_previous_full_hidden.clone()
            full_hidden.index_copy_(1, active_positions, hidden_states)
        else:
            full_hidden = hidden_states
        if current_importance is None:
            raise RuntimeError("VLA-Cache did not capture reference-layer attention")

        self._vla_cache_previous_full_hidden = full_hidden.detach()
        self._vla_cache_previous_source = source_embeds.detach().clone()
        self._vla_cache_previous_importance = current_importance
        self._vla_cache_previous_entropies = current_entropies
        self._vla_cache_query_index += 1

        full_token_layers = sequence_length * len(self.h)
        computed_token_layers = sum(active_tokens_per_layer)
        report = {
            "mode": str(config["mode"]),
            "first_query": bool(first_query),
            "query_index": int(self._vla_cache_query_index - 1),
            "selected_action_timestep": int(selected_timestep),
            "sequence_length": sequence_length,
            "visual_tokens": len(config["visual_positions"]),
            "reusable_candidates": int(reusable.numel()),
            "removed_final": int(removed_positions.numel()),
            "active_tokens_per_layer": active_tokens_per_layer,
            "selected_positions_per_pruning_layer": selected_per_pruning_layer,
            "selection_by_timestep_camera": selection_diagnostics,
            "full_token_layers": full_token_layers,
            "computed_token_layers": computed_token_layers,
            "skipped_token_layers": full_token_layers - computed_token_layers,
            "token_layer_reduction": 1.0
            - computed_token_layers / full_token_layers,
            "actual_kv_reuse": bool(removed_positions.numel()),
            "output_reconstructed_from_previous_hidden": bool(
                removed_positions.numel()
            ),
        }
        self.last_vla_cache_report = report
        stats = self._vla_cache_statistics
        stats["calls"] += 1
        stats["first_queries"] += int(first_query)
        stats["actual_kv_reuse_calls"] += int(report["actual_kv_reuse"])
        stats["reusable_candidates"] += report["reusable_candidates"]
        stats["removed_final"] += report["removed_final"]
        stats["full_token_layers"] += full_token_layers
        stats["computed_token_layers"] += computed_token_layers
        return full_hidden

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vla_cache_config=None,
        vla_cache_source_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if vla_cache_config and bool(vla_cache_config.get("enabled", False)):
            return self._forward_vla_cache(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                source_embeds=vla_cache_source_embeds,
                config=vla_cache_config,
            )
        
        input_shape = inputs_embeds.size()[:-1]

        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        for i, block in enumerate(self.h):

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                )

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        
        return hidden_states
