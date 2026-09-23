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
        return_attention: bool = False,
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

        if return_attention:
            return attn_output, attn_weights
        return attn_output


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
        return_attention: bool = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attention_result = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            return_attention=return_attention,
        )
        if return_attention:
            attn_output, attention_weights = attention_result
        else:
            attn_output = attention_result
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states  # TODO

        if return_attention:
            return hidden_states, attention_weights
        return hidden_states 


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
        self.last_fastv_stats = None
        self.fastv_retention_calls = 0
        self.fastv_retained_token_sum_by_timestep_camera = None
        self.fastv_zero_retention_calls_by_timestep_camera = None
        self.fastv_min_retained_tokens_by_timestep_camera = None
        self.fastv_max_retained_tokens_by_timestep_camera = None

    @staticmethod
    def _prune_attention_mask(attention_mask, keep_indices):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask.index_select(0, keep_indices).index_select(
                1, keep_indices
            )
        if attention_mask.dim() == 3:
            return attention_mask.index_select(1, keep_indices).index_select(
                2, keep_indices
            )
        if attention_mask.dim() == 4:
            return attention_mask.index_select(2, keep_indices).index_select(
                3, keep_indices
            )
        raise ValueError(
            "FastV supports 2D, 3D, or 4D attention masks; "
            f"got shape={tuple(attention_mask.shape)}"
        )

    @classmethod
    def _score_fastv_visual_tokens(
        cls,
        attention_history,
        visual_indices,
        query_indices,
        score_mode,
    ):
        if not attention_history:
            raise ValueError("FastV requires at least one early-layer attention tensor")
        expected_shape = attention_history[0].shape
        if len(expected_shape) != 4:
            raise ValueError(
                "FastV attention must have shape [B,H,Q,K], "
                f"got {tuple(expected_shape)}"
            )
        if any(item.shape != expected_shape for item in attention_history):
            raise ValueError("FastV early-layer attention tensors must share one shape")

        if score_mode in {"text_mean_first_l", "action_mean_first_l"}:
            stacked = torch.stack(attention_history, dim=0)
            selected = stacked.index_select(-2, query_indices).index_select(
                -1, visual_indices
            )
            return selected.mean(dim=(0, 1, 2, 3))
        if score_mode in {"last_token_at_l", "hf_last_action_at_l"}:
            selected = attention_history[-1].index_select(
                -2, query_indices
            ).index_select(-1, visual_indices)
            return selected.mean(dim=(0, 1, 2))
        raise ValueError(f"Unsupported FastV score mode: {score_mode}")

    @classmethod
    def _apply_fastv_pruning(
        cls,
        hidden_states,
        attention_mask,
        attention_history,
        fastv_config,
    ):
        if hidden_states.shape[0] != 1:
            raise ValueError(
                "Seer FastV currently requires per-rank batch size 1 so each policy "
                "context can select its own visual tokens"
            )
        sequence_length = hidden_states.shape[1]
        device = hidden_states.device
        visual_indices = torch.as_tensor(
            fastv_config["visual_token_indices"], dtype=torch.long, device=device
        )
        query_indices = torch.as_tensor(
            fastv_config["score_query_indices"], dtype=torch.long, device=device
        )
        retention_diagnostics = bool(
            fastv_config.get("retention_diagnostics", False)
        )
        visual_timesteps = None
        visual_cameras = None
        num_timesteps = int(fastv_config["sequence_length"])
        num_cameras = len(fastv_config["visual_camera_names"])
        if retention_diagnostics:
            visual_timesteps = torch.as_tensor(
                fastv_config["visual_token_timesteps"],
                dtype=torch.long,
                device=device,
            )
            visual_cameras = torch.as_tensor(
                fastv_config["visual_token_cameras"],
                dtype=torch.long,
                device=device,
            )
            if visual_timesteps.numel() != visual_indices.numel():
                raise ValueError("FastV visual timestep metadata length mismatch")
            if visual_cameras.numel() != visual_indices.numel():
                raise ValueError("FastV visual camera metadata length mismatch")

        # The explicit score mode selects both the query family and whether the
        # first L attention maps or only the final pre-pruning map are aggregated.
        visual_scores = cls._score_fastv_visual_tokens(
            attention_history,
            visual_indices,
            query_indices,
            fastv_config["score_mode"],
        )
        keep_visual_count = max(
            1,
            int(
                round(
                    visual_indices.numel()
                    * (1.0 - float(fastv_config["prune_ratio"]))
                )
            ),
        )
        selected_relative = torch.topk(
            visual_scores,
            keep_visual_count,
            largest=True,
            sorted=False,
        ).indices
        selected_visual = visual_indices.index_select(0, selected_relative)
        retained_group_counts = None
        if retention_diagnostics:
            selected_group_ids = (
                visual_timesteps.index_select(0, selected_relative) * num_cameras
                + visual_cameras.index_select(0, selected_relative)
            )
            retained_group_counts = torch.bincount(
                selected_group_ids,
                minlength=num_timesteps * num_cameras,
            ).view(num_timesteps, num_cameras)

        keep_mask = torch.ones(sequence_length, dtype=torch.bool, device=device)
        keep_mask[visual_indices] = False
        keep_mask[selected_visual] = True
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()

        hidden_states = hidden_states.index_select(1, keep_indices)
        attention_mask = cls._prune_attention_mask(attention_mask, keep_indices)
        return hidden_states, attention_mask, keep_indices, retained_group_counts

    def get_fastv_retention_stats(self):
        if self.fastv_retention_calls == 0:
            return {
                "calls": 0,
                "retained_token_sum_by_timestep_camera": [],
                "zero_retention_calls_by_timestep_camera": [],
                "min_retained_tokens_by_timestep_camera": [],
                "max_retained_tokens_by_timestep_camera": [],
            }

        def as_list(value):
            return value.detach().cpu().tolist()

        return {
            "calls": int(self.fastv_retention_calls),
            "retained_token_sum_by_timestep_camera": as_list(
                self.fastv_retained_token_sum_by_timestep_camera
            ),
            "zero_retention_calls_by_timestep_camera": as_list(
                self.fastv_zero_retention_calls_by_timestep_camera
            ),
            "min_retained_tokens_by_timestep_camera": as_list(
                self.fastv_min_retained_tokens_by_timestep_camera
            ),
            "max_retained_tokens_by_timestep_camera": as_list(
                self.fastv_max_retained_tokens_by_timestep_camera
            ),
        }

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        fastv_config=None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        input_shape = inputs_embeds.size()[:-1]

        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        original_sequence_length = hidden_states.shape[1]
        fastv_enabled = bool(fastv_config and fastv_config.get("enabled", False))
        if fastv_enabled and self.training:
            raise ValueError("FastV is an inference-only token-pruning path")
        prune_layer = int(fastv_config["prune_layer"]) if fastv_enabled else -1
        if fastv_enabled and not 1 <= prune_layer < len(self.h):
            raise ValueError(
                f"FastV prune_layer must be in [1, {len(self.h) - 1}], got {prune_layer}"
            )
        keep_indices = torch.arange(
            original_sequence_length, device=hidden_states.device
        )
        retained_group_counts = None
        selection_attentions = []

        for i, block in enumerate(self.h):

            if fastv_enabled and i == prune_layer:
                if len(selection_attentions) != prune_layer:
                    raise RuntimeError(
                        "FastV did not capture every full-sequence early-layer attention"
                    )
                (
                    hidden_states,
                    attention_mask,
                    keep_indices,
                    retained_group_counts,
                ) = self._apply_fastv_pruning(
                    hidden_states,
                    attention_mask,
                    selection_attentions,
                    fastv_config,
                )
                selection_attentions = []

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    attention_mask,
                )
            else:
                block_output = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    return_attention=(fastv_enabled and i < prune_layer),
                )
                if fastv_enabled and i < prune_layer:
                    hidden_states, selection_attention = block_output
                    selection_attentions.append(selection_attention)
                else:
                    hidden_states = block_output

        hidden_states = self.ln_f(hidden_states)
        if fastv_enabled and hidden_states.shape[1] != original_sequence_length:
            restored_hidden_states = hidden_states.new_zeros(
                hidden_states.shape[0],
                original_sequence_length,
                hidden_states.shape[-1],
            )
            restored_hidden_states.index_copy_(1, keep_indices, hidden_states)
            hidden_states = restored_hidden_states
        if fastv_enabled:
            retention_diagnostics = bool(
                fastv_config.get("retention_diagnostics", False)
            )
            if retention_diagnostics and retained_group_counts is None:
                raise RuntimeError("FastV did not produce visual retention diagnostics")
            if not retention_diagnostics and retained_group_counts is not None:
                raise RuntimeError("FastV unexpectedly produced retention diagnostics")
            if retention_diagnostics:
                retained_group_counts = retained_group_counts.detach().to(dtype=torch.long)
                if self.fastv_retained_token_sum_by_timestep_camera is None:
                    self.fastv_retained_token_sum_by_timestep_camera = torch.zeros_like(
                        retained_group_counts
                    )
                    self.fastv_zero_retention_calls_by_timestep_camera = torch.zeros_like(
                        retained_group_counts
                    )
                    self.fastv_min_retained_tokens_by_timestep_camera = (
                        retained_group_counts.clone()
                    )
                    self.fastv_max_retained_tokens_by_timestep_camera = (
                        retained_group_counts.clone()
                    )
                self.fastv_retention_calls += 1
                self.fastv_retained_token_sum_by_timestep_camera.add_(
                    retained_group_counts
                )
                self.fastv_zero_retention_calls_by_timestep_camera.add_(
                    retained_group_counts.eq(0).to(dtype=torch.long)
                )
                self.fastv_min_retained_tokens_by_timestep_camera.copy_(
                    torch.minimum(
                        self.fastv_min_retained_tokens_by_timestep_camera,
                        retained_group_counts,
                    )
                )
                self.fastv_max_retained_tokens_by_timestep_camera.copy_(
                    torch.maximum(
                        self.fastv_max_retained_tokens_by_timestep_camera,
                        retained_group_counts,
                    )
                )
            self.last_fastv_stats = {
                "enabled": True,
                "prune_layer": prune_layer,
                "prune_ratio": float(fastv_config["prune_ratio"]),
                "tokens_before_pruning": original_sequence_length,
                "tokens_after_pruning": int(keep_indices.numel()),
                "visual_tokens_before_pruning": len(
                    fastv_config["visual_token_indices"]
                ),
                "visual_tokens_after_pruning": int(
                    fastv_config["visual_tokens_after_pruning"]
                ),
                "score_mode": fastv_config["score_mode"],
                "score_layer_count": prune_layer,
                "selection_scope": fastv_config["selection_scope"],
                "retention_diagnostics": retention_diagnostics,
            }
        else:
            self.last_fastv_stats = {
                "enabled": False,
                "tokens_before_pruning": original_sequence_length,
                "tokens_after_pruning": original_sequence_length,
            }
        hidden_states = hidden_states.view(output_shape)
        
        return hidden_states
