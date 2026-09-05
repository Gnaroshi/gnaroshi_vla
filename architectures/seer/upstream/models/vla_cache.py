"""Seer-specific VLA-Cache contract and token-layout helpers.

The published VLA-Cache implementation operates on spatial image tokens in an
autoregressive VLM decoder.  Seer instead exposes learned Perceiver latents and
two image CLS tokens to its causal action transformer.  This module preserves
the published stable-minus-task-relevant selection ratios and layer schedule,
but applies temporal similarity to Seer's actual projected visual-condition
tokens.  It deliberately does not pretend that Perceiver latent indices are
spatial image patches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


VLA_CACHE_REPOSITORY = "https://github.com/siyuhsu/VLA-Cache"
VLA_CACHE_COMMIT = "a4909880573868dee2769343d52e793c0341678b"
VLA_CACHE_TRANSFORMERS_REPOSITORY = "https://github.com/siyuhsu/transformers"
VLA_CACHE_TRANSFORMERS_BRANCH = "vla-cache-openvla"
VLA_CACHE_TRANSFORMERS_COMMIT = "2302fce58afa3a4f8461625b1394f9e9c8a7f1ea"

VLA_CACHE_MODES = {"off", "matched_full", "reuse"}


@dataclass(frozen=True)
class SeerVLACacheLayout:
    sequence_length: int
    tokens_per_timestep: int
    total_tokens: int
    visual_tokens_per_view: int
    visual_positions: tuple[int, ...]
    visual_groups: tuple[tuple[int, ...], ...]
    visual_group_timesteps: tuple[int, ...]
    visual_group_cameras: tuple[str, ...]
    action_query_groups: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["visual_positions"] = list(self.visual_positions)
        payload["visual_groups"] = [list(group) for group in self.visual_groups]
        payload["visual_group_timesteps"] = list(self.visual_group_timesteps)
        payload["visual_group_cameras"] = list(self.visual_group_cameras)
        payload["action_query_groups"] = [
            list(group) for group in self.action_query_groups
        ]
        return payload


def parse_pruning_layers(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        fields = [field.strip() for field in value.split(",") if field.strip()]
        if not fields:
            raise ValueError("VLA-Cache pruning layer list cannot be empty")
        layers = tuple(int(field) for field in fields)
    else:
        layers = tuple(int(layer) for layer in value)
    if tuple(sorted(set(layers))) != layers:
        raise ValueError(
            "VLA-Cache pruning layers must be unique and strictly increasing"
        )
    return layers


def build_seer_vla_cache_layout(
    *,
    sequence_length: int,
    num_resampler_query: int,
    num_obs_token_per_image: int,
    obs_pred: bool,
    action_pred_steps: int,
) -> SeerVLACacheLayout:
    sequence_length = int(sequence_length)
    num_resampler_query = int(num_resampler_query)
    num_obs_token_per_image = int(num_obs_token_per_image)
    action_pred_steps = int(action_pred_steps)
    if sequence_length < 1:
        raise ValueError("VLA-Cache requires sequence_length >= 1")
    if num_resampler_query < 1:
        raise ValueError("VLA-Cache requires num_resampler_query >= 1")
    if action_pred_steps < 1:
        raise ValueError("VLA-Cache requires action_pred_steps >= 1")

    # Seer token order in every timestep is:
    # language, proprioception, primary resampler, wrist resampler,
    # primary CLS, wrist CLS, optional observation queries, action queries.
    visual_start = 2
    primary_resampler = tuple(
        range(visual_start, visual_start + num_resampler_query)
    )
    wrist_resampler = tuple(
        range(
            visual_start + num_resampler_query,
            visual_start + 2 * num_resampler_query,
        )
    )
    primary_cls = visual_start + 2 * num_resampler_query
    wrist_cls = primary_cls + 1
    context_tokens = wrist_cls + 1
    observation_queries = 2 * num_obs_token_per_image if bool(obs_pred) else 0
    action_start = context_tokens + observation_queries
    tokens_per_timestep = action_start + action_pred_steps

    visual_positions: list[int] = []
    visual_groups: list[tuple[int, ...]] = []
    visual_group_timesteps: list[int] = []
    visual_group_cameras: list[str] = []
    action_query_groups: list[tuple[int, ...]] = []
    for timestep in range(sequence_length):
        base = timestep * tokens_per_timestep
        primary = tuple(base + position for position in primary_resampler) + (
            base + primary_cls,
        )
        wrist = tuple(base + position for position in wrist_resampler) + (
            base + wrist_cls,
        )
        for camera, group in (("primary", primary), ("wrist", wrist)):
            visual_groups.append(group)
            visual_positions.extend(group)
            visual_group_timesteps.append(timestep)
            visual_group_cameras.append(camera)
        action_query_groups.append(
            tuple(
                range(
                    base + action_start,
                    base + action_start + action_pred_steps,
                )
            )
        )

    return SeerVLACacheLayout(
        sequence_length=sequence_length,
        tokens_per_timestep=tokens_per_timestep,
        total_tokens=sequence_length * tokens_per_timestep,
        visual_tokens_per_view=num_resampler_query + 1,
        visual_positions=tuple(visual_positions),
        visual_groups=tuple(visual_groups),
        visual_group_timesteps=tuple(visual_group_timesteps),
        visual_group_cameras=tuple(visual_group_cameras),
        action_query_groups=tuple(action_query_groups),
    )


def build_seer_vla_cache_config(
    *,
    mode: str,
    pruning_layers: str | tuple[int, ...] | list[int],
    reference_attention_layer: int,
    similarity_threshold: float,
    positive_growth_factor: float,
    transformer_layers: int,
    sequence_length: int,
    num_resampler_query: int,
    num_obs_token_per_image: int,
    obs_pred: bool,
    action_pred_steps: int,
) -> dict[str, object]:
    mode = str(mode)
    if mode not in VLA_CACHE_MODES:
        raise ValueError(
            f"Unknown VLA-Cache mode {mode!r}; expected one of {sorted(VLA_CACHE_MODES)}"
        )
    layers = parse_pruning_layers(pruning_layers)
    transformer_layers = int(transformer_layers)
    reference_attention_layer = int(reference_attention_layer)
    if mode != "off":
        if min(layers) < 1 or max(layers) >= transformer_layers - 1:
            raise ValueError(
                "VLA-Cache pruning layers must be in [1, transformer_layers - 2] "
                "because the official entropy schedule omits the final layer"
            )
        if not 0 <= reference_attention_layer < transformer_layers:
            raise ValueError("VLA-Cache reference attention layer is out of range")
    similarity_threshold = float(similarity_threshold)
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("VLA-Cache similarity threshold must be in [-1, 1]")
    positive_growth_factor = float(positive_growth_factor)
    if not 0.0 <= positive_growth_factor <= 1.0:
        raise ValueError("VLA-Cache growth factor must be in [0, 1]")

    layout = build_seer_vla_cache_layout(
        sequence_length=sequence_length,
        num_resampler_query=num_resampler_query,
        num_obs_token_per_image=num_obs_token_per_image,
        obs_pred=obs_pred,
        action_pred_steps=action_pred_steps,
    )
    source_visual_tokens_per_view = 256
    source_stable_top_k = 150
    source_task_relevant_top_k = 100
    stable_top_k = max(
        1,
        round(
            source_stable_top_k
            * layout.visual_tokens_per_view
            / source_visual_tokens_per_view
        ),
    )
    task_relevant_top_k = max(
        1,
        round(
            source_task_relevant_top_k
            * layout.visual_tokens_per_view
            / source_visual_tokens_per_view
        ),
    )
    if stable_top_k > layout.visual_tokens_per_view:
        raise ValueError("Scaled stable-token count exceeds Seer visual group size")
    if task_relevant_top_k > layout.visual_tokens_per_view:
        raise ValueError("Scaled task-relevant count exceeds Seer visual group size")

    payload: dict[str, object] = {
        "mode": mode,
        "enabled": mode != "off",
        "reuse_enabled": mode == "reuse",
        "pruning_layers": list(layers),
        "reference_attention_layer": reference_attention_layer,
        "similarity_threshold": similarity_threshold,
        "positive_growth_factor": positive_growth_factor,
        "source_visual_tokens_per_view": source_visual_tokens_per_view,
        "source_stable_top_k": source_stable_top_k,
        "source_task_relevant_top_k": source_task_relevant_top_k,
        "stable_top_k": stable_top_k,
        "task_relevant_top_k": task_relevant_top_k,
        "selection_signal": "projected_visual_condition_token_cosine",
        "task_relevance_signal": "previous_selected_action_query_attention",
        "token_alignment": "fixed_relative_history_slot_and_camera",
        "official_repository": VLA_CACHE_REPOSITORY,
        "official_commit": VLA_CACHE_COMMIT,
        "official_transformers_repository": VLA_CACHE_TRANSFORMERS_REPOSITORY,
        "official_transformers_branch": VLA_CACHE_TRANSFORMERS_BRANCH,
        "official_transformers_commit": VLA_CACHE_TRANSFORMERS_COMMIT,
        "architecture_adaptation": (
            "Seer Perceiver latents are learned condition tokens, not spatial patches; "
            "stability is therefore measured on the actual projected condition tokens."
        ),
    }
    payload.update(layout.to_dict())
    return payload
