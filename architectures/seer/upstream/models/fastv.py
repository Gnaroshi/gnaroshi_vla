from dataclasses import asdict, dataclass


FASTV_SCORE_MODES = {
    "text_mean_first_l",
    "last_token_at_l",
    "action_mean_first_l",
    # Backward-compatible name used by earlier local FastV experiments.
    "hf_last_action_at_l",
}


@dataclass(frozen=True)
class SeerFastVLayout:
    sequence_length: int
    tokens_per_timestep: int
    total_tokens: int
    visual_tokens_per_timestep: int
    total_visual_tokens: int
    visual_token_indices: tuple
    visual_token_timesteps: tuple
    visual_token_cameras: tuple
    visual_token_kinds: tuple
    score_query_indices: tuple
    score_mode: str

    def to_dict(self):
        payload = asdict(self)
        payload["visual_token_indices"] = list(self.visual_token_indices)
        payload["visual_token_timesteps"] = list(self.visual_token_timesteps)
        payload["visual_token_cameras"] = list(self.visual_token_cameras)
        payload["visual_token_kinds"] = list(self.visual_token_kinds)
        payload["score_query_indices"] = list(self.score_query_indices)
        return payload


def build_seer_fastv_layout(
    *,
    sequence_length,
    num_resampler_query,
    num_obs_token_per_image,
    obs_pred,
    action_pred_steps,
    score_mode,
):
    sequence_length = int(sequence_length)
    num_resampler_query = int(num_resampler_query)
    num_obs_token_per_image = int(num_obs_token_per_image)
    action_pred_steps = int(action_pred_steps)
    score_mode = str(score_mode)

    if sequence_length < 1:
        raise ValueError("FastV requires sequence_length >= 1")
    if num_resampler_query < 1:
        raise ValueError("FastV requires num_resampler_query >= 1")
    if action_pred_steps < 1:
        raise ValueError("FastV requires action_pred_steps >= 1")
    if score_mode not in FASTV_SCORE_MODES:
        raise ValueError(
            f"Unknown FastV score mode {score_mode!r}; expected one of "
            f"{sorted(FASTV_SCORE_MODES)}"
        )

    # Per timestep: language, proprioception, primary/wrist resampler tokens,
    # primary/wrist CLS tokens, optional observation-prediction tokens, actions.
    visual_start = 2
    visual_tokens_per_timestep = 2 * num_resampler_query + 2
    context_tokens = visual_start + visual_tokens_per_timestep
    observation_tokens = 2 * num_obs_token_per_image if bool(obs_pred) else 0
    action_start = context_tokens + observation_tokens
    tokens_per_timestep = action_start + action_pred_steps

    visual_indices = []
    visual_timesteps = []
    visual_cameras = []
    visual_kinds = []
    local_visual_groups = (
        [(0, "resampler")] * num_resampler_query
        + [(1, "resampler")] * num_resampler_query
        + [(0, "cls"), (1, "cls")]
    )
    for timestep in range(sequence_length):
        base = timestep * tokens_per_timestep
        timestep_indices = list(
            range(base + visual_start, base + visual_start + visual_tokens_per_timestep)
        )
        visual_indices.extend(timestep_indices)
        for camera, token_kind in local_visual_groups:
            visual_timesteps.append(timestep)
            visual_cameras.append(camera)
            visual_kinds.append(token_kind)

    language_indices = tuple(
        timestep * tokens_per_timestep for timestep in range(sequence_length)
    )
    last_step_base = (sequence_length - 1) * tokens_per_timestep
    last_step_actions = tuple(
        range(
            last_step_base + action_start,
            last_step_base + action_start + action_pred_steps,
        )
    )
    if score_mode == "text_mean_first_l":
        score_query_indices = language_indices
    elif score_mode == "action_mean_first_l":
        score_query_indices = last_step_actions
    else:
        score_query_indices = (last_step_actions[-1],)

    return SeerFastVLayout(
        sequence_length=sequence_length,
        tokens_per_timestep=tokens_per_timestep,
        total_tokens=sequence_length * tokens_per_timestep,
        visual_tokens_per_timestep=visual_tokens_per_timestep,
        total_visual_tokens=len(visual_indices),
        visual_token_indices=tuple(visual_indices),
        visual_token_timesteps=tuple(visual_timesteps),
        visual_token_cameras=tuple(visual_cameras),
        visual_token_kinds=tuple(visual_kinds),
        score_query_indices=tuple(score_query_indices),
        score_mode=score_mode,
    )


def build_seer_fastv_config(
    *,
    enabled,
    prune_layer,
    prune_ratio,
    transformer_layers,
    sequence_length,
    num_resampler_query,
    num_obs_token_per_image,
    obs_pred,
    action_pred_steps,
    score_mode,
    retention_diagnostics=False,
):
    layout = build_seer_fastv_layout(
        sequence_length=sequence_length,
        num_resampler_query=num_resampler_query,
        num_obs_token_per_image=num_obs_token_per_image,
        obs_pred=obs_pred,
        action_pred_steps=action_pred_steps,
        score_mode=score_mode,
    )
    visual_indices = list(layout.visual_token_indices)
    score_query_indices = list(layout.score_query_indices)
    visual_timesteps = list(layout.visual_token_timesteps)
    visual_cameras = list(layout.visual_token_cameras)
    if not visual_indices or len(set(visual_indices)) != len(visual_indices):
        raise ValueError("FastV visual-token indices must be non-empty and unique")
    if not score_query_indices:
        raise ValueError("FastV score-query indices must be non-empty")
    if min(visual_indices) < 0 or max(visual_indices) >= layout.total_tokens:
        raise ValueError("FastV visual-token indices are outside the input sequence")
    if min(score_query_indices) < 0 or max(score_query_indices) >= layout.total_tokens:
        raise ValueError("FastV score-query indices are outside the input sequence")
    if len(visual_timesteps) != len(visual_indices):
        raise ValueError("FastV visual timestep metadata length mismatch")
    if len(visual_cameras) != len(visual_indices):
        raise ValueError("FastV visual camera metadata length mismatch")
    if min(visual_timesteps) < 0 or max(visual_timesteps) >= layout.sequence_length:
        raise ValueError("FastV visual timestep metadata is outside the context")
    if min(visual_cameras) < 0 or max(visual_cameras) >= 2:
        raise ValueError("FastV visual camera metadata is outside the camera set")
    enabled = bool(enabled)
    prune_layer = int(prune_layer)
    prune_ratio = float(prune_ratio)
    transformer_layers = int(transformer_layers)

    if not 1 <= prune_layer < transformer_layers:
        raise ValueError(
            "FastV prune_layer must satisfy 1 <= prune_layer < transformer_layers; "
            f"got prune_layer={prune_layer}, transformer_layers={transformer_layers}"
        )
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError(
            f"FastV prune_ratio must be in [0, 1), got {prune_ratio}"
        )

    keep_visual_tokens = max(
        1, int(round(layout.total_visual_tokens * (1.0 - prune_ratio)))
    )
    dropped_visual_tokens = layout.total_visual_tokens - keep_visual_tokens
    active_keep_visual_tokens = (
        keep_visual_tokens if enabled else layout.total_visual_tokens
    )
    active_dropped_visual_tokens = dropped_visual_tokens if enabled else 0
    return {
        "enabled": enabled,
        "prune_layer": prune_layer,
        "prune_ratio": prune_ratio,
        "score_mode": layout.score_mode,
        "retention_diagnostics": bool(enabled and retention_diagnostics),
        "score_layer_count": prune_layer if enabled else 0,
        "visual_token_indices": visual_indices,
        "visual_token_timesteps": visual_timesteps,
        "visual_token_cameras": visual_cameras,
        "visual_token_kinds": list(layout.visual_token_kinds),
        "visual_camera_names": ["primary", "wrist"],
        "selection_scope": "global_visual",
        "original_visual_tokens_by_timestep_camera": [
            [num_resampler_query + 1, num_resampler_query + 1]
            for _ in range(layout.sequence_length)
        ],
        "score_query_indices": score_query_indices,
        "tokens_before_pruning": layout.total_tokens,
        "tokens_after_pruning": layout.total_tokens - active_dropped_visual_tokens,
        "visual_tokens_before_pruning": layout.total_visual_tokens,
        "visual_tokens_after_pruning": active_keep_visual_tokens,
        "dropped_visual_tokens": active_dropped_visual_tokens,
        "tokens_per_timestep": layout.tokens_per_timestep,
        "visual_tokens_per_timestep": layout.visual_tokens_per_timestep,
        "sequence_length": layout.sequence_length,
        "transformer_layers": transformer_layers,
        "pruned_transformer_layers": transformer_layers - prune_layer if enabled else 0,
    }
