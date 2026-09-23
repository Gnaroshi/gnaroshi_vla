"""Default-off attachment and parameter audit for Seer's joint protocol."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from methods.joint_latent_action_surrogate.modules import (
    joint_surrogate_parameter_count,
    wide_control_parameter_count,
)

from .seer_joint import SeerJointLatentActionAdapter


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _find_wide_hidden(target: int, model: nn.Module) -> tuple[int, int, float]:
    candidates = []
    for hidden in range(8, 513):
        count = wide_control_parameter_count(
            latent_dim=int(model.hidden_dim),
            motion_dim=int(model.lrnode_motion_dim),
            action_pred_steps=int(model.action_pred_steps),
            hidden_dim=hidden,
        )
        candidates.append((abs(count - target), hidden, count))
    _, hidden, count = min(candidates)
    return hidden, count, abs(count - target) / float(target)


def attach_joint_latent_action_surrogate(model: nn.Module, args: Any) -> dict[str, object]:
    mode = str(getattr(args, "joint_latent_action_surrogate_mode", "off"))
    if mode == "off":
        return {"mode": "off", "attached": False}
    if mode not in {"joint", "wide"}:
        raise ValueError(f"Unknown joint_latent_action_surrogate_mode={mode!r}")
    if not bool(getattr(model, "use_lrnode_latent_update", False)):
        raise ValueError("The joint protocol requires canonical LatentLoop modules")
    if int(model.action_pred_steps) != 3:
        raise ValueError("The source-locked Seer joint protocol requires P=3")

    canonical_encoder = _count(model.lrnode_delta_encoder)
    canonical_updater = _count(model.lrnode_dynamics)
    canonical_total = canonical_encoder + canonical_updater
    surrogate_hidden = int(
        getattr(args, "joint_latent_action_surrogate_hidden_dim", 192)
    )
    surrogate_count = joint_surrogate_parameter_count(
        latent_dim=int(model.hidden_dim),
        motion_dim=int(model.lrnode_motion_dim),
        action_pred_steps=int(model.action_pred_steps),
        hidden_dim=surrogate_hidden,
    )
    wide_hidden, wide_count, wide_error = _find_wide_hidden(surrogate_count, model)
    match_tolerance = float(
        getattr(args, "joint_latent_action_surrogate_parameter_match_tolerance", 0.02)
    )
    if wide_error > match_tolerance:
        raise ValueError(
            "Wide-control parameter match failed: "
            f"target={surrogate_count}, actual={wide_count}, error={wide_error:.6f}"
        )

    seed = int(getattr(args, "seed", 42))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adapter = SeerJointLatentActionAdapter(
            mode=mode,
            latent_dim=int(model.hidden_dim),
            motion_dim=int(model.lrnode_motion_dim),
            action_pred_steps=int(model.action_pred_steps),
            surrogate_hidden_dim=surrogate_hidden,
            wide_hidden_dim=wide_hidden,
        )
        adapter.apply(model._init_weights)
        if adapter.wide_control is not None:
            nn.init.zeros_(adapter.wide_control.network[-1].weight)
            nn.init.zeros_(adapter.wide_control.network[-1].bias)

    model.joint_latent_action_surrogate = adapter
    added = _count(adapter)
    total_trainable = canonical_total + added
    ratio = total_trainable / float(canonical_total)
    if ratio > 1.25:
        raise ValueError(
            f"Joint trainable budget exceeds 1.25x canonical: {ratio:.6f}"
        )
    report = {
        "mode": mode,
        "scientific_name": (
            "joint_latent_anchored_action_surrogate"
            if mode == "joint"
            else "parameter_matched_wide_latentloop"
        ),
        "attached": True,
        "initialization_seed": seed,
        "canonical_encoder_parameters": canonical_encoder,
        "canonical_updater_parameters": canonical_updater,
        "canonical_total_parameters": canonical_total,
        "added_parameters": added,
        "total_trainable_stage_b": total_trainable,
        "parameter_ratio_to_canonical": ratio,
        "surrogate_hidden_dim": surrogate_hidden,
        "wide_hidden_dim": wide_hidden,
        "joint_added_reference": surrogate_count,
        "wide_added_reference": wide_count,
        "joint_wide_relative_error": wide_error,
        "match_tolerance": match_tolerance,
    }
    model.joint_latent_action_surrogate_report = report
    return report
