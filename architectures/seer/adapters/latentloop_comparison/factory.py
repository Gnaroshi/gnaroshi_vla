"""Attach default-off comparison adapters to a constructed Seer model."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .seer_action_correction import SeerActionCorrectionAdapter
from .seer_nonrecurrent_latent import SeerNonRecurrentLatentAdapter


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def attach_comparison_adapter(model: nn.Module, args: Any) -> dict[str, object]:
    """Attach one selected baseline and return a parameter audit."""

    mode = str(getattr(args, "latentloop_plan_adapter_mode", "off"))
    if mode == "off":
        return {"mode": "off", "attached": False}
    if mode not in {"action_correction", "anchor_bridge"}:
        raise ValueError(f"Unknown latentloop_plan_adapter_mode={mode}")
    if not bool(getattr(model, "use_lrnode_latent_update", False)):
        raise ValueError(f"{mode} requires the matched LatentLoop dimensions")
    if not hasattr(model, "lrnode_delta_encoder") or not hasattr(
        model, "lrnode_dynamics"
    ):
        raise ValueError("Seer model does not expose the locked LatentLoop modules")

    delta_encoder = copy.deepcopy(model.lrnode_delta_encoder)
    target_predictor_parameters = _count(model.lrnode_dynamics)
    common = {
        "delta_encoder": delta_encoder,
        "target_predictor_parameters": target_predictor_parameters,
        "action_pred_steps": int(model.action_pred_steps),
        "motion_dim": int(model.lrnode_motion_dim),
        "hidden_dim": int(getattr(args, "latentloop_plan_adapter_hidden_dim", 0)),
        "maximum_relative_error": float(
            getattr(args, "latentloop_plan_parameter_match_tolerance", 0.05)
        ),
    }
    initialization_seed = int(getattr(args, "seed", 42))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        if mode == "action_correction":
            adapter: nn.Module = SeerActionCorrectionAdapter(**common)
            scientific_name = "matched_action_space_correction"
        else:
            adapter = SeerNonRecurrentLatentAdapter(
                **common,
                latent_dim=int(model.hidden_dim),
            )
            scientific_name = "nonrecurrent_anchor_to_current_latent"
        adapter.apply(model._init_weights)
    model.latentloop_plan_adapter = adapter

    delta_count = _count(adapter.delta_encoder)
    actual_total = _count(adapter)
    predictor_count = actual_total - delta_count
    reference_delta = _count(model.lrnode_delta_encoder)
    reference_total = reference_delta + target_predictor_parameters
    report = {
        "mode": mode,
        "scientific_name": scientific_name,
        "attached": True,
        "initialization_seed": initialization_seed,
        "hidden_dim": int(adapter.parameter_match.hidden_dim),
        "reference_delta_encoder_parameters": reference_delta,
        "baseline_delta_encoder_parameters": delta_count,
        "reference_predictor_parameters": target_predictor_parameters,
        "baseline_predictor_parameters": predictor_count,
        "reference_total_parameters": reference_total,
        "baseline_total_parameters": actual_total,
        "total_relative_error": abs(actual_total - reference_total)
        / float(reference_total),
    }
    model.latentloop_plan_adapter_report = report
    return report


def attach_plan_adapter(model: nn.Module, args: Any) -> dict[str, object]:
    """Compatibility alias for the existing Seer entry point."""

    return attach_comparison_adapter(model, args)
