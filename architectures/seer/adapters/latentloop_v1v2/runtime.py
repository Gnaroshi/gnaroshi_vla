"""Runtime helpers that keep locked Seer V0 source unchanged."""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from methods.latentloop_v1v2.transition import (
    VariableTimeLatentLoopTransition,
    count_trainable_parameters,
)


V0_TRAINABLE_PARAMETERS = 470_146
V1_RATIO_CAP = int(1.25 * V0_TRAINABLE_PARAMETERS)
V1_ABSOLUTE_CAP = 600_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_state(path: Path) -> OrderedDict[str, Tensor]:
    payload = torch.load(path, map_location="cpu")
    raw = payload.get("model_state_dict", payload)
    return OrderedDict(
        (name.removeprefix("module."), value) for name, value in raw.items()
    )


def load_base_and_v0(model: nn.Module, teacher: Path, adapter: Path) -> dict[str, Any]:
    teacher_state = {
        name: value
        for name, value in _checkpoint_state(teacher).items()
        if not name.startswith(("lrnode_", "latentloop_plan_adapter."))
    }
    missing, unexpected = model.load_state_dict(teacher_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected teacher keys: {unexpected[:20]}")
    illegal_missing = [
        name
        for name in missing
        if not name.startswith(("lrnode_delta_encoder.", "lrnode_dynamics."))
    ]
    if illegal_missing:
        raise RuntimeError(f"teacher checkpoint is missing core Seer keys: {illegal_missing[:20]}")
    adapter_state = {
        name: value
        for name, value in _checkpoint_state(adapter).items()
        if name.startswith(("lrnode_delta_encoder.", "lrnode_dynamics."))
    }
    if not adapter_state:
        raise RuntimeError("canonical adapter has no V0 transition parameters")
    expected_adapter_keys = {
        name
        for name in model.state_dict()
        if name.startswith(("lrnode_delta_encoder.", "lrnode_dynamics."))
    }
    if set(adapter_state) != expected_adapter_keys:
        missing_adapter = sorted(expected_adapter_keys - set(adapter_state))
        extra_adapter = sorted(set(adapter_state) - expected_adapter_keys)
        raise RuntimeError(
            "canonical V0 adapter key mismatch: "
            f"missing={missing_adapter[:20]}, extra={extra_adapter[:20]}"
        )
    _, adapter_unexpected = model.load_state_dict(adapter_state, strict=False)
    if adapter_unexpected:
        raise RuntimeError(f"unexpected V0 adapter keys: {adapter_unexpected[:20]}")
    return {
        "teacher_loaded_keys": len(teacher_state),
        "v0_loaded_keys": len(adapter_state),
        "teacher_missing_keys": list(missing),
        "teacher_sha256": sha256_file(teacher),
        "v0_adapter_sha256": sha256_file(adapter),
    }


def attach_v1_transition(model: nn.Module) -> VariableTimeLatentLoopTransition:
    if not hasattr(model, "lrnode_delta_encoder") or not hasattr(model, "lrnode_dynamics"):
        raise RuntimeError("Seer model must construct canonical V0 modules before V1 attach")
    transition = VariableTimeLatentLoopTransition(
        delta_encoder=copy.deepcopy(model.lrnode_delta_encoder),
        dynamics=copy.deepcopy(model.lrnode_dynamics),
        motion_dim=int(model.lrnode_motion_dim),
    )
    transition.mode = "v1_transition"
    model.latentloop_plan_adapter = transition
    model.requires_grad_(False)
    transition.requires_grad_(True)
    report = trainable_parameter_report(model)
    if not report["within_ratio_cap"] or not report["within_absolute_cap"]:
        raise RuntimeError(f"V1 parameter budget exceeded: {report}")
    model.latentloop_v1_parameter_report = report
    return transition


def trainable_parameter_report(model: nn.Module) -> dict[str, int | float | bool]:
    transition = model.latentloop_plan_adapter
    trainable = count_trainable_parameters(transition)
    outside = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("latentloop_plan_adapter.")
    ]
    if outside:
        raise RuntimeError(f"non-V1 parameters remain trainable: {outside[:20]}")
    return {
        "v0_trainable_parameters": V0_TRAINABLE_PARAMETERS,
        "v1_trainable_parameters": trainable,
        "parameter_ratio": trainable / V0_TRAINABLE_PARAMETERS,
        "ratio_cap": V1_RATIO_CAP,
        "absolute_cap": V1_ABSOLUTE_CAP,
        "within_ratio_cap": trainable <= V1_RATIO_CAP,
        "within_absolute_cap": trainable <= V1_ABSOLUTE_CAP,
    }


def adapter_checkpoint_state(transition: nn.Module) -> OrderedDict[str, Tensor]:
    return OrderedDict(
        (
            f"module.latentloop_plan_adapter.{name}",
            value.detach().cpu(),
        )
        for name, value in transition.state_dict().items()
    )


def load_v1_checkpoint(model: nn.Module, checkpoint: Path) -> dict[str, Any]:
    state = _checkpoint_state(checkpoint)
    prefix = "latentloop_plan_adapter."
    adapter_state = {
        name[len(prefix) :]: value for name, value in state.items() if name.startswith(prefix)
    }
    if not adapter_state:
        raise RuntimeError("V1 checkpoint has no latentloop_plan_adapter parameters")
    missing, unexpected = model.latentloop_plan_adapter.load_state_dict(
        adapter_state, strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"V1 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "loaded_keys": len(adapter_state),
        "checkpoint_sha256": sha256_file(checkpoint),
        "state_sha256": state_dict_sha256(adapter_state),
    }
