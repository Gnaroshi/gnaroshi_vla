"""Known-range RGB delta encoding without per-view CUDA scalar synchronization."""

from __future__ import annotations

import types
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair


def _ordered_views(images: Tensor, *, normalized_float: bool = False) -> list[Tensor]:
    if not torch.is_tensor(images) or images.ndim != 5:
        raise ValueError("efficient V0 requires uint8 image views [B,V,H,W,C]")
    expected = torch.float32 if normalized_float else torch.uint8
    if images.dtype != expected or images.shape[-1] != 3:
        raise ValueError(f"efficient V0 image cache must use {expected} RGB")
    return [images[:, index] for index in range(images.shape[1])]


def _source_chw(image: Tensor, *, normalized_float: bool) -> Tensor:
    value = image.permute(0, 3, 1, 2).contiguous().float()
    return value if normalized_float else value.div_(255.0)


def _forward_without_scalar_sync(self: Any, pair: NativeV0ObservationPair) -> Tensor:
    normalized_float = getattr(self, "_exact_normalized_float_rgb", False)
    previous_views = _ordered_views(pair.previous_images, normalized_float=normalized_float)
    current_views = _ordered_views(pair.current_images, normalized_float=normalized_float)
    if len(previous_views) != self.num_views or len(current_views) != self.num_views:
        raise ValueError("efficient delta input changed the configured camera count")
    previous_q = pair.previous_proprio
    current_q = pair.current_proprio
    if previous_q.ndim != 2 or current_q.shape != previous_q.shape:
        raise ValueError("previous/current proprio must have identical [B,Q] shapes")
    if previous_q.shape[-1] != self.proprio_dim:
        raise ValueError("efficient delta input changed proprio dimension")

    ordered_features: list[Tensor] = []
    for previous, current in zip(previous_views, current_views):
        previous_f = _source_chw(previous, normalized_float=normalized_float)
        current_f = _source_chw(current, normalized_float=normalized_float).to(previous_f.device)
        previous_f = F.interpolate(
            previous_f,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        current_f = F.interpolate(
            current_f,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        ordered_features.append(
            self.image_encoder(
                torch.cat((previous_f, current_f, current_f - previous_f), dim=1)
            )
        )
    reference = ordered_features[0]
    previous_q = previous_q.to(device=reference.device, dtype=reference.dtype)
    current_q = current_q.to(device=reference.device, dtype=reference.dtype)
    proprio_feature = self.proprio_encoder(
        torch.cat((previous_q, current_q, current_q - previous_q), dim=-1)
    )
    return self.fusion(torch.cat((*ordered_features, proprio_feature), dim=-1))


def install_exact_uint8_delta_path(model: Any) -> dict[str, Any]:
    """Replace only the Python input-conversion branch; parameters stay unchanged."""

    encoder = model.delta_encoder
    encoder._exact_normalized_float_rgb = False
    encoder.forward = types.MethodType(_forward_without_scalar_sync, encoder)
    return {
        "installed": True,
        "parameter_objects_replaced": False,
        "accepted_input": "source uint8 RGB [B,V,H,W,C]",
        "removed_cuda_scalar_sync": "per-view amax().item() range probe",
        "mathematical_conversion": "uint8.float()/255, identical to parent uint8 branch",
    }


def use_normalized_float_delta_inputs(model: Any) -> None:
    """Use only with the live policy's raw_rgb_to_tensor output in [0,1].

    That preprocessing already scales uint8 RGB. Scaling again would change
    the policy. The recorded-input path keeps the strict uint8 default.
    """
    model.delta_encoder._exact_normalized_float_rgb = True
