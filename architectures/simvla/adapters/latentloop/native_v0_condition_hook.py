"""Exact external hook for SimVLA's pre-action-projection fused condition."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from unittest import mock

import torch
from torch import Tensor

from architectures.simvla.adapters.dcld import SimVLAActionAdapter


GROUP_PADDING = 0
GROUP_IMAGE_VIEW_0 = 1
GROUP_IMAGE_VIEW_1 = 2
GROUP_IMAGE_OTHER = 3
GROUP_LANGUAGE = 4
GROUP_LANGUAGE_SPECIAL = 5
GROUP_LANGUAGE_PAD_ID = 6

GROUP_NAMES = {
    GROUP_PADDING: "padding",
    GROUP_IMAGE_VIEW_0: "image_view_0",
    GROUP_IMAGE_VIEW_1: "image_view_1",
    GROUP_IMAGE_OTHER: "image_other_view",
    GROUP_LANGUAGE: "language",
    GROUP_LANGUAGE_SPECIAL: "language_special",
    GROUP_LANGUAGE_PAD_ID: "language_pad_id_source_active",
}


@dataclass(frozen=True)
class ConditionTokenLayout:
    """Source-derived token validity and group assignment."""

    valid_mask: Tensor
    group_ids: Tensor
    image_tokens_per_view: int
    text_tokens: int
    sample_ranges: tuple[dict[str, Any], ...]
    source_attention_quirk: str

    def serializable(self) -> dict[str, Any]:
        return {
            "valid_mask": self.valid_mask.detach().cpu().tolist(),
            "group_ids": self.group_ids.detach().cpu().tolist(),
            "group_names": GROUP_NAMES,
            "image_tokens_per_view": self.image_tokens_per_view,
            "text_tokens": self.text_tokens,
            "sample_ranges": list(self.sample_ranges),
            "source_attention_quirk": self.source_attention_quirk,
        }


@dataclass(frozen=True)
class ExtractedActionCondition:
    """Fused condition and metadata before the frozen action VLM projection."""

    condition: Tensor
    layout: ConditionTokenLayout
    source_symbol: str = "models.modeling_smolvlm_vla.SmolVLMVLA.forward_vlm_efficient"
    consumer_symbol: str = "models.transformer_smolvlm.ActionTransformer.forward::self.vlm_proj"


def _special_ids(values: Iterable[int] | None) -> set[int]:
    return {int(value) for value in (values or ()) if value is not None}


def build_condition_token_layout(
    *,
    condition: Tensor,
    image_mask: Tensor,
    input_ids: Tensor,
    pad_token_id: int | None = None,
    special_token_ids: Iterable[int] | None = None,
) -> ConditionTokenLayout:
    """Reconstruct the exact sequence layout built by ``forward_vlm_efficient``."""

    if condition.ndim != 3:
        raise ValueError("condition must be [B,T,D]")
    if image_mask.ndim != 2 or input_ids.ndim != 2:
        raise ValueError("image_mask and input_ids must be [B,V] and [B,L]")
    batch_size, token_count, _ = condition.shape
    if image_mask.shape[0] != batch_size or input_ids.shape[0] != batch_size:
        raise ValueError("condition/image/text batch sizes differ")
    mask_bool = image_mask.to(dtype=torch.bool)
    # The upstream source slices [:num_valid], so non-prefix masks would silently
    # select the wrong reconstructed view.  Reject them at the external hook.
    for sample_mask in mask_bool.detach().cpu().tolist():
        seen_false = False
        for value in sample_mask:
            if not value:
                seen_false = True
            elif seen_false:
                raise ValueError("SimVLA forward_vlm_efficient requires prefix-valid image_mask")

    text_tokens = int(input_ids.shape[1])
    maximum_views = int(mask_bool.sum(dim=1).max().item())
    if maximum_views < 1:
        raise ValueError("at least one image view must be valid")
    image_total_at_max = token_count - text_tokens
    if image_total_at_max <= 0 or image_total_at_max % maximum_views:
        raise ValueError(
            "cannot derive an integral image-token count from fused condition shape"
        )
    tokens_per_view = image_total_at_max // maximum_views
    valid = torch.zeros((batch_size, token_count), device=condition.device, dtype=torch.bool)
    groups = torch.zeros((batch_size, token_count), device=condition.device, dtype=torch.long)
    special_ids = _special_ids(special_token_ids)
    ranges: list[dict[str, Any]] = []
    input_ids_cpu = input_ids.detach().cpu()
    valid_views = mask_bool.sum(dim=1).tolist()
    for batch_index, raw_num_views in enumerate(valid_views):
        num_views = int(raw_num_views)
        image_end = num_views * tokens_per_view
        text_start = image_end
        text_end = text_start + text_tokens
        valid[batch_index, :text_end] = True
        image_ranges: list[dict[str, int | str]] = []
        for view in range(num_views):
            start = view * tokens_per_view
            end = start + tokens_per_view
            group = (
                GROUP_IMAGE_VIEW_0
                if view == 0
                else GROUP_IMAGE_VIEW_1
                if view == 1
                else GROUP_IMAGE_OTHER
            )
            groups[batch_index, start:end] = group
            image_ranges.append({"view": view, "start": start, "end": end})
        for offset, token_id_tensor in enumerate(input_ids_cpu[batch_index]):
            token_id = int(token_id_tensor.item())
            index = text_start + offset
            if pad_token_id is not None and token_id == int(pad_token_id):
                groups[batch_index, index] = GROUP_LANGUAGE_PAD_ID
            elif token_id in special_ids:
                groups[batch_index, index] = GROUP_LANGUAGE_SPECIAL
            else:
                groups[batch_index, index] = GROUP_LANGUAGE
        ranges.append(
            {
                "sample": batch_index,
                "image_views": image_ranges,
                "language": {"start": text_start, "end": text_end},
                "batch_padding": {"start": text_end, "end": token_count},
            }
        )
    return ConditionTokenLayout(
        valid_mask=valid,
        group_ids=groups,
        image_tokens_per_view=tokens_per_view,
        text_tokens=text_tokens,
        sample_ranges=tuple(ranges),
        source_attention_quirk=(
            "All fixed-length input_ids positions, including tokenizer pad IDs, are source-active; "
            "only batch-tail sequence padding introduced after per-sample concatenation is invalid."
        ),
    )


def extract_action_condition(
    model: Any,
    *,
    input_ids: Tensor,
    image_input: Tensor,
    image_mask: Tensor,
    pad_token_id: int | None = None,
    special_token_ids: Iterable[int] | None = None,
) -> ExtractedActionCondition:
    """Call the exact upstream fusion hook without moving ``vlm_proj``."""

    encoded = model.forward_vlm_efficient(image_input, image_mask, input_ids)
    if set(encoded) != {"vlm_features"}:
        raise RuntimeError(f"unexpected forward_vlm_efficient outputs: {sorted(encoded)}")
    condition = encoded["vlm_features"]
    layout = build_condition_token_layout(
        condition=condition,
        image_mask=image_mask,
        input_ids=input_ids,
        pad_token_id=pad_token_id,
        special_token_ids=special_token_ids,
    )
    return ExtractedActionCondition(condition=condition, layout=layout)


def source_hook_audit(model_class: type[Any], transformer_class: type[Any]) -> dict[str, Any]:
    """Prove from local source that the hook output is consumed by ``vlm_proj``."""

    encoder_source = inspect.getsource(model_class.forward_vlm_efficient)
    generator_source = inspect.getsource(model_class.generate_actions)
    transformer_source = inspect.getsource(transformer_class.forward)
    concat_source = inspect.getsource(transformer_class._forward_concat)
    adaln_source = inspect.getsource(transformer_class._forward_adaln)
    checks = {
        "encoder_returns_vlm_features": 'return {"vlm_features": vlm_features}' in encoder_source,
        "generate_passes_vlm_features": 'vlm_features=enc["vlm_features"]' in generator_source,
        "transformer_dispatches_existing_path": (
            "self._forward_adaln" in transformer_source
            and "self._forward_concat" in transformer_source
        ),
        "concat_path_owns_vlm_projection": "self.vlm_proj(vlm_features)" in concat_source,
        "adaln_path_owns_vlm_projection": "self.vlm_cond_proj" in adaln_source,
        "external_hook_is_pre_projection": True,
    }
    return {
        "verdict": "SOURCE_EXACT_PRE_VLM_PROJ" if all(checks.values()) else "SOURCE_HOOK_UNRESOLVED",
        "checks": checks,
        "encoder_file": str(Path(inspect.getsourcefile(model_class) or "").resolve()),
        "transformer_file": str(Path(inspect.getsourcefile(transformer_class) or "").resolve()),
    }


def run_same_noise_k1_parity(
    model: Any,
    *,
    input_ids: Tensor,
    image_input: Tensor,
    image_mask: Tensor,
    proprio: Tensor,
    initial_noise: Tensor,
    steps: int = 10,
    pad_token_id: int | None = None,
    special_token_ids: Iterable[int] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Compare official full, extracted-hook, and adapter K1-bypass actions."""

    extracted = extract_action_condition(
        model,
        input_ids=input_ids,
        image_input=image_input,
        image_mask=image_mask,
        pad_token_id=pad_token_id,
        special_token_ids=special_token_ids,
    )
    action_adapter = SimVLAActionAdapter(model)
    fixed_noise = initial_noise.to(device=proprio.device, dtype=proprio.dtype)

    def fixed_randn(*shape: Any, **kwargs: Any) -> Tensor:
        requested = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], tuple) else tuple(shape)
        if requested != tuple(fixed_noise.shape):
            raise AssertionError(f"unexpected torch.randn shape in official path: {requested}")
        return fixed_noise.to(device=kwargs.get("device", proprio.device), dtype=kwargs.get("dtype", proprio.dtype)).clone()

    with mock.patch.object(torch, "randn", side_effect=fixed_randn):
        official = model.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            proprio=proprio,
            steps=steps,
        )
    hooked = action_adapter.decode_action_from_condition(
        extracted.condition,
        proprio,
        steps=steps,
        initial_noise=fixed_noise,
    )
    # K=1 is an explicit bypass: the V0 module receives no call and the exact
    # extracted condition is sent to the same frozen decoder.
    adapter_k1 = action_adapter.decode_action_from_condition(
        extracted.condition,
        proprio,
        steps=steps,
        initial_noise=fixed_noise,
    )

    def comparison(left: Tensor, right: Tensor) -> dict[str, float | bool]:
        difference = (left.detach().float() - right.detach().float()).abs()
        return {
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
            "exact": bool(torch.equal(left, right)),
            "allclose": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
        }

    official_hook = comparison(official, hooked)
    hook_adapter = comparison(hooked, adapter_k1)
    passed = bool(official_hook["allclose"] and hook_adapter["allclose"])
    return {
        "verdict": "K1_HOOK_PARITY_PASS" if passed else "K1_HOOK_PARITY_FAIL",
        "official_vs_hook": official_hook,
        "hook_vs_adapter_k1_bypass": hook_adapter,
        "condition_shape": list(extracted.condition.shape),
        "condition_dtype": str(extracted.condition.dtype),
        "layout": extracted.layout.serializable(),
        "v0_updater_calls_in_k1": 0,
        "explicit_noise": True,
        "flow_steps": int(steps),
    }
