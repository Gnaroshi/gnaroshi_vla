"""Expose native Condition internals without changing its computation."""

from __future__ import annotations

import torch
from torch import Tensor

from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair
from methods.latentloop.modules.shared_refinement import RefinementContext


@torch.no_grad()
def condition_query(adapter, sequence: dict, age: int):
    if age not in (1, 2, 3):
        raise ValueError(age)
    previous = sequence["anchor_condition"] if age == 1 else sequence["teacher_conditions"][:, 1]
    batch, tokens, _ = previous.shape
    if age == 2:
        condition = previous
        global_code = previous.new_zeros(batch, adapter.delta_dim)
        token_code = previous.new_zeros(batch, tokens, adapter.condition_updater.down.out_features + 1)
    else:
        captured = []
        handle = adapter.condition_updater.up.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
        try:
            pair = NativeV0ObservationPair(
                previous_images=sequence["image_sequence"][:, age - 1],
                current_images=sequence["image_sequence"][:, age],
                previous_proprio=sequence["proprio_sequence"][:, age - 1],
                current_proprio=sequence["proprio_sequence"][:, age],
            )
            global_code = adapter.delta_encoder(pair)
            update = adapter.condition_updater(previous, global_code,
                valid_mask=sequence["valid_mask"], group_ids=sequence["group_ids"], age=1)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected one native Condition hidden capture")
        condition = update.condition
        # Carry gate separately: up(g*h) alone would mishandle the up-projection bias.
        token_code = torch.cat((captured[0] * update.gate, update.gate), -1)
        token_code = token_code * sequence["valid_mask"].unsqueeze(-1)
    return RefinementContext(
        condition=condition, valid_mask=sequence["valid_mask"].bool(),
        proprio=sequence["proprio_sequence"][:, age], global_code=global_code,
        token_code=token_code,
        updated=torch.full((batch,), age != 2, device=previous.device, dtype=torch.bool),
    )


@torch.no_grad()
def frozen_anchor(transformer, context: RefinementContext, noise: Tensor):
    captures = []
    handle = transformer.action_decoder.register_forward_pre_hook(lambda _m, args: captures.append(args[0]))
    try:
        velocity = transformer(vlm_features=context.condition, action_with_noise=noise,
                               proprio=context.proprio, t=noise.new_ones(noise.shape[0]))
    finally:
        handle.remove()
    if len(captures) != 1:
        raise RuntimeError("Expected one frozen action hidden capture")
    return captures[0].detach(), velocity.detach()
