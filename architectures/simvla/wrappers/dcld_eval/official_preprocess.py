"""Official-style SimVLA LIBERO preprocessing helpers.

Purpose:
    Document the single source of truth for official-style image resize,
    tensor construction, prompt tokenization, and LIBERO state vector building.

Inputs/outputs:
    The canonical runtime implementations currently live in `rollout_runner`
    for behavior preservation and are re-exported here for new diagnostics.

Official-match scope:
    These helpers are intended to match the official WebSocket client/server
    path: 224 resize-with-pad, PIL 384 resize, ImageNet normalization, two real
    views plus one zero-padded view, and an 8D proprio vector.

DCLD scope:
    Not DCLD-specific. DCLD consumes the resulting SimVLA condition/action path.

Caveat:
    This module is an API surface for future callers; the rollout runner remains
    the behavior-preserving implementation in this refactor.
"""

from __future__ import annotations

from .rollout_runner import _quat2axisangle, build_env_obs, resize_with_pad_uint8, tensor_action_diff

__all__ = ["_quat2axisangle", "build_env_obs", "resize_with_pad_uint8", "tensor_action_diff"]
