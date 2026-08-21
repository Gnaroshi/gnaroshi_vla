"""Runtime invariants for SimVLA DCLD evaluation modes.

Purpose:
    Provide explicit assertions for mode semantics that were previously implicit
    in counters and calibration reports.

Inputs/outputs:
    `assert_mode_invariants` consumes a row name, mode, K value, and aggregated
    rollout counters. It raises `AssertionError` on a semantic violation and
    returns `None` otherwise.

Official-match scope:
    Full K=1 rows should perform a full SimVLA condition refresh at every
    policy query and should not call DCLD.

DCLD scope:
    Non-full modes assert DCLD/ablation-specific counter relationships. These
    checks do not change the algorithm; they fail early when counters indicate a
    silent baseline or ablation semantics change.

Caveat:
    The `no_delta` zero-delta norm is guaranteed by `DCLDCore.encode_delta`;
    this wrapper-level invariant checks the observable counter proxy because
    the rollout code does not persist every delta tensor norm.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _count(counters: Mapping[str, Any], name: str) -> int:
    return int(counters.get(name, 0) or 0)


def assert_mode_invariants(
    *,
    row_name: str,
    mode: str,
    refresh_every: int,
    counters: Mapping[str, Any],
) -> None:
    """Assert semantic invariants for a completed eval row.

    Args:
        row_name: Output table row name, such as `baseline_full_k1`.
        mode: Policy mode used by the row.
        refresh_every: DCLD K measured in policy-query/action-queue refills.
        counters: Aggregated counters for the row.
    """

    policy_queries = _count(counters, "num_policy_queries")
    full_vlm_calls = _count(counters, "num_full_vlm_calls")
    dcld_updates = _count(counters, "num_dcld_updates")
    fast_encoder_calls = _count(counters, "num_fast_encoder_calls")
    action_transformer_calls = _count(counters, "num_action_transformer_calls")

    if mode == "full":
        assert dcld_updates == 0, f"{row_name}: full mode must not call DCLD"
        assert fast_encoder_calls == 0, f"{row_name}: full mode must not call FastEncoder"
        assert full_vlm_calls == policy_queries, (
            f"{row_name}: full mode must refresh full VLM every policy query "
            f"({full_vlm_calls=} {policy_queries=})"
        )
        if int(refresh_every) == 1 or row_name in {"baseline_full_k1", "ours_full_k1", "baseline_k1_full"}:
            assert dcld_updates == 0, f"{row_name}: K=1 full row must not call DCLD"
            assert fast_encoder_calls == 0, f"{row_name}: K=1 full row must not call FastEncoder"

    if mode == "stepwise_dcld":
        assert full_vlm_calls >= 1, f"{row_name}: stepwise_dcld must have full-refresh anchors"
        assert dcld_updates == max(0, policy_queries - full_vlm_calls), (
            f"{row_name}: DCLD updates must happen only on non-refresh policy queries "
            f"({dcld_updates=} {policy_queries=} {full_vlm_calls=})"
        )
        assert full_vlm_calls <= policy_queries, f"{row_name}: full VLM calls cannot exceed policy queries"

    if mode == "hold_condition":
        assert dcld_updates == 0, f"{row_name}: hold_condition must not call DCLD"
        assert fast_encoder_calls == 0, f"{row_name}: hold_condition must not call FastEncoder"
        assert action_transformer_calls > 0 or policy_queries == 0, (
            f"{row_name}: hold_condition should decode actions from held condition on skipped queries"
        )

    if mode == "native_action_chunk":
        assert dcld_updates == 0, f"{row_name}: native_action_chunk must not call DCLD"
        assert fast_encoder_calls == 0, f"{row_name}: native_action_chunk must not call FastEncoder"
        assert action_transformer_calls <= full_vlm_calls * 11, (
            f"{row_name}: native_action_chunk should not decode on cached chunk replay steps"
        )

    if mode == "no_delta":
        no_delta_steps = _count(counters, "num_no_delta_steps")
        assert dcld_updates > 0 or policy_queries <= full_vlm_calls, (
            f"{row_name}: no_delta mode should exercise DCLD updates on non-refresh queries"
        )
        assert no_delta_steps == dcld_updates, (
            f"{row_name}: no_delta counter should match DCLD updates; "
            "DCLDCore.encode_delta returns an exact zero delta tensor in this mode"
        )
