"""Verified Seer action-token alignment used by action correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass


ACTION_LABEL_OFFSET = 0


@dataclass(frozen=True)
class TokenTime:
    """Intended environment time for one query/token pair."""

    query_step: int
    token_index: int
    intended_execution_time: int


def intended_execution_time(
    query_step: int,
    token_index: int,
    *,
    action_label_offset: int = ACTION_LABEL_OFFSET,
) -> int:
    """Return the training-label time represented by one action token."""

    if token_index < 0:
        raise ValueError("token_index must be non-negative")
    return int(query_step) + int(action_label_offset) + int(token_index)


def action_token_time_mapping(
    query_step: int,
    action_pred_steps: int,
    *,
    action_label_offset: int = ACTION_LABEL_OFFSET,
) -> list[dict[str, int]]:
    """Return a serializable mapping for all tokens in one horizon."""

    if action_pred_steps < 1:
        raise ValueError("action_pred_steps must be positive")
    return [
        asdict(
            TokenTime(
                query_step=int(query_step),
                token_index=token,
                intended_execution_time=intended_execution_time(
                    query_step,
                    token,
                    action_label_offset=action_label_offset,
                ),
            )
        )
        for token in range(int(action_pred_steps))
    ]


def verified_overlap_pairs(action_pred_steps: int = 3) -> list[tuple[int, int]]:
    """Return previous/current token pairs with the same execution time."""

    previous = action_token_time_mapping(0, action_pred_steps)
    current = action_token_time_mapping(1, action_pred_steps)
    return [
        (left["token_index"], right["token_index"])
        for left in previous
        for right in current
        if left["intended_execution_time"] == right["intended_execution_time"]
    ]


def assert_canonical_seer_alignment(action_pred_steps: int = 3) -> None:
    """Fail unless P=3 alignment is exactly previous 1,2 to current 0,1."""

    expected = [(index, index - 1) for index in range(1, action_pred_steps)]
    actual = verified_overlap_pairs(action_pred_steps)
    if actual != expected:
        raise RuntimeError(
            f"Seer action-token alignment mismatch: expected {expected}, got {actual}"
        )
