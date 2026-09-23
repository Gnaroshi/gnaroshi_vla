"""Verified Seer action-token to environment-time alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


# No action_label_offset argument exists in the locked Seer source. Training
# constructs token h from actions[:, h:...+h], so the effective offset is zero.
ACTION_LABEL_OFFSET = 0


@dataclass(frozen=True)
class TokenTime:
    """One action token's intended environment execution time."""

    query_step: int
    token_index: int
    intended_execution_time: int


def intended_execution_time(
    query_step: int, token_index: int, *, action_label_offset: int = ACTION_LABEL_OFFSET
) -> int:
    """Map a query/token pair to the supervised environment step."""

    if token_index < 0:
        raise ValueError("token_index must be non-negative")
    return int(query_step) + int(action_label_offset) + int(token_index)


def action_token_time_mapping(
    query_step: int, action_pred_steps: int, *, action_label_offset: int = ACTION_LABEL_OFFSET
) -> list[dict[str, int]]:
    """Return the complete mapping for one Seer raw action horizon."""

    if action_pred_steps < 1:
        raise ValueError("action_pred_steps must be positive")
    return [
        asdict(
            TokenTime(
                query_step=int(query_step),
                token_index=token,
                intended_execution_time=intended_execution_time(
                    query_step, token, action_label_offset=action_label_offset
                ),
            )
        )
        for token in range(int(action_pred_steps))
    ]


def verified_overlap_pairs(
    action_pred_steps: int,
    *,
    previous_query_step: int = 0,
    current_query_step: int = 1,
    action_label_offset: int = ACTION_LABEL_OFFSET,
) -> list[tuple[int, int]]:
    """Return ``(previous_token,current_token)`` pairs with equal target time."""

    if action_pred_steps < 2:
        return []
    pairs: list[tuple[int, int]] = []
    for previous_token in range(int(action_pred_steps)):
        previous_time = intended_execution_time(
            previous_query_step,
            previous_token,
            action_label_offset=action_label_offset,
        )
        for current_token in range(int(action_pred_steps)):
            current_time = intended_execution_time(
                current_query_step,
                current_token,
                action_label_offset=action_label_offset,
            )
            if previous_time == current_time:
                pairs.append((previous_token, current_token))
    return pairs


def assert_canonical_seer_alignment(action_pred_steps: int = 3) -> None:
    """Fail unless the locked P=3 Seer overlap is exactly ``(1,0),(2,1)``."""

    pairs = verified_overlap_pairs(action_pred_steps)
    expected = [(index, index - 1) for index in range(1, int(action_pred_steps))]
    if pairs != expected:
        raise RuntimeError(f"Seer token-time alignment mismatch: expected {expected}, got {pairs}")
