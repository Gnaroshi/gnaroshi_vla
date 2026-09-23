"""Pure execution-contract helpers for LatentLoop V0/V1/V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Operation(str, Enum):
    FULL = "full"
    UPDATE = "update"
    DIRECT_REANCHOR = "direct_reanchor"


def operation_for_step(step: int, query_interval: int, updater_enabled: bool = True) -> Operation:
    if step < 0:
        raise ValueError("step must be nonnegative")
    if query_interval < 1:
        raise ValueError("query_interval must be positive")
    if not updater_enabled or query_interval == 1 or step % query_interval == 0:
        return Operation.FULL
    return Operation.UPDATE


def periodic_schedule(length: int, query_interval: int, updater_enabled: bool = True) -> tuple[Operation, ...]:
    if length < 0:
        raise ValueError("length must be nonnegative")
    return tuple(operation_for_step(step, query_interval, updater_enabled) for step in range(length))


@dataclass
class OperationCounters:
    policy_steps: int = 0
    full_calls: int = 0
    transition_calls: int = 0
    direct_transition_calls: int = 0
    direct_reanchors: int = 0
    action_generator_calls: int = 0

    def record(self, operation: Operation, *, direct_was_evaluated: bool = False) -> None:
        self.policy_steps += 1
        self.action_generator_calls += 1
        if operation is Operation.FULL:
            self.full_calls += 1
        else:
            self.transition_calls += 1
            if direct_was_evaluated:
                self.direct_transition_calls += 1
            if operation is Operation.DIRECT_REANCHOR:
                self.direct_reanchors += 1

    @property
    def effective_k(self) -> float:
        return self.policy_steps / self.full_calls if self.full_calls else 0.0

    @property
    def full_query_reduction(self) -> float:
        return 1.0 - self.full_calls / self.policy_steps if self.policy_steps else 0.0

    def to_dict(self) -> dict[str, int | float]:
        payload = asdict(self)
        payload.update(
            effective_k=self.effective_k,
            full_query_reduction=self.full_query_reduction,
        )
        return payload
