"""Predeclared two-timescale execution schedule."""

from __future__ import annotations

from enum import IntEnum


class JointExecutionLevel(IntEnum):
    FAST_SURROGATE = 0
    EXACT_ACTION_HEAD = 1
    FULL_SEER = 2


class JointHierarchySchedule:
    def __init__(self, full_interval: int = 8, regeneration_interval: int = 3) -> None:
        self.full_interval = int(full_interval)
        self.regeneration_interval = int(regeneration_interval)
        if self.full_interval < 2:
            raise ValueError("K_F must be at least 2")
        if not 1 < self.regeneration_interval < self.full_interval:
            raise ValueError("The joint schedule requires 1 < K_G < K_F")

    def level(self, timestep: int, *, has_cache: bool) -> JointExecutionLevel:
        step = int(timestep)
        if step < 0:
            raise ValueError("timestep must be non-negative")
        if not has_cache or step % self.full_interval == 0:
            return JointExecutionLevel.FULL_SEER
        if step % self.full_interval % self.regeneration_interval == 0:
            return JointExecutionLevel.EXACT_ACTION_HEAD
        return JointExecutionLevel.FAST_SURROGATE

    def cycle(self) -> list[int]:
        return [int(self.level(step, has_cache=step > 0)) for step in range(self.full_interval + 1)]
