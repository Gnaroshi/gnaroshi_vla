"""Exact default-off execution schedule for the Seer intervention."""

from __future__ import annotations

from enum import Enum, IntEnum


class ExecutionLevel(IntEnum):
    ACTION_CORRECTION = 0
    HORIZON_REGENERATION = 1
    FULL_SEER = 2


class ExecutionMode(str, Enum):
    OFF = "off"
    FULL_SEER = "full_seer"
    PURE_LATENTLOOP = "pure_latentloop"
    PURE_ACTION_CORRECTION = "pure_action_correction"
    HYBRID = "hybrid"


class HierarchicalSchedule:
    """Map an environment step to one of the three predeclared levels.

    ``PURE_ACTION_CORRECTION`` is explicit rather than represented as
    ``K_G == K_F``; this avoids an endpoint off-by-one ambiguity.
    """

    def __init__(self, mode: str, full_interval: int = 8, regeneration_interval: int = 3):
        self.mode = ExecutionMode(mode)
        self.full_interval = int(full_interval)
        self.regeneration_interval = int(regeneration_interval)
        if self.full_interval < 1:
            raise ValueError("full_interval must be positive")
        if self.mode == ExecutionMode.HYBRID and not (
            1 < self.regeneration_interval < self.full_interval
        ):
            raise ValueError("hybrid requires 1 < regeneration_interval < full_interval")

    @property
    def enabled(self) -> bool:
        return self.mode != ExecutionMode.OFF

    def level(self, timestep: int, *, has_latent_cache: bool) -> ExecutionLevel:
        timestep = int(timestep)
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        if not has_latent_cache or self.mode == ExecutionMode.FULL_SEER:
            return ExecutionLevel.FULL_SEER
        if timestep % self.full_interval == 0:
            return ExecutionLevel.FULL_SEER
        if self.mode == ExecutionMode.PURE_LATENTLOOP:
            return ExecutionLevel.HORIZON_REGENERATION
        if self.mode == ExecutionMode.PURE_ACTION_CORRECTION:
            return ExecutionLevel.ACTION_CORRECTION
        if self.mode == ExecutionMode.HYBRID:
            offset = timestep % self.full_interval
            if offset % self.regeneration_interval == 0:
                return ExecutionLevel.HORIZON_REGENERATION
            return ExecutionLevel.ACTION_CORRECTION
        raise RuntimeError("ExecutionMode.OFF has no intervention schedule")

    def cycle(self) -> list[int]:
        if not self.enabled:
            return []
        return [
            int(self.level(step, has_latent_cache=step > 0))
            for step in range(self.full_interval + 1)
        ]
