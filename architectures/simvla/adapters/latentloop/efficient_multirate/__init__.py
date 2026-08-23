"""Fail-closed preparation utilities for efficient multirate LatentLoop."""

from .contracts import (
    CACHE_SCHEMA,
    GENERATION_SCHEDULES,
    STAGE_GRAPH,
    balanced_mode_d_age,
    native_nfe_time_grid,
    project_exact_teacher_cache,
)

__all__ = [
    "CACHE_SCHEMA",
    "GENERATION_SCHEDULES",
    "STAGE_GRAPH",
    "balanced_mode_d_age",
    "native_nfe_time_grid",
    "project_exact_teacher_cache",
]
