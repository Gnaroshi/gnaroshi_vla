"""Named DCLD ablation modes."""

from __future__ import annotations

from enum import Enum


class AblationMode(str, Enum):
    REAL_DELTA = "real_delta"
    NO_DELTA = "no_delta"
    SHUFFLED_DELTA = "shuffled_delta"
    PROPRIO_ONLY = "proprio_only"
    IMAGE_ONLY = "image_only"
