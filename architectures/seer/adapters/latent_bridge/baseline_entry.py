"""Behavior-preserving baseline wrapper that adds peak-memory metadata."""

from __future__ import annotations

import os

import torch

from .determinism import configure_deterministic_inference
from .provenance import PUBLIC_SEER_33_SHA256, require_file_hash
from .rendering import configured_renderer_backend
from .runtime import seer_import_context


def _build_profiled_wrapper(base_wrapper, determinism):
    renderer = configured_renderer_backend()

    class ProfiledBaselineWrapper(base_wrapper):
        def __init__(self, *args, **kwargs):
            kwargs["lrnode_eval_profile_full_action_head"] = 1
            super().__init__(*args, **kwargs)
            self._baseline_peak_memory_bytes = 0

        def reset(self):
            result = super().reset()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            return result

        def step(self, *args, **kwargs):
            action = super().step(*args, **kwargs)
            if torch.cuda.is_available():
                self._baseline_peak_memory_bytes = max(
                    self._baseline_peak_memory_bytes,
                    int(torch.cuda.max_memory_allocated()),
                )
            return action

        def get_lrnode_stats(self):
            stats = super().get_lrnode_stats()
            stats.update(
                {
                    "method": "frozen_seer_baseline",
                    "peak_gpu_memory_bytes": int(self._baseline_peak_memory_bytes),
                    "renderer_backend": renderer,
                    "determinism": determinism,
                }
            )
            return stats

    return ProfiledBaselineWrapper


def main() -> None:
    checkpoint = os.environ.get("SEER_LATENT_BRIDGE_BASE_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("SEER_LATENT_BRIDGE_BASE_CHECKPOINT is required")
    checkpoint_hash = require_file_hash(
        checkpoint,
        PUBLIC_SEER_33_SHA256,
        "Latent Bridge baseline Seer checkpoint",
    )
    determinism = configure_deterministic_inference()
    with seer_import_context():
        import eval_libero
        import utils.eval_utils_libero as eval_utils

        profiled_wrapper = _build_profiled_wrapper(eval_utils.ModelWrapper, determinism)

        class ProvenanceLockedBaselineWrapper(profiled_wrapper):
            def get_lrnode_stats(self):
                stats = super().get_lrnode_stats()
                stats.update(
                    {
                        "base_checkpoint_sha256": checkpoint_hash,
                        "refresh_period": 1,
                        "action_protocol": (
                            "three-token prediction with temporal ensembling; "
                            "one executed action"
                        ),
                    }
                )
                return stats

        eval_utils.ModelWrapper = ProvenanceLockedBaselineWrapper
        eval_libero.main()


if __name__ == "__main__":
    main()
