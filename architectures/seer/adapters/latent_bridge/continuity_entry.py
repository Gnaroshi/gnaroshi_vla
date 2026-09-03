"""Run upstream Seer evaluation with only an external continuity wrapper."""

from __future__ import annotations

import os

from .determinism import configure_deterministic_inference
from .runtime import seer_import_context


def main() -> None:
    if "LATENT_BRIDGE_CONTINUITY_OUTPUT" not in os.environ:
        raise RuntimeError("LATENT_BRIDGE_CONTINUITY_OUTPUT is required")
    configure_deterministic_inference()
    with seer_import_context():
        import eval_libero
        import utils.eval_utils_libero as eval_utils

        from .continuity import build_continuity_wrapper

        eval_utils.ModelWrapper = build_continuity_wrapper(eval_utils.ModelWrapper)
        eval_libero.main()


if __name__ == "__main__":
    main()
