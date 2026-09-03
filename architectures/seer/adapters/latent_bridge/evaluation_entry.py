"""Run upstream Seer evaluation with an external Latent Bridge policy."""

from __future__ import annotations

import os

from .determinism import configure_deterministic_inference
from .runtime import seer_import_context


def main() -> None:
    required = ("SEER_LATENT_BRIDGE_CHECKPOINT", "SEER_LATENT_BRIDGE_REFRESH_PERIOD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing Latent Bridge runtime variables: {missing}")
    configure_deterministic_inference()
    with seer_import_context():
        import eval_libero
        import utils.eval_utils_libero as eval_utils

        from .policy import build_latent_bridge_wrapper

        eval_utils.ModelWrapper = build_latent_bridge_wrapper(eval_utils.ModelWrapper)
        eval_libero.main()


if __name__ == "__main__":
    main()
