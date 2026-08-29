#!/usr/bin/env python3
"""Serve a validation-selected V0 checkpoint from an explicitly declared budget."""

from __future__ import annotations

import os
from typing import Any

import serve_pi05_v0_streaming as legacy_server
from verify_pi05_v0_training_budget_eval import verify_evaluation_inputs


def _budget_verifier(**kwargs: Any) -> dict[str, Any]:
    value = os.environ.get("OPENPI_PI05_V0_EXPECTED_TRAINING_STEPS")
    if value is None:
        raise RuntimeError("OPENPI_PI05_V0_EXPECTED_TRAINING_STEPS is required")
    return verify_evaluation_inputs(expected_training_steps=int(value), **kwargs)


def main() -> None:
    legacy_server.verify_evaluation_inputs = _budget_verifier
    legacy_server.main()


if __name__ == "__main__":
    main()
