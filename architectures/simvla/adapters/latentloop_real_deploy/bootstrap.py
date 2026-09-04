"""Import helpers for the vendored official SimVLA model package."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
VENDORED_UPSTREAM = (
    ROOT / "architectures" / "simvla" / "third_party" / "simvla_upstream_32700d0"
)


def configure_model_imports() -> Path:
    """Make the tracked model snapshot importable as the upstream `models` package."""

    models = VENDORED_UPSTREAM / "models"
    if not (models / "modeling_smolvlm_vla.py").is_file():
        raise FileNotFoundError(f"Vendored SimVLA model package is incomplete: {models}")
    for path in (ROOT, VENDORED_UPSTREAM):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return VENDORED_UPSTREAM
