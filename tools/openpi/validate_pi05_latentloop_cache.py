#!/usr/bin/env python3
"""Validate cache schema, episode split isolation, tensor shapes, and hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from architectures.openpi.adapters.latentloop.cache_io import validate_cache


def main() -> None:
    raise RuntimeError(
        "DISABLED_SUPERSEDED_CACHE_VALIDATOR_V1: use validate_pi05_cache_v2.py"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = validate_cache(args.cache, verify_hashes=args.verify_hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
