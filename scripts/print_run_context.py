#!/usr/bin/env python3
"""Print selected run context as JSON."""

from __future__ import annotations

import json
import os
import sys


def parse_key_values(argv: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in argv:
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
    return values


def main() -> int:
    cli = parse_key_values(sys.argv[1:])
    context = {
        "cli": cli,
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "env": {
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }
    print(json.dumps(context, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

