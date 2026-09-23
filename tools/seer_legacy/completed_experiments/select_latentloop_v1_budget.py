#!/usr/bin/env python3
"""Select V1 e20/e40 from validation metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from methods.latentloop_v1v2.selection import select_v1_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e20", type=Path, required=True)
    parser.add_argument("--e40", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = select_v1_budget(
        json.loads(args.e20.read_text(encoding="utf-8")),
        json.loads(args.e40.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["verdict"])


if __name__ == "__main__":
    main()
