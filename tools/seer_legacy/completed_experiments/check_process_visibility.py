#!/usr/bin/env python3
"""Reject process-visible paths and labels that obscure experiment identity."""

from __future__ import annotations

import argparse
import re


FORBIDDEN = (
    (re.compile(r"(?:^|/)codex_outputs(?:/|$)", re.IGNORECASE), "codex_outputs path"),
    (re.compile(r"(?:^|[-_/])20\d{6}(?:[-_/]|$)"), "date-stamped identifier"),
    (re.compile(r"(?:^|[-_/])20\d{6}[_-]\d{6}(?:[-_/]|$)"), "timestamped identifier"),
)


def validate(values: list[str]) -> None:
    for value in values:
        for pattern, label in FORBIDDEN:
            if pattern.search(value):
                raise ValueError(f"process visibility contract rejected {label}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("values", nargs="+")
    args = parser.parse_args()
    validate(args.values)
    print("PROCESS_VISIBILITY_PASS")


if __name__ == "__main__":
    main()
