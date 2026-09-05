"""Verify immutable source snapshots before model or hardware initialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import runtime_source_identity, sha256_file


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = Path(__file__).with_name("source_manifest.json")


def verify_source_snapshots() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported source snapshot manifest")
    observed: dict[str, str] = {}
    for relative, expected in payload["sources"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Tracked deployment source is missing: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(
                f"Tracked deployment source changed: {relative}: "
                f"observed={digest} expected={expected}"
            )
        observed[relative] = digest

    for copied_relative, reference_relative in payload[
        "byte_identical_reference_pairs"
    ].items():
        copied = ROOT / copied_relative
        reference = ROOT / reference_relative
        if not reference.is_file():
            raise FileNotFoundError(f"Tracked reference source is missing: {reference}")
        if copied.read_bytes() != reference.read_bytes():
            raise ValueError(
                f"Copied source is no longer byte-identical to reference: "
                f"{copied_relative} != {reference_relative}"
            )

    runtime_identity = runtime_source_identity()
    return {
        "verdict": "SOURCE_PREFLIGHT_PASS",
        "manifest": str(MANIFEST),
        "verified_files": len(observed),
        "byte_identical_pairs": len(payload["byte_identical_reference_pairs"]),
        "sha256": observed,
        "runtime_source_identity_sha256": runtime_identity["combined_sha256"],
        "runtime_source_file_count": runtime_identity["file_count"],
        "runtime_source_sha256": runtime_identity["files"],
    }
