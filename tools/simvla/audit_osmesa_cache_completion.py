#!/usr/bin/env python3
"""Write the immutable cache-pipeline audit required before exact-q2 work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architectures.simvla.adapters.exact_q2.cache_contract import (  # noqa: E402
    audit_completed_osmesa_pipeline,
)
from architectures.simvla.adapters.latentloop.source_lock import require_empty_output  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(audit: dict) -> str:
    lines = [
        "# SimVLA OSMesa cache-pipeline completion audit",
        "",
        f"- Verdict: **{audit['verdict']}**",
        f"- Pipeline verdict: `{audit['pipeline_completion_verdict']}`",
        f"- R5 cache: `{audit['paths']['r5_production_cache']}`",
        f"- R5 records / shards / episodes: {audit['r5']['records']} / {audit['r5']['shards']} / {audit['r5']['episodes']}",
        f"- Exact-q2 tuples: {audit['r5']['tuple_count']} (train {audit['r5']['train_tuples']}, validation {audit['r5']['validation_tuples']})",
        f"- Completed pipeline time: {audit['pipeline_elapsed_seconds'] / 3600.0:.3f} h",
        "",
        "## Pipeline phases",
        "",
        "| Phase | R | Episodes | Records | Renderer | Validation |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, row in audit["phases"].items():
        lines.append(
            f"| {name} | {row['execution_horizon']} | {row['episodes']} | {row['records']} | {row['renderer']} | {'PASS' if row['validation_passed'] and not row['errors'] else 'FAIL'} |"
        )
    lines.extend(["", "## Hard gate", ""])
    for name, passed in audit["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Source lock",
            "",
            f"- Root commit: `{audit['r5']['source_signature']['root_commit']}`",
            f"- SimVLA upstream commit: `{audit['r5']['source_signature']['simvla_upstream_commit']}`",
            f"- Checkpoint revision: `{audit['r5']['source_signature']['checkpoint_revision']}`",
            f"- Norm SHA-256: `{audit['r5']['source_signature']['norm_stats_sha256']}`",
            f"- Cache manifest SHA-256: `{audit['r5']['manifest_sha256']}`",
            "",
            "No cache generation was run by this audit.",
        ]
    )
    if audit["errors"]:
        lines.extend(["", "## Errors", ""] + [f"- {item}" for item in audit["errors"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-shard-hashes", action="store_true")
    args = parser.parse_args()
    output = require_empty_output(args.output)
    audit = audit_completed_osmesa_pipeline(
        args.launcher,
        verify_shard_hashes=not args.skip_shard_hashes,
    )
    _write_json(output / "cache_pipeline_manifest.json", audit)
    (output / "cache_pipeline_completion_audit.md").write_text(
        _markdown(audit), encoding="utf-8"
    )
    print(json.dumps({"verdict": audit["verdict"], "output": str(output)}, indent=2))
    return 0 if audit["verdict"] == "R5_EXACT_Q2_CACHE_GATE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
