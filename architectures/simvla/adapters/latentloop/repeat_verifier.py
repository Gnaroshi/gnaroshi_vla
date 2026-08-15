"""Exact repeat verifier for strict deterministic SimVLA online evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .determinism import DETERMINISM_PROTOCOL, compare_manifest_contracts


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_repeat(reference_dir: str | Path, candidate_dir: str | Path) -> dict[str, Any]:
    """Require exact contracts and exact deterministic episode traces."""

    reference = Path(reference_dir).expanduser().resolve()
    candidate = Path(candidate_dir).expanduser().resolve()
    reference_manifest = _read_json(reference / "determinism_manifest.json")
    candidate_manifest = _read_json(candidate / "determinism_manifest.json")
    compare_manifest_contracts(candidate_manifest, reference_manifest)

    reference_results = _read_json(reference / "deterministic_results.json")
    candidate_results = _read_json(candidate / "deterministic_results.json")
    reference_rows = {
        (row["row"], int(row["task_id"]), int(row["episode"])): row
        for row in reference_results.get("episodes", [])
    }
    candidate_rows = {
        (row["row"], int(row["task_id"]), int(row["episode"])): row
        for row in candidate_results.get("episodes", [])
    }
    all_keys = sorted(set(reference_rows) | set(candidate_rows))
    mismatches: list[dict[str, Any]] = []
    for key in all_keys:
        left = reference_rows.get(key)
        right = candidate_rows.get(key)
        if left == right:
            continue
        differing_fields = sorted(
            field
            for field in set(left or {}) | set(right or {})
            if (left or {}).get(field) != (right or {}).get(field)
        )
        mismatches.append(
            {
                "row": key[0],
                "task_id": key[1],
                "episode": key[2],
                "differing_fields": differing_fields,
                "reference_episode_trace_hash": (left or {}).get("episode_trace_hash"),
                "candidate_episode_trace_hash": (right or {}).get("episode_trace_hash"),
            }
        )

    aggregate_match = (
        reference_results.get("trajectory_sha256")
        == candidate_results.get("trajectory_sha256")
    )
    passed = bool(
        reference_results.get("protocol") == DETERMINISM_PROTOCOL
        and candidate_results.get("protocol") == DETERMINISM_PROTOCOL
        and reference_results.get("run_contract_sha256")
        == candidate_results.get("run_contract_sha256")
        and aggregate_match
        and not mismatches
    )
    return {
        "protocol": DETERMINISM_PROTOCOL,
        "verdict": "DETERMINISTIC_REPEAT_PASS" if passed else "DETERMINISTIC_REPEAT_FAIL",
        "passed": passed,
        "reference": str(reference),
        "candidate": str(candidate),
        "run_contract_sha256": candidate_results.get("run_contract_sha256"),
        "reference_trajectory_sha256": reference_results.get("trajectory_sha256"),
        "candidate_trajectory_sha256": candidate_results.get("trajectory_sha256"),
        "aggregate_match": aggregate_match,
        "reference_episode_count": len(reference_rows),
        "candidate_episode_count": len(candidate_rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "excluded_from_exact_comparison": [
            "wall-clock duration",
            "latency measurements",
            "progress timestamps",
            "encoded video container bytes",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = verify_repeat(args.reference, args.candidate)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
