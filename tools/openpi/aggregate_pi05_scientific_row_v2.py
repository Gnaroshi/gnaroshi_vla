#!/usr/bin/env python3
"""Validate primary artifacts, then aggregate four paired 500-episode suites."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    SUITES,
    load_final_evaluation_manifest,
)
from source_lock_v2 import verify_lock
from verify_pi05_final_manifest_v2 import verify_manifest


MARKERS = {
    "paired-full": "PAIRED_FULL_BASELINE_PASS",
    "v0": "V0_PAIRED_ROW_PASS",
    "v1": "V1_PAIRED_ROW_PASS",
    "v2": "V2_PAIRED_ROW_PASS",
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _episode_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["suite"]), int(row["task_id"]), int(row["trial"]))


def _manifest_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["suite"]),
        int(row["benchmark_task_index"]),
        int(row["trial"]),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _require_artifact(summary: dict[str, Any], path_key: str, hash_key: str) -> Path:
    artifacts = summary.get("protocol_artifacts", {})
    path = Path(str(artifacts.get(path_key, ""))).resolve()
    if not path.is_file() or artifacts.get(hash_key) != _sha256(path):
        raise RuntimeError(f"suite summary artifact is absent or changed: {path_key}")
    return path


def validate_suite_summary(
    summary_path: str | Path,
    *,
    method: str,
    source_lock_id: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    suite = str(summary.get("suite"))
    if suite not in SUITES:
        raise ValueError(f"unknown suite in summary: {suite}")
    expected_label = "original" if method == "paired-full" else method
    expected_manifest_rows = [row for row in manifest["episodes"] if row["suite"] == suite]
    expected_keys = {_manifest_key(row) for row in expected_manifest_rows}
    if (
        summary.get("complete") is not True
        or summary.get("rollouts") != 500
        or summary.get("tasks") != 10
        or summary.get("trials_per_task") != 50
        or summary.get("method_label") != expected_label
        or summary.get("source_lock_id") != source_lock_id
        or summary.get("final_evaluation_manifest_sha256") != manifest_sha256
        or summary.get("final_evaluation_manifest_id") != manifest["manifest_id"]
        or summary.get("seed") != 7
        or summary.get("replan_steps") != 5
        or summary.get("resize_size") != 224
        or summary.get("wait_steps") != 10
    ):
        raise RuntimeError(f"invalid or stale suite summary: {suite}")

    outcomes_path = _require_artifact(
        summary, "episode_outcomes", "episode_outcomes_sha256"
    )
    query_path = _require_artifact(summary, "query_metrics", "query_metrics_sha256")
    episode_manifest_path = _require_artifact(
        summary, "episode_manifest", "episode_manifest_sha256"
    )
    environment_path = _require_artifact(
        summary, "environment_metadata", "environment_metadata_sha256"
    )
    outcomes = list(csv.DictReader(outcomes_path.open(encoding="utf-8")))
    if len(outcomes) != 500:
        raise RuntimeError(f"{suite} outcome table has {len(outcomes)} rows, expected 500")
    outcome_keys = {_episode_key(row) for row in outcomes}
    if len(outcome_keys) != 500 or outcome_keys != expected_keys:
        raise RuntimeError(f"{suite} outcome identities differ from the final manifest")
    if any(row.get("policy_path") != expected_label for row in outcomes):
        raise RuntimeError(f"{suite} outcome method labels are mixed")
    successes = sum(str(row["success"]).lower() in {"1", "true", "yes"} for row in outcomes)
    if successes != int(summary["successes"]):
        raise RuntimeError(f"{suite} success total differs from its outcome table")

    observed_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
    observed_manifest_by_key = {_manifest_key(row): row for row in observed_manifest}
    if len(observed_manifest) != 500 or set(observed_manifest_by_key) != expected_keys:
        raise RuntimeError(f"{suite} executed episode manifest is incomplete or duplicated")
    expected_by_key = {_manifest_key(row): row for row in expected_manifest_rows}
    for key in expected_keys:
        for field in (
            "environment_seed",
            "initial_state_identifier",
            "query_noise_key_prefix",
            "max_episode_steps",
        ):
            if observed_manifest_by_key[key].get(field) != expected_by_key[key].get(field):
                raise RuntimeError(f"{suite} executed manifest differs at {key}/{field}")

    queries = _read_jsonl(query_path)
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for row in queries:
        key = (
            suite,
            int(row.get("task_id", -1)),
            int(row.get("episode_id", -1)),
        )
        if key not in expected_keys or row.get("policy_path") != expected_label:
            raise RuntimeError(f"{suite} query metrics contain another episode or method")
        grouped.setdefault(key, []).append(int(row.get("query_index", -1)))
    if set(grouped) != expected_keys:
        raise RuntimeError(f"{suite} query metrics omit one or more evaluated episodes")
    for key, indices in grouped.items():
        ordered = sorted(indices)
        if ordered != list(range(len(ordered))):
            raise RuntimeError(f"{suite} query indices are not unique and contiguous for {key}")

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    server = environment.get("policy_server_metadata", {})
    if (
        environment.get("render_backend") != "egl"
        or server.get("source_lock_id") != source_lock_id
        or server.get("final_evaluation_manifest_sha256") != manifest_sha256
        or server.get("final_evaluation_manifest_id") != manifest["manifest_id"]
        or server.get("suite") != suite
    ):
        raise RuntimeError(f"{suite} environment/server metadata is stale or inconsistent")
    return {
        "suite": suite,
        "summary": summary,
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "successes": successes,
        "episodes": 500,
        "query_rows": len(queries),
        "episode_keys": outcome_keys,
        "artifact_hashes": summary["protocol_artifacts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(MARKERS), required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--suite-summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite row aggregate: {output}")
    lock = verify_lock(args.source_lock)
    manifest_gate = verify_manifest(args.final_evaluation_manifest, args.source_lock)
    manifest = load_final_evaluation_manifest(args.final_evaluation_manifest)
    validated = [
        validate_suite_summary(
            path,
            method=args.method,
            source_lock_id=lock["source_lock_id"],
            manifest=manifest,
            manifest_sha256=manifest_gate["manifest_sha256"],
        )
        for path in args.suite_summary
    ]
    if len(validated) != 4 or {row["suite"] for row in validated} != set(SUITES):
        raise ValueError("one scientific row requires exactly one validated summary per suite")
    all_keys = set().union(*(row["episode_keys"] for row in validated))
    if len(all_keys) != 2000:
        raise RuntimeError("four-suite scientific row does not contain 2,000 unique episodes")
    successes = sum(int(row["successes"]) for row in validated)
    marker = MARKERS[args.method]
    payload = {
        "schema_version": 2,
        marker: True,
        "markers": [marker],
        "source_lock_id": lock["source_lock_id"],
        "final_evaluation_manifest_id": manifest["manifest_id"],
        "final_evaluation_manifest_sha256": manifest_gate["manifest_sha256"],
        "method": args.method,
        "episodes": 2000,
        "successes": successes,
        "success_rate": successes / 2000,
        "underlying_artifacts_revalidated": True,
        "suite_results": {
            row["suite"]: {
                "successes": row["successes"],
                "episodes": row["episodes"],
                "success_rate": row["successes"] / row["episodes"],
                "query_rows": row["query_rows"],
                "summary": row["summary_path"],
                "summary_sha256": row["summary_sha256"],
                "artifact_hashes": row["artifact_hashes"],
            }
            for row in validated
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
