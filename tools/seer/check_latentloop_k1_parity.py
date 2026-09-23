#!/usr/bin/env python3
"""Check full-Seer versus adapter-loaded K1 parity on paired episodes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

from methods.latentloop_segment_grid.metrics import episode_key, paired_flip_counts
from methods.latentloop_segment_grid.serialization import atomic_write_json, read_json


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _outcomes(path: Path) -> Dict[Tuple[object, ...], int]:
    return {
        episode_key(row): int(row["success"])
        for row in _csv_rows(path)
    }


def _step_actions(step_log_dir: Path) -> Dict[Tuple[int, int, int], np.ndarray]:
    actions: Dict[Tuple[int, int, int], np.ndarray] = {}
    for path in sorted(step_log_dir.glob("*.csv")):
        for row in _csv_rows(path):
            key = (
                int(row["task_id"]),
                int(row["episode_id"]),
                int(row["timestep"]),
            )
            actions[key] = np.asarray(
                [float(row[f"action_{index}"]) for index in range(7)],
                dtype=np.float64,
            )
    return actions


def compare_k1_rows(
    baseline_row: Mapping[str, object],
    adapter_row: Mapping[str, object],
    *,
    action_tolerance: float,
) -> Dict[str, object]:
    """Return exact paired outcome and executed-action parity diagnostics."""

    baseline_outcomes = _outcomes(Path(str(baseline_row["episode_metrics_path"])))
    adapter_outcomes = _outcomes(Path(str(adapter_row["episode_metrics_path"])))
    flips = paired_flip_counts(baseline_outcomes, adapter_outcomes)
    episode_keys_identical = set(baseline_outcomes) == set(adapter_outcomes)
    outcome_parity = (
        episode_keys_identical
        and flips["baseline_success_candidate_failure"] == 0
        and flips["baseline_failure_candidate_success"] == 0
    )

    baseline_actions = _step_actions(Path(str(baseline_row["step_log_dir"])))
    adapter_actions = _step_actions(Path(str(adapter_row["step_log_dir"])))
    action_keys_identical = set(baseline_actions) == set(adapter_actions)
    common = set(baseline_actions).intersection(adapter_actions)
    max_action_difference = max(
        (
            float(np.max(np.abs(baseline_actions[key] - adapter_actions[key])))
            for key in common
        ),
        default=0.0,
    )
    action_parity = (
        action_keys_identical
        and bool(common)
        and max_action_difference <= float(action_tolerance)
    )
    return {
        "pass": bool(outcome_parity and action_parity),
        "episode_keys_identical": episode_keys_identical,
        "outcome_parity": outcome_parity,
        "action_keys_identical": action_keys_identical,
        "paired_action_steps": len(common),
        "max_executed_action_difference": max_action_difference,
        "action_tolerance": float(action_tolerance),
        "paired_flips": flips,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--checkpoint-id", type=int, required=True)
    parser.add_argument("--action-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    campaign_path = args.campaign_root / "segment_grid_campaign.json"
    campaign = read_json(campaign_path)
    candidates = [
        row
        for row in campaign.get("rows", [])
        if int(row["checkpoint_id"]) == int(args.checkpoint_id)
        and int(row["segment_length"]) == 1
    ]
    baseline = next(
        (row for row in candidates if row["baseline_kind"] == "full_replanning"),
        None,
    )
    adapter = next(
        (row for row in candidates if row["baseline_kind"] == "adapter_k1_parity"),
        None,
    )
    if baseline is None or adapter is None:
        raise RuntimeError(
            f"Missing K1 parity rows for checkpoint {args.checkpoint_id}"
        )
    identity_fields = (
        "checkpoint_profile",
        "checkpoint_source",
        "baseline_checkpoint_path",
        "baseline_checkpoint_sha256",
        "adapter_checkpoint_path",
        "adapter_checkpoint_sha256",
    )
    identity = {field: baseline.get(field) for field in identity_fields}
    mismatched_identity = {
        field: (baseline.get(field), adapter.get(field))
        for field in identity_fields
        if baseline.get(field) != adapter.get(field)
    }
    if mismatched_identity:
        raise RuntimeError(
            "K1 parity rows use different checkpoint identities: "
            f"{mismatched_identity}"
        )
    result = {
        "schema_version": 1,
        "checkpoint_id": int(args.checkpoint_id),
        "baseline_row_id": baseline["row_id"],
        "adapter_row_id": adapter["row_id"],
        **identity,
        **compare_k1_rows(
            baseline, adapter, action_tolerance=args.action_tolerance
        ),
    }
    output = args.output or (args.campaign_root / "k1_parity.json")
    atomic_write_json(output, result, refuse_overwrite=True)
    if not result["pass"]:
        raise SystemExit(f"[VERIFY][FAIL] K1 parity failed: {result}")
    print(f"[VERIFY][OK] K1 parity: {output}")


if __name__ == "__main__":
    main()
