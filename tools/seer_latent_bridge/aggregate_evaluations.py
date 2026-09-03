#!/usr/bin/env python3
"""Validate exact manifests and aggregate paired Seer bridge evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from architectures.seer.adapters.latent_bridge.provenance import PUBLIC_SEER_33_SHA256


ACTION_PROTOCOL = "three-token prediction with temporal ensembling; one executed action"


def _require_equal(name: str, field: str, actual, expected) -> None:
    if actual != expected:
        raise RuntimeError(
            f"evaluation contract mismatch for {name}: "
            f"{field} expected={expected!r}, actual={actual!r}"
        )


def _validate_runtime_contract(name: str, summary: dict, manifest: dict) -> dict:
    expected_period = 1 if name == "f1" else int(name.removeprefix("f"))
    expected_method = "frozen_seer_baseline" if expected_period == 1 else "seer_latent_bridge"
    environment = summary.get("environment", {})
    runtime = summary.get("lrnode", {})
    _require_equal(name, "suite", summary.get("suite"), manifest["suite"])
    _require_equal(
        name,
        "control_hz",
        float(environment.get("control_hz", -1)),
        float(manifest["policy_contract"]["control_frequency_hz"]),
    )
    _require_equal(
        name,
        "eval_max_steps",
        int(environment.get("eval_max_steps", -1)),
        int(manifest["policy_contract"]["max_policy_steps"]),
    )
    _require_equal(name, "renderer", runtime.get("renderer_backend"), manifest["renderer"])
    _require_equal(name, "method", runtime.get("method"), expected_method)
    _require_equal(name, "refresh_period", int(runtime.get("refresh_period", -1)), expected_period)
    _require_equal(
        name,
        "base_checkpoint_sha256",
        runtime.get("base_checkpoint_sha256"),
        manifest["checkpoint_sha256"],
    )
    _require_equal(name, "action_protocol", runtime.get("action_protocol"), ACTION_PROTOCOL)
    if manifest["checkpoint_sha256"] != PUBLIC_SEER_33_SHA256:
        raise RuntimeError(
            "manifest is not locked to public Seer checkpoint 33: "
            f"{manifest['checkpoint_sha256']}"
        )
    if expected_period == 1:
        _require_equal(name, "bridge_calls", int(runtime.get("bridge_calls", 0)), 0)
    else:
        bridge_calls = int(runtime.get("bridge_calls", 0))
        if bridge_calls <= 0:
            raise RuntimeError(f"evaluation contract mismatch for {name}: no bridge calls recorded")
    return {
        "suite": manifest["suite"],
        "renderer": manifest["renderer"],
        "control_hz": float(environment["control_hz"]),
        "eval_max_steps": int(environment["eval_max_steps"]),
        "method": expected_method,
        "refresh_period": expected_period,
        "base_checkpoint_sha256": runtime["base_checkpoint_sha256"],
        "action_protocol": runtime["action_protocol"],
    }


def _load_row(spec: str, expected_ids: set[tuple[int, int]], manifest: dict) -> dict:
    name, raw_root = spec.split("=", 1)
    root = Path(raw_root)
    csv_path = root / "analysis/eval_episode_metrics.csv"
    summary_path = root / "analysis/eval_summary.json"
    latency_path = root / "analysis/eval_latency_profile.json"
    for path in (csv_path, summary_path, latency_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        episodes = list(csv.DictReader(stream))
    ids = [(int(row["task_id"]), int(row["episode_id"])) for row in episodes]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate task/trial identities in {name}")
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))[:10]
        extra = sorted(set(ids) - expected_ids)[:10]
        raise RuntimeError(f"manifest mismatch for {name}: missing={missing}, extra={extra}")
    if {int(row["seed"]) for row in episodes} != {int(manifest["seed"])}:
        raise RuntimeError(f"seed mismatch for {name}")
    by_id = {(int(row["task_id"]), int(row["episode_id"])): row for row in episodes}
    success = np.asarray([int(row["success"]) for row in episodes], dtype=np.int64)
    steps = np.asarray([int(row["num_steps"]) for row in episodes], dtype=np.int64)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    validated_contract = _validate_runtime_contract(name, summary, manifest)
    return {
        "name": name,
        "root": str(root),
        "episodes": len(episodes),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "total_env_steps": int(steps.sum()),
        "success_episode_mean_steps": float(steps[success == 1].mean()) if success.any() else None,
        "by_id": by_id,
        "summary": summary,
        "latency": latency,
        "validated_contract": validated_contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row", action="append", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected_ids = {(int(row["task_id"]), int(row["trial_id"])) for row in manifest["episodes"]}
    rows = [_load_row(spec, expected_ids, manifest) for spec in args.row]
    by_name = {row["name"]: row for row in rows}
    if args.baseline not in by_name:
        raise KeyError(f"baseline row is absent: {args.baseline}")
    baseline = by_name[args.baseline]
    comparisons = []
    for row in rows:
        both_success = baseline_only = row_only = both_fail = 0
        step_deltas = []
        for identity in sorted(expected_ids):
            b = baseline["by_id"][identity]
            r = row["by_id"][identity]
            b_success = int(b["success"])
            r_success = int(r["success"])
            both_success += b_success and r_success
            baseline_only += b_success and not r_success
            row_only += r_success and not b_success
            both_fail += not b_success and not r_success
            if b_success and r_success:
                step_deltas.append(int(r["num_steps"]) - int(b["num_steps"]))
        comparisons.append(
            {
                "row": row["name"],
                "success_rate": row["success_rate"],
                "delta_success_rate_pp": 100.0 * (row["success_rate"] - baseline["success_rate"]),
                "both_success": both_success,
                "baseline_only_success": baseline_only,
                "row_only_success": row_only,
                "both_fail": both_fail,
                "paired_both_success_mean_step_delta": (
                    float(np.mean(step_deltas)) if step_deltas else None
                ),
            }
        )
    serializable_rows = []
    for row in rows:
        serializable_rows.append({key: value for key, value in row.items() if key != "by_id"})
    payload = {
        "status": "SEER_LATENT_BRIDGE_EVALUATION_AGGREGATION_PASS",
        "manifest": manifest,
        "baseline": args.baseline,
        "rows": serializable_rows,
        "paired_comparisons": comparisons,
    }
    (output / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output / "paired_success.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
