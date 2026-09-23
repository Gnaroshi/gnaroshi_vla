#!/usr/bin/env python3
"""Build immutable q0-q1-q2 locators and the episode-disjoint split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architectures.simvla.adapters.latentloop.source_lock import require_empty_output  # noqa: E402
from methods.simvla_exact_q2.dataset import build_exact_q2_index  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-refresh-interval", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    args = parser.parse_args()
    audit = json.loads(Path(args.cache_audit).read_text(encoding="utf-8"))
    if audit.get("verdict") != "R5_EXACT_Q2_CACHE_GATE_PASS":
        raise RuntimeError("dataset build is blocked by the cache hard gate")
    if Path(args.cache).resolve() != Path(audit["paths"]["r5_production_cache"]).resolve():
        raise RuntimeError("--cache differs from the hard-gated R5 production cache")
    output = require_empty_output(args.output)
    manifest, split = build_exact_q2_index(
        args.cache,
        full_refresh_interval=args.full_refresh_interval,
        split_seed=args.split_seed,
        heldout_fraction=args.heldout_fraction,
    )
    _write_json(output / "r5_exact_q2_dataset_manifest.json", manifest)
    _write_json(output / "r5_exact_q2_split.json", split)
    per_task = Counter(int(row["task_id"]) for row in manifest["tuples"])
    report = [
        "# R5 exact-q2 dataset report",
        "",
        f"- Cache: `{manifest['cache_root']}`",
        f"- Cache manifest SHA-256: `{manifest['cache_manifest_sha256']}`",
        f"- Tuple anchor: `{manifest['native_semantics']['anchor_rule']}`",
        f"- H / R: {manifest['native_semantics']['action_horizon_H']} / {manifest['native_semantics']['execution_horizon_R']}",
        f"- Total tuples: {manifest['tuple_count']}",
        f"- Train: {split['train_tuple_count']} tuples in {split['train_episode_count']} episodes",
        f"- Validation: {split['validation_tuple_count']} tuples in {split['validation_episode_count']} episodes",
        f"- Split seed: {split['split_seed']}",
        f"- Episode overlap: {len(set(split['train_episode_ids']) & set(split['validation_episode_ids']))}",
        "",
        "## Tuples per task",
        "",
    ]
    report.extend(f"- task {task}: {per_task[task]}" for task in sorted(per_task))
    report.extend(
        [
            "",
            "Each tuple contains q0/q1/q2 RGB and proprioception, C0/C1/C2 full-policy reference conditions, ordered executed X0/X1, same-noise A1/A2 targets, elapsed times, and source hashes through immutable cache record locators.",
        ]
    )
    (output / "r5_exact_q2_dataset_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "tuples": manifest["tuple_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
