#!/usr/bin/env python3
"""Aggregate multi-rank Seer continuity shards and choose a measured context."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--official-threshold", type=float, default=0.999)
    return parser.parse_args()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main():
    args = parse_args()
    root = Path(args.input_dir)
    shards = sorted(root.glob("continuity_rank*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no continuity shards in {root}")
    rows = []
    for shard in shards:
        rows.extend(json.loads(line) for line in shard.read_text().splitlines() if line.strip())
    groups = defaultdict(list)
    for row in rows:
        base = (row["layer"], row["token_group"], int(row["offset"]))
        groups[("overall", "all", *base)].append(row)
        groups[("task", str(row["task_id"]), *base)].append(row)
        groups[("outcome", "success" if row["success"] else "failure", *base)].append(row)
    summaries = []
    for key, items in sorted(groups.items()):
        breakdown, value, layer, token_group, offset = key
        summary = {
            "breakdown": breakdown,
            "breakdown_value": value,
            "layer": layer,
            "token_group": token_group,
            "offset": offset,
        }
        for metric in ("cosine", "delta_l2", "delta_mse"):
            for name, number in summarize([item[metric] for item in items]).items():
                summary[f"{metric}_{name}"] = number
        summaries.append(summary)
    output_csv = root / "seer_layer_similarity.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    overall_adjacent = [
        row
        for row in summaries
        if row["breakdown"] == "overall" and row["offset"] == 1
    ]
    # Official Latent Bridge uses an image-token stable context. Restricting the
    # decision to Seer's combined primary+wrist visual group prevents a nearly
    # constant learned query token from winning merely because it is trivial to
    # copy. Other groups remain in the audit CSV but cannot become the context.
    eligible_groups = {"visual"}
    eligible = [row for row in overall_adjacent if row["token_group"] in eligible_groups]
    passing = [row for row in eligible if row["cosine_p05"] > args.official_threshold]
    pool = passing or eligible
    decision = max(pool, key=lambda row: (row["cosine_p05"], row["cosine_mean"]))
    threshold_passed = bool(passing)
    text_output = next(
        row
        for row in overall_adjacent
        if row["layer"] == "ln_f" and row["token_group"] == "text"
    )
    text_threshold = 0.9999
    text_threshold_passed = text_output["cosine_p05"] > text_threshold
    episodes = sorted({row["episode_id"] for row in rows})
    tasks = sorted({int(row["task_id"]) for row in rows})
    meta_paths = sorted(root.glob("continuity_rank*.meta.json"))
    if not meta_paths:
        raise FileNotFoundError(f"no continuity metadata shards in {root}")
    metas = [json.loads(path.read_text(encoding="utf-8")) for path in meta_paths]
    token_layout = metas[0]["token_layout"]
    if any(meta["token_layout"] != token_layout for meta in metas[1:]):
        raise RuntimeError("continuity ranks disagree on the Seer token layout")
    payload = {
        "num_raw_metric_rows": len(rows),
        "num_episodes": len(episodes),
        "num_tasks": len(tasks),
        "episode_ids": episodes,
        "task_ids": tasks,
        "token_layout": token_layout,
        "official_stable_threshold": args.official_threshold,
        "any_candidate_p05_strictly_above_threshold": threshold_passed,
        "official_assumption_checks": {
            "final_output_text_adjacent_cosine": text_output,
            "text_p05_threshold": text_threshold,
            "text_p05_strictly_above_threshold": text_threshold_passed,
            "visual_stable_p05_threshold": args.official_threshold,
            "visual_stable_candidate_strictly_above_threshold": threshold_passed,
        },
        "selected_stable_context": decision,
        "selection_rule": (
            "maximum adjacent-query p05 among the combined primary+wrist visual tokens; "
            "prefer only candidates strictly above the official 0.999 threshold when any exist"
        ),
        "overall": overall_adjacent,
    }
    (root / "seer_latent_continuity_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    verdict = "official threshold passed" if threshold_passed else "Seer adaptation required"
    lines = [
        "# Seer Stable-Layer and Token-Mode Decision",
        "",
        f"- Verdict: **{verdict}**.",
        f"- Episodes/tasks: `{len(episodes)}` / `{len(tasks)}`.",
        f"- Selected layer: `{decision['layer']}`.",
        f"- Selected token group: `{decision['token_group']}`.",
        f"- Adjacent cosine mean/median/p05/min: `{decision['cosine_mean']:.6f}` / "
        f"`{decision['cosine_median']:.6f}` / `{decision['cosine_p05']:.6f}` / "
        f"`{decision['cosine_minimum']:.6f}`.",
        f"- Final-output text-token p05: `{text_output['cosine_p05']:.6f}`; "
        f"official `> {text_threshold}` assumption passed: `{text_threshold_passed}`.",
        "",
        "The decision does not lower the official threshold. If no eligible candidate has p05 > 0.999, "
        "the highest-p05 measured candidate is recorded as a Seer-specific adaptation rather than an "
        "official stable-context match.",
    ]
    (root / "seer_token_mode_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "selected": decision, "output": str(root)}, indent=2))


if __name__ == "__main__":
    main()
