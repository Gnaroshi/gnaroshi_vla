#!/usr/bin/env python3
"""Aggregate fixed-checkpoint execution seeds without calling them train seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.comparison]
    grouped = defaultdict(list)
    for payload in payloads:
        seed = int(payload["manifest"]["seed"])
        for row in payload["rows"]:
            grouped[row["name"]].append(
                {
                    "seed": seed,
                    "success_rate": float(row["success_rate"]),
                    "successes": int(row["successes"]),
                    "episodes": int(row["episodes"]),
                    "total_env_steps": int(row["total_env_steps"]),
                    "success_episode_mean_steps": row["success_episode_mean_steps"],
                    "summary": row["summary"],
                    "latency": row["latency"],
                }
            )
    rows = []
    for name, seeds in sorted(grouped.items()):
        rates = np.asarray([item["success_rate"] for item in seeds], dtype=np.float64)
        rows.append(
            {
                "row": name,
                "num_execution_seeds": len(seeds),
                "episodes_per_seed": seeds[0]["episodes"],
                "total_episodes": sum(item["episodes"] for item in seeds),
                "pooled_success_rate": sum(item["successes"] for item in seeds)
                / sum(item["episodes"] for item in seeds),
                "seed_success_rate_mean": float(rates.mean()),
                "seed_success_rate_sample_sd": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
                "seed_success_rates": [float(value) for value in rates],
                "total_env_steps": sum(item["total_env_steps"] for item in seeds),
            }
        )
    result = {
        "status": "SEER_LATENT_BRIDGE_THREE_EXECUTION_SEEDS_COMPLETE",
        "replication_unit": (
            "three execution seeds for one frozen Seer checkpoint and one trained bridge; "
            "not three independent training seeds"
        ),
        "rows": rows,
        "per_seed_payloads": payloads,
    }
    (output / "three_execution_seed_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output / "three_execution_seed_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "seed_success_rates": json.dumps(row["seed_success_rates"])})
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
