"""Aggregate paired three-seed Latent Bridge evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .recipe import EVALUATION_ROWS, scientific_contract


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in EVALUATION_ROWS}
    seed_metadata: dict[str, Any] = {}
    for seed in (0, 1, 2):
        seed_root = root / f"seed{seed}"
        result = _load(seed_root / "comparison_summary.json")
        metadata = _load(seed_root / "environment_metadata.json")
        if result.get("verdict") != "SIMVLA_LATENT_BRIDGE_EVAL_COMPLETE":
            raise RuntimeError(f"seed {seed} evaluation is incomplete")
        if int(metadata.get("evaluation_seed", -1)) != seed:
            raise RuntimeError(f"seed {seed} metadata identity differs")
        if metadata.get("rows") != list(EVALUATION_ROWS):
            raise RuntimeError(f"seed {seed} row order/identity differs")
        seed_metadata[str(seed)] = {
            "episodes_per_task": metadata["trials_per_task"],
            "renderer": metadata["renderer"],
            "checkpoint": metadata["checkpoint"],
            "bridge_checkpoint": metadata["bridge_checkpoint"],
            "norm_stats_sha256": metadata["norm_stats_sha256"],
        }
        for row_name in EVALUATION_ROWS:
            summary = result["summaries"][row_name]
            if int(summary["episodes"]) != 200:
                raise RuntimeError(
                    f"seed {seed} row {row_name} has {summary['episodes']} episodes, expected 200"
                )
            rows[row_name].append(summary)

    aggregate: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for row_name, seed_rows in rows.items():
        success_rates = np.asarray([row["success_rate"] for row in seed_rows])
        latencies = np.asarray(
            [row["latency_per_executed_action_ms"] for row in seed_rows]
        )
        speedups = np.asarray([row["speedup_vs_baseline"] for row in seed_rows])
        aggregate[row_name] = {
            "seeds": [0, 1, 2],
            "episodes": 600,
            "success_rate_mean": float(success_rates.mean()),
            "success_rate_std": float(success_rates.std(ddof=0)),
            "latency_per_executed_action_ms_mean": float(latencies.mean()),
            "latency_per_executed_action_ms_std": float(latencies.std(ddof=0)),
            "speedup_vs_baseline_mean": float(speedups.mean()),
            "refresh_every": seed_rows[0]["refresh_every"],
            "expected_full_vlm_call_saving": seed_rows[0][
                "expected_full_vlm_call_saving"
            ],
            "observed_full_vlm_call_saving_mean": float(
                np.mean([row["observed_full_vlm_call_saving"] for row in seed_rows])
            ),
        }
        csv_rows.append({"row": row_name, **aggregate[row_name]})

    payload = {
        "verdict": "SIMVLA_LATENT_BRIDGE_THREE_SEED_SUMMARY_COMPLETE",
        "scope": (
            "Matched RTX runtime comparison on SimVLA; latent_bridge_f4 matches "
            "ours K_C=4 only on full-VLM call period, not total coupled compute."
        ),
        "scientific_contract": scientific_contract(),
        "seed_metadata": seed_metadata,
        "rows": aggregate,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "three_seed_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "three_seed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--eval-root", required=True)
    value.add_argument("--output", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(summarize(args.eval_root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
