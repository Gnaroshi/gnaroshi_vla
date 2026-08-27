"""Aggregate paired SimVLA mechanical controls without imposing a stop gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate import (
    _key,
    _paired,
    _read_csv,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    atomic_write_json,
    load_json,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    MECHANICAL_CONTROL_ROWS,
)


PRIMARY_ROW = "condition_kc2_ng3"
REFERENCE_ROWS = (
    "full_nfe10",
    "generation_ng3",
    "naive_nfe3",
    "condition_kc2_ng10",
    PRIMARY_ROW,
)


def _wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> list[float]:
    if episodes <= 0:
        return [float("nan"), float("nan")]
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / episodes + z * z / (4.0 * episodes * episodes))
        / denominator
    )
    return [center - margin, center + margin]


def _parse_rows(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"--row must be NAME=PATH, got: {value}")
        if name in output:
            raise ValueError(f"duplicate row: {name}")
        output[name] = Path(path).expanduser().resolve()
    required = {PRIMARY_ROW, *MECHANICAL_CONTROL_ROWS}
    missing = sorted(required - set(output))
    if missing:
        raise ValueError(f"missing required rows: {missing}")
    return output


def _validated_row(
    name: str, root: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary = load_json(root / "row_summary.json")
    rows = _read_csv(root / "episode_metrics.csv")
    keys = {_key(row) for row in rows}
    expected_keys = {(task, trial) for task in range(10) for trial in range(50)}
    failures = []
    if summary.get("row") != name:
        failures.append("row name mismatch")
    if summary.get("manifest_sha256") != expected_manifest_sha256:
        failures.append("manifest SHA-256 mismatch")
    if len(rows) != 500 or keys != expected_keys:
        failures.append("episode set is not exact LIBERO-Long 10x50")
    if not str(summary.get("verdict", "")).endswith("_ROW_PASS"):
        failures.append("row verdict did not pass")
    if failures:
        raise RuntimeError(f"invalid row {name} at {root}: {failures}")
    return summary, rows


def _per_task(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    output = {}
    for task_id in range(10):
        task_rows = [row for row in rows if int(row["task_id"]) == task_id]
        successes = sum(int(row["success"]) for row in task_rows)
        output[str(task_id)] = {
            "episodes": len(task_rows),
            "successes": successes,
            "success_rate": successes / len(task_rows),
        }
    return output


def _mean_field(summary: Mapping[str, Any], name: str) -> float | None:
    value = summary.get(name)
    return None if value is None else float(value)


def _first_field(summary: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _mean_field(summary, name)
        if value is not None:
            return value
    return None


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    row_roots = _parse_rows(args.row)
    summaries: dict[str, dict[str, Any]] = {}
    tables: dict[str, list[dict[str, str]]] = {}
    for name, root in row_roots.items():
        summaries[name], tables[name] = _validated_row(
            name, root, args.expected_manifest_sha256
        )

    primary_by_key = {_key(row): row for row in tables[PRIMARY_ROW]}
    paired = {}
    for name in MECHANICAL_CONTROL_ROWS:
        candidate = {_key(row): row for row in tables[name]}
        result = _paired(primary_by_key, candidate)
        result["primary_minus_control_successes"] = (
            result["reference_only"] - result["candidate_only"]
        )
        result["per_task"] = {
            str(task_id): _paired(
                {key: value for key, value in primary_by_key.items() if key[0] == task_id},
                {key: value for key, value in candidate.items() if key[0] == task_id},
            )
            for task_id in range(10)
        }
        paired[name] = result

    row_table = []
    for name, summary in summaries.items():
        episodes = int(summary["episodes"])
        successes = int(summary["successes"])
        row_table.append(
            {
                "row": name,
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes,
                "wilson95_low": _wilson(successes, episodes)[0],
                "wilson95_high": _wilson(successes, episodes)[1],
                "effective_k_c": _mean_field(summary, "effective_k_c"),
                "full_vlm_calls": summary.get("full_vlm_calls"),
                "condition_updater_calls": summary.get("condition_updater_calls"),
                "full_action_transformer_evaluations": summary.get(
                    "full_action_transformer_evaluations"
                ),
                "generation_loop_updates": summary.get("generation_loop_updates"),
                "latency_per_policy_query_ms": _mean_field(
                    summary, "latency_per_policy_query_ms"
                ),
                "latency_per_executed_action_ms": _mean_field(
                    summary, "latency_per_executed_action_ms"
                ),
                "model_vlm_encoder_per_call_ms": _first_field(
                    summary,
                    "model_vlm_encoder_per_call_ms",
                    "model_vlm_encoder_per_query_ms",
                ),
                "model_vlm_encoder_amortized_per_query_ms": _first_field(
                    summary,
                    "model_vlm_encoder_amortized_per_query_ms",
                    "model_vlm_encoder_per_query_ms",
                ),
                "model_action_generation_per_decode_ms": _first_field(
                    summary,
                    "model_action_generation_per_decode_ms",
                    "model_action_generation_per_query_ms",
                ),
                "model_action_generation_amortized_per_query_ms": _first_field(
                    summary,
                    "model_action_generation_amortized_per_query_ms",
                    "model_action_generation_per_query_ms",
                ),
            }
        )

    primary_success = int(summaries[PRIMARY_ROW]["successes"])
    control_successes = {
        name: int(summaries[name]["successes"]) for name in MECHANICAL_CONTROL_ROWS
    }
    point_estimate_verdict = (
        "LEARNED_UPDATE_OUTPERFORMS_ALL_MECHANICAL_POINT_ESTIMATES"
        if all(primary_success > value for value in control_successes.values())
        else "MECHANICAL_CONTROL_NOT_DOMINATED_BY_PRIMARY_POINT_ESTIMATE"
    )

    result = {
        "verdict": "MECHANICAL_CONTROL_COMPARISON_COMPLETE",
        "point_estimate_verdict": point_estimate_verdict,
        "statistical_note": (
            "Point estimates are descriptive. Exact paired McNemar p-values are "
            "reported and no significance claim is made automatically."
        ),
        "primary_row": PRIMARY_ROW,
        "manifest_sha256": args.expected_manifest_sha256,
        "rows": {item["row"]: item for item in row_table},
        "per_task": {name: _per_task(rows) for name, rows in tables.items()},
        "paired_primary_vs_control": paired,
        "row_roots": {name: str(path) for name, path in row_roots.items()},
    }
    output.mkdir(parents=True)
    atomic_write_json(output / "mechanical_control_summary.json", result)
    with (output / "mechanical_control_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_table[0]))
        writer.writeheader()
        writer.writerows(row_table)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--row", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(aggregate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
