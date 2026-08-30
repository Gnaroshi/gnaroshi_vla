"""Validate and aggregate the three-seed SimVLA replay-control campaign."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    atomic_write_json,
    exact_mcnemar,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    FULL_NFE10_REPLAY_ROW,
    expected_call_counts,
    row_spec,
)


SEEDS = ("seed01", "seed02", "seed03")
ROWS = (
    "full_nfe10",
    "condition_kc2_ng3_coupled",
    "condition_kc2_naive_nfe3",
    "mechanical_native_chunk_replay_kc2_ng3",
    FULL_NFE10_REPLAY_ROW,
)
ACCEPTED_VERDICTS = {
    "GENERATION_CONTROL_ROW_PASS",
    "FIXED_2X2_ROW_PASS",
    "KC_FRONTIER_ROW_PASS",
    "MECHANICAL_CONTROL_ROW_PASS",
}


def _integer(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no"}:
        return 0
    return int(float(text))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _parse_cells(values: list[str]) -> dict[tuple[str, str], Path]:
    output: dict[tuple[str, str], Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        seed, marker, row = key.partition(":")
        if not separator or not marker or seed not in SEEDS or row not in ROWS:
            raise ValueError(f"invalid --cell: {value}")
        output[(seed, row)] = Path(raw_path).expanduser().resolve()
    expected = {(seed, row) for seed in SEEDS for row in ROWS}
    if set(output) != expected:
        raise ValueError(f"cell set mismatch: missing={sorted(expected - set(output))}")
    return output


def _parse_manifests(values: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        seed, separator, digest = value.partition("=")
        if not separator or seed not in SEEDS or len(digest) != 64:
            raise ValueError(f"invalid --manifest-sha256: {value}")
        output[seed] = digest
    if set(output) != set(SEEDS):
        raise ValueError("manifest hashes must cover all three inference seeds")
    return output


def _validate_cell(
    seed: str, row: str, root: Path, manifest_sha256: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary = json.loads((root / "row_summary.json").read_text(encoding="utf-8"))
    episodes = _read_csv(root / "episode_metrics.csv")
    expected_ids = {(task, trial) for task in range(10) for trial in range(50)}
    observed_ids = {
        (_integer(item["task_id"]), _integer(item["trial_id"]))
        for item in episodes
    }
    checks = {
        "row": summary.get("row") == row,
        "seed": summary.get("inference_seed") == seed,
        "manifest": summary.get("manifest_sha256") == manifest_sha256,
        "verdict": summary.get("verdict") in ACCEPTED_VERDICTS,
        "classification": summary.get("classification") == "RB2_CONFIRMATORY_EGL",
        "paper_runtime": summary.get("paper_runtime_match") is True,
        "episode_count": len(episodes) == 500,
        "episode_ids": observed_ids == expected_ids,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid cell {seed}:{row}: {checks}")
    successes = sum(_integer(item["success"]) for item in episodes)
    if successes != int(summary["successes"]):
        raise RuntimeError(f"success count mismatch for {seed}:{row}")
    for item in episodes:
        expected = expected_call_counts(row, _integer(item["num_policy_queries"]))
        observed = {
            "full_vlm_calls": _integer(item.get("num_full_vlm_calls", 0)),
            "condition_updater_calls": _integer(
                item.get("num_condition_updater_calls", 0)
            ),
            "full_action_transformer_calls": _integer(
                item.get("num_full_action_transformer_evaluations", 0)
            ),
            "generation_loop_updates": _integer(
                item.get("num_generation_loop_updates", 0)
            ),
            "integration_updates": _integer(item.get("num_integration_updates", 0)),
        }
        if observed != expected:
            raise RuntimeError(
                f"counter mismatch {seed}:{row} "
                f"task={item['task_id']} trial={item['trial_id']}"
            )
    return summary, episodes


def _episode_map(
    rows: list[dict[str, str]],
) -> dict[tuple[int, int], bool]:
    return {
        (_integer(item["task_id"]), _integer(item["trial_id"])): bool(
            _integer(item["success"])
        )
        for item in rows
    }


def _paired(left: Mapping[Any, bool], right: Mapping[Any, bool]) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("paired episode keys differ")
    left_only = right_only = both_success = both_fail = 0
    for key in sorted(left):
        if left[key] and right[key]:
            both_success += 1
        elif left[key]:
            left_only += 1
        elif right[key]:
            right_only += 1
        else:
            both_fail += 1
    return {
        "episodes": len(left),
        "both_success": both_success,
        "both_fail": both_fail,
        "left_only": left_only,
        "right_only": right_only,
        "right_minus_left_successes": right_only - left_only,
        "exact_mcnemar_p": exact_mcnemar(left_only, right_only),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    roots = _parse_cells(args.cell)
    manifests = _parse_manifests(args.manifest_sha256)
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    episodes: dict[tuple[str, str], list[dict[str, str]]] = {}
    cells: list[dict[str, Any]] = []
    for seed in SEEDS:
        for row in ROWS:
            summary, rows = _validate_cell(seed, row, roots[(seed, row)], manifests[seed])
            summaries[(seed, row)] = summary
            episodes[(seed, row)] = rows
            spec = row_spec(row)
            successes = int(summary["successes"])
            cells.append(
                {
                    "seed": seed,
                    "row": row,
                    "k_c": spec.k_c,
                    "n_g": spec.n_g,
                    "episodes": 500,
                    "successes": successes,
                    "success_percent": successes / 5.0,
                    "latency_per_executed_action_ms": float(
                        summary["latency_per_executed_action_ms"]
                    ),
                    "root": str(roots[(seed, row)]),
                }
            )

    row_summaries = []
    for row in ROWS:
        selected = [item for item in cells if item["row"] == row]
        rates = [float(item["success_percent"]) for item in selected]
        latencies = [float(item["latency_per_executed_action_ms"]) for item in selected]
        row_summaries.append(
            {
                "row": row,
                "episodes": 1500,
                "successes": sum(int(item["successes"]) for item in selected),
                "seed_mean_success_percent": statistics.mean(rates),
                "seed_sample_std_success_pp": statistics.stdev(rates),
                "seed_mean_latency_per_action_ms": statistics.mean(latencies),
                "seed_sample_std_latency_per_action_ms": statistics.stdev(latencies),
            }
        )

    comparisons = {
        "full_vs_pure_replay": ("full_nfe10", FULL_NFE10_REPLAY_ROW),
        "pure_vs_learned_replay": (
            FULL_NFE10_REPLAY_ROW,
            "mechanical_native_chunk_replay_kc2_ng3",
        ),
        "pure_replay_vs_coupled": (
            FULL_NFE10_REPLAY_ROW,
            "condition_kc2_ng3_coupled",
        ),
        "learned_replay_vs_coupled": (
            "mechanical_native_chunk_replay_kc2_ng3",
            "condition_kc2_ng3_coupled",
        ),
        "kc2_naive_vs_coupled": (
            "condition_kc2_naive_nfe3",
            "condition_kc2_ng3_coupled",
        ),
    }
    paired: dict[str, Any] = {}
    for name, (left_row, right_row) in comparisons.items():
        per_seed = {}
        pooled_left: dict[tuple[str, int, int], bool] = {}
        pooled_right: dict[tuple[str, int, int], bool] = {}
        for seed in SEEDS:
            left = _episode_map(episodes[(seed, left_row)])
            right = _episode_map(episodes[(seed, right_row)])
            per_seed[seed] = _paired(left, right)
            pooled_left.update({(seed, *key): value for key, value in left.items()})
            pooled_right.update({(seed, *key): value for key, value in right.items()})
        paired[name] = {
            "left_row": left_row,
            "right_row": right_row,
            "per_seed": per_seed,
            "pooled": _paired(pooled_left, pooled_right),
        }

    result = {
        "verdict": "REPLAY_CONTROL_THREE_SEED_COMPLETE",
        "replication_unit": (
            "Three deterministic inference/action-noise seeds for frozen checkpoints; "
            "not three independent training seeds."
        ),
        "manifests": manifests,
        "rows": row_summaries,
        "paired": paired,
    }
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "replay_control_cells.csv", cells)
    _write_csv(output / "replay_control_rows.csv", row_summaries)
    atomic_write_json(output / "replay_control_three_seed_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--manifest-sha256", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(aggregate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
