"""Validate, plan, and aggregate the selected three-seed SimVLA paper follow-up."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    atomic_write_json,
    exact_mcnemar,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    expected_call_counts,
    row_spec,
)


SEEDS = ("seed01", "seed02", "seed03")
ROWS = (
    "full_nfe10",
    "naive_nfe3",
    "generation_ng3",
    "condition_kc2_ng10",
    "condition_kc2_ng3",
    "condition_kc2_ng2_coupled",
    "condition_kc2_ng3_coupled",
    "condition_kc2_ng5_coupled",
)
CLASSIFICATION = "RB2_CONFIRMATORY_EGL"
ACCEPTED_VERDICTS = {
    "GENERATION_CONTROL_ROW_PASS",
    "FIXED_2X2_ROW_PASS",
    "KC_FRONTIER_ROW_PASS",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _integer(row: Mapping[str, Any], name: str, default: int = 0) -> int:
    value = row.get(name)
    if value in {None, ""}:
        return int(default)
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no"}:
        return 0
    return int(float(text))


def _finite_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.mean(finite) if finite else None


def _sample_std(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.stdev(finite) if len(finite) >= 2 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_keyed_paths(values: Iterable[str], *, separator: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, marker, raw_path = value.partition(separator)
        if not marker or not key or not raw_path:
            raise ValueError(f"expected KEY{separator}PATH: {value}")
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = Path(raw_path).expanduser().resolve()
    return result


def _parse_cell_roots(values: Iterable[str]) -> dict[tuple[str, str], Path]:
    keyed = _parse_keyed_paths(values, separator="=")
    roots: dict[tuple[str, str], Path] = {}
    for key, path in keyed.items():
        seed, marker, row = key.partition(":")
        if not marker or seed not in SEEDS or row not in ROWS:
            raise ValueError(f"invalid cell key: {key}")
        roots[(seed, row)] = path
    return roots


def _parse_manifest_hashes(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        seed, marker, digest = value.partition("=")
        if not marker or seed not in SEEDS or len(digest) != 64:
            raise ValueError(f"invalid manifest SHA mapping: {value}")
        if seed in result:
            raise ValueError(f"duplicate manifest seed: {seed}")
        result[seed] = digest
    if set(result) != set(SEEDS):
        raise ValueError(f"manifest hashes must cover {SEEDS}")
    return result


def _expected_episode_ids() -> set[tuple[int, int]]:
    return {(task_id, trial_id) for task_id in range(10) for trial_id in range(50)}


def validate_cell(
    seed: str,
    row: str,
    root: str | Path,
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"unknown seed: {seed}")
    if row not in ROWS:
        raise ValueError(f"row is outside selected follow-up: {row}")
    destination = Path(root).expanduser().resolve()
    summary_path = destination / "row_summary.json"
    episode_path = destination / "episode_metrics.csv"
    failures: list[str] = []
    summary: dict[str, Any] = {}
    episodes: list[dict[str, str]] = []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"row_summary unreadable: {exc}")
    try:
        episodes = _read_csv(episode_path)
    except Exception as exc:
        failures.append(f"episode_metrics unreadable: {exc}")

    expected_ids = _expected_episode_ids()
    observed_ids: set[tuple[int, int]] = set()
    if episodes:
        try:
            observed_ids = {
                (_integer(item, "task_id"), _integer(item, "trial_id"))
                for item in episodes
            }
        except Exception as exc:
            failures.append(f"episode IDs invalid: {exc}")
    if len(episodes) != 500 or observed_ids != expected_ids:
        failures.append(
            f"episode set mismatch: rows={len(episodes)} unique_ids={len(observed_ids)}"
        )

    if summary:
        checks = {
            "row": summary.get("row") == row,
            "verdict": summary.get("verdict") in ACCEPTED_VERDICTS,
            "episodes": int(summary.get("episodes", -1)) == 500,
            "manifest": summary.get("manifest_sha256") == manifest_sha256,
            "seed": summary.get("inference_seed") == seed,
            "classification": summary.get("classification") == CLASSIFICATION,
            "paper_runtime": summary.get("paper_runtime_match") is True,
        }
        failures.extend(name for name, passed in checks.items() if not passed)

    if len(episodes) == 500 and observed_ids == expected_ids:
        try:
            successes = 0
            for item in episodes:
                successes += _integer(item, "success")
                expected = expected_call_counts(row, _integer(item, "num_policy_queries"))
                observed = {
                    "full_vlm_calls": _integer(item, "num_full_vlm_calls"),
                    "condition_updater_calls": _integer(
                        item, "num_condition_updater_calls"
                    ),
                    "full_action_transformer_calls": _integer(
                        item, "num_full_action_transformer_evaluations"
                    ),
                    "generation_loop_updates": _integer(
                        item, "num_generation_loop_updates"
                    ),
                    "integration_updates": _integer(item, "num_integration_updates"),
                }
                if observed != expected:
                    raise RuntimeError(
                        f"counter mismatch task={item.get('task_id')} "
                        f"trial={item.get('trial_id')} observed={observed} expected={expected}"
                    )
            if summary and successes != int(summary.get("successes", -1)):
                failures.append(
                    f"success count mismatch: csv={successes} summary={summary.get('successes')}"
                )
        except Exception as exc:
            failures.append(str(exc))

    return {
        "verdict": "PAPER_FOLLOWUP_CELL_PASS" if not failures else "PAPER_FOLLOWUP_CELL_FAIL",
        "seed": seed,
        "row": row,
        "root": str(destination),
        "episodes": len(episodes),
        "successes": int(summary.get("successes", -1)) if summary else -1,
        "success_rate": float(summary.get("success_rate", -1.0)) if summary else -1.0,
        "failures": failures,
    }


def build_plan(
    cell_roots: Mapping[tuple[str, str], Path],
    manifest_hashes: Mapping[str, str],
) -> dict[str, Any]:
    complete: list[str] = []
    missing: list[str] = []
    validations: dict[str, Any] = {}
    for seed in SEEDS:
        for row in ROWS:
            key = f"{seed}:{row}"
            root = cell_roots.get((seed, row))
            if root is None:
                report = {
                    "verdict": "PAPER_FOLLOWUP_CELL_MISSING",
                    "seed": seed,
                    "row": row,
                    "failures": ["no cell root provided"],
                }
            else:
                report = validate_cell(
                    seed, row, root, manifest_sha256=manifest_hashes[seed]
                )
            validations[key] = report
            (complete if report["verdict"] == "PAPER_FOLLOWUP_CELL_PASS" else missing).append(key)
    return {
        "verdict": (
            "PAPER_FOLLOWUP_COMPLETE"
            if not missing
            else "PAPER_FOLLOWUP_EXECUTION_REQUIRED"
        ),
        "schema_version": "simvla_selected_three_inference_seed_followup_v1",
        "replication_unit": (
            "Three deterministic inference/action-noise seeds for fixed trained "
            "checkpoints; these are not three independent training seeds."
        ),
        "seeds": list(SEEDS),
        "rows": list(ROWS),
        "target_cell_count": len(SEEDS) * len(ROWS),
        "complete_cells": complete,
        "complete_cell_count": len(complete),
        "missing_cells": missing,
        "missing_cell_count": len(missing),
        "new_episode_count": 500 * len(missing),
        "manifest_sha256": dict(manifest_hashes),
        "validations": validations,
    }


def _episode_map(root: Path) -> dict[tuple[int, int], dict[str, str]]:
    rows = _read_csv(root / "episode_metrics.csv")
    return {
        (_integer(item, "task_id"), _integer(item, "trial_id")): item
        for item in rows
    }


def _paired(
    baseline: Mapping[tuple[int, int], Mapping[str, Any]],
    candidate: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise RuntimeError("paired episode IDs differ")
    baseline_only = candidate_only = both_success = both_fail = 0
    for key in sorted(baseline):
        left = bool(_integer(baseline[key], "success"))
        right = bool(_integer(candidate[key], "success"))
        if left and right:
            both_success += 1
        elif left:
            baseline_only += 1
        elif right:
            candidate_only += 1
        else:
            both_fail += 1
    return {
        "episodes": len(baseline),
        "both_success": both_success,
        "both_fail": both_fail,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "exact_mcnemar_p": exact_mcnemar(baseline_only, candidate_only),
    }


def _component(summary: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _finite_float(summary.get(name))
        if value is not None:
            return value
    return None


def _cell_table_row(seed: str, row: str, root: Path) -> dict[str, Any]:
    summary = json.loads((root / "row_summary.json").read_text(encoding="utf-8"))
    episodes = _read_csv(root / "episode_metrics.csv")
    spec = row_spec(row)
    successes = sum(_integer(item, "success") for item in episodes)
    return {
        "seed": seed,
        "row": row,
        "k_c": spec.k_c,
        "compute": spec.n_g,
        "compute_axis": "nfe" if spec.naive_nfe else "n_g",
        "coupled": spec.coupled,
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "success_percent": 100.0 * successes / len(episodes),
        "latency_per_executed_action_ms": _finite_float(
            summary.get("latency_per_executed_action_ms")
        ),
        "latency_per_policy_query_ms": _finite_float(
            summary.get("latency_per_policy_query_ms")
        ),
        "model_vlm_encoder_per_call_ms": _component(
            summary,
            "model_vlm_encoder_per_call_ms",
            "model_vlm_encoder_per_query_ms",
        ),
        "model_condition_updater_per_call_ms": _component(
            summary, "model_condition_updater_per_call_ms"
        ),
        "model_action_generation_per_query_ms": _component(
            summary, "model_action_generation_per_query_ms"
        ),
        "full_vlm_calls": sum(_integer(item, "num_full_vlm_calls") for item in episodes),
        "condition_updater_calls": sum(
            _integer(item, "num_condition_updater_calls") for item in episodes
        ),
        "full_action_transformer_evaluations": sum(
            _integer(item, "num_full_action_transformer_evaluations")
            for item in episodes
        ),
        "generation_loop_updates": sum(
            _integer(item, "num_generation_loop_updates") for item in episodes
        ),
        "root": str(root),
    }


def _external_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "verdict": payload.get("verdict"),
    }


def aggregate(
    cell_roots: Mapping[tuple[str, str], Path],
    manifest_hashes: Mapping[str, str],
    output: str | Path,
    *,
    external_artifacts: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    plan = build_plan(cell_roots, manifest_hashes)
    if plan["verdict"] != "PAPER_FOLLOWUP_COMPLETE":
        raise RuntimeError(json.dumps(plan, indent=2, sort_keys=True))
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    cells = [
        _cell_table_row(seed, row, cell_roots[(seed, row)])
        for seed in SEEDS
        for row in ROWS
    ]
    _write_csv(destination / "selected_three_seed_cells.csv", cells)
    by_key = {(item["seed"], item["row"]): item for item in cells}
    episode_maps = {
        (seed, row): _episode_map(cell_roots[(seed, row)])
        for seed in SEEDS
        for row in ROWS
    }

    row_summaries: list[dict[str, Any]] = []
    paired_reports: dict[str, Any] = {}
    per_task: list[dict[str, Any]] = []
    for row in ROWS:
        selected = [by_key[(seed, row)] for seed in SEEDS]
        rates = [float(item["success_rate"]) for item in selected]
        latencies = [item["latency_per_executed_action_ms"] for item in selected]
        pooled_successes = sum(int(item["successes"]) for item in selected)
        baseline_latencies = [
            by_key[(seed, "full_nfe10")]["latency_per_executed_action_ms"]
            for seed in SEEDS
        ]
        reductions = [
            1.0 - float(latency) / float(baseline)
            for latency, baseline in zip(latencies, baseline_latencies)
            if latency is not None and baseline is not None
        ]
        row_summaries.append(
            {
                "row": row,
                "k_c": row_spec(row).k_c,
                "compute": row_spec(row).n_g,
                "compute_axis": "nfe" if row_spec(row).naive_nfe else "n_g",
                "coupled": row_spec(row).coupled,
                "seeds": 3,
                "episodes": 1500,
                "successes": pooled_successes,
                "pooled_success_rate": pooled_successes / 1500.0,
                "seed_mean_success_rate": statistics.mean(rates),
                "seed_sample_std_success_rate": statistics.stdev(rates),
                "seed_mean_latency_per_action_ms": _mean(latencies),
                "seed_sample_std_latency_per_action_ms": _sample_std(latencies),
                "seed_mean_latency_reduction_vs_full_fraction": _mean(reductions),
                "seed_sample_std_latency_reduction_vs_full_fraction": _sample_std(
                    reductions
                ),
                "seed_mean_vlm_encoder_per_call_ms": _mean(
                    item["model_vlm_encoder_per_call_ms"] for item in selected
                ),
                "seed_mean_condition_updater_per_call_ms": _mean(
                    item["model_condition_updater_per_call_ms"] for item in selected
                ),
                "seed_mean_action_generation_per_query_ms": _mean(
                    item["model_action_generation_per_query_ms"] for item in selected
                ),
                "full_vlm_calls": sum(int(item["full_vlm_calls"]) for item in selected),
                "condition_updater_calls": sum(
                    int(item["condition_updater_calls"]) for item in selected
                ),
                "full_action_transformer_evaluations": sum(
                    int(item["full_action_transformer_evaluations"])
                    for item in selected
                ),
                "generation_loop_updates": sum(
                    int(item["generation_loop_updates"]) for item in selected
                ),
            }
        )

        discordant_baseline = discordant_candidate = 0
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            report = _paired(
                episode_maps[(seed, "full_nfe10")], episode_maps[(seed, row)]
            )
            per_seed[seed] = report
            discordant_baseline += int(report["baseline_only"])
            discordant_candidate += int(report["candidate_only"])
        paired_reports[row] = {
            "per_seed": per_seed,
            "pooled": {
                "episodes": 1500,
                "baseline_only": discordant_baseline,
                "candidate_only": discordant_candidate,
                "exact_mcnemar_p": exact_mcnemar(
                    discordant_baseline, discordant_candidate
                ),
            },
        }

        for task_id in range(10):
            task_rates = []
            task_successes = 0
            for seed in SEEDS:
                rows = [
                    item
                    for (task, _), item in episode_maps[(seed, row)].items()
                    if task == task_id
                ]
                successes = sum(_integer(item, "success") for item in rows)
                task_successes += successes
                task_rates.append(successes / 50.0)
            per_task.append(
                {
                    "row": row,
                    "task_id": task_id,
                    "episodes": 150,
                    "successes": task_successes,
                    "pooled_success_rate": task_successes / 150.0,
                    "seed_mean_success_rate": statistics.mean(task_rates),
                    "seed_sample_std_success_rate": statistics.stdev(task_rates),
                }
            )

    _write_csv(destination / "selected_three_seed_summary.csv", row_summaries)
    _write_csv(destination / "selected_three_seed_per_task.csv", per_task)
    externals = {
        name: _external_artifact(path)
        for name, path in (external_artifacts or {}).items()
    }
    summary = {
        "verdict": "PAPER_FOLLOWUP_THREE_INFERENCE_SEED_COMPLETE",
        "schema_version": "simvla_selected_three_inference_seed_followup_v1",
        "replication_unit": plan["replication_unit"],
        "suite": "libero_10",
        "seeds": list(SEEDS),
        "rows": list(ROWS),
        "cell_count": len(cells),
        "episodes_per_cell": 500,
        "episodes_per_row": 1500,
        "manifest_sha256": dict(manifest_hashes),
        "row_summaries": row_summaries,
        "paired_vs_full_nfe10": paired_reports,
        "external_evidence": externals,
    }
    atomic_write_json(destination / "paper_followup_three_seed_summary.json", summary)
    atomic_write_json(destination / "paper_followup_validation.json", plan)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--cell-root", action="append", default=[])
        command.add_argument("--manifest-sha256", action="append", default=[])
        command.add_argument("--output", required=True)
        if name == "aggregate":
            command.add_argument("--external-artifact", action="append", default=[])
    validate = commands.add_parser("validate-cell")
    validate.add_argument("--seed", choices=SEEDS, required=True)
    validate.add_argument("--row", choices=ROWS, required=True)
    validate.add_argument("--root", required=True)
    validate.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()

    if args.command == "validate-cell":
        result = validate_cell(
            args.seed, args.row, args.root, manifest_sha256=args.manifest_sha256
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["verdict"] == "PAPER_FOLLOWUP_CELL_PASS" else 1

    roots = _parse_cell_roots(args.cell_root)
    manifest_hashes = _parse_manifest_hashes(args.manifest_sha256)
    if args.command == "plan":
        result = build_plan(roots, manifest_hashes)
        atomic_write_json(args.output, result)
    else:
        externals = _parse_keyed_paths(args.external_artifact, separator="=")
        result = aggregate(
            roots,
            manifest_hashes,
            args.output,
            external_artifacts=externals,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
