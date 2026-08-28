"""Plan, validate, and aggregate the single-seed SimVLA paper grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    atomic_write_json,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    PAPER_ANCHOR_ROWS,
    PAPER_COUPLED_ROWS,
    PAPER_GRID_ROWS,
    PAPER_LEARNED_ROWS,
    PAPER_NAIVE_ROWS,
    expected_call_counts,
    row_spec,
)


MANIFEST_SHA256 = "9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48"
INFERENCE_SEED = "seed02"
CLASSIFICATION = "RB2_CONFIRMATORY_EGL"
ACCEPTED_VERDICTS = {
    "GENERATION_CONTROL_ROW_PASS",
    "FIXED_2X2_ROW_PASS",
    "KC_FRONTIER_ROW_PASS",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def _integer(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if value in {None, ""}:
        return 0
    return int(float(str(value)))


def _parse_row_roots(values: Iterable[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        row, separator, raw_path = value.partition("=")
        if not separator or not row or not raw_path:
            raise ValueError(f"--row-root must be ROW=PATH: {value}")
        if row not in PAPER_GRID_ROWS:
            raise ValueError(f"row is outside the paper grid: {row}")
        if row in roots:
            raise ValueError(f"duplicate --row-root: {row}")
        roots[row] = Path(raw_path).expanduser().resolve()
    return roots


def validate_row(
    row: str,
    root: str | Path,
    *,
    manifest_sha256: str = MANIFEST_SHA256,
) -> dict[str, Any]:
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

    expected_ids = {(task, trial) for task in range(10) for trial in range(50)}
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
            "seed": summary.get("inference_seed") == INFERENCE_SEED,
            "classification": summary.get("classification") == CLASSIFICATION,
            "paper_runtime": summary.get("paper_runtime_match") is True,
        }
        failures.extend(name for name, passed in checks.items() if not passed)

    if len(episodes) == 500 and observed_ids == expected_ids:
        try:
            for item in episodes:
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
        except Exception as exc:
            failures.append(str(exc))

    return {
        "verdict": "PAPER_GRID_ROW_PASS" if not failures else "PAPER_GRID_ROW_FAIL",
        "row": row,
        "root": str(destination),
        "summary": str(summary_path),
        "episodes": len(episodes),
        "successes": int(summary.get("successes", -1)) if summary else -1,
        "success_rate": float(summary.get("success_rate", -1.0)) if summary else -1.0,
        "failures": failures,
    }


def build_plan(
    row_roots: Mapping[str, Path],
    *,
    manifest_sha256: str = MANIFEST_SHA256,
) -> dict[str, Any]:
    complete: list[str] = []
    missing: list[str] = []
    validations: dict[str, dict[str, Any]] = {}
    for row in PAPER_GRID_ROWS:
        root = row_roots.get(row)
        if root is None:
            missing.append(row)
            validations[row] = {
                "verdict": "PAPER_GRID_ROW_MISSING",
                "row": row,
                "failures": ["no row root was provided"],
            }
            continue
        report = validate_row(row, root, manifest_sha256=manifest_sha256)
        validations[row] = report
        if report["verdict"] == "PAPER_GRID_ROW_PASS":
            complete.append(row)
        else:
            missing.append(row)
    return {
        "verdict": (
            "PAPER_GRID_COMPLETE" if not missing else "PAPER_GRID_EXECUTION_REQUIRED"
        ),
        "schema_version": "simvla_paper_grid_seed02_v1",
        "manifest_sha256": manifest_sha256,
        "inference_seed": INFERENCE_SEED,
        "classification": CLASSIFICATION,
        "expected_rows": list(PAPER_GRID_ROWS),
        "expected_row_count": len(PAPER_GRID_ROWS),
        "complete_rows": complete,
        "complete_row_count": len(complete),
        "missing_rows": missing,
        "missing_row_count": len(missing),
        "validations": validations,
    }


def _family(row: str) -> str:
    if row in PAPER_ANCHOR_ROWS:
        return "anchor"
    if row in PAPER_NAIVE_ROWS:
        return "naive_nfe"
    if row in PAPER_LEARNED_ROWS:
        return "learned_generation"
    if row in PAPER_COUPLED_ROWS:
        return "coupled_projection"
    raise ValueError(f"unknown paper row: {row}")


def _table_row(row: str, root: Path) -> dict[str, Any]:
    summary = json.loads((root / "row_summary.json").read_text(encoding="utf-8"))
    spec = row_spec(row)
    return {
        "row": row,
        "family": _family(row),
        "k_c": spec.k_c,
        "compute": spec.n_g,
        "compute_axis": "nfe" if spec.naive_nfe else "n_g",
        "generation_parent_schedule": (
            "trained_schedule"
            if spec.uses_generation and spec.n_g == 2
            else "model_supported_transfer"
            if spec.uses_generation and spec.n_g == 3
            else "schedule_extrapolation"
            if spec.uses_generation and spec.n_g == 5
            else "native_nfe"
        ),
        "coupled_projection_schedule": "trained" if spec.coupled else "not_applicable",
        "episodes": int(summary["episodes"]),
        "successes": int(summary["successes"]),
        "success_rate": float(summary["success_rate"]),
        "success_percent": 100.0 * float(summary["success_rate"]),
        "latency_per_executed_action_ms": float(
            summary["latency_per_executed_action_ms"]
        ),
        "latency_per_policy_query_ms": float(summary["latency_per_policy_query_ms"]),
        "full_vlm_calls": int(summary["full_vlm_calls"]),
        "condition_updater_calls": int(summary["condition_updater_calls"]),
        "full_action_transformer_evaluations": int(
            summary["full_action_transformer_evaluations"]
        ),
        "generation_loop_updates": int(summary["generation_loop_updates"]),
        "root": str(root),
    }


def aggregate(
    row_roots: Mapping[str, Path],
    output: str | Path,
    *,
    manifest_sha256: str = MANIFEST_SHA256,
) -> dict[str, Any]:
    plan = build_plan(row_roots, manifest_sha256=manifest_sha256)
    if plan["verdict"] != "PAPER_GRID_COMPLETE":
        raise RuntimeError(json.dumps(plan, indent=2, sort_keys=True))
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = [_table_row(row, row_roots[row]) for row in PAPER_GRID_ROWS]
    _write_csv(destination / "paper_grid_all_rows.csv", rows)

    anchors = {row_spec(row).k_c: row for row in PAPER_ANCHOR_ROWS}
    for family, family_rows in (
        ("naive_nfe", PAPER_NAIVE_ROWS),
        ("learned_generation", PAPER_LEARNED_ROWS),
    ):
        selected = [
            _table_row(anchors[k_c], row_roots[anchors[k_c]])
            for k_c in (1, 2, 3)
        ] + [_table_row(row, row_roots[row]) for row in family_rows]
        selected.sort(key=lambda item: (int(item["k_c"]), int(item["compute"])))
        _write_csv(destination / f"heatmap_{family}.csv", selected)
    coupled = [_table_row(row, row_roots[row]) for row in PAPER_COUPLED_ROWS]
    coupled.sort(key=lambda item: (int(item["k_c"]), int(item["compute"])))
    _write_csv(destination / "heatmap_coupled_projection.csv", coupled)

    summary = {
        "verdict": "PAPER_GRID_SEED02_COMPLETE",
        "schema_version": "simvla_paper_grid_seed02_v1",
        "manifest_sha256": manifest_sha256,
        "inference_seed": INFERENCE_SEED,
        "classification": CLASSIFICATION,
        "row_count": len(rows),
        "families": {
            "anchor": len(PAPER_ANCHOR_ROWS),
            "naive_nfe": len(PAPER_NAIVE_ROWS),
            "learned_generation": len(PAPER_LEARNED_ROWS),
            "coupled_projection": len(PAPER_COUPLED_ROWS),
        },
        "generation_schedule_extrapolation_rows": [
            item["row"]
            for item in rows
            if item["generation_parent_schedule"] == "schedule_extrapolation"
        ],
        "rows": rows,
    }
    atomic_write_json(destination / "paper_grid_summary.json", summary)
    atomic_write_json(destination / "paper_grid_validation.json", plan)
    return summary


def validate_coupled_artifact(
    *,
    k_c: int,
    n_g: int,
    train_root: str | Path,
    offline_root: str | Path,
) -> dict[str, Any]:
    train = Path(train_root).expanduser().resolve()
    offline = Path(offline_root).expanduser().resolve()
    failures: list[str] = []
    try:
        config = json.loads((train / "training_config.json").read_text(encoding="utf-8"))
        summary = json.loads((train / "run_summary.json").read_text(encoding="utf-8"))
    except Exception as exc:
        config = {}
        summary = {}
        failures.append(f"training artifact unreadable: {exc}")
    checkpoint = train / "checkpoints" / "coupled_generation_step_010000.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        failures.append(f"checkpoint missing: {checkpoint}")
    checks = {
        "training_verdict": summary.get("verdict")
        == "COUPLED_PROJECTION_TRAINING_COMPLETE",
        "optimizer_step": int(summary.get("optimizer_step", -1)) == 10_000,
        "k_c": int(config.get("k_c", -1)) == int(k_c),
        "n_g": int(config.get("n_g", -1)) == int(n_g),
        "projection_parameters": int(
            config.get("projection_audit", {}).get("trainable_parameters", -1)
        )
        == 16_384,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    try:
        screen = json.loads((offline / "offline_screen.json").read_text(encoding="utf-8"))
        if screen.get("verdict") != "COUPLED_OFFLINE_INTEGRITY_PASS":
            failures.append("offline_verdict")
        if int(screen.get("k_c", -1)) != int(k_c):
            failures.append("offline_k_c")
        if int(screen.get("n_g", -1)) != int(n_g):
            failures.append("offline_n_g")
        if int(screen.get("queries", -1)) != 512:
            failures.append("offline_queries")
    except Exception as exc:
        failures.append(f"offline artifact unreadable: {exc}")
    return {
        "verdict": (
            "COUPLED_PAPER_ARTIFACT_PASS"
            if not failures
            else "COUPLED_PAPER_ARTIFACT_FAIL"
        ),
        "k_c": int(k_c),
        "n_g": int(n_g),
        "train_root": str(train),
        "offline_root": str(offline),
        "checkpoint": str(checkpoint),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "aggregate"):
        command = subparsers.add_parser(name)
        command.add_argument("--row-root", action="append", default=[])
        command.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
        command.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate-row")
    validate.add_argument("--row", choices=PAPER_GRID_ROWS, required=True)
    validate.add_argument("--root", required=True)
    validate.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    coupled = subparsers.add_parser("validate-coupled")
    coupled.add_argument("--k-c", type=int, choices=(2, 3), required=True)
    coupled.add_argument("--n-g", type=int, choices=(2, 3, 5), required=True)
    coupled.add_argument("--train-root", required=True)
    coupled.add_argument("--offline-root", required=True)
    args = parser.parse_args()

    if args.command == "validate-row":
        result = validate_row(
            args.row, args.root, manifest_sha256=args.manifest_sha256
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["verdict"] == "PAPER_GRID_ROW_PASS" else 1
    if args.command == "validate-coupled":
        result = validate_coupled_artifact(
            k_c=args.k_c,
            n_g=args.n_g,
            train_root=args.train_root,
            offline_root=args.offline_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["verdict"] == "COUPLED_PAPER_ARTIFACT_PASS" else 1

    roots = _parse_row_roots(args.row_root)
    if args.command == "plan":
        result = build_plan(roots, manifest_sha256=args.manifest_sha256)
        atomic_write_json(args.output, result)
    else:
        result = aggregate(
            roots, args.output, manifest_sha256=args.manifest_sha256
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
